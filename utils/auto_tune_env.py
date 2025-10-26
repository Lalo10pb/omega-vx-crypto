#!/usr/bin/env python3
"""
Nightly auto-tuning job for Omega-VX-Crypto.

This script mines scanner logs, derives refreshed guardrails, and updates .env overrides
once a minimum number of fresh samples is available. Run from the repository root, e.g.:

    python3 utils/auto_tune_env.py
    python3 utils/auto_tune_env.py --dry-run --lookback-days 3

The job will:
1. Load `scanner_evaluation_log.csv` and `scanner_near_miss_log.csv`.
2. Filter entries to the requested lookback window.
3. Compute adaptive thresholds (volume, spread, depth, score, buffers).
4. Update `.env` (with a timestamped backup) unless `--dry-run` is passed.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_PATH = REPO_ROOT / ".env"
EVALUATION_LOG_PATH = REPO_ROOT / "scanner_evaluation_log.csv"
NEAR_MISS_LOG_PATH = REPO_ROOT / "scanner_near_miss_log.csv"


@dataclass
class AutoTuneConfig:
    env_path: Path = DEFAULT_ENV_PATH
    lookback_days: int = 7
    min_samples: int = 120
    dry_run: bool = False
    verbose: bool = True
    backup_dir: Path = REPO_ROOT / "env" / "backups"


def _load_csv(path: Path, parse_dates: Optional[List[str]] = None) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception as err:
        raise RuntimeError(f"Failed to load {path}: {err}") from err
    if parse_dates:
        for col in parse_dates:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _coerce_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    if df.empty:
        return df
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _filter_lookback(df: pd.DataFrame, lookback_days: int, timestamp_col: str = "timestamp") -> pd.DataFrame:
    if df.empty or timestamp_col not in df.columns:
        return df
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    filtered = df[df[timestamp_col] >= cutoff]
    return filtered


def _read_env(env_path: Path) -> Dict[str, str]:
    env_data: Dict[str, str] = {}
    if not env_path.exists():
        return env_data
    with env_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            env_data[key.strip()] = value.strip()
    return env_data


def _format_value(key: str, value: float) -> str:
    if "PERCENT" in key or key.endswith("_PCT"):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    if "MULTIPLIER" in key or "SLOPE" in key:
        return f"{value:.4f}".rstrip("0").rstrip(".")
    if "SCORE" in key:
        return f"{value:.2f}"
    if value >= 1000:
        return f"{round(value):.0f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _should_update(old_value: Optional[float], new_value: float, rel_tol: float = 0.05, abs_tol: float = 1e-3) -> bool:
    if old_value is None:
        return True
    if math.isnan(new_value):
        return False
    if abs(new_value - old_value) <= abs_tol:
        return False
    denominator = max(abs(old_value), abs(new_value), 1e-6)
    return abs(new_value - old_value) / denominator >= rel_tol


def _safe_float(value: Optional[str], default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def compute_recommendations(
    evaluation_df: pd.DataFrame,
    near_miss_df: pd.DataFrame,
    config: AutoTuneConfig,
) -> Tuple[Dict[str, float], Dict[str, str]]:
    recommendations: Dict[str, float] = {}
    notes: Dict[str, str] = {}

    if evaluation_df.empty or len(evaluation_df) < config.min_samples:
        raise RuntimeError(
            f"Insufficient evaluation samples ({len(evaluation_df)}) "
            f"for auto-tuning (need ≥ {config.min_samples})."
        )

    volume_series = evaluation_df["quote_volume"].dropna()
    if len(volume_series) >= config.min_samples:
        min_volume = float(volume_series.quantile(0.35))
        min_volume = max(20_000.0, min(min_volume, 500_000.0))
        fallback_volume = max(10_000.0, min_volume * 0.6)
        recommendations["MIN_QUOTE_VOLUME_24H"] = round(min_volume, -2)
        recommendations["MIN_QUOTE_VOLUME_FALLBACK"] = round(fallback_volume, -2)
        notes["MIN_QUOTE_VOLUME_24H"] = f"35th percentile quote volume: {min_volume:,.0f}"
    elif config.verbose:
        notes["MIN_QUOTE_VOLUME_24H"] = "Skipped volume tuning (not enough samples)."

    spread_series = evaluation_df["spread_pct"].dropna()
    if len(spread_series) >= config.min_samples:
        max_spread = float(spread_series.quantile(0.8))
        soft_buffer = float(spread_series.quantile(0.95)) - max_spread
        max_spread = min(max(max_spread, 0.1), 1.0)
        soft_buffer = min(max(soft_buffer, 0.05), 0.5)
        recommendations["MAX_SPREAD_PERCENT"] = round(max_spread, 3)
        recommendations["SOFT_SPREAD_BUFFER"] = round(soft_buffer, 3)
        notes["MAX_SPREAD_PERCENT"] = (
            f"80th percentile spread: {max_spread:.3f}%, soft buffer {soft_buffer:.3f}%"
        )
    elif config.verbose:
        notes["MAX_SPREAD_PERCENT"] = "Skipped spread tuning (not enough samples)."

    depth_series = evaluation_df["depth_ratio"].dropna()
    if len(depth_series) >= config.min_samples:
        min_depth = float(depth_series.quantile(0.25))
        min_depth = min(max(min_depth, 1.05), 2.5)
        soft_depth = max(min_depth - 0.05, 1.0)
        recommendations["MIN_DEPTH_IMBALANCE"] = round(min_depth, 3)
        recommendations["SOFT_DEPTH_BUFFER"] = round(
            max(recommendations["MIN_DEPTH_IMBALANCE"] - soft_depth, 0.0),
            3,
        )
        notes["MIN_DEPTH_IMBALANCE"] = f"25th percentile depth ratio: {min_depth:.3f}"
    elif config.verbose:
        notes["MIN_DEPTH_IMBALANCE"] = "Skipped depth tuning (not enough samples)."

    score_series = evaluation_df["score"].dropna()
    if len(score_series) >= config.min_samples:
        lower_quartile = float(score_series.quantile(0.25))
        min_candidate_score = max(3.0, min(lower_quartile, 8.0))
        recommendations["MIN_CANDIDATE_SCORE"] = round(min_candidate_score, 2)
        notes["MIN_CANDIDATE_SCORE"] = f"25th percentile model score: {lower_quartile:.2f}"

    ema_slope_series = evaluation_df["ema_slope"].dropna()
    if len(ema_slope_series) >= config.min_samples:
        slope_weight = np.clip(float(ema_slope_series.abs().median()) * 40, 0.6, 2.5)
        recommendations["WEIGHT_EMA_SLOPE"] = round(slope_weight, 2)
        notes["WEIGHT_EMA_SLOPE"] = f"Median |EMA slope| implied weight: {slope_weight:.2f}"

    if not near_miss_df.empty:
        soft_volume_series = near_miss_df.loc[
            near_miss_df["reason"].str.contains("volume", case=False, na=False), "quote_volume"
        ].dropna()
        base_volume = recommendations.get("MIN_QUOTE_VOLUME_24H")
        if base_volume and len(soft_volume_series) >= config.min_samples // 3:
            soft_multiplier = float(np.clip(soft_volume_series.median() / base_volume, 0.3, 0.9))
            recommendations["SOFT_VOLUME_MULTIPLIER"] = round(soft_multiplier, 2)
            notes["SOFT_VOLUME_MULTIPLIER"] = f"Median near-miss volume ratio: {soft_multiplier:.2f}"

    return recommendations, notes


def apply_updates(config: AutoTuneConfig, updates: Dict[str, float]) -> Tuple[bool, List[str]]:
    env_path = config.env_path
    current_env = _read_env(env_path)
    updates_to_apply: Dict[str, str] = {}
    summary: List[str] = []
    changed = False

    for key, new_value in updates.items():
        old = _safe_float(current_env.get(key), math.nan)
        if not _should_update(old, new_value):
            continue
        formatted = _format_value(key, new_value)
        updates_to_apply[key] = formatted
        summary.append(f"{key}: {current_env.get(key, 'unset')} → {formatted}")
        changed = True

    if not changed or config.dry_run:
        return changed, summary

    env_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_dir = config.backup_dir
    backup_dir.mkdir(parents=True, exist_ok=True)
    if env_path.exists():
        backup_path = backup_dir / f"{env_path.name}.bak-{timestamp}"
        shutil.copy2(env_path, backup_path)

    lines: List[str] = []
    seen_keys = set()
    if env_path.exists():
        with env_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.rstrip("\n")
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    lines.append(line)
                    continue
                key, _, _ = stripped.partition("=")
                key = key.strip()
                if key in updates_to_apply:
                    lines.append(f"{key}={updates_to_apply[key]}")
                    seen_keys.add(key)
                else:
                    lines.append(line)
    for key, value in updates_to_apply.items():
        if key not in seen_keys:
            lines.append(f"{key}={value}")

    lines.append(f"# Auto-tuned on {timestamp}")
    with env_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    return changed, summary


def run_auto_tune(config: AutoTuneConfig) -> None:
    evaluation_df = _load_csv(EVALUATION_LOG_PATH, parse_dates=["timestamp"])
    evaluation_df = _filter_lookback(evaluation_df, config.lookback_days)
    evaluation_df = _coerce_numeric(
        evaluation_df,
        ["quote_volume", "spread_pct", "depth_ratio", "score", "ema_slope"],
    )

    near_miss_df = _load_csv(NEAR_MISS_LOG_PATH, parse_dates=["timestamp"])
    near_miss_df = _filter_lookback(near_miss_df, config.lookback_days)
    near_miss_df = _coerce_numeric(
        near_miss_df,
        ["quote_volume", "spread_pct", "depth_ratio", "atr_pct"],
    )

    recommendations, notes = compute_recommendations(evaluation_df, near_miss_df, config)
    updated, summary = apply_updates(config, recommendations)

    if config.verbose:
        print("📊 Auto-tune recommendations:")
        for key, value in recommendations.items():
            note = notes.get(key, "")
            print(f"  - {key}: {value:.4f} {note}")
        if summary:
            status = "(dry run)" if config.dry_run else ""
            print(f"\n🛠️ Pending updates {status}:")
            for line in summary:
                print(f"  * {line}")
        else:
            print("\nℹ️ No updates required; thresholds already aligned.")

    if not updated and config.dry_run:
        print("✅ Dry run complete (no changes written).")
    elif not updated:
        print("✅ Auto-tune complete; no thresholds changed.")
    else:
        print(f"✅ Auto-tune applied to {config.env_path}")


def parse_args() -> AutoTuneConfig:
    parser = argparse.ArgumentParser(description="Nightly auto-tune job for Omega-VX-Crypto.")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_PATH), help="Path to the .env file to update.")
    parser.add_argument("--lookback-days", type=int, default=7, help="Number of days to include in sample window.")
    parser.add_argument("--min-samples", type=int, default=120, help="Minimum samples required to tune thresholds.")
    parser.add_argument("--dry-run", action="store_true", help="Run computations without updating .env.")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output.")
    args = parser.parse_args()
    return AutoTuneConfig(
        env_path=Path(args.env_file).resolve(),
        lookback_days=max(1, args.lookback_days),
        min_samples=max(20, args.min_samples),
        dry_run=bool(args.dry_run),
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    config = parse_args()
    run_auto_tune(config)
