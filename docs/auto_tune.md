# Nightly Auto-Tune Job

The bot now ships with an adaptive tuner that mines the scanner logs and refreshes key guardrails each night.

## What It Does
- Reads `scanner_evaluation_log.csv` and `scanner_near_miss_log.csv` for the last **7 days**.
- Requires at least **120** evaluation samples before attempting to tune.
- Derives refreshed values for:
  - `MIN_QUOTE_VOLUME_24H` and `MIN_QUOTE_VOLUME_FALLBACK`
  - `MAX_SPREAD_PERCENT` and `SOFT_SPREAD_BUFFER`
  - `MIN_DEPTH_IMBALANCE` and `SOFT_DEPTH_BUFFER`
  - `MIN_CANDIDATE_SCORE`
  - `SOFT_VOLUME_MULTIPLIER`
  - `WEIGHT_EMA_SLOPE`
- Writes updates directly into `.env`, creating a timestamped backup under `env/backups/`.

## Running Manually
From the repository root:

```bash
python3 utils/auto_tune_env.py              # apply updates when thresholds move enough
python3 utils/auto_tune_env.py --dry-run    # preview changes without writing
python3 utils/auto_tune_env.py --lookback-days 3 --min-samples 60
```

## Scheduled Job (Render)
`render.yaml` now defines a cron job:

```yaml
cronJobs:
  - name: omega-vx-auto-tune
    schedule: "30 4 * * *"
    command: "python3 utils/auto_tune_env.py --env-file .env"
```

This runs daily at 04:30 UTC. Adjust the schedule as needed. The job shares the same build/runtime environment as the worker service, so dependencies from `requirements.txt` are available.

## Operational Notes
- Backups live under `env/backups/.env.bak-YYYYMMDD-HHMMSS`.
- The tuner skips updates when thresholds have not shifted by at least 5%.
- If the log files are empty or too small, the job exits without touching `.env`.
- Keep the bot in dry-run mode the first day after enabling auto-tuning to observe the new thresholds before trading live capital.
