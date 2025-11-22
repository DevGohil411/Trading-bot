# Submission Guide (XAUUSD-bot)

Include these files/folders in your submission:
- `README.md` (setup, run, DB init, dump instructions)
- `ENV.example` (copy to `.env.local`)
- `config.example.json` (copy to `config.json` and fill secrets locally)
- `schema.sql` (SQLite schema)
- `DEPENDENCIES.md` (system + pip)
- `db-dump.sql` (optional, if generated)
- `api_server.py`, `STOCKDATA/`, `utils/`, `requirements.txt`
- `logs/` and `trades/` (optional; remove sensitive data if any)

Quick run:
1) `python -m venv venv && venv\\Scripts\\activate && pip install -r requirements.txt`
2) Create DB: `python` snippet in `README.md` (uses `schema.sql`)
3) Start API: `uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload`
4) Start bot: `python -m STOCKDATA` 