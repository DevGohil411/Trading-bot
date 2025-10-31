LLM Setup and Safe Usage

Overview
--------
This document explains how to safely configure the Gemini / Google Generative AI key for the bot, how to test available models, and what to expect in the bot logs when LLM features are enabled or disabled.

Important: Do NOT paste API keys into chat, public issues, or commit them into the repository. If you have accidentally committed a key (this repository contains a `config.json` entry for `gemini.api_key`), rotate or revoke that key immediately.

1) Environment variables (recommended)
------------------------------------
- Preferred: set the key in an environment variable. The analyzer checks these in order:
  - `GEMINI_API_KEY`
  - `GENAI_API_KEY`
- Example (PowerShell, temporary for current session):

```powershell
$env:GEMINI_API_KEY = 'your_real_api_key_here'
```

- To persist, add it to Windows user environment variables (System > Advanced > Environment Variables) or set it in your PowerShell profile.

2) Repository config (not recommended)
-------------------------------------
- The analyzer will attempt to read `config.json` at the repository root if no env var is present. This is convenient but insecure if the repo is shared. If you currently store a key in `config.json`:
  - Remove it and replace with a placeholder like `"api_key": "YOUR_GEMINI_API_KEY"`.
  - Rotate the key in your provider console.

3) Install client library
-------------------------
- Install or upgrade the official client:

```powershell
pip install --upgrade google-generativeai
```

4) List available models (safe test)
-----------------------------------
- Use `scripts/list_models.py` which reads the env var and prints the model list.

```powershell
$env:GEMINI_API_KEY = 'your_real_api_key_here'
python scripts/list_models.py
```

- If the client returns a 400 API_KEY_INVALID, check that the key is valid and has the right permissions for the Generative Language API.

5) Health-check a model
-----------------------
- Use `scripts/test_model.py` to attempt a tiny generation against a model name you choose (or let it try defaults):

```powershell
$env:GEMINI_API_KEY = 'your_real_api_key_here'
python scripts/test_model.py gemini-1.5
```

- If the chosen model succeeds the script prints the response text. If not, try a different model name returned by `list_models.py`.

6) How the bot logs LLM status
------------------------------
- The `llm_sentiment_analyzer` module logs helpful messages. Examples:
  - When package missing (no client installed):
    "google.generativeai package not available; LLM features will be disabled until installed."
  - When no API key found: 
    "Gemini API Key is missing or invalid. LLM sentiment analysis will be disabled."
  - When model initialization succeeds:
    "Model <name> initialized and passed health check."
  - If model fails or no fallback found:
    "Failed to initialize any LLM model. ... LLM sentiment analysis will be disabled."

- When LLM is disabled, `get_sentiment()` returns a neutral result: `{"sentiment": "NEUTRAL", "score": 0.0, "reason": ...}`. The rest of the bot continues to operate normally.

7) If you want automatic config (advanced)
-----------------------------------------
- You may keep keys in an external secrets manager and load them at runtime, or use environment-based deployment (e.g., CI/CD secretes, container secrets). Do NOT store keys in repo files.

8) Troubleshooting checklist
----------------------------
- Ensure `google-generativeai` is installed and up-to-date.
- Ensure the API key is the correct type and not a different API key (e.g., a Google Cloud service account key vs generative API key) and has permissions for `generativelanguage.googleapis.com`.
- Use `scripts/list_models.py` to see which models are available for your key and API version.
- If `list_models` returns a list but health-check calls fail, the model might not support `generate_content()` — check the model metadata or try a different API method.

9) Security note
----------------
- If you pasted a key in a public chat or committed it, rotate/revoke it ASAP.
- Prefer environment variables or secret managers.

10) Where to look in logs for problems
-------------------------------------
- Search for lines in logs with `modules.llm_sentiment_analyzer` or logger name that show `Failed to initialize` or `API Key not found`. Those lines explain why LLM is disabled.


If you want, I can also:
- Add a small command-line helper to remove the gemini key from `config.json` and replace it with a placeholder (I will not modify your repo to insert real keys), or
- Add explicit checks in your main bot initialization that log a clear warning if repo `config.json` still contains a non-placeholder key.

