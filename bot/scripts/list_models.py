"""
List available models using google.generativeai. Reads API key from GEMINI_API_KEY or GENAI_API_KEY environment variable.

Usage:
  - Set environment variable in PowerShell:
      $env:GEMINI_API_KEY = 'your_api_key_here'
  - Run:
      python scripts/list_models.py

This script does NOT store your API key; it only uses the env var at runtime.
"""

import os

try:
    import google.generativeai as genai
except Exception as e:
    print('google.generativeai is not installed or failed to import:', e)
    raise SystemExit(1)

key = os.getenv('GEMINI_API_KEY') or os.getenv('GENAI_API_KEY')
if not key:
    print('No API key found. Set GEMINI_API_KEY or GENAI_API_KEY environment variable and rerun.')
    raise SystemExit(1)

genai.configure(api_key=key)

try:
    models = genai.list_models()
    print('Available models (raw):')
    for m in models:
        print(m)
except Exception as e:
    print('Error while listing models:', e)
    raise SystemExit(1)
