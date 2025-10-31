"""
Test a specific model name with a tiny health-check generation call.
Usage:
  $env:GEMINI_API_KEY = 'your_api_key'
  python scripts/test_model.py gemini-1.5

If model argument is omitted, it will try a short default list.
"""
import os
import sys

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

candidates = sys.argv[1:] or ['gemini-1.5-flash', 'gemini-1.5', 'text-bison-001']

for model_name in candidates:
    print(f'Testing model: {model_name}')
    try:
        m = genai.GenerativeModel(model_name)
        r = m.generate_content('Ping')
        print('Response text:', getattr(r, 'text', None))
        print('--- OK ---')
        break
    except Exception as e:
        print(f'Model {model_name} failed: {e}')
        continue
else:
    print('No candidate model succeeded; try running scripts/list_models.py to see available models.')
