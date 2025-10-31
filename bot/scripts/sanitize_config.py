"""
Backup config.json and replace gemini.api_key with a placeholder to avoid committing real keys.

Usage:
  python scripts/sanitize_config.py

This script will create a backup `config.json.bak_TIMESTAMP` and then replace the gemini.api_key
with `"YOUR_GEMINI_API_KEY"` if it exists.
"""
import json
import os
import time

repo_cfg = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'XAUUSD-bot', 'config.json')
if not os.path.exists(repo_cfg):
    print('config.json not found at expected path:', repo_cfg)
    raise SystemExit(1)

bak = repo_cfg + '.bak_' + str(int(time.time()))
print('Backing up', repo_cfg, 'to', bak)
os.rename(repo_cfg, bak)

with open(bak, 'r', encoding='utf-8') as f:
    cfg = json.load(f)

if 'gemini' in cfg and isinstance(cfg['gemini'], dict) and 'api_key' in cfg['gemini']:
    print('Found gemini.api_key in config.json; sanitizing...')
    cfg['gemini']['api_key'] = 'YOUR_GEMINI_API_KEY'
else:
    print('No gemini.api_key found in config.json; nothing to sanitize.')

with open(repo_cfg, 'w', encoding='utf-8') as f:
    json.dump(cfg, f, indent=2)

print('Sanitization complete. New config.json written. Original backed up.')
