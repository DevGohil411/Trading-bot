import os
import importlib.util
import importlib.machinery

# Ensure env keys are not present for a safe offline import
os.environ.pop('GEMINI_API_KEY', None)
os.environ.pop('GENAI_API_KEY', None)

path = r'd:\bot\XAUUSD-bot\STOCKDATA\modules\llm_sentiment_analyzer.py'
loader = importlib.machinery.SourceFileLoader('llm_test', path)
spec = importlib.util.spec_from_loader(loader.name, loader)
mod = importlib.util.module_from_spec(spec)
try:
    loader.exec_module(mod)
    print('Imported module OK')
    cls = getattr(mod, 'LLMSentimentAnalyzer')
    inst = cls(api_key='')
    print('Instantiated LLMSentimentAnalyzer, model:', getattr(inst, 'model', None))
except Exception as e:
    print('ERROR during import/instantiation:', e)
