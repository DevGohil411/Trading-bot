try:
    import google.generativeai as genai
except Exception:
    genai = None
    # We can't use llm_logger here because it's defined after imports in this module,
    # but the module-level logger is created next; use a basic fallback print as well.
    try:
        import logging as _logging
        _log = _logging.getLogger(__name__)
        _log.warning("google.generativeai package not available; LLM features will be disabled until installed.")
    except Exception:
        print("Warning: google.generativeai package not available; LLM features disabled.")
import logging
import json
import os
import time

# Setup a dedicated logger for LLM sentiment analyzer
llm_logger = logging.getLogger(__name__)
llm_logger.setLevel(logging.INFO) # Default level, can be configured later

# Ensure handlers are attached only once for this logger
if not llm_logger.handlers:
    # This handler might be managed by main.py's logging setup,
    # but defining a console handler here as a fallback or for module-specific output
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    console_handler.setFormatter(formatter)
    llm_logger.addHandler(console_handler)
    llm_logger.propagate = False # Prevent duplication if main logger also captures

class LLMSentimentAnalyzer:
    def __init__(self, api_key: str):
        # Prefer environment variable if provided; fall back to passed api_key
        env_key = os.getenv("GEMINI_API_KEY") or os.getenv("GENAI_API_KEY")
        key_to_use = env_key or api_key

        # If still missing, try to read from repo config.json (do not commit keys!)
        if not key_to_use:
            try:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                cfg_path = os.path.join(current_dir, '..', '..', 'config.json')
                if os.path.exists(cfg_path):
                    with open(cfg_path, 'r') as f:
                        cfg = json.load(f)
                    cfg_key = cfg.get('gemini', {}).get('api_key')
                    if cfg_key and cfg_key != 'YOUR_GEMINI_API_KEY':
                        key_to_use = cfg_key
                        llm_logger.info('Loaded GEMINI API key from config.json')
            except Exception:
                # Silent fallback - just continue with None and let subsequent checks handle it
                pass

        if not key_to_use or key_to_use == "YOUR_GEMINI_API_KEY":
            llm_logger.error("Gemini API Key is missing or invalid. LLM sentiment analysis will be disabled.")
            self.model = None
            return

        # Configure the client
        if genai is None:
            llm_logger.error("google.generativeai package is not installed. Install it with 'pip install google-generativeai' to enable LLM features. Disabling LLM.")
            self.model = None
            return

        try:
            genai.configure(api_key=key_to_use)
        except Exception as e:
            llm_logger.error(f"Failed to configure generative AI client: {e}. Disabling LLM.")
            self.model = None
            return

        # Candidate model names to try (in order). This list can be extended as needed.
        candidate_models = [
            'gemini-2.5-flash',
            'gemini-2.5',
            'gemini-2.5-pro',
            'gemini-1.5-flash',
            'gemini-1.5',
            'text-bison-001',
            'bison'
        ]

        # Try to initialize a working model. We'll try candidates and fall back to listing available models.
        self.model = None
        last_exc = None
        for model_name in candidate_models:
            try:
                llm_logger.info(f"Attempting to initialize model: {model_name}")
                self.model = genai.GenerativeModel(model_name)
                # quick health-check generation (low-cost prompt)
                try:
                    resp = self.model.generate_content("Ping")
                    text = getattr(resp, 'text', None)
                    if not text:
                        raise ValueError("Empty response from model health check")
                except Exception as e:
                    llm_logger.warning(f"Model {model_name} initialized but health check failed: {e}")
                    # try next candidate
                    self.model = None
                    last_exc = e
                    continue

                llm_logger.info(f"Model {model_name} initialized and passed health check.")
                break

            except Exception as e:
                llm_logger.warning(f"Could not initialize model {model_name}: {e}")
                self.model = None
                last_exc = e

        # If no candidate worked, try listing available models to pick a compatible fallback
        if not self.model:
            try:
                llm_logger.info("Attempting to list available models to find a compatible fallback...")
                models = genai.list_models()
                # models may be a list of objects; try to extract names
                available_names = []
                for m in models:
                    name = None
                    if isinstance(m, dict):
                        name = m.get('name') or m.get('model')
                    else:
                        # try attribute access
                        name = getattr(m, 'name', None) or getattr(m, 'model', None)
                    if name:
                        available_names.append(name)

                llm_logger.info(f"Available models: {available_names}")

                # prefer any model that contains 'gemini' or 'bison'
                fallback = None
                for n in available_names:
                    lname = n.lower()
                    if 'gemini' in lname or 'bison' in lname or 'bison' in lname:
                        fallback = n
                        break

                if fallback:
                    llm_logger.info(f"Attempting to initialize fallback model from list: {fallback}")
                    try:
                        self.model = genai.GenerativeModel(fallback)
                        resp = self.model.generate_content("Ping")
                        if not getattr(resp, 'text', None):
                            raise ValueError("Fallback model health check returned empty response")
                        llm_logger.info(f"Fallback model {fallback} initialized successfully.")
                    except Exception as e:
                        llm_logger.error(f"Fallback model {fallback} failed health check: {e}")
                        self.model = None
                else:
                    llm_logger.error("No suitable fallback model found in model list.")

            except Exception as e:
                llm_logger.error(f"Failed to list or inspect available models: {e}")

        if not self.model:
            # Final fallback: disable LLM features but keep the program running
            llm_logger.error(f"Failed to initialize any LLM model. Last exception: {last_exc}. LLM sentiment analysis will be disabled.")
            self.model = None

    def get_sentiment(self, text: str) -> dict:
        """
        Analyzes the sentiment of the given text using the Google Gemini model.
        Returns a dictionary with 'sentiment' (BULLISH, BEARISH, NEUTRAL) and 'score' (-1.0 to 1.0).
        """
        if not self.model:
            llm_logger.warning("LLM model is not initialized. Cannot perform sentiment analysis.")
            return {"sentiment": "NEUTRAL", "score": 0.0, "reason": "LLM not initialized."}

        # A more robust prompt designed to ensure JSON output and focus on financial impact.
        prompt = (
            "You are a financial analyst AI. Your task is to analyze the sentiment of the provided text "
            "based on its likely impact on Gold (XAUUSD) and major USD currency pairs. "
            "Your response MUST be a single, clean JSON object and nothing else. Do not add any extra text or markdown formatting.\n\n"
            "Analyze the following text:\n"
            f"'{text}'\n\n"
            "Provide your analysis in this exact JSON format:\n"
            "{\n"
            '  "sentiment": "BULLISH | BEARISH | NEUTRAL",\n'
            '  "score": <a float between -1.0 and 1.0>,\n'
            '  "explanation": "<A brief explanation of your reasoning>"\n'
            "}"
        )

        try:
            # Use generation_config for more controlled output
            response = self.model.generate_content(prompt,
                                                  generation_config=genai.GenerationConfig(
                                                      temperature=0.1, # Very low temperature for consistency
                                                      max_output_tokens=200 # Limit output length
                                                  ))
            
            # Clean up the response to ensure it's valid JSON
            response_text = response.text.strip()
            # Remove markdown fences if the model adds them
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            # Attempt to parse the JSON response
            sentiment_data = json.loads(response_text)
            
            sentiment = sentiment_data.get("sentiment", "NEUTRAL").upper()
            score = float(sentiment_data.get("score", 0.0))
            explanation = sentiment_data.get("explanation", "")

            # Basic validation of score and sentiment
            if not (-1.0 <= score <= 1.0):
                llm_logger.warning(f"LLM returned invalid score: {score}. Clamping to range.")
                score = max(-1.0, min(1.0, score))
            
            if sentiment not in ["BULLISH", "BEARISH", "NEUTRAL"]:
                llm_logger.warning(f"LLM returned unknown sentiment: {sentiment}. Setting to NEUTRAL and score 0.0.")
                sentiment = "NEUTRAL"
                score = 0.0 

            return {"sentiment": sentiment, "score": score, "reason": explanation}

        except json.JSONDecodeError:
            llm_logger.error(f"LLM response was not valid JSON: '{response_text}'. Returning neutral sentiment.", exc_info=True)
            return {"sentiment": "NEUTRAL", "score": 0.0, "reason": "Invalid LLM response format."}
        except Exception as e:
            llm_logger.error(f"Error calling Gemini API for sentiment analysis: {e}. Returning neutral sentiment.", exc_info=True)
            return {"sentiment": "NEUTRAL", "score": 0.0, "reason": f"API Error: {e}"}

# Example Usage (for testing this module independently)
if __name__ == "__main__":
    # For independent testing, load API key from config.json or environment variable
    try:
        # Adjust path as needed if you run this directly from modules folder
        # For this example, assuming it's in a subfolder from where script is run
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, '..', '..', 'config.json') # Go up two dirs to find config.json

        with open(config_path, 'r') as f:
            test_config = json.load(f)
        test_api_key = test_config.get("gemini", {}).get("api_key")
        
        if not test_api_key or test_api_key == "YOUR_GEMINI_API_KEY":
            raise ValueError("API Key not found in config.json or is default value. Please update config.json.")
            
    except FileNotFoundError:
        print(f"Error: config.json not found at {config_path}. Please create it or adjust path for testing.")
        test_api_key = os.getenv("GEMINI_API_KEY") # Try environment variable as fallback
        if not test_api_key:
            print("GEMINI_API_KEY environment variable not set either. Exiting test.")
            exit()
    except Exception as e:
        print(f"Error loading config for test: {e}. Exiting.")
        exit()

    sentiment_analyzer = LLMSentimentAnalyzer(test_api_key)

    if sentiment_analyzer.model:
        print("\n--- Testing LLM Sentiment Analyzer ---")
        
        test_news = [
            "U.S. inflation data comes in much higher than expected, sparking fears of rate hikes.", # Bearish for Gold, Bullish for USD
            "Gold prices surge as global economic uncertainty increases, safe-haven demand rises.", # Bullish for Gold
            "Central bank holds interest rates steady, as expected, with no new forward guidance.", # Neutral
            "Unexpected job losses reported, signalling potential recession.", # Bearish for USD, Bullish for Gold
            "Strong manufacturing PMI boosts investor confidence and drives equity markets higher." # Bullish for equities/USD, Bearish for Gold
        ]

        for i, news_text in enumerate(test_news):
            print(f"\n--- Test Case {i+1} ---")
            print(f"News: '{news_text}'")
            sentiment_result = sentiment_analyzer.get_sentiment(news_text)
            print(f"Sentiment: {sentiment_result['sentiment']}, Score: {sentiment_result['score']:.2f}")
            print(f"Reason: {sentiment_result['reason']}")
            time.sleep(3) # Small delay to respect potential API rate limits
    else:
        print("LLM Sentiment Analyzer could not be initialized due to API key or model issues. Skipping tests.")