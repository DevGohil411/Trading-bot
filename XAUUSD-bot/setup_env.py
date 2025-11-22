{
  "instruction": "Create setup script to initialize API keys",
  "TargetFile": "d:\\bot\\XAUUSD-bot\\setup_env.py",
  "CodeMarkdownLanguage": "python",
  "CodeEdit">#!/usr/bin/env python3
"""
Setup script to initialize environment variables for the trading bot.
Run this script to create the .env file with your API keys.
"""

import os
import sys

def create_env_file():
    """Create .env file with API keys"""

    env_content = """# Trading Bot Environment Variables
NEXT_PUBLIC_API_URL=http://localhost:8000
TELEGRAM_BOT_TOKEN=
TELEGRAM_BOT_USERNAME=
GEMINI_API_KEY=AIzaSyD43qNjRNmecJyNiiqF5Yerri27D9U89Y8
"""

    env_file_path = os.path.join(os.path.dirname(__file__), '.env')

    try:
        with open(env_file_path, 'w') as f:
            f.write(env_content)
        print("✅ .env file created successfully!"        print(f"📁 Location: {env_file_path}")
        print("\n📝 Note: Please update the following if needed:")
        print("   - TELEGRAM_BOT_TOKEN (if using Telegram notifications)")
        print("   - TELEGRAM_BOT_USERNAME (if using Telegram notifications)")
        print("   - GEMINI_API_KEY (✅ Already configured)")
        return True
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return False

def validate_api_keys():
    """Validate that required API keys are present"""

    print("\n🔍 Validating API keys...")

    gemini_key = os.environ.get('GEMINI_API_KEY')
    if gemini_key:
        print("✅ Gemini API Key: Found")
        if len(gemini_key) > 10:  # Basic validation
            print("✅ Gemini API Key: Format looks correct")
        else:
            print("⚠️  Gemini API Key: Format may be incorrect")
    else:
        print("❌ Gemini API Key: Not found in environment")

    return gemini_key is not None

def main():
    """Main setup function"""

    print("🚀 SniprX Trading Bot - Environment Setup")
    print("=" * 50)

    # Create .env file
    if create_env_file():
        print("\n✅ Environment setup completed!")

        # Validate API keys
        if validate_api_keys():
            print("\n🎉 All API keys are properly configured!")
            print("💡 You can now run your trading bot with:")
            print("   python -m STOCKDATA")
        else:
            print("\n⚠️  Some API keys are missing. Please check the .env file.")
    else:
        print("\n❌ Environment setup failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
