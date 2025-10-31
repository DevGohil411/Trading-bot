# Quick Start - AI Assistant

## 🚀 Get Your AI Bot Running in 3 Steps

### Step 1: Add API Key (2 minutes)

Create `.env.local` in your BOT--master folder:

```bash
# Choose ONE of these:

# Option 1: Google Gemini (FREE) - Get from https://aistudio.google.com/app/apikey
GOOGLE_GENERATIVE_AI_API_KEY=your_google_api_key_here

# Option 2: OpenAI (PAID) - Get from https://platform.openai.com/api-keys
OPENAI_API_KEY=your_openai_api_key_here
```

### Step 2: Start Development Server

```powershell
cd F:\bot\BOT--master
npm run dev
```

### Step 3: Test It!

Open: http://localhost:3000/chatbot

Try asking:

- "Explain MMXM strategy"
- "How to manage risk in trading?"
- "What's the best time to trade XAUUSD?"

## ✅ That's It!

Your AI assistant is now live on your website at `/chatbot`

**No API Key Yet?** The bot will still work with mock responses for testing!

---

📖 **For detailed info**, see: `docs/AI_ASSISTANT_SETUP.md`
