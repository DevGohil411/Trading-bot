{
  "instruction": "Create environment setup guide for Next.js",
  "TargetFile": "d:\\bot\\BOT--master\\server\\SETUP_API.md",
  "CodeMarkdownLanguage": "markdown",
  "CodeEdit"># 🔧 Next.js Server - API Key Setup

## 📋 Environment Variables Setup

Create or update your `server/.env` file with the following content:

```env
PORT=8000
SESSION_SECRET=your-super-secret-session-key-change-in-production
TELEGRAM_BOT_TOKEN=8435934615:AAEDS34JKRkjBcPRrdtOrd5i3NKs4OLFzJY
TELEGRAM_BOT_USERNAME=SniprXBot
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
GEMINI_API_KEY=eOTn0m18D2RUTdvyATakDGoZNzlfKmJR
```

## 🚀 How to Apply

1. **Navigate to server directory:**
   ```bash
   cd server
   ```

2. **Create .env file:**
   ```bash
   cp env.example .env
   ```

3. **Add the Gemini API key:**
   Edit `.env` and add:
   ```
   GEMINI_API_KEY=eOTn0m18D2RUTdvyATakDGoZNzlfKmJR
   ```

4. **Restart the server:**
   ```bash
   # Stop current server (Ctrl+C)
   # Then restart
   npm run dev
   # or
   node server.js
   ```

## ✅ Verification

After setup, you should see in server logs:
```
✅ LLM Sentiment Analyzer initialized successfully with Gemini API.
```

## 📞 Need Help?

If you encounter issues:
1. Check that the `.env` file exists in the `server/` directory
2. Verify the API key is correctly formatted
3. Restart the server after making changes
