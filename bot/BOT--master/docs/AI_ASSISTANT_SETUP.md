# AI Assistant Integration Guide

## ✅ Integration Complete!

Your AI assistant from `my-app` has been successfully integrated into your SniprX website at `/chatbot`. The bot now features a modern, ChatGPT-like interface with full conversation threading, markdown support, and file attachments.

## 🎯 What Was Done

### 1. **Dependencies Installed**

The following packages were added to your BOT--master project:

- `@assistant-ui/react` - Core UI components for the assistant
- `@assistant-ui/react-ai-sdk` - AI SDK integration
- `@assistant-ui/react-markdown` - Markdown rendering with syntax highlighting
- `@ai-sdk/google` - Google Gemini AI integration
- `@ai-sdk/openai` - OpenAI integration (fallback option)
- `ai` - Vercel AI SDK for streaming responses
- `remark-gfm` - GitHub Flavored Markdown support
- `zustand` - State management

### 2. **Components Created**

New components in `components/assistant-ui/`:

- `thread.tsx` - Main chat interface with messages, composer, and suggestions
- `markdown-text.tsx` - Markdown renderer for AI responses
- `tooltip-icon-button.tsx` - Reusable button component
- `tool-fallback.tsx` - Display for tool calls
- `attachment.tsx` - File attachment support
- `thread-list.tsx` - Chat thread list
- `threadlist-sidebar.tsx` - Sidebar with thread navigation

### 3. **API Route Updated**

- `/app/api/chat/route.ts` - Updated to use AI SDK with Google Gemini or OpenAI
- Includes fallback mock responses if no API key is configured
- Specialized system prompt for trading assistant context

### 4. **Chatbot Page Replaced**

- `/app/chatbot/page.tsx` - Completely redesigned with the assistant-ui interface
- Maintains your website's theme (dark mode, protected route)
- Features sidebar navigation, breadcrumbs, and clean UI

## 🚀 Setup Instructions

### Step 1: Get an API Key

You have two options:

#### Option A: Google Gemini (Recommended - Free tier available)

1. Go to https://aistudio.google.com/app/apikey
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated key

#### Option B: OpenAI (Paid)

1. Go to https://platform.openai.com/api-keys
2. Sign in and create a new API key
3. Copy the key

### Step 2: Configure Environment Variables

1. In your `BOT--master` folder, create a file named `.env.local` (if it doesn't exist)
2. Add one of these lines depending on which API you chose:

```env
# For Google Gemini (free tier available)
GOOGLE_GENERATIVE_AI_API_KEY=your_key_here

# OR for OpenAI (paid)
OPENAI_API_KEY=your_key_here
```

**Note:** You can also find the template in `ENV.example`

### Step 3: Run Your Development Server

```bash
cd F:\bot\BOT--master
npm run dev
```

### Step 4: Test the AI Assistant

1. Open http://localhost:3000/chatbot
2. You should see the new AI assistant interface
3. Try asking questions like:
   - "Explain MMXM strategy"
   - "How do I manage risk in algorithmic trading?"
   - "What are the best entry points for XAUUSD?"

## 🎨 Theme Matching

The AI assistant interface automatically matches your website's theme:

- ✅ Uses your existing dark mode settings
- ✅ Maintains glassmorphism effects
- ✅ Protected by authentication (ProtectedRoute)
- ✅ Integrated with your navigation breadcrumbs
- ✅ Consistent with your color scheme and spacing

## 📱 Features

### Chat Interface

- **Real-time streaming responses** - Messages appear token-by-token
- **Conversation history** - Multiple chat threads with sidebar navigation
- **Markdown support** - Code blocks, tables, lists, and formatting
- **File attachments** - Upload and preview images
- **Suggested prompts** - Quick-start questions for users
- **Branch picker** - Navigate between different response versions

### User Experience

- **Auto-scroll** - Automatically scrolls to latest message
- **Copy messages** - Copy AI responses to clipboard
- **Edit messages** - Re-edit and resend user messages
- **Refresh responses** - Regenerate AI responses
- **Mobile responsive** - Works on all screen sizes

## 🔧 Customization

### Change Welcome Message

Edit `components/assistant-ui/thread.tsx` around line 73:

```tsx
<m.div className="...">
  Hello! I'm SniprX AI Assistant {/* Change this */}
</m.div>
```

### Change Suggested Questions

Edit `components/assistant-ui/thread.tsx` around line 91:

```tsx
{
  [
    {
      title: "Your custom title",
      label: "your description",
      action: "The full prompt to send",
    },
    // Add more suggestions...
  ];
}
```

### Customize AI System Prompt

Edit `app/api/chat/route.ts` around line 25:

```tsx
system: `You are SniprX AI Assistant...`;
```

### Switch AI Models

#### For Google Gemini:

```tsx
model: google("gemini-2.0-flash"); // Fast, good for chat
model: google("gemini-1.5-pro"); // More powerful, slower
```

#### For OpenAI:

```tsx
model: openai("gpt-4o-mini"); // Fast and cheap
model: openai("gpt-4o"); // Most powerful
```

## 🐛 Troubleshooting

### Issue: "No responses from AI"

- **Check:** Ensure you added the API key to `.env.local`
- **Check:** Restart the dev server after adding environment variables
- **Check:** Look for errors in the browser console (F12)

### Issue: "API rate limit exceeded"

- **Google Gemini:** Free tier has 60 requests per minute
- **OpenAI:** Check your billing and rate limits at platform.openai.com

### Issue: "Components not styled correctly"

- **Check:** Ensure all UI components exist in `components/ui/`
- **Run:** `npm install` to ensure all dependencies are installed

### Issue: "Sidebar not showing"

- **Check:** The sidebar component requires certain UI primitives
- **Verify:** `components/ui/sidebar.tsx` exists and is properly imported

## 💡 Tips

1. **Cost Optimization**: Google Gemini's free tier is generous. Start there before paying for OpenAI.

2. **Testing**: The chatbot works with mock responses even without an API key - great for UI testing.

3. **Production**: Remember to add your API keys to your production environment variables (Vercel, Netlify, etc.)

4. **Multiple Threads**: Users can create multiple chat threads - each maintains its own conversation history.

5. **Custom Tools**: You can extend the AI with custom tools/functions - check the assistant-ui documentation.

## 📚 Resources

- [assistant-ui Documentation](https://www.assistant-ui.com/docs)
- [Vercel AI SDK Docs](https://sdk.vercel.ai/docs)
- [Google Gemini API](https://ai.google.dev)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)

## ✨ Next Steps

1. **Add custom tools** - Integrate with your trading data APIs
2. **Enhance system prompt** - Add more trading-specific knowledge
3. **Connect to database** - Persist chat history across sessions
4. **Add voice input** - Integrate speech-to-text
5. **Custom themes** - Further customize colors and styling

## 🎉 You're All Set!

Your AI assistant is now fully integrated and ready to help your users with trading questions. The interface matches your website's theme and provides a professional, modern chat experience.

For any questions or issues, refer to this guide or check the documentation links above.
