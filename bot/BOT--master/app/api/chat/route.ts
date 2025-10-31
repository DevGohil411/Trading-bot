import { google } from "@ai-sdk/google";
import { openai } from "@ai-sdk/openai";
import { convertToModelMessages, streamText } from "ai";

export const maxDuration = 30;

export async function POST(req: Request) {
  const { messages } = await req.json();

  // Use Google Gemini if available, fall back to OpenAI, then mock
  const GOOGLE_API_KEY = process.env.GOOGLE_GENERATIVE_AI_API_KEY;
  const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

  let model;

  if (GOOGLE_API_KEY) {
    model = google("gemini-2.0-flash");
  } else if (OPENAI_API_KEY) {
    model = openai("gpt-4o-mini");
  } else {
    // Fallback to mock response if no API keys configured
    return mockStreamResponse(messages);
  }

  const result = streamText({
    model,
    messages: convertToModelMessages(messages),
    system: `You are SniprX AI Assistant, a helpful trading bot assistant specializing in algorithmic trading, technical analysis, and trading strategies.
    
You help users with:
- Understanding trading strategies like MMXM, Judas Swing, and OTE
- Technical analysis and chart interpretation
- Risk management and position sizing
- MT5 platform features and bot configuration
- Market analysis and trading psychology

Be professional, accurate, and provide actionable insights for traders. Keep responses concise but informative.`,
  });

  return result.toUIMessageStreamResponse();
}

// Mock fallback when no API keys are configured
function mockStreamResponse(messages: any[]) {
  const lastMessage =
    messages[messages.length - 1]?.content?.toLowerCase() || "";

  let reply =
    "Hello! I'm SniprX AI Assistant. How can I help you with trading today?";

  if (lastMessage.includes("mmxm")) {
    reply =
      "MMXM is a market-maker strategy combining liquidity hunts, order-block alignment, and CHOCH confirmation. Use HTF order blocks for bias and price reaction for entries.";
  } else if (lastMessage.includes("judas")) {
    reply =
      "Judas Swing identifies fake breakouts around session opens; wait for the reversal and trade with institutional bias.";
  } else if (lastMessage.includes("ote")) {
    reply =
      "OTE uses Fibonacci 0.62-0.79 zones aligned with order blocks for high-probability entries.";
  } else if (lastMessage.includes("risk") || lastMessage.includes("stop")) {
    reply =
      "Manage risk: size to 1-2% of account, place stops beyond liquidity clusters, and keep a risk-reward plan.";
  }

  const stream = new ReadableStream({
    start(controller) {
      const encoder = new TextEncoder();
      let i = 0;
      const chunkSize = 8;
      const interval = setInterval(() => {
        if (i >= reply.length) {
          controller.close();
          clearInterval(interval);
          return;
        }
        const chunk = reply.slice(i, i + chunkSize);
        controller.enqueue(encoder.encode(chunk));
        i += chunkSize;
      }, 40);
    },
  });

  return new Response(stream, {
    headers: { "Content-Type": "text/plain; charset=utf-8" },
  });
}
