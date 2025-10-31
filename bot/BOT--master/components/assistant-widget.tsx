"use client";

import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { MessageCircle, X, Send, Bot } from "lucide-react";

// Minimal, self-contained assistant widget that mimics ChatGPT-like behavior using a
// mock streaming response. It intentionally avoids heavy external deps so it's easy to run.
export default function AssistantWidget() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<
    Array<{ type: "bot" | "user"; content: string; time: string }>
  >(() => {
    try {
      const raw = localStorage.getItem("sniprx_assistant_messages");
      if (raw) return JSON.parse(raw);
    } catch (e) {
      // ignore
    }
    return [
      {
        type: "bot",
        content:
          "Hi! I'm your SniprX assistant. Ask me about strategies or the dashboard.",
        time: new Date().toLocaleTimeString([], {
          hour: "2-digit",
          minute: "2-digit",
        }),
      },
    ];
  });
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessionName, setSessionName] = useState<string>("Conversation");
  const streamingRef = useRef<NodeJS.Timeout | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  // persist messages
  useEffect(() => {
    try {
      localStorage.setItem(
        "sniprx_assistant_messages",
        JSON.stringify(messages)
      );
    } catch (e) {
      // ignore
    }
  }, [messages]);

  // auto-scroll to bottom when messages change
  useEffect(() => {
    if (!scrollRef.current) return;
    scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages, isOpen]);

  // load or create a server session
  useEffect(() => {
    const load = async () => {
      try {
        // check if an existing session id is stored
        const sid = localStorage.getItem("sniprx_assistant_session");
        if (sid) {
          const r = await fetch(`/api/assistant?id=${sid}`);
          if (r.ok) {
            const j = await r.json();
            setSessionId(sid);
            setSessionName(j.session?.name || "Conversation");
            // optionally load server messages into UI
            if (j.session?.messages) setMessages(j.session.messages);
            return;
          }
        }

        // create a new session
        const c = await fetch("/api/assistant", { method: "POST" });
        const cj = await c.json();
        if (cj.id) {
          setSessionId(cj.id);
          localStorage.setItem("sniprx_assistant_session", cj.id);
        }
      } catch (e) {
        // ignore
      }
    };
    load();
  }, []);

  const sendMessage = async (text: string) => {
    const userMsg = {
      type: "user" as const,
      content: text,
      time: new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      }),
    };
    setMessages((s) => [...s, userMsg]);

    // Mock streaming reply: gradually append characters for UX

    // create server-side bot message placeholder
    const botMsgIndex = (messages.length || 0) + 1;
    setMessages((s) => [
      ...s,
      {
        type: "bot",
        content: "",
        time: new Date().toLocaleTimeString([], {
          hour: "2-digit",
          minute: "2-digit",
        }),
      },
    ]);

    // Build conversation for context
    const conv = messages.map((m) => ({
      role: m.type === "user" ? "user" : "assistant",
      content: m.content,
    }));
    conv.push({ role: "user", content: text });

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: conv, sessionId }),
      });

      if (!res.body) {
        // fallback to mock
        const replyText = generateMockReply(text);
        setMessages((prev) => {
          const copy = [...prev];
          copy[copy.length - 1].content = replyText;
          return copy;
        });
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let done = false;
      while (!done) {
        const { value, done: d } = await reader.read();
        done = d;
        if (value) {
          const chunk = decoder.decode(value);
          setMessages((prev) => {
            const copy = [...prev];
            const last = copy[copy.length - 1];
            last.content = (last.content || "") + chunk;
            return copy;
          });
        }
      }
    } catch (err) {
      // fallback
      const replyText = generateMockReply(text);
      setMessages((prev) => {
        const copy = [...prev];
        copy[copy.length - 1].content = replyText;
        return copy;
      });
    } finally {
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  };

  const handleSend = () => {
    if (!input.trim()) return;
    const txt = input.trim();
    setInput("");
    sendMessage(txt);
  };

  return (
    <>
      <Button
        aria-label="Open assistant"
        className="fixed bottom-6 right-6 w-14 h-14 rounded-full gradient-primary text-black shadow-2xl hover:opacity-90 z-50"
        onClick={() => setIsOpen(true)}
      >
        <MessageCircle className="w-6 h-6" />
      </Button>

      {isOpen && (
        <Card className="fixed bottom-24 right-6 w-96 h-[520px] glassmorphism z-50 flex flex-col border-primary/20 shadow-lg transition-transform duration-200 ease-out transform translate-y-0">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle className="text-lg flex items-center">
                <Bot className="w-5 h-5 mr-2 text-primary" />
                SniprX Assistant
              </CardTitle>
              <div className="flex items-center gap-2">
                <input
                  className="bg-transparent text-sm text-foreground px-2 py-1 rounded"
                  value={sessionName}
                  onChange={(e) => setSessionName(e.target.value)}
                />
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={async () => {
                    if (!sessionId) return;
                    // save name server-side
                    await fetch(`/api/assistant?id=${sessionId}`, {
                      method: "POST",
                      headers: { "Content-Type": "application/json" },
                      body: JSON.stringify({ name: sessionName }),
                    });
                  }}
                >
                  Save
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={async () => {
                    if (!sessionId) return;
                    await fetch(`/api/assistant?id=${sessionId}`, {
                      method: "DELETE",
                    });
                    setMessages([]);
                    localStorage.removeItem("sniprx_assistant_session");
                    setSessionId(null);
                  }}
                >
                  Clear
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => setIsOpen(false)}
                >
                  <X className="w-4 h-4" />
                </Button>
              </div>
            </div>
          </CardHeader>

          <CardContent className="flex-1 flex flex-col p-3">
            <div
              ref={scrollRef}
              className="flex-1 overflow-y-auto space-y-3 mb-3"
              id="assistant-scroll"
            >
              {messages.map((m, i) => (
                <div
                  key={i}
                  className={`p-3 rounded-lg text-sm ${
                    m.type === "bot"
                      ? "bg-muted/50 text-muted-foreground"
                      : "bg-primary/20 text-primary-foreground ml-8"
                  }`}
                >
                  <div className="flex items-start space-x-2">
                    {m.type === "bot" && (
                      <Bot className="w-4 h-4 mt-1 flex-shrink-0 text-primary" />
                    )}
                    <div className="flex-1">
                      <p className="whitespace-pre-wrap">{m.content}</p>
                      <p className="text-xs opacity-70 mt-1">{m.time}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <div className="flex gap-2">
              <Input
                ref={inputRef}
                placeholder="Ask about strategies..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSend()}
              />
              <Button
                onClick={handleSend}
                className="gradient-primary text-black"
                aria-label="Send"
              >
                <Send className="w-4 h-4" />
              </Button>
            </div>
            <div className="text-xs text-muted-foreground mt-2">
              Tip: Try 'Explain MMXM' or 'Which strategy for NY session?'
            </div>
          </CardContent>
        </Card>
      )}
    </>
  );
}

function generateMockReply(user: string) {
  const lc = user.toLowerCase();
  if (lc.includes("mmxm"))
    return "MMXM is a market-maker focused strategy that combines liquidity hunts and CHOCH signals. Use higher timeframe order blocks for bias.";
  if (lc.includes("judas"))
    return "Judas Swing finds fake breakouts and enters after the reversal. Watch for liquidity sweeps around session opens.";
  if (lc.includes("ote"))
    return "OTE uses Fibonacci retracements and aligns with order blocks for higher probability entries.";
  if (lc.includes("risk") || lc.includes("stop"))
    return "Manage risk by sizing trades to 1-2% of account and placing stops beyond liquidity clusters.";
  if (lc.includes("ny session"))
    return "NY session has high liquidity — MMXM and Judas Swing perform well due to institutional participation.";
  // default reply
  return "I can explain strategies (MMXM, Judas Swing, OTE, MMC), give session guidance, and help with platform features. Ask me a specific question like 'Explain MMXM' or 'Which strategy for NY session?'.";
}
