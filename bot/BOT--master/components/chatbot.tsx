"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { MessageCircle, X, Send, Bot } from "lucide-react"

const botResponses: { [key: string]: string } = {
  mmxm: "MMXM (Market Maker X Manipulation) combines market maker manipulation detection, liquidity sweeps, and CHOCH confirmation for sharp entries. It uses volume filters and ML algorithms to identify high-probability reversal zones where institutional players are manipulating price before the real move.",
  judas:
    "Judas Swing Setup detects fake breakouts during key sessions (London/NY open). It identifies when price sweeps liquidity above/below key levels, then reverses. The strategy waits for the 'Judas swing' - a false move that traps retail traders before the real institutional move begins.",
  ote: "Optimal Trade Entry (OTE) uses Fibonacci retracements (0.62-0.79 zone) combined with Order Block alignment and CHOCH confirmation. It waits for institutional retracement into premium/discount zones before entering with the higher timeframe bias.",
  amd: "AMD (Accumulation-Manipulation-Distribution) follows the institutional market cycle. Accumulation phase builds positions, Manipulation creates false moves, Distribution releases positions. Perfect for swing trades during high/low volatility transitions.",
  "order block":
    "Order Block + CHOCH Combo identifies institutional order blocks on higher timeframes, then waits for Change of Character (CHOCH) on lower timeframes to confirm entry. It combines structure analysis with momentum confirmation.",
  msb: "MSB Retest strategy enters after Market Structure Breaks, waiting for price to retest the broken level. It looks for entries into Order Blocks or Fair Value Gaps with proper trend alignment and volume confirmation.",
  mmc: "MMC Combo is our most comprehensive strategy, combining Order Blocks, CHOCH, liquidity grabs, and engulfing patterns. It's designed specifically for XAUUSD volatility with multiple confirmation filters for high-accuracy entries.",
  "ny session":
    "The New York session (8 AM - 5 PM EST) is ideal for MMXM and Judas Swing strategies due to high volatility and institutional activity. OTE works well during overlap periods.",
  "best strategy":
    "For beginners, start with MSB Retest. For advanced traders, MMXM and Judas Swing offer the highest win rates. OTE is perfect for swing traders.",
}

export default function ChatBot() {
  const [isOpen, setIsOpen] = useState(false)
  const [messages, setMessages] = useState([
    {
      type: "bot",
      content:
        "Hi! I'm your SniprX AI assistant. Ask me about our strategies like 'What is MMXM strategy?' or 'Which strategy is best for NY session?'",
      time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    },
  ])
  const [input, setInput] = useState("")

  const handleSend = () => {
    if (!input.trim()) return

    const userMessage = {
      type: "user",
      content: input,
      time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    }
    setMessages((prev) => [...prev, userMessage])

    // Simple keyword matching for bot responses
    const lowerInput = input.toLowerCase()
    let botResponse =
      "I can help explain our strategies like MMXM, Judas Swing Setup, OTE, AMD, Order Block + CHOCH, MSB Retest, and MMC Combo. I can also advise on session timing and strategy selection. What would you like to know?"

    for (const [keyword, response] of Object.entries(botResponses)) {
      if (lowerInput.includes(keyword)) {
        botResponse = response
        break
      }
    }

    setTimeout(() => {
      setMessages((prev) => [
        ...prev,
        {
          type: "bot",
          content: botResponse,
          time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        },
      ])
    }, 1000)

    setInput("")
  }

  return (
    <>
      {/* Floating Chat Button */}
      <Button
        className="fixed bottom-6 right-6 w-14 h-14 rounded-full gradient-primary text-black shadow-2xl hover:opacity-90 z-50 animate-pulse-slow"
        onClick={() => setIsOpen(true)}
      >
        <MessageCircle className="w-6 h-6" />
      </Button>

      {/* Chat Window */}
      {isOpen && (
        <Card className="fixed bottom-24 right-6 w-80 h-96 glassmorphism z-50 flex flex-col border-primary/20">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-lg flex items-center">
                <Bot className="w-5 h-5 mr-2 text-primary" />
                SniprX AI
              </CardTitle>
              <Button size="sm" variant="ghost" onClick={() => setIsOpen(false)}>
                <X className="w-4 h-4" />
              </Button>
            </div>
          </CardHeader>
          <CardContent className="flex-1 flex flex-col">
            <div className="flex-1 overflow-y-auto space-y-3 mb-4">
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`p-3 rounded-lg text-sm ${
                    message.type === "bot"
                      ? "bg-muted/50 text-muted-foreground"
                      : "bg-primary/20 text-primary-foreground ml-8"
                  }`}
                >
                  <div className="flex items-start space-x-2">
                    {message.type === "bot" && <Bot className="w-4 h-4 mt-1 flex-shrink-0 text-primary" />}
                    <div className="flex-1">
                      <p>{message.content}</p>
                      <p className="text-xs opacity-70 mt-1">{message.time}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
            <div className="flex space-x-2">
              <Input
                placeholder="Ask about strategies..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => e.key === "Enter" && handleSend()}
                className="flex-1"
              />
              <Button size="sm" onClick={handleSend} className="gradient-primary text-black">
                <Send className="w-4 h-4" />
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </>
  )
}
