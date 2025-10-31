declare global {
  interface Window {
    TradingView: any;
  }
}

"use client"

import { useState, useEffect, useRef } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Search, TrendingUp, Maximize2 } from "lucide-react"

const timeframes = ["1", "5", "15", "60", "240", "1D"]
const timeframeLabels = ["1M", "5M", "15M", "1H", "4H", "D1"]
const popularSymbols = ["XAUUSD", "BTCUSD", "SPX500", "USDJPY", "ETHUSD", "AAPL"]

// TradingView symbol mappings
const symbolMappings: { [key: string]: string } = {
  XAUUSD: "FX:XAUUSD",
  BTCUSD: "BITSTAMP:BTCUSD",
  SPX500: "SP:SPX",
  USDJPY: "FX:USDJPY",
  ETHUSD: "BITSTAMP:ETHUSD",
  AAPL: "NASDAQ:AAPL",
}

export default function TradingChart() {
  const [selectedSymbol, setSelectedSymbol] = useState("XAUUSD")
  const [selectedTimeframe, setSelectedTimeframe] = useState("60")
  const [searchQuery, setSearchQuery] = useState("")
  const [currentPrice, setCurrentPrice] = useState(2045.67)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [scriptLoaded, setScriptLoaded] = useState(false)
  const chartContainerRef = useRef<HTMLDivElement>(null)

  // Load TradingView script
  useEffect(() => {
    if (typeof window !== "undefined" && !window.TradingView) {
      const script = document.createElement("script")
      script.src = "https://s3.tradingview.com/tv.js"
      script.async = true
      script.onload = () => setScriptLoaded(true)
      script.onerror = () => console.error("Failed to load TradingView script")
      document.head.appendChild(script)

      return () => {
        if (document.head.contains(script)) {
          document.head.removeChild(script)
        }
      }
    } else if (window.TradingView) {
      setScriptLoaded(true)
    }
  }, [])

  // Initialize TradingView widget
  useEffect(() => {
    if (scriptLoaded && chartContainerRef.current && window.TradingView) {
      const tradingViewSymbol = symbolMappings[selectedSymbol] || `FX:${selectedSymbol}`

      // Clear previous content
      chartContainerRef.current.innerHTML = ""

      try {
        new window.TradingView.widget({
          width: "100%",
          height: isFullscreen ? "calc(100vh - 200px)" : 400,
          symbol: tradingViewSymbol,
          interval: selectedTimeframe,
          timezone: "Etc/UTC",
          theme: "dark",
          style: "1",
          locale: "en",
          toolbar_bg: "#1a1a1a",
          enable_publishing: false,
          allow_symbol_change: false,
          container_id: chartContainerRef.current.id || "tradingview_chart",
          studies: ["Volume@tv-basicstudies", "RSI@tv-basicstudies"],
          disabled_features: ["use_localstorage_for_settings", "volume_force_overlay"],
          enabled_features: ["study_templates"],
          overrides: {
            "paneProperties.background": "#0a0a0a",
            "paneProperties.vertGridProperties.color": "#1a1a1a",
            "paneProperties.horzGridProperties.color": "#1a1a1a",
            "symbolWatermarkProperties.transparency": 90,
            "scalesProperties.textColor": "#888888",
          },
        })
      } catch (error) {
        console.error("TradingView widget error:", error)
      }
    }
  }, [scriptLoaded, selectedSymbol, selectedTimeframe, isFullscreen])

  // Simulate price updates
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentPrice((prev) => prev + (Math.random() - 0.5) * 2)
    }, 2000)

    return () => clearInterval(interval)
  }, [])

  const filteredSymbols = popularSymbols.filter((symbol) => symbol.toLowerCase().includes(searchQuery.toLowerCase()))

  return (
    <Card className="glassmorphism">
      <CardHeader>
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between space-y-4 sm:space-y-0">
          <CardTitle className="flex items-center">
            <TrendingUp className="w-5 h-5 mr-2 text-primary" />
            Live TradingView Charts
          </CardTitle>
          <div className="flex items-center space-x-2">
            <Badge className="bg-green-500/10 text-green-500 border-green-500/20">Live</Badge>
            <span className="text-sm font-mono">
              {selectedSymbol}: ${currentPrice.toFixed(2)}
            </span>
            <Button size="sm" variant="outline" onClick={() => setIsFullscreen(!isFullscreen)}>
              <Maximize2 className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Symbol Search */}
        <div className="flex flex-col sm:flex-row space-y-2 sm:space-y-0 sm:space-x-2">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4" />
            <Input
              placeholder="Search symbols (XAUUSD, BTCUSD, SPX500...)"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10"
            />
          </div>
        </div>

        {/* Popular Symbols */}
        <div className="flex flex-wrap gap-2">
          {(searchQuery ? filteredSymbols : popularSymbols).map((symbol) => (
            <Button
              key={symbol}
              size="sm"
              variant={selectedSymbol === symbol ? "default" : "outline"}
              onClick={() => setSelectedSymbol(symbol)}
              className={selectedSymbol === symbol ? "gradient-primary text-black" : ""}
            >
              {symbol}
            </Button>
          ))}
        </div>

        {/* Timeframe Selection */}
        <div className="flex space-x-2">
          {timeframes.map((tf, index) => (
            <Button
              key={tf}
              size="sm"
              variant={selectedTimeframe === tf ? "default" : "outline"}
              onClick={() => setSelectedTimeframe(tf)}
              className={selectedTimeframe === tf ? "gradient-primary text-black" : ""}
            >
              {timeframeLabels[index]}
            </Button>
          ))}
        </div>

        {/* TradingView Chart */}
        <div
          className={`${isFullscreen ? "fixed inset-0 z-50 bg-background p-4" : "h-96"} rounded-2xl overflow-hidden`}
        >
          {isFullscreen && (
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold">{selectedSymbol} Chart</h3>
              <Button onClick={() => setIsFullscreen(false)}>Close Fullscreen</Button>
            </div>
          )}

          {!scriptLoaded ? (
            <div className="w-full h-full bg-muted/20 rounded-2xl flex items-center justify-center">
              <div className="text-center">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
                <p className="text-muted-foreground">Loading TradingView Chart...</p>
              </div>
            </div>
          ) : (
            <div ref={chartContainerRef} id="tradingview_chart" className="w-full h-full rounded-2xl" />
          )}
        </div>

        {/* Chart Info */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <span className="text-muted-foreground">Open:</span>
            <div className="font-mono">{(currentPrice - 5.23).toFixed(2)}</div>
          </div>
          <div>
            <span className="text-muted-foreground">High:</span>
            <div className="font-mono text-green-500">{(currentPrice + 2.45).toFixed(2)}</div>
          </div>
          <div>
            <span className="text-muted-foreground">Low:</span>
            <div className="font-mono text-red-500">{(currentPrice - 8.12).toFixed(2)}</div>
          </div>
          <div>
            <span className="text-muted-foreground">Volume:</span>
            <div className="font-mono">1.2M</div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
