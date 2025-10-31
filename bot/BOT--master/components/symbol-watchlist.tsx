"use client"

import { useEffect, useRef, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { TrendingUp } from "lucide-react"

export default function SymbolWatchlist() {
  const containerRef = useRef<HTMLDivElement>(null)
  const [isLoaded, setIsLoaded] = useState(false)

  useEffect(() => {
    if (typeof window === "undefined") return

    const loadWidget = () => {
      if (containerRef.current) {
        containerRef.current.innerHTML = `
          <div class="tradingview-widget-container__widget"></div>
        `

        const script = document.createElement("script")
        script.type = "text/javascript"
        script.src = "https://s3.tradingview.com/external-embedding/embed-widget-symbol-overview.js"
        script.async = true
        script.innerHTML = JSON.stringify({
          symbols: [
            ["Gold|1D|FX:XAUUSD"],
            ["Bitcoin|1D|BITSTAMP:BTCUSD"],
            ["S&P 500|1D|SP:SPX"],
          ],
          chartOnly: false,
          width: "100%",
          height: "400",
          locale: "en",
          colorTheme: "dark",
          autosize: false,
          showVolume: false,
          showMA: false,
          hideDateRanges: false,
          hideMarketStatus: false,
          hideSymbolLogo: false,
          scalePosition: "right",
          scaleMode: "Normal",
          fontFamily: "-apple-system, BlinkMacSystemFont, Trebuchet MS, Roboto, Ubuntu, sans-serif",
          fontSize: "10",
          noTimeScale: false,
          valuesTracking: "1",
          changeMode: "price-and-percent",
          chartType: "area",
          maLineColor: "#2962FF",
          maLineWidth: 1,
          maLength: 9,
          lineWidth: 2,
          lineType: 0,
          dateRanges: ["1d|1", "1m|30", "3m|60", "12m|1D", "60m|1W", "all|1M"],
        })

        script.onload = () => setIsLoaded(true)
        script.onerror = () => {
          console.error("Failed to load TradingView symbol overview widget")
          setIsLoaded(false)
        }

        containerRef.current.appendChild(script)
      }
    }

    const timer = setTimeout(loadWidget, 100)
    return () => clearTimeout(timer)
  }, [])

  return (
    <Card className="glassmorphism">
      <CardHeader>
        <CardTitle className="flex items-center">
          <TrendingUp className="w-5 h-5 mr-2 text-primary" />
          Market Watchlist
        </CardTitle>
      </CardHeader>
      <CardContent>
        {!isLoaded ? (
          <div className="w-full h-96 bg-muted/20 rounded-lg flex items-center justify-center">
            <div className="text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
              <p className="text-muted-foreground">Loading Watchlist...</p>
            </div>
          </div>
        ) : (
          <div className="tradingview-widget-container" ref={containerRef}>
            <div className="tradingview-widget-container__widget"></div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
