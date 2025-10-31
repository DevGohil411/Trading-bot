"use client"

import { useEffect, useRef, useState } from "react"

export default function MarketOverviewWidget() {
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
        script.src = "https://s3.tradingview.com/external-embedding/embed-widget-market-overview.js"
        script.async = true
        script.innerHTML = JSON.stringify({
          colorTheme: "dark",
          dateRange: "12M",
          showChart: true,
          locale: "en",
          width: "100%",
          height: "400",
          largeChartUrl: "",
          isTransparent: true,
          showSymbolLogo: true,
          showFloatingTooltip: false,
          plotLineColorGrowing: "rgba(255, 215, 0, 1)",
          plotLineColorFalling: "rgba(255, 69, 0, 1)",
          gridLineColor: "rgba(240, 243, 250, 0.06)",
          scaleFontColor: "rgba(209, 212, 220, 1)",
          belowLineFillColorGrowing: "rgba(255, 215, 0, 0.12)",
          belowLineFillColorFalling: "rgba(255, 69, 0, 0.12)",
          belowLineFillColorGrowingBottom: "rgba(255, 215, 0, 0)",
          belowLineFillColorFallingBottom: "rgba(255, 69, 0, 0)",
          symbolActiveColor: "rgba(255, 215, 0, 0.12)",
          tabs: [
            {
              title: "Forex",
              symbols: [
                { s: "FX:XAUUSD", d: "Gold" },
                { s: "FX:USDJPY", d: "USD/JPY" },
              ],
              originalTitle: "Forex",
            },
            {
              title: "Crypto",
              symbols: [
                { s: "BITSTAMP:BTCUSD", d: "Bitcoin" },
                { s: "BITSTAMP:ETHUSD", d: "Ethereum" },
              ],
              originalTitle: "Crypto",
            },
          ],
        })

        script.onload = () => setIsLoaded(true)
        script.onerror = () => {
          console.error("Failed to load TradingView market overview widget")
          setIsLoaded(false)
        }

        containerRef.current.appendChild(script)
      }
    }

    const timer = setTimeout(loadWidget, 100)
    return () => clearTimeout(timer)
  }, [])

  if (!isLoaded) {
    return (
      <div className="w-full h-96 bg-muted/20 rounded-lg flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading Market Overview...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="tradingview-widget-container w-full" ref={containerRef}>
      <div className="tradingview-widget-container__widget"></div>
    </div>
  )
}
