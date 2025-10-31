"use client"

import { useEffect, useRef, useState } from "react"

interface MiniChartWidgetProps {
  symbol?: string
  width?: string
  height?: string
}

export default function MiniChartWidget({
  symbol = "FX:XAUUSD",
  width = "100%",
  height = "300",
}: MiniChartWidgetProps) {
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
        script.src = "https://s3.tradingview.com/external-embedding/embed-widget-mini-symbol-overview.js"
        script.async = true
        script.innerHTML = JSON.stringify({
          symbol: symbol,
          width: width,
          height: height,
          locale: "en",
          dateRange: "12M",
          colorTheme: "dark",
          isTransparent: true,
          autosize: false,
          largeChartUrl: "",
        })

        script.onload = () => setIsLoaded(true)
        script.onerror = () => {
          console.error("Failed to load TradingView mini chart widget")
          setIsLoaded(false)
        }

        containerRef.current.appendChild(script)
      }
    }

    const timer = setTimeout(loadWidget, 100)
    return () => clearTimeout(timer)
  }, [symbol, width, height])

  if (!isLoaded) {
    return (
      <div className="w-full h-72 bg-muted/20 rounded-lg flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading Chart...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="tradingview-widget-container" ref={containerRef}>
      <div className="tradingview-widget-container__widget"></div>
    </div>
  )
}
