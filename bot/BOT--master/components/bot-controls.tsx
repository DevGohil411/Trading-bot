"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Bot, Settings, Play, Pause, Wifi, WifiOff, AlertTriangle } from "lucide-react"
import { Checkbox } from "@/components/ui/checkbox"

// Function to send Telegram notification
const sendTelegramNotification = async (type: string, data: any) => {
  try {
    await fetch('/api/telegram/notify', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ type, data }),
    });
  } catch (error) {
    console.error('Failed to send Telegram notification:', error);
  }
};

export default function BotControls() {
  const [botActive, setBotActive] = useState(false)
  const [mode, setMode] = useState("live")
  const [mt5Connected, setMt5Connected] = useState(true)
  const [selectedStrategy, setSelectedStrategy] = useState("mmxm")
  const [killzoneFilter, setKillzoneFilter] = useState(false)
  const [allStrategies, setAllStrategies] = useState(false)
  const [botRunning, setBotRunning] = useState(false)
  const [loading, setLoading] = useState(false)
  const [selectedStrategies, setSelectedStrategies] = useState<string[]>(["mmxm"])
  const [killzoneMap, setKillzoneMap] = useState<{ [key: string]: boolean }>({})

  const strategies = [
    { id: "mmxm", name: "MMXM Strategy", icon: "🎯" },
    { id: "judas_swing", name: "Judas Swing", icon: "📉" },
    { id: "ote", name: "OTE Strategy", icon: "⏳" },
    { id: "amd_strategy", name: "AMD Cycle", icon: "🔄" },
    { id: "msb_retest", name: "MSB Retest", icon: "📊" },
    { id: "order_block", name: "Order Block + CHOCH", icon: "🧱" },
    { id: "mmc_combo_strategy", name: "MMC Combo", icon: "⚡" },
    { id: "mmc", name: "MMC Classic", icon: "💡" },
  ]

  // Fetch settings and bot status from backend
  useEffect(() => {
    let isMounted = true;
    const fetchSettings = () => {
      fetch('/api/bot/settings')
        .then(res => res.json())
        .then(s => {
          if (isMounted) {
            setBotActive(s.bot_active)
            setKillzoneFilter(s.killzone)
            setSelectedStrategy(s.strategy)
            setAllStrategies(s.all_strategies)
            setSelectedStrategies(s.selected_strategies || ["mmxm"])
            setKillzoneMap(s.killzone_map || {})
          }
        })
      fetch('/api/bot/status')
        .then(res => res.json())
        .then(s => {
          if (isMounted) setBotRunning(s.running)
        })
    }
    fetchSettings()
    const interval = setInterval(fetchSettings, 5000)
    return () => { isMounted = false; clearInterval(interval) }
  }, [])

  // Update backend on any change
  const updateSettings = (updates: any) => {
    fetch('/api/bot/settings', {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        bot_active: updates.bot_active ?? botActive,
        killzone: updates.killzone ?? killzoneFilter,
        strategy: updates.strategy ?? selectedStrategy,
        all_strategies: updates.all_strategies ?? allStrategies,
        selected_strategies: updates.selected_strategies ?? selectedStrategies,
        killzone_map: updates.killzone_map ?? killzoneMap,
      })
    })
  }

  const handleStartBot = async () => {
    setLoading(true)
    try {
      // First set bot_active true
      await fetch('/api/bot/settings', {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          bot_active: true,
          killzone: killzoneFilter,
          strategy: selectedStrategy,
          all_strategies: allStrategies,
        })
      })
      // Then start the process
      await fetch('/api/bot/start', { method: "POST" })
      
      // Send Telegram notification
      await sendTelegramNotification('bot_status', {
        status: 'Bot Started',
        details: `Strategy: ${selectedStrategy}, Killzone: ${killzoneFilter ? 'Enabled' : 'Disabled'}`
      });
    } catch (error) {
      console.error('Error starting bot:', error);
      // Send error notification
      await sendTelegramNotification('error', {
        error: 'Failed to start trading bot'
      });
    } finally {
      setLoading(false)
    }
  }

  const handleStopBot = async () => {
    setLoading(true)
    try {
      // First set bot_active false
      await fetch('/api/bot/settings', {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          bot_active: false,
          killzone: killzoneFilter,
          strategy: selectedStrategy,
          all_strategies: allStrategies,
        })
      })
      // Then stop the process
      await fetch('/api/bot/stop', { method: "POST" })
      
      // Send Telegram notification
      await sendTelegramNotification('bot_status', {
        status: 'Bot Stopped',
        details: 'Trading bot has been stopped'
      });
    } catch (error) {
      console.error('Error stopping bot:', error);
      // Send error notification
      await sendTelegramNotification('error', {
        error: 'Failed to stop trading bot'
      });
    } finally {
      setLoading(false)
    }
  }

  const handleBotToggle = () => {
    if (!mt5Connected) {
      alert("Cannot start bot: MT5 not connected")
      return
    }
    if (!botRunning) {
      handleStartBot()
    } else {
      handleStopBot()
    }
  }

  const handleKillzoneToggle = () => {
    setKillzoneFilter((prev) => {
      updateSettings({ killzone: !prev })
      return !prev
    })
  }

  const handleStrategyChange = (val: string) => {
    setSelectedStrategy(val)
    updateSettings({ strategy: val, all_strategies: false })
    setAllStrategies(false)
  }

  const handleAllStrategies = (checked: boolean) => {
    setAllStrategies(checked)
    if (checked) {
      const allIds = strategies.map(s => s.id)
      setSelectedStrategies(allIds)
      const newMap = { ...killzoneMap }
      allIds.forEach(id => { if (!(id in newMap)) newMap[id] = false })
      setKillzoneMap(newMap)
      updateSettings({ all_strategies: true, selected_strategies: allIds, killzone_map: newMap })
    } else {
      setSelectedStrategies([])
      updateSettings({ all_strategies: false, selected_strategies: [] })
    }
  }

  // Multi-select strategy handler
  const handleStrategySelect = (id: string, checked: boolean) => {
    let newSelected: string[]
    if (checked) {
      newSelected = [...selectedStrategies, id]
    } else {
      newSelected = selectedStrategies.filter(s => s !== id)
    }
    setSelectedStrategies(newSelected)
    updateSettings({ selected_strategies: newSelected, all_strategies: false })
    setAllStrategies(false)
  }

  // Per-strategy killzone toggle
  const handleKillzoneToggleStrategy = (id: string, checked: boolean) => {
    const newMap = { ...killzoneMap, [id]: checked }
    setKillzoneMap(newMap)
    updateSettings({ killzone_map: newMap })
  }

  return (
    <Card className="glassmorphism">
      <CardHeader>
        <CardTitle className="flex items-center">
          <Bot className="w-5 h-5 mr-2 text-primary" />
          Bot Control Center
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* MT5 Connection Status */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              {mt5Connected ? (
                <Wifi className="w-4 h-4 text-green-500" />
              ) : (
                <WifiOff className="w-4 h-4 text-red-500" />
              )}
              <span className="text-sm font-medium">MT5 Status</span>
            </div>
            <Badge
              className={
                mt5Connected
                  ? "bg-green-500/10 text-green-500 border-green-500/20"
                  : "bg-red-500/10 text-red-500 border-red-500/20"
              }
            >
              {mt5Connected ? "Connected" : "Disconnected"}
            </Badge>
          </div>
          {!mt5Connected && (
            <div className="flex items-center space-x-2 p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
              <AlertTriangle className="w-4 h-4 text-red-500" />
              <span className="text-sm text-red-500">Bot disabled - MT5 connection required</span>
            </div>
          )}
        </div>

        {/* Mode Toggle */}
        <div className="space-y-3">
          <label className="text-sm font-medium">Trading Mode</label>
          <div className="flex space-x-2">
            <Button
              size="sm"
              variant={mode === "live" ? "default" : "outline"}
              onClick={() => setMode("live")}
              className={mode === "live" ? "gradient-primary text-black" : ""}
            >
              ✅ Live Mode
            </Button>
            <Button
              size="sm"
              variant={mode === "backtest" ? "default" : "outline"}
              onClick={() => setMode("backtest")}
              className={mode === "backtest" ? "gradient-primary text-black" : ""}
            >
              🧪 Backtest
            </Button>
          </div>
        </div>

        {/* Bot Status */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div
                className={`w-3 h-3 rounded-full ${botRunning && mt5Connected ? "bg-green-500" : "bg-red-500"} animate-pulse`}
              />
              <div>
                <p className="font-medium">{botRunning && mt5Connected ? "Bot Running" : "Bot Stopped"}</p>
                <p className="text-sm text-muted-foreground">
                  {botRunning && mt5Connected ? "Hunting for entries" : "Standby mode"}
                </p>
              </div>
            </div>
            <Switch
              checked={botRunning && mt5Connected}
              onCheckedChange={handleBotToggle}
              disabled={!mt5Connected || loading}
              className="data-[state=checked]:bg-primary"
            />
          </div>

          <Button
            size="sm"
            variant={botRunning ? "destructive" : "default"}
            className="w-full"
            onClick={handleBotToggle}
            disabled={!mt5Connected || loading}
          >
            {botRunning ? (
              <>
                <Pause className="w-4 h-4 mr-2" />
                Stop Bot
              </>
            ) : (
              <>
                <Play className="w-4 h-4 mr-2" />
                Start Bot
              </>
            )}
          </Button>
        </div>

        {/* Strategy Selection */}
        <div className="space-y-3">
          <label className="text-sm font-medium">Active Strategy</label>
          <Select value={selectedStrategy} onValueChange={handleStrategyChange}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {strategies.map((strategy) => (
                <SelectItem key={strategy.id} value={strategy.id}>
                  <div className="flex items-center space-x-2">
                    <span>{strategy.icon}</span>
                    <span>{strategy.name}</span>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Killzone Filter */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium">Killzone Filter</label>
            <Switch
              checked={killzoneFilter}
              onCheckedChange={handleKillzoneToggle}
              className="data-[state=checked]:bg-primary"
            />
          </div>
          <p className="text-xs text-muted-foreground">
            Only trade during high-probability sessions (London/NY overlap)
          </p>
        </div>

        {/* Strategy Filter & Killzone Toggles */}
        <div className="space-y-2">
          <label className="text-xs font-medium">Select Strategies</label>
          <div className="flex flex-col gap-1">
            {strategies.map(strategy => (
              <div key={strategy.id} className="flex items-center gap-2 bg-black/5 rounded px-2 py-1 min-h-7">
                <Checkbox
                  checked={selectedStrategies.includes(strategy.id)}
                  onCheckedChange={checked => handleStrategySelect(strategy.id, !!checked)}
                  className="mr-1 w-4 h-4 min-w-4 min-h-4"
                />
                <span className="text-xs leading-tight">{strategy.icon} {strategy.name}</span>
                <Switch
                  checked={!!killzoneMap[strategy.id]}
                  onCheckedChange={checked => handleKillzoneToggleStrategy(strategy.id, !!checked)}
                  disabled={!selectedStrategies.includes(strategy.id)}
                  className="ml-auto scale-75"
                />
                <span className="text-[10px] text-muted-foreground ml-1">Killzone</span>
              </div>
            ))}
          </div>
        </div>

        {/* All Strategies Button */}
        <div className="flex justify-center mt-2">
          <Button
            className="w-full gradient-primary text-black text-base font-semibold py-1"
            variant={allStrategies ? "default" : "outline"}
            onClick={() => handleAllStrategies(!allStrategies)}
            style={{ minHeight: 32 }}
          >
            {allStrategies ? "All Strategies ON" : "All Strategies"}
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
