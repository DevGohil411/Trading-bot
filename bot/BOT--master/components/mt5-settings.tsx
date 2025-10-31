"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Separator } from "@/components/ui/separator"
import { Settings, DollarSign, Shield, Clock, AlertTriangle, Target } from "lucide-react"
import { toast } from "sonner"

const recommended = {
  max_daily_loss: 500,
  max_daily_profit: 1000,
  risk_per_trade: 1.5,
  default_lot_size: 0.05,
  max_open_trades: 20,
  max_slippage: 3,
  max_spread: 20,
}

// Default settings structure
const defaultSettings = {
  bot_active: false,
  killzone: true,
  strategy: "mmxm",
  all_strategies: false,
  selected_strategies: ["mmxm"],
  killzone_map: { "mmxm": true },
  risk_settings: {
    max_daily_loss: 500,
    max_daily_profit: 1000,
    risk_per_trade: 1.5,
    default_lot_size: 0.05,
    max_open_trades: 20,
    max_daily_trades: 50,
    max_slippage: 3,
    max_spread: 20,
  },
  trading_sessions: {
    enabled: true,
    london_session: true,
    new_york_session: true,
    tokyo_session: false,
    sydney_session: false,
  },
  notifications: {
    email_notifications: false,
    telegram_notifications: true,
    trade_alerts: true,
    error_alerts: true,
    push_notifications: false,
  },
  news_filter: false,
  volatility_filter: true,
  trend_filter: true,
  advanced_settings: {
    max_slippage: 3,
    max_spread: 20,
    auto_reconnect: true,
    emergency_stop: false,
  }
}

export default function MT5Settings() {
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [settings, setSettings] = useState<any>(defaultSettings)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)

  // Fetch settings from backend
  const fetchSettings = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch("/api/bot/settings")
      const data = await res.json()
      // Merge with default settings to ensure all properties exist
      const mergedSettings = {
        ...defaultSettings,
        ...data,
        risk_settings: {
          ...defaultSettings.risk_settings,
          ...data.risk_settings
        },
        trading_sessions: {
          ...defaultSettings.trading_sessions,
          ...data.trading_sessions
        },
        notifications: {
          ...defaultSettings.notifications,
          ...data.notifications
        }
      }
      setSettings(mergedSettings)
    } catch (e) {
      setError("Failed to load settings.")
      // Use default settings if fetch fails
      setSettings(defaultSettings)
    }
    setLoading(false)
  }

  useEffect(() => {
    fetchSettings()
  }, [])

  // Helper to update nested settings
  const updateSetting = (path: string[], value: any) => {
    setSettings((prev: any) => {
      const updated = { ...prev }
      let obj = updated
      for (let i = 0; i < path.length - 1; i++) {
        if (!obj[path[i]]) {
          obj[path[i]] = {}
        }
        obj[path[i]] = { ...obj[path[i]] }
        obj = obj[path[i]]
      }
      obj[path[path.length - 1]] = value
      return updated
    })
  }

  // Save settings to backend
  const saveSettings = async () => {
    setSaving(true)
    setError(null)
    setSuccess(null)
    try {
      const res = await fetch("/api/bot/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(settings),
      })
      const data = await res.json()
      if (data.success) {
        setSuccess("Settings saved successfully!")
        toast.success("Settings saved successfully!")
      } else {
        setError(data.error || "Failed to save settings.")
        toast.error(data.error || "Failed to save settings.")
      }
    } catch (e) {
      setError("Failed to save settings.")
      toast.error("Failed to save settings.")
    }
    setSaving(false)
  }

  // Reset to backend values
  const resetSettings = () => {
    fetchSettings()
    setSuccess(null)
    setError(null)
    toast("Settings reset to last saved values.")
  }

  if (loading) return <div className="py-12 text-center">Loading settings...</div>

  // Helper for recommended text
  const rec = (key: keyof typeof recommended) => (
    <span className="text-xs text-muted-foreground ml-2">
      (Recommended: {recommended[key]})
    </span>
  )

  return (
    <div className="space-y-6">
      {/* Risk Management */}
      <Card className="glassmorphism border-primary/20">
        <CardHeader>
          <CardTitle className="flex items-center">
            <Shield className="w-5 h-5 mr-2 text-primary" />
            Risk Management
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <Label>Max Daily Loss ($){rec("max_daily_loss")}</Label>
              <Input
                type="number"
                value={settings.risk_settings.max_daily_loss}
                onChange={e => updateSetting(["risk_settings", "max_daily_loss"], Number(e.target.value))}
                min={0}
              />
              <p className="text-xs text-muted-foreground">Bot stops trading if daily loss exceeds this amount</p>
            </div>
            <div className="space-y-2">
              <Label>Max Daily Profit ($){rec("max_daily_profit")}</Label>
              <Input
                type="number"
                value={settings.risk_settings.max_daily_profit}
                onChange={e => updateSetting(["risk_settings", "max_daily_profit"], Number(e.target.value))}
                min={0}
              />
              <p className="text-xs text-muted-foreground">Bot stops trading after reaching daily profit target</p>
            </div>
          </div>
          <div className="space-y-4">
            <div>
              <Label>Risk Per Trade: {settings.risk_settings.risk_per_trade}%{rec("risk_per_trade")}</Label>
              <Slider
                value={[settings.risk_settings.risk_per_trade]}
                onValueChange={value => updateSetting(["risk_settings", "risk_per_trade"], value[0])}
                max={10}
                min={0.5}
                step={0.1}
                className="mt-2"
              />
              <div className="flex justify-between text-xs text-muted-foreground mt-1">
                <span>Conservative (0.5%)</span>
                <span>Aggressive (10%)</span>
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Default Lot Size{rec("default_lot_size")}</Label>
                <Select
                  value={settings.risk_settings.default_lot_size.toString()}
                  onValueChange={value => updateSetting(["risk_settings", "default_lot_size"], Number(value))}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="0.01">0.01 (Micro)</SelectItem>
                    <SelectItem value="0.05">0.05 (Small)</SelectItem>
                    <SelectItem value="0.1">0.1 (Standard)</SelectItem>
                    <SelectItem value="0.2">0.2 (Medium)</SelectItem>
                    <SelectItem value="0.5">0.5 (Large)</SelectItem>
                    <SelectItem value="1">1.0 (Full)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Max Open Trades{rec("max_open_trades")}</Label>
                <Input
                  type="number"
                  value={settings.risk_settings.max_open_trades}
                  onChange={e => updateSetting(["risk_settings", "max_open_trades"], Number(e.target.value))}
                  min={1}
                />
              </div>
            </div>
            <div className="space-y-2">
              <Label>Max Trades Per Day</Label>
              <Input
                type="number"
                value={settings.risk_settings.max_daily_trades}
                onChange={e => updateSetting(["risk_settings", "max_daily_trades"], Number(e.target.value))}
                min={1}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Trading Sessions */}
      <Card className="glassmorphism border-primary/20">
        <CardHeader>
          <CardTitle className="flex items-center">
            <Clock className="w-5 h-5 mr-2 text-primary" />
            Trading Sessions
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-base">Enable Trading</Label>
              <p className="text-sm text-muted-foreground">Master switch for all trading activity</p>
            </div>
            <Switch
              checked={settings.trading_sessions.enabled}
              onCheckedChange={checked => updateSetting(["trading_sessions", "enabled"], checked)}
            />
          </div>
          <Separator />
          <div className="space-y-4">
            <h4 className="font-medium">Active Sessions</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="flex items-center justify-between p-3 border rounded-lg">
                <div>
                  <Label>London Session</Label>
                  <p className="text-xs text-muted-foreground">{settings.trading_sessions.london_session ? "Active" : "Inactive"}</p>
                </div>
                <Switch
                  checked={settings.trading_sessions.london_session}
                  onCheckedChange={checked => updateSetting(["trading_sessions", "london_session"], checked)}
                />
              </div>
              <div className="flex items-center justify-between p-3 border rounded-lg">
                <div>
                  <Label>New York Session</Label>
                  <p className="text-xs text-muted-foreground">{settings.trading_sessions.new_york_session ? "Active" : "Inactive"}</p>
                </div>
                <Switch
                  checked={settings.trading_sessions.new_york_session}
                  onCheckedChange={checked => updateSetting(["trading_sessions", "new_york_session"], checked)}
                />
              </div>
              <div className="flex items-center justify-between p-3 border rounded-lg">
                <div>
                  <Label>Asian Session</Label>
                  <p className="text-xs text-muted-foreground">{settings.trading_sessions.tokyo_session ? "Active" : "Inactive"}</p>
                </div>
                <Switch
                  checked={settings.trading_sessions.tokyo_session}
                  onCheckedChange={checked => updateSetting(["trading_sessions", "tokyo_session"], checked)}
                />
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Strategy Filters */}
      <Card className="glassmorphism border-primary/20">
        <CardHeader>
          <CardTitle className="flex items-center">
            <Target className="w-5 h-5 mr-2 text-primary" />
            Strategy Filters
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="flex items-center justify-between">
              <div>
                <Label>Killzone Filter</Label>
                <p className="text-sm text-muted-foreground">Only trade during high-probability zones</p>
              </div>
              <Switch
                checked={settings.killzone}
                onCheckedChange={checked => updateSetting(["killzone"], checked)}
              />
            </div>
            <div className="flex items-center justify-between">
              <div>
                <Label>News Filter</Label>
                <p className="text-sm text-muted-foreground">Avoid trading during high-impact news</p>
              </div>
              <Switch
                checked={settings.news_filter}
                onCheckedChange={checked => updateSetting(["news_filter"], checked)}
              />
            </div>
            <div className="flex items-center justify-between">
              <div>
                <Label>Volatility Filter</Label>
                <p className="text-sm text-muted-foreground">Filter trades based on market volatility</p>
              </div>
              <Switch
                checked={settings.volatility_filter}
                onCheckedChange={checked => updateSetting(["volatility_filter"], checked)}
              />
            </div>
            <div className="flex items-center justify-between">
              <div>
                <Label>Trend Filter</Label>
                <p className="text-sm text-muted-foreground">Align trades with higher timeframe trend</p>
              </div>
              <Switch
                checked={settings.trend_filter}
                onCheckedChange={checked => updateSetting(["trend_filter"], checked)}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Notifications */}
      <Card className="glassmorphism border-primary/20">
        <CardHeader>
          <CardTitle className="flex items-center">
            <AlertTriangle className="w-5 h-5 mr-2 text-primary" />
            Notifications & Alerts
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="flex items-center justify-between">
              <div>
                <Label>Email Alerts</Label>
                <p className="text-sm text-muted-foreground">Get trade notifications via email</p>
              </div>
              <Switch
                checked={settings.notifications.email_notifications}
                onCheckedChange={checked => updateSetting(["notifications", "email_notifications"], checked)}
              />
            </div>
            <div className="flex items-center justify-between">
              <div>
                <Label>Telegram Alerts</Label>
                <p className="text-sm text-muted-foreground">Real-time alerts on Telegram</p>
              </div>
              <Switch
                checked={settings.notifications.telegram_notifications}
                onCheckedChange={checked => updateSetting(["notifications", "telegram_notifications"], checked)}
              />
            </div>
            <div className="flex items-center justify-between">
              <div>
                <Label>Push Notifications</Label>
                <p className="text-sm text-muted-foreground">Browser push notifications</p>
              </div>
              <Switch
                checked={settings.notifications.push_notifications}
                onCheckedChange={checked => updateSetting(["notifications", "push_notifications"], checked)}
              />
            </div>
            <div className="flex items-center justify-between">
              <div>
                <Label>Trade Alerts</Label>
                <p className="text-sm text-muted-foreground">Alerts for trade entries and exits</p>
              </div>
              <Switch
                checked={settings.notifications.trade_alerts}
                onCheckedChange={checked => updateSetting(["notifications", "trade_alerts"], checked)}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Advanced Settings */}
      <Card className="glassmorphism border-primary/20">
        <CardHeader>
          <CardTitle className="flex items-center">
            <Settings className="w-5 h-5 mr-2 text-primary" />
            Advanced Settings
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <Label>Max Slippage (Points){rec("max_slippage")}</Label>
              <Input
                type="number"
                value={settings.advanced_settings.max_slippage}
                onChange={e => updateSetting(["advanced_settings", "max_slippage"], Number(e.target.value))}
                min={0}
              />
              <p className="text-xs text-muted-foreground">Maximum acceptable slippage for trade execution</p>
            </div>
            <div className="space-y-2">
              <Label>Max Spread (Points){rec("max_spread")}</Label>
              <Input
                type="number"
                value={settings.advanced_settings.max_spread}
                onChange={e => updateSetting(["advanced_settings", "max_spread"], Number(e.target.value))}
                min={0}
              />
              <p className="text-xs text-muted-foreground">Don't trade if spread exceeds this value</p>
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="flex items-center justify-between">
              <div>
                <Label>Auto Reconnect</Label>
                <p className="text-sm text-muted-foreground">Automatically reconnect if connection drops</p>
              </div>
              <Switch
                checked={settings.advanced_settings.auto_reconnect}
                onCheckedChange={checked => updateSetting(["advanced_settings", "auto_reconnect"], checked)}
              />
            </div>
            <div className="flex items-center justify-between">
              <div>
                <Label>Emergency Stop</Label>
                <p className="text-sm text-muted-foreground">Enable emergency stop functionality</p>
              </div>
              <Switch
                checked={settings.advanced_settings.emergency_stop}
                onCheckedChange={checked => updateSetting(["advanced_settings", "emergency_stop"], checked)}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Save Settings */}
      <div className="flex justify-end space-x-4">
        <Button variant="outline" onClick={resetSettings} disabled={saving}>Reset to Default</Button>
        <Button className="gradient-primary text-black" onClick={saveSettings} disabled={saving}>
          <DollarSign className="w-4 h-4 mr-2" />
          {saving ? "Saving..." : "Save Settings"}
        </Button>
      </div>
      {error && <div className="text-red-500 text-sm mt-2">{error}</div>}
      {success && <div className="text-green-600 text-sm mt-2">{success}</div>}
    </div>
  )
}
