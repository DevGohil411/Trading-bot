"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { Bot, MessageSquare, Bell, Settings, Users, Copy, Check, ExternalLink } from "lucide-react"
import { toast } from "sonner"

interface TelegramSettings {
  enabled: boolean
  authorizedUsers: number[]
  notifications: {
    login: boolean
    logout: boolean
    trades: boolean
    botStatus: boolean
    errors: boolean
  }
}

export default function TelegramBotManager() {
  const [settings, setSettings] = useState<TelegramSettings>({
    enabled: true,
    authorizedUsers: [],
    notifications: {
      login: true,
      logout: true,
      trades: true,
      botStatus: true,
      errors: true
    }
  })
  const [loading, setLoading] = useState(false)
  const [copied, setCopied] = useState(false)
  const [botInfo, setBotInfo] = useState<{username?: string, token?: string}>({})

  useEffect(() => {
    // Load settings from localStorage or API
    const savedSettings = localStorage.getItem('telegram_settings')
    if (savedSettings) {
      setSettings(JSON.parse(savedSettings))
    }

    // Get bot info from environment (this will be handled server-side)
    fetch('/api/telegram/bot-info')
      .then(res => res.json())
      .then(data => {
        if (data.success) {
          setBotInfo(data.botInfo)
        }
      })
      .catch(error => {
        console.error('Failed to fetch bot info:', error)
      })
  }, [])

  const saveSettings = (newSettings: TelegramSettings) => {
    setSettings(newSettings)
    localStorage.setItem('telegram_settings', JSON.stringify(newSettings))
  }

  const handleToggleNotifications = (type: keyof TelegramSettings['notifications']) => {
    const newSettings = {
      ...settings,
      notifications: {
        ...settings.notifications,
        [type]: !settings.notifications[type]
      }
    }
    saveSettings(newSettings)
  }

  const handleToggleBot = () => {
    const newSettings = {
      ...settings,
      enabled: !settings.enabled
    }
    saveSettings(newSettings)
    
    if (newSettings.enabled) {
      toast.success("Telegram bot enabled")
    } else {
      toast.success("Telegram bot disabled")
    }
  }

  const copyBotLink = () => {
    if (!botInfo.username) {
      toast.error("Bot username not available")
      return
    }
    
    const botLink = `https://t.me/${botInfo.username.replace('@', '')}`
    navigator.clipboard.writeText(botLink)
    setCopied(true)
    toast.success("Bot link copied to clipboard")
    setTimeout(() => setCopied(false), 2000)
  }

  const openBotInTelegram = () => {
    if (!botInfo.username) {
      toast.error("Bot username not available")
      return
    }
    
    const botLink = `https://t.me/${botInfo.username.replace('@', '')}`
    window.open(botLink, '_blank', 'noopener,noreferrer')
  }

  const testNotification = async () => {
    setLoading(true)
    try {
      await fetch('/api/telegram/notify', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          type: 'bot_status',
          data: {
            status: 'Test Notification',
            details: 'This is a test notification from SniprX Trading Bot'
          }
        }),
      })
      toast.success("Test notification sent!")
    } catch (error) {
      toast.error("Failed to send test notification")
    } finally {
      setLoading(false)
    }
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Bot className="h-5 w-5" />
          Telegram Bot Manager
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Bot Status */}
        <div className="flex items-center justify-between">
          <div>
            <h3 className="font-medium">Bot Status</h3>
            <p className="text-sm text-muted-foreground">
              {settings.enabled ? "Bot is active and ready" : "Bot is disabled"}
            </p>
          </div>
          <Switch
            checked={settings.enabled}
            onCheckedChange={handleToggleBot}
          />
        </div>

        <Separator />

        {/* Bot Information */}
        <div className="space-y-3">
          <h3 className="font-medium">Bot Information</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <Label htmlFor="bot-username">Bot Username</Label>
              <div className="flex items-center gap-2">
                <Input
                  id="bot-username"
                  value={botInfo.username || 'Loading...'}
                  readOnly
                />
                <Button
                  variant="outline"
                  size="sm"
                  onClick={copyBotLink}
                  disabled={!botInfo.username}
                >
                  {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                </Button>
              </div>
            </div>
            <div>
              <Label>Bot Link</Label>
              <div className="flex items-center gap-2">
                <Input
                  value={botInfo.username ? `https://t.me/${botInfo.username.replace('@', '')}` : 'Loading...'}
                  readOnly
                />
                <Button
                  variant="outline"
                  size="sm"
                  onClick={openBotInTelegram}
                  disabled={!botInfo.username}
                >
                  <ExternalLink className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </div>
        </div>

        <Separator />

        {/* Notification Settings */}
        <div className="space-y-3">
          <h3 className="font-medium">Notification Settings</h3>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <MessageSquare className="h-4 w-4" />
                <span>Login/Logout Notifications</span>
              </div>
              <Switch
                checked={settings.notifications.login && settings.notifications.logout}
                onCheckedChange={() => {
                  handleToggleNotifications('login')
                  handleToggleNotifications('logout')
                }}
              />
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Bell className="h-4 w-4" />
                <span>Trade Notifications</span>
              </div>
              <Switch
                checked={settings.notifications.trades}
                onCheckedChange={() => handleToggleNotifications('trades')}
              />
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Bot className="h-4 w-4" />
                <span>Bot Status Updates</span>
              </div>
              <Switch
                checked={settings.notifications.botStatus}
                onCheckedChange={() => handleToggleNotifications('botStatus')}
              />
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Settings className="h-4 w-4" />
                <span>Error Alerts</span>
              </div>
              <Switch
                checked={settings.notifications.errors}
                onCheckedChange={() => handleToggleNotifications('errors')}
              />
            </div>
          </div>
        </div>

        <Separator />

        {/* Authorized Users */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="font-medium">Authorized Users</h3>
            <Badge variant="secondary">
              {settings.authorizedUsers.length} users
            </Badge>
          </div>
          {settings.authorizedUsers.length > 0 ? (
            <div className="space-y-2">
              {settings.authorizedUsers.map((userId) => (
                <div key={userId} className="flex items-center justify-between p-2 bg-muted rounded">
                  <span className="font-mono text-sm">Chat ID: {userId}</span>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      const newUsers = settings.authorizedUsers.filter(id => id !== userId)
                      saveSettings({ ...settings, authorizedUsers: newUsers })
                    }}
                  >
                    Remove
                  </Button>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">
              No authorized users. Users need to contact admin to be authorized.
            </p>
          )}
        </div>

        <Separator />

        {/* Test Notification */}
        <div className="flex items-center justify-between">
          <div>
            <h3 className="font-medium">Test Notification</h3>
            <p className="text-sm text-muted-foreground">
              Send a test notification to all authorized users
            </p>
          </div>
          <Button
            onClick={testNotification}
            disabled={loading || !settings.enabled}
            variant="outline"
          >
            {loading ? "Sending..." : "Send Test"}
          </Button>
        </div>
      </CardContent>
    </Card>
  )
} 