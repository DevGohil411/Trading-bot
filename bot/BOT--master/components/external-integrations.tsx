"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { Switch } from "@/components/ui/switch"
import { Textarea } from "@/components/ui/textarea"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  Webhook,
  MessageCircle,
  TrendingUp,
  Server,
  Link,
  CheckCircle,
  AlertTriangle,
  Settings,
  Copy,
  ExternalLink,
} from "lucide-react"

interface Integration {
  id: string
  name: string
  type: "webhook" | "telegram" | "tradingview" | "broker" | "api"
  status: "connected" | "disconnected" | "error"
  description: string
  lastSync?: string
}

export default function ExternalIntegrations() {
  const [integrations, setIntegrations] = useState<Integration[]>([
    {
      id: "tv-webhook",
      name: "TradingView Webhooks",
      type: "tradingview",
      status: "connected",
      description: "Receive signals from TradingView alerts",
      lastSync: "2 minutes ago",
    },
    {
      id: "telegram-bot",
      name: "Telegram Bot",
      type: "telegram",
      status: "connected",
      description: "Control bots and receive alerts via Telegram",
      lastSync: "5 minutes ago",
    },
    {
      id: "mt4-bridge",
      name: "MetaTrader 4",
      type: "broker",
      status: "disconnected",
      description: "Connect MT4 accounts for trading",
    },
    {
      id: "ctrader",
      name: "cTrader",
      type: "broker",
      status: "disconnected",
      description: "cTrader platform integration",
    },
  ])

  const [webhookUrl] = useState("https://api.sniprx.com/webhook/tv/user123")
  const [telegramBotToken, setTelegramBotToken] = useState("")
  const [telegramChatId, setTelegramChatId] = useState("")
  const [showTelegramSetup, setShowTelegramSetup] = useState(false)
  const [showWebhookSetup, setShowWebhookSetup] = useState(false)

  const toggleIntegration = (id: string) => {
    setIntegrations((prev) =>
      prev.map((integration) =>
        integration.id === id
          ? {
              ...integration,
              status: integration.status === "connected" ? "disconnected" : "connected",
              lastSync: integration.status === "disconnected" ? "Just now" : integration.lastSync,
            }
          : integration,
      ),
    )
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case "connected":
        return "bg-green-500/10 text-green-500 border-green-500/20"
      case "error":
        return "bg-red-500/10 text-red-500 border-red-500/20"
      default:
        return "bg-gray-500/10 text-gray-500 border-gray-500/20"
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "connected":
        return <CheckCircle className="w-4 h-4" />
      case "error":
        return <AlertTriangle className="w-4 h-4" />
      default:
        return <Server className="w-4 h-4" />
    }
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
  }

  return (
    <div className="space-y-6">
      {/* Integration Overview */}
      <Card className="glassmorphism border-primary/20">
        <CardHeader>
          <CardTitle className="flex items-center">
            <Link className="w-5 h-5 mr-2 text-primary" />
            External Integrations
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {integrations.map((integration) => (
              <div key={integration.id} className="border border-border/40 rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-2">
                    {integration.type === "tradingview" && <TrendingUp className="w-4 h-4 text-blue-500" />}
                    {integration.type === "telegram" && <MessageCircle className="w-4 h-4 text-blue-500" />}
                    {integration.type === "broker" && <Server className="w-4 h-4 text-green-500" />}
                    {integration.type === "webhook" && <Webhook className="w-4 h-4 text-purple-500" />}
                    <span className="font-medium text-sm">{integration.name}</span>
                  </div>
                  <Switch
                    checked={integration.status === "connected"}
                    onCheckedChange={() => toggleIntegration(integration.id)}
                    className="data-[state=checked]:bg-primary"
                  />
                </div>

                <p className="text-xs text-muted-foreground mb-3">{integration.description}</p>

                <div className="flex items-center justify-between">
                  <Badge className={getStatusColor(integration.status)}>
                    {getStatusIcon(integration.status)}
                    <span className="ml-1 capitalize">{integration.status}</span>
                  </Badge>
                  {integration.lastSync && (
                    <span className="text-xs text-muted-foreground">{integration.lastSync}</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Integration Setup */}
      <Card className="glassmorphism border-primary/20">
        <CardHeader>
          <CardTitle className="flex items-center">
            <Settings className="w-5 h-5 mr-2 text-primary" />
            Integration Setup
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="tradingview" className="space-y-6">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="tradingview">TradingView</TabsTrigger>
              <TabsTrigger value="telegram">Telegram</TabsTrigger>
              <TabsTrigger value="brokers">Brokers</TabsTrigger>
              <TabsTrigger value="api">API Access</TabsTrigger>
            </TabsList>

            {/* TradingView Webhooks */}
            <TabsContent value="tradingview" className="space-y-4">
              <div className="space-y-4">
                <div className="p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg">
                  <h4 className="font-semibold mb-2 flex items-center">
                    <TrendingUp className="w-4 h-4 mr-2" />
                    TradingView Webhook Integration
                  </h4>
                  <p className="text-sm text-muted-foreground mb-4">
                    Connect your TradingView alerts to automatically trigger trades based on your custom indicators and
                    strategies.
                  </p>

                  <div className="space-y-3">
                    <div>
                      <Label>Your Webhook URL</Label>
                      <div className="flex items-center space-x-2 mt-1">
                        <Input value={webhookUrl} readOnly className="font-mono text-sm" />
                        <Button size="sm" variant="outline" onClick={() => copyToClipboard(webhookUrl)}>
                          <Copy className="w-4 h-4" />
                        </Button>
                      </div>
                      <p className="text-xs text-muted-foreground mt-1">
                        Use this URL in your TradingView alert webhook settings
                      </p>
                    </div>

                    <div>
                      <Label>Webhook Message Format</Label>
                      <Textarea
                        readOnly
                        value={`{
  "strategy": "{{strategy.order.action}}",
  "symbol": "{{ticker}}",
  "price": "{{close}}",
  "time": "{{time}}",
  "message": "{{strategy.order.comment}}"
}`}
                        className="font-mono text-xs h-32 mt-1"
                      />
                    </div>

                    <div className="flex items-center space-x-4">
                      <Button className="gradient-primary text-black">
                        <ExternalLink className="w-4 h-4 mr-2" />
                        Setup Guide
                      </Button>
                      <Button variant="outline">Test Webhook</Button>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="p-4 border border-border/40 rounded-lg">
                    <h5 className="font-medium mb-2">Supported Signals</h5>
                    <ul className="text-sm text-muted-foreground space-y-1">
                      <li>• Buy/Sell signals</li>
                      <li>• Stop Loss updates</li>
                      <li>• Take Profit levels</li>
                      <li>• Position sizing</li>
                      <li>• Custom parameters</li>
                    </ul>
                  </div>

                  <div className="p-4 border border-border/40 rounded-lg">
                    <h5 className="font-medium mb-2">Recent Signals</h5>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span>XAUUSD BUY</span>
                        <span className="text-muted-foreground">2m ago</span>
                      </div>
                      <div className="flex justify-between">
                        <span>BTCUSD BUY</span>
                        <span className="text-muted-foreground">1h ago</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </TabsContent>

            {/* Telegram Bot */}
            <TabsContent value="telegram" className="space-y-4">
              <div className="space-y-4">
                <div className="p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg">
                  <h4 className="font-semibold mb-2 flex items-center">
                    <MessageCircle className="w-4 h-4 mr-2" />
                    Telegram Bot Commands
                  </h4>
                  <p className="text-sm text-muted-foreground mb-4">
                    Control your trading bots and receive real-time alerts directly in Telegram.
                  </p>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <Label>Bot Token</Label>
                      <Input
                        type="password"
                        placeholder="Enter your bot token"
                        value={telegramBotToken}
                        onChange={(e) => setTelegramBotToken(e.target.value)}
                      />
                    </div>
                    <div>
                      <Label>Chat ID</Label>
                      <Input
                        placeholder="Your Telegram chat ID"
                        value={telegramChatId}
                        onChange={(e) => setTelegramChatId(e.target.value)}
                      />
                    </div>
                  </div>

                  <div className="mt-4">
                    <Button className="gradient-primary text-black mr-2">Connect Telegram</Button>
                    <Button variant="outline">
                      <ExternalLink className="w-4 h-4 mr-2" />
                      Setup Guide
                    </Button>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="p-4 border border-border/40 rounded-lg">
                    <h5 className="font-medium mb-3">Available Commands</h5>
                    <div className="space-y-2 text-sm font-mono">
                      <div>/status - Bot status</div>
                      <div>/pnl - Current P&L</div>
                      <div>/start - Start bot</div>
                      <div>/stop - Stop bot</div>
                      <div>/trades - Recent trades</div>
                      <div>/settings - Bot settings</div>
                    </div>
                  </div>

                  <div className="p-4 border border-border/40 rounded-lg">
                    <h5 className="font-medium mb-3">Alert Types</h5>
                    <div className="space-y-2 text-sm">
                      <div className="flex items-center space-x-2">
                        <span>🟢</span>
                        <span>Trade entries/exits</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span>🔴</span>
                        <span>Risk warnings</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span>📊</span>
                        <span>Daily P&L reports</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span>⚠️</span>
                        <span>System alerts</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span>🎯</span>
                        <span>Strategy signals</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </TabsContent>

            {/* Brokers */}
            <TabsContent value="brokers" className="space-y-4">
              <div className="space-y-4">
                <div className="p-4 bg-green-500/10 border border-green-500/20 rounded-lg">
                  <h4 className="font-semibold mb-2 flex items-center">
                    <Server className="w-4 h-4 mr-2" />
                    Multi-Broker Support
                  </h4>
                  <p className="text-sm text-muted-foreground mb-4">
                    Connect multiple broker accounts and platforms for diversified trading execution.
                  </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* MT4 Integration */}
                  <div className="space-y-4">
                    <h5 className="font-medium flex items-center">
                      <Server className="w-4 h-4 mr-2 text-blue-500" />
                      MetaTrader 4
                    </h5>
                    <div className="p-4 border border-border/40 rounded-lg space-y-3">
                      <div>
                        <Label>MT4 Server</Label>
                        <Input placeholder="e.g., ICMarkets-Live01" />
                      </div>
                      <div>
                        <Label>Account Number</Label>
                        <Input placeholder="Your MT4 account number" />
                      </div>
                      <div>
                        <Label>Password</Label>
                        <Input type="password" placeholder="MT4 password" />
                      </div>
                      <Button className="w-full gradient-primary text-black">Connect MT4</Button>
                    </div>
                  </div>

                  {/* cTrader Integration */}
                  <div className="space-y-4">
                    <h5 className="font-medium flex items-center">
                      <Server className="w-4 h-4 mr-2 text-green-500" />
                      cTrader
                    </h5>
                    <div className="p-4 border border-border/40 rounded-lg space-y-3">
                      <div>
                        <Label>cTrader ID</Label>
                        <Input placeholder="Your cTrader ID" />
                      </div>
                      <div>
                        <Label>Access Token</Label>
                        <Input type="password" placeholder="API access token" />
                      </div>
                      <div>
                        <Label>Environment</Label>
                        <select className="w-full p-2 border rounded">
                          <option>Live</option>
                          <option>Demo</option>
                        </select>
                      </div>
                      <Button className="w-full gradient-primary text-black">Connect cTrader</Button>
                    </div>
                  </div>
                </div>

                <div className="p-4 border border-border/40 rounded-lg">
                  <h5 className="font-medium mb-3">Supported Brokers</h5>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
                    <div className="flex items-center space-x-2">
                      <CheckCircle className="w-4 h-4 text-green-500" />
                      <span>IC Markets</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <CheckCircle className="w-4 h-4 text-green-500" />
                      <span>Pepperstone</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <CheckCircle className="w-4 h-4 text-green-500" />
                      <span>FTMO</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <CheckCircle className="w-4 h-4 text-green-500" />
                      <span>MyFundedFX</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <CheckCircle className="w-4 h-4 text-green-500" />
                      <span>Admirals</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <CheckCircle className="w-4 h-4 text-green-500" />
                      <span>XM</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <CheckCircle className="w-4 h-4 text-green-500" />
                      <span>FXCM</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <CheckCircle className="w-4 h-4 text-green-500" />
                      <span>AvaTrade</span>
                    </div>
                  </div>
                </div>
              </div>
            </TabsContent>

            {/* API Access */}
            <TabsContent value="api" className="space-y-4">
              <div className="space-y-4">
                <div className="p-4 bg-purple-500/10 border border-purple-500/20 rounded-lg">
                  <h4 className="font-semibold mb-2 flex items-center">
                    <Webhook className="w-4 h-4 mr-2" />
                    SniprX API Access
                  </h4>
                  <p className="text-sm text-muted-foreground mb-4">
                    Integrate SniprX with your own applications using our REST API and WebSocket connections.
                  </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <h5 className="font-medium">API Credentials</h5>
                    <div className="space-y-3">
                      <div>
                        <Label>API Key</Label>
                        <div className="flex items-center space-x-2">
                          <Input value="sk_live_****************************" readOnly className="font-mono" />
                          <Button size="sm" variant="outline">
                            <Copy className="w-4 h-4" />
                          </Button>
                        </div>
                      </div>
                      <div>
                        <Label>Secret Key</Label>
                        <div className="flex items-center space-x-2">
                          <Input type="password" value="****************************" readOnly className="font-mono" />
                          <Button size="sm" variant="outline">
                            <Copy className="w-4 h-4" />
                          </Button>
                        </div>
                      </div>
                      <Button variant="outline" className="w-full">
                        Generate New Keys
                      </Button>
                    </div>
                  </div>

                  <div className="space-y-4">
                    <h5 className="font-medium">API Endpoints</h5>
                    <div className="space-y-2 text-sm font-mono">
                      <div className="p-2 bg-muted/20 rounded">GET /api/v1/bots</div>
                      <div className="p-2 bg-muted/20 rounded">POST /api/v1/bots/start</div>
                      <div className="p-2 bg-muted/20 rounded">GET /api/v1/trades</div>
                      <div className="p-2 bg-muted/20 rounded">GET /api/v1/performance</div>
                      <div className="p-2 bg-muted/20 rounded">WebSocket: wss://api.sniprx.com/ws</div>
                    </div>
                  </div>
                </div>

                <div className="p-4 border border-border/40 rounded-lg">
                  <h5 className="font-medium mb-3">Usage Statistics</h5>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                    <div>
                      <div className="text-2xl font-bold text-primary">1,247</div>
                      <div className="text-sm text-muted-foreground">API Calls Today</div>
                    </div>
                    <div>
                      <div className="text-2xl font-bold">5,000</div>
                      <div className="text-sm text-muted-foreground">Monthly Limit</div>
                    </div>
                    <div>
                      <div className="text-2xl font-bold text-green-500">99.9%</div>
                      <div className="text-sm text-muted-foreground">Uptime</div>
                    </div>
                    <div>
                      <div className="text-2xl font-bold">45ms</div>
                      <div className="text-sm text-muted-foreground">Avg Response</div>
                    </div>
                  </div>
                </div>

                <div className="flex items-center space-x-4">
                  <Button className="gradient-primary text-black">
                    <ExternalLink className="w-4 h-4 mr-2" />
                    API Documentation
                  </Button>
                  <Button variant="outline">Download SDK</Button>
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  )
}
