"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Separator } from "@/components/ui/separator"
import { Wifi, WifiOff, Shield, AlertTriangle, CheckCircle, Loader2, Server, User, Key } from "lucide-react"

interface MT5Account {
  login: string
  server: string
  name: string
  balance: number
  equity: number
  margin: number
  freeMargin: number
  connected: boolean
}

export default function MT5Connector() {
  const [isConnecting, setIsConnecting] = useState(false)
  const [showConnector, setShowConnector] = useState(false)
  const [mt5Accounts, setMt5Accounts] = useState<MT5Account[]>([])
  const [newAccount, setNewAccount] = useState({
    login: "",
    password: "",
    server: "",
    name: "",
  })

  const popularServers = [
    "MetaQuotes-Demo",
    "ICMarkets-Live01",
    "ICMarkets-Live02",
    "Pepperstone-Live",
    "FTMO-Server",
    "MyFundedFX-Live",
    "Admirals-Live",
    "XM-Real",
  ]

  // Fetch accounts from backend on mount
  useEffect(() => {
    fetch(`/api/mt5/accounts`)
      .then(res => res.json())
      .then(data => setMt5Accounts(data))
  }, [])

  const handleConnect = async () => {
    setIsConnecting(true)
    try {
      const res = await fetch(`/api/mt5/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(newAccount),
      })
      const result = await res.json()
      if (result.success) {
        // Fetch updated accounts list
        const accRes = await fetch(`/api/mt5/accounts`)
        const accData = await accRes.json()
        setMt5Accounts(accData)
        setShowConnector(false)
        setNewAccount({ login: "", password: "", server: "", name: "" })
      }
    } finally {
      setIsConnecting(false)
    }
  }

  const handleDisconnect = (login: string) => {
    setMt5Accounts((prev) => prev.map((acc) => (acc.login === login ? { ...acc, connected: false } : acc)))
  }

  const handleReconnect = async (login: string) => {
    setMt5Accounts((prev) => prev.map((acc) => (acc.login === login ? { ...acc, connected: true } : acc)))
  }

  return (
    <Card className="glassmorphism border-primary/20">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center">
            <Server className="w-5 h-5 mr-2 text-primary" />
            MT5 Accounts
          </CardTitle>
          <Dialog open={showConnector} onOpenChange={setShowConnector}>
            <DialogTrigger asChild>
              <Button size="sm" className="gradient-primary text-black">
                <Wifi className="w-4 h-4 mr-2" />
                Add Account
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-md">
              <DialogHeader>
                <DialogTitle className="flex items-center">
                  <Shield className="w-5 h-5 mr-2 text-primary" />
                  Connect MT5 Account
                </DialogTitle>
              </DialogHeader>

              <div className="space-y-4">
                <div className="p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg">
                  <div className="flex items-start space-x-2">
                    <Shield className="w-4 h-4 text-blue-500 mt-0.5" />
                    <div className="text-sm">
                      <p className="font-medium text-blue-500">Secure Connection</p>
                      <p className="text-muted-foreground">
                        Your credentials are encrypted and never stored on our servers.
                      </p>
                    </div>
                  </div>
                </div>

                <div className="space-y-3">
                  <div>
                    <Label htmlFor="account-name" className="flex items-center">
                      <User className="w-4 h-4 mr-2" />
                      Account Name (Optional)
                    </Label>
                    <Input
                      id="account-name"
                      placeholder="My Trading Account"
                      value={newAccount.name}
                      onChange={(e) => setNewAccount((prev) => ({ ...prev, name: e.target.value }))}
                    />
                  </div>

                  <div>
                    <Label htmlFor="mt5-login" className="flex items-center">
                      <User className="w-4 h-4 mr-2" />
                      MT5 Login Number
                    </Label>
                    <Input
                      id="mt5-login"
                      placeholder="12345678"
                      value={newAccount.login}
                      onChange={(e) => setNewAccount((prev) => ({ ...prev, login: e.target.value }))}
                      required
                    />
                  </div>

                  <div>
                    <Label htmlFor="mt5-password" className="flex items-center">
                      <Key className="w-4 h-4 mr-2" />
                      MT5 Password
                    </Label>
                    <Input
                      id="mt5-password"
                      type="password"
                      placeholder="Your MT5 password"
                      value={newAccount.password}
                      onChange={(e) => setNewAccount((prev) => ({ ...prev, password: e.target.value }))}
                      required
                    />
                  </div>

                  <div>
                    <Label htmlFor="mt5-server" className="flex items-center">
                      <Server className="w-4 h-4 mr-2" />
                      MT5 Server
                    </Label>
                    <Select
                      value={newAccount.server}
                      onValueChange={(value) => setNewAccount((prev) => ({ ...prev, server: value }))}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select your broker server" />
                      </SelectTrigger>
                      <SelectContent>
                        {popularServers.map((server) => (
                          <SelectItem key={server} value={server}>
                            {server}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <Separator />

                <Button
                  onClick={handleConnect}
                  disabled={!newAccount.login || !newAccount.password || !newAccount.server || isConnecting}
                  className="w-full gradient-primary text-black"
                >
                  {isConnecting ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Connecting...
                    </>
                  ) : (
                    <>
                      <Wifi className="w-4 h-4 mr-2" />
                      Connect Account
                    </>
                  )}
                </Button>

                <p className="text-xs text-muted-foreground text-center">
                  🔒 Connection is secured with bank-level encryption
                </p>
              </div>
            </DialogContent>
          </Dialog>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {mt5Accounts.length === 0 ? (
          <div className="text-center py-8">
            <WifiOff className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
            <h3 className="font-semibold mb-2">No MT5 Accounts Connected</h3>
            <p className="text-muted-foreground text-sm mb-4">Connect your MT5 account to start automated trading</p>
            <Button onClick={() => setShowConnector(true)} className="gradient-primary text-black">
              <Wifi className="w-4 h-4 mr-2" />
              Connect First Account
            </Button>
          </div>
        ) : (
          mt5Accounts.map((account, index) => (
            <div key={index} className="border border-border/40 rounded-lg p-4 space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div
                    className={`w-3 h-3 rounded-full ${account.connected ? "bg-green-500" : "bg-red-500"} animate-pulse`}
                  />
                  <div>
                    <h4 className="font-medium">{account.name}</h4>
                    <p className="text-sm text-muted-foreground">
                      {account.login} • {account.server}
                    </p>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <Badge
                    className={
                      account.connected
                        ? "bg-green-500/10 text-green-500 border-green-500/20"
                        : "bg-red-500/10 text-red-500 border-red-500/20"
                    }
                  >
                    {account.connected ? (
                      <>
                        <CheckCircle className="w-3 h-3 mr-1" />
                        Connected
                      </>
                    ) : (
                      <>
                        <AlertTriangle className="w-3 h-3 mr-1" />
                        Disconnected
                      </>
                    )}
                  </Badge>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() =>
                      account.connected ? handleDisconnect(account.login) : handleReconnect(account.login)
                    }
                  >
                    {account.connected ? "Disconnect" : "Reconnect"}
                  </Button>
                </div>
              </div>

              {account.connected && (
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-muted-foreground">Balance:</span>
                    <div className="font-mono font-medium">${account.balance.toFixed(2)}</div>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Equity:</span>
                    <div
                      className={`font-mono font-medium ${account.equity >= account.balance ? "text-green-500" : "text-red-500"}`}
                    >
                      ${account.equity.toFixed(2)}
                    </div>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Margin:</span>
                    <div className="font-mono font-medium">${account.margin.toFixed(2)}</div>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Free Margin:</span>
                    <div className="font-mono font-medium">${account.freeMargin.toFixed(2)}</div>
                  </div>
                </div>
              )}
            </div>
          ))
        )}

        <div className="pt-4 border-t border-border/40">
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Total Connected:</span>
            <span className="font-medium">{mt5Accounts.filter((acc) => acc.connected).length} accounts</span>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
