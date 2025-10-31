"use client"

import type React from "react"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import { Chrome, Mail, Shield, Loader2 } from "lucide-react"
import { useAuth } from "@/hooks/use-auth"

export default function LoginForm() {
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [mt5Login, setMt5Login] = useState("")
  const [mt5Password, setMt5Password] = useState("")
  const [mt5Server, setMt5Server] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState("")

  const { login } = useAuth()
  const router = useRouter()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)
    setError("")

    try {
      const success = await login(email, password)
      if (success) {
        router.push("/dashboard")
      } else {
        setError("Invalid email or password")
      }
    } catch (err) {
      setError("Login failed. Please try again.")
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <Card className="glassmorphism">
      <CardHeader className="text-center">
        <CardTitle className="text-2xl">Sign In to SniprX</CardTitle>
        <p className="text-muted-foreground">Access your trading dashboard</p>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="relative">
          <div className="absolute inset-0 flex items-center">
            <Separator />
          </div>
          <div className="relative flex justify-center text-xs uppercase">
            <span className="bg-background px-2 text-muted-foreground">Or continue with email</span>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="email">Email</Label>
            <Input
              id="email"
              type="email"
              placeholder="trader@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              disabled={isLoading}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="password">Password</Label>
            <Input
              id="password"
              type="password"
              placeholder="Enter your password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              disabled={isLoading}
            />
          </div>

          <Separator />

          <div className="space-y-4">
            <div className="flex items-center space-x-2">
              <Shield className="w-4 h-4 text-primary" />
              <Label className="text-sm font-medium">MT5 Credentials (Optional)</Label>
            </div>

            <div className="space-y-2">
              <Label htmlFor="mt5-login">MT5 Login</Label>
              <Input
                id="mt5-login"
                placeholder="Your MT5 account number"
                value={mt5Login}
                onChange={(e) => setMt5Login(e.target.value)}
                disabled={isLoading}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="mt5-password">MT5 Password</Label>
              <Input
                id="mt5-password"
                type="password"
                placeholder="Your MT5 password"
                value={mt5Password}
                onChange={(e) => setMt5Password(e.target.value)}
                disabled={isLoading}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="mt5-server">MT5 Server</Label>
              <Input
                id="mt5-server"
                placeholder="e.g., MetaQuotes-Demo"
                value={mt5Server}
                onChange={(e) => setMt5Server(e.target.value)}
                disabled={isLoading}
              />
            </div>

            <p className="text-xs text-muted-foreground">
              🔒 Your MT5 credentials are encrypted and stored securely. We never access your funds.
            </p>
          </div>

          {error && <div className="text-sm text-red-500 text-center">{error}</div>}

          <Button type="submit" className="w-full gradient-primary text-white hover:opacity-90" disabled={isLoading}>
            {isLoading ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Signing In...
              </>
            ) : (
              <>
                <Mail className="w-4 h-4 mr-2" />
                Sign In & Access Dashboard
              </>
            )}
          </Button>
        </form>

        <div className="text-center text-sm">
          <span className="text-muted-foreground">Don't have an account? </span>
          <button className="text-primary hover:underline">Contact support for access</button>
        </div>
      </CardContent>
    </Card>
  )
}
