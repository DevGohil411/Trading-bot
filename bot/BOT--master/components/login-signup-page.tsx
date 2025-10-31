"use client"

import type React from "react"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Chrome, Mail, Shield, Loader2, Target, Zap } from "lucide-react"
import { useAuth } from "@/hooks/use-auth"
import { ThemeToggle } from "@/components/theme-toggle"

export default function LoginSignupPage() {
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState("")
  const [activeTab, setActiveTab] = useState("login")

  const { login } = useAuth()
  const router = useRouter()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)
    setError("")

    if (activeTab === "signup" && password !== confirmPassword) {
      setError("Passwords don't match")
      setIsLoading(false)
      return
    }

    try {
      const success = await login(email, password)
      if (success) {
        router.push("/dashboard")
      } else {
        setError("Invalid credentials")
      }
    } catch (err) {
      setError("Authentication failed. Please try again.")
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen tech-dark-bg flex items-center justify-center p-4 relative overflow-hidden">
      {/* Background Effects */}
      <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-transparent to-primary/10" />
      <div className="absolute top-10 right-10">
        <ThemeToggle />
      </div>

      {/* Floating Elements */}
      <div className="absolute top-20 left-20 w-32 h-32 bg-primary/10 rounded-full blur-xl animate-float" />
      <div
        className="absolute bottom-20 right-20 w-24 h-24 bg-primary/5 rounded-full blur-xl animate-float"
        style={{ animationDelay: "2s" }}
      />

      <div className="w-full max-w-md relative z-10">
        {/* Logo */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center space-x-3 mb-6">
            <div className="w-16 h-16 gradient-primary rounded-2xl flex items-center justify-center shadow-2xl">
              <Target className="w-8 h-8 text-black" />
            </div>
            <div>
              <h1 className="text-4xl font-bold">SniprX</h1>
              <p className="text-sm text-muted-foreground">Premium Trading Bot</p>
            </div>
          </div>
          <div className="flex items-center justify-center space-x-2 text-primary">
            <Zap className="w-4 h-4" />
            <span className="text-sm font-medium">Algorithmic Trading Platform</span>
          </div>
        </div>

        <Card className="glassmorphism shadow-2xl">
          <CardHeader className="text-center">
            <CardTitle className="text-2xl">Access SniprX Platform</CardTitle>
            <p className="text-muted-foreground">Connect your MT5 and automate your trading</p>
          </CardHeader>
          <CardContent>
            <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
              <TabsList className="grid w-full grid-cols-2 mb-6">
                <TabsTrigger value="login">Login</TabsTrigger>
                <TabsTrigger value="signup">Sign Up</TabsTrigger>
              </TabsList>

              <TabsContent value="login" className="space-y-4">
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

                  <Button className="w-full" variant="outline" disabled={isLoading} onClick={() => window.location.href = '/auth/google'}> 
                    <Chrome className="w-4 h-4 mr-2" />
                    Continue with Google
                  </Button>

                  {error && <div className="text-sm text-red-500 text-center">{error}</div>}

                  <Button
                    type="submit"
                    className="w-full gradient-primary text-black hover:opacity-90"
                    disabled={isLoading}
                  >
                    {isLoading ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        Signing In...
                      </>
                    ) : (
                      <>
                        <Mail className="w-4 h-4 mr-2" />
                        Access Dashboard
                      </>
                    )}
                  </Button>
                </form>
              </TabsContent>

              <TabsContent value="signup" className="space-y-4">
                <div className="relative">
                  <div className="absolute inset-0 flex items-center">
                    <Separator />
                  </div>
                  <div className="relative flex justify-center text-xs uppercase">
                    <span className="bg-background px-2 text-muted-foreground">Or create account</span>
                  </div>
                </div>

                <form onSubmit={handleSubmit} className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="signup-email">Email</Label>
                    <Input
                      id="signup-email"
                      type="email"
                      placeholder="trader@example.com"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      required
                      disabled={isLoading}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="signup-password">Password</Label>
                    <Input
                      id="signup-password"
                      type="password"
                      placeholder="Create a strong password"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      required
                      disabled={isLoading}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="confirm-password">Confirm Password</Label>
                    <Input
                      id="confirm-password"
                      type="password"
                      placeholder="Confirm your password"
                      value={confirmPassword}
                      onChange={(e) => setConfirmPassword(e.target.value)}
                      required
                      disabled={isLoading}
                    />
                  </div>

                  <Button className="w-full" variant="outline" disabled={isLoading} onClick={() => window.location.href = '/auth/google'}> 
                    <Chrome className="w-4 h-4 mr-2" />
                    Continue with Google
                  </Button>

                  {error && <div className="text-sm text-red-500 text-center">{error}</div>}

                  <Button
                    type="submit"
                    className="w-full gradient-primary text-black hover:opacity-90"
                    disabled={isLoading}
                  >
                    {isLoading ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        Creating Account...
                      </>
                    ) : (
                      <>
                        <Shield className="w-4 h-4 mr-2" />
                        Create Account
                      </>
                    )}
                  </Button>
                </form>
              </TabsContent>
            </Tabs>

            <div className="mt-6 text-center text-xs text-muted-foreground">
              By continuing, you agree to our Terms of Service and Privacy Policy
            </div>
          </CardContent>
        </Card>

        {/* Features Preview */}
        <div className="mt-8 grid grid-cols-3 gap-4 text-center">
          <div className="glassmorphism p-4 rounded-2xl">
            <div className="text-2xl mb-2">🎯</div>
            <p className="text-xs text-muted-foreground">Precision Trading</p>
          </div>
          <div className="glassmorphism p-4 rounded-2xl">
            <div className="text-2xl mb-2">🤖</div>
            <p className="text-xs text-muted-foreground">AI Automation</p>
          </div>
          <div className="glassmorphism p-4 rounded-2xl">
            <div className="text-2xl mb-2">📊</div>
            <p className="text-xs text-muted-foreground">Live Analytics</p>
          </div>
        </div>
      </div>
    </div>
  )
}
