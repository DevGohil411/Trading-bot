import Header from "@/components/header"
import MT5Connector from "@/components/mt5-connector"
import ProtectedRoute from "@/components/protected-route"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent } from "@/components/ui/card"
import { Shield, Zap, CheckCircle, Server } from "lucide-react"

export default function MT5Page() {
  return (
    <ProtectedRoute>
      <div className="min-h-screen tech-dark-bg">
        <Header />
        <main className="container mx-auto px-4 py-16">
          <div className="text-center mb-12">
            <Badge className="mb-4 bg-primary/10 text-primary border-primary/20">MT5 Integration</Badge>
            <h1 className="text-4xl font-bold mb-4">Connect Your MT5 Accounts</h1>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Securely connect multiple MT5 accounts and start automated trading with institutional strategies
            </p>
          </div>

          <div className="max-w-4xl mx-auto space-y-8">
            {/* Features */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
              <Card className="glassmorphism border-primary/20">
                <CardContent className="p-6 text-center">
                  <Shield className="w-8 h-8 text-primary mx-auto mb-3" />
                  <h3 className="font-semibold mb-2">Bank-Level Security</h3>
                  <p className="text-sm text-muted-foreground">
                    Your credentials are encrypted and never stored on our servers
                  </p>
                </CardContent>
              </Card>

              <Card className="glassmorphism border-primary/20">
                <CardContent className="p-6 text-center">
                  <Zap className="w-8 h-8 text-primary mx-auto mb-3" />
                  <h3 className="font-semibold mb-2">Lightning Fast</h3>
                  <p className="text-sm text-muted-foreground">Sub-second trade execution with minimal slippage</p>
                </CardContent>
              </Card>

              <Card className="glassmorphism border-primary/20">
                <CardContent className="p-6 text-center">
                  <Server className="w-8 h-8 text-primary mx-auto mb-3" />
                  <h3 className="font-semibold mb-2">Multi-Account</h3>
                  <p className="text-sm text-muted-foreground">
                    Connect multiple accounts and manage them from one dashboard
                  </p>
                </CardContent>
              </Card>
            </div>

            {/* MT5 Connector */}
            <MT5Connector />

            {/* Supported Brokers */}
            <Card className="glassmorphism border-primary/20">
              <CardContent className="p-6">
                <h3 className="text-lg font-semibold mb-4 flex items-center">
                  <CheckCircle className="w-5 h-5 mr-2 text-green-500" />
                  Supported Brokers
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
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
                    <span>And many more...</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </main>
      </div>
    </ProtectedRoute>
  )
}
