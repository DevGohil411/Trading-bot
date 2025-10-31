import Header from "@/components/header"
import ProtectedRoute from "@/components/protected-route"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent } from "@/components/ui/card"
import { Target, Users, TrendingUp, Shield, Zap, Brain } from "lucide-react"

export default function AboutPage() {
  return (
    <ProtectedRoute>
      <div className="min-h-screen tech-dark-bg">
        <Header />
        <main className="container mx-auto px-4 py-16">
          <div className="text-center mb-12">
            <Badge className="mb-4 bg-primary/10 text-primary border-primary/20">About SniprX</Badge>
            <h1 className="text-4xl font-bold mb-4">Built for Traders, By Traders</h1>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              The premium algorithmic trading platform democratizing institutional-level strategies
            </p>
          </div>

          <div className="max-w-6xl mx-auto space-y-12">
            {/* Mission */}
            <div className="text-center">
              <h2 className="text-3xl font-bold mb-6">Our Mission</h2>
              <p className="text-lg text-muted-foreground max-w-4xl mx-auto">
                To bridge the gap between manual trading expertise and automated execution. SniprX combines years of
                institutional trading experience with cutting-edge AI technology to deliver consistent, profitable
                results while you focus on what matters most.
              </p>
            </div>

            {/* Features Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
              <Card className="glassmorphism border-primary/20">
                <CardContent className="p-6">
                  <div className="flex items-center mb-4">
                    <Target className="w-8 h-8 text-primary mr-3" />
                    <h3 className="text-xl font-semibold">Precision Trading</h3>
                  </div>
                  <p className="text-muted-foreground">
                    Our algorithms are specifically tuned to capture unique market patterns and volatility cycles in
                    XAUUSD, delivering sniper-like precision in entry and exit points.
                  </p>
                </CardContent>
              </Card>

              <Card className="glassmorphism border-primary/20">
                <CardContent className="p-6">
                  <div className="flex items-center mb-4">
                    <Brain className="w-8 h-8 text-primary mr-3" />
                    <h3 className="text-xl font-semibold">AI-Powered Intelligence</h3>
                  </div>
                  <p className="text-muted-foreground">
                    Advanced machine learning algorithms continuously analyze market conditions, adapting to changing
                    environments and improving performance over time.
                  </p>
                </CardContent>
              </Card>

              <Card className="glassmorphism border-primary/20">
                <CardContent className="p-6">
                  <div className="flex items-center mb-4">
                    <Shield className="w-8 h-8 text-primary mr-3" />
                    <h3 className="text-xl font-semibold">Risk Management</h3>
                  </div>
                  <p className="text-muted-foreground">
                    Every strategy includes institutional-grade risk management protocols with intelligent position
                    sizing, dynamic stop losses, and daily drawdown limits.
                  </p>
                </CardContent>
              </Card>

              <Card className="glassmorphism border-primary/20">
                <CardContent className="p-6">
                  <div className="flex items-center mb-4">
                    <Users className="w-8 h-8 text-primary mr-3" />
                    <h3 className="text-xl font-semibold">Community Driven</h3>
                  </div>
                  <p className="text-muted-foreground">
                    Our strategies are developed and refined based on feedback from our community of successful traders,
                    ensuring real-world performance optimization.
                  </p>
                </CardContent>
              </Card>

              <Card className="glassmorphism border-primary/20">
                <CardContent className="p-6">
                  <div className="flex items-center mb-4">
                    <TrendingUp className="w-8 h-8 text-primary mr-3" />
                    <h3 className="text-xl font-semibold">Proven Results</h3>
                  </div>
                  <p className="text-muted-foreground">
                    Backtested across multiple market conditions with consistently superior risk-adjusted returns
                    compared to traditional trading approaches.
                  </p>
                </CardContent>
              </Card>

              <Card className="glassmorphism border-primary/20">
                <CardContent className="p-6">
                  <div className="flex items-center mb-4">
                    <Zap className="w-8 h-8 text-primary mr-3" />
                    <h3 className="text-xl font-semibold">Lightning Execution</h3>
                  </div>
                  <p className="text-muted-foreground">
                    Sub-second trade execution with direct MT5 integration ensures minimal slippage and maximum profit
                    capture during volatile market conditions.
                  </p>
                </CardContent>
              </Card>
            </div>

            {/* Stats */}
            <Card className="glassmorphism border-primary/20">
              <CardContent className="p-8">
                <h3 className="text-2xl font-semibold mb-6 text-center">Trusted by Premium Traders Worldwide</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-8 text-center">
                  <div>
                    <div className="text-3xl font-bold text-primary">2,800+</div>
                    <div className="text-sm text-muted-foreground">Premium Users</div>
                  </div>
                  <div>
                    <div className="text-3xl font-bold text-primary">$25M+</div>
                    <div className="text-sm text-muted-foreground">Volume Traded</div>
                  </div>
                  <div>
                    <div className="text-3xl font-bold text-primary">84%</div>
                    <div className="text-sm text-muted-foreground">Average Win Rate</div>
                  </div>
                  <div>
                    <div className="text-3xl font-bold text-primary">24/7</div>
                    <div className="text-sm text-muted-foreground">Market Coverage</div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Team */}
            <div className="text-center">
              <h2 className="text-3xl font-bold mb-6">Our Team</h2>
              <p className="text-lg text-muted-foreground max-w-4xl mx-auto">
                SniprX was founded by a team of institutional traders and AI engineers with over 20 years of combined
                experience in financial markets and algorithmic trading. We understand the challenges traders face
                because we've been there ourselves - from prop trading floors to hedge funds, we've seen what works and
                what doesn't.
              </p>
            </div>
          </div>
        </main>
      </div>
    </ProtectedRoute>
  )
}
