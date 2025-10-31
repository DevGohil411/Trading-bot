import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Shield, TrendingUp, Clock, Target, Brain, Activity } from "lucide-react"

const features = [
  {
    icon: Target,
    title: "Sniper Entries",
    description: "Precision market timing with institutional-level entry logic and smart money concepts.",
  },
  {
    icon: Brain,
    title: "AI-Powered Strategies",
    description: "Machine learning algorithms trained on years of XAUUSD and crypto market data.",
  },
  {
    icon: Shield,
    title: "Advanced Risk Management",
    description: "Built-in stop losses, position sizing, and daily drawdown protection.",
  },
  {
    icon: Activity,
    title: "Real-Time Execution",
    description: "Lightning-fast trade execution with direct MT5 integration for minimal slippage.",
  },
  {
    icon: Clock,
    title: "24/7 Market Monitoring",
    description: "Never miss a trading opportunity with round-the-clock market surveillance.",
  },
  {
    icon: TrendingUp,
    title: "Live Performance Tracking",
    description: "Real-time analytics with detailed statistics and trade history analysis.",
  },
]

export default function FeaturesSection() {
  return (
    <section className="py-20">
      <div className="container mx-auto px-4">
        <div className="text-center mb-16">
          <Badge className="mb-4 bg-primary/10 text-primary border-primary/20">Features</Badge>
          <h2 className="text-3xl md:text-4xl font-bold mb-4">Everything You Need to Trade Like a Pro</h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Powerful features designed to maximize your trading potential while minimizing risk
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <Card key={index} className="glassmorphism hover:shadow-lg transition-shadow">
              <CardHeader>
                <div className="w-12 h-12 rounded-2xl bg-primary/10 flex items-center justify-center mb-4">
                  <feature.icon className="w-6 h-6 text-primary" />
                </div>
                <CardTitle className="text-xl">{feature.title}</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground">{feature.description}</p>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  )
}
