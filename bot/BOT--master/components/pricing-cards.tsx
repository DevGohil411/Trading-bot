import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Check, Star, Zap } from "lucide-react"

const plans = [
  {
    name: "Strategy Trial",
    price: "$0",
    period: "3 days",
    description: "Test any single strategy",
    features: [
      "Choose 1 strategy to test",
      "Demo account trading only",
      "Basic performance tracking",
      "Email support",
      "Strategy documentation",
    ],
    popular: false,
    cta: "Start Free Trial",
    highlight: false,
  },
  {
    name: "Premium Monthly",
    price: "$197",
    period: "per month",
    description: "Full access to all strategies",
    features: [
      "All 7 battle-tested strategies",
      "Live MT5 account trading",
      "Advanced risk management",
      "Real-time analytics dashboard",
      "Priority support",
      "Killzone filters",
      "Multi-bot support",
      "Strategy customization",
    ],
    popular: true,
    cta: "Go Premium",
    highlight: true,
  },
  {
    name: "Premium Annual",
    price: "$1,497",
    period: "per year",
    description: "Best value for serious traders",
    features: [
      "Everything in Monthly Plan",
      "4 months free (33% savings)",
      "VIP support & priority",
      "Advanced strategy access",
      "Custom indicator development",
      "1-on-1 strategy sessions",
      "Referral program access",
      "TradingView webhook integration",
      "White-label bot options",
    ],
    popular: false,
    cta: "Save 33%",
    highlight: false,
  },
]

export default function PricingCards() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-6xl mx-auto">
      {plans.map((plan, index) => (
        <Card
          key={index}
          className={`glassmorphism relative ${plan.popular ? "border-primary/40 scale-105" : "border-primary/20"}`}
        >
          {plan.popular && (
            <div className="absolute -top-3 left-1/2 transform -translate-x-1/2">
              <Badge className="gradient-primary text-black">
                <Star className="w-3 h-3 mr-1" />
                Most Popular
              </Badge>
            </div>
          )}
          <CardHeader className="text-center">
            <CardTitle className="text-2xl">{plan.name}</CardTitle>
            <div className="mt-4">
              <span className="text-4xl font-bold">{plan.price}</span>
              <span className="text-muted-foreground">/{plan.period}</span>
            </div>
            <p className="text-muted-foreground mt-2">{plan.description}</p>
          </CardHeader>
          <CardContent className="space-y-6">
            <ul className="space-y-3">
              {plan.features.map((feature, featureIndex) => (
                <li key={featureIndex} className="flex items-center">
                  <Check className="w-4 h-4 text-primary mr-3 flex-shrink-0" />
                  <span className="text-sm">{feature}</span>
                </li>
              ))}
            </ul>
            <Button
              className={`w-full ${plan.popular ? "gradient-primary text-black hover:opacity-90" : ""}`}
              variant={plan.popular ? "default" : "outline"}
            >
              {plan.highlight && <Zap className="w-4 h-4 mr-2" />}
              {plan.cta}
            </Button>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}
