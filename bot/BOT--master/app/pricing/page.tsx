import Header from "@/components/header"
import PricingCards from "@/components/pricing-cards"
import ProtectedRoute from "@/components/protected-route"
import { Badge } from "@/components/ui/badge"

export default function PricingPage() {
  return (
    <ProtectedRoute>
      <div className="min-h-screen tech-dark-bg">
        <Header />
        <main className="container mx-auto px-4 py-16">
          <div className="text-center mb-12">
            <Badge className="mb-4 bg-primary/10 text-primary border-primary/20">Premium Pricing</Badge>
            <h1 className="text-4xl font-bold mb-4">Choose Your Trading Plan</h1>
            <p className="text-xl text-muted-foreground">
              Start with a free trial, then scale with our premium algorithmic strategies
            </p>
          </div>
          <PricingCards />
        </main>
      </div>
    </ProtectedRoute>
  )
}
