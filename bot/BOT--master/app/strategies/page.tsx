import Header from "@/components/header"
import StrategiesSection from "@/components/strategies-section"
import ProtectedRoute from "@/components/protected-route"
import { Badge } from "@/components/ui/badge"

export default function StrategiesPage() {
  return (
    <ProtectedRoute>
      <div className="min-h-screen tech-dark-bg">
        <Header />
        <main className="container mx-auto px-4 py-16">
          <div className="text-center mb-12">
            <Badge className="mb-4 bg-primary/10 text-primary border-primary/20">Battle-Tested</Badge>
            <h1 className="text-4xl font-bold mb-4">Our Battle-Tested Strategies</h1>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Premium algorithmic strategies designed for institutional-level performance
            </p>
          </div>
          <StrategiesSection />
        </main>
      </div>
    </ProtectedRoute>
  )
}
