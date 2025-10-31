import Header from "@/components/header"
import AIInsightsPanel from "@/components/ai-insights-panel"
import ProtectedRoute from "@/components/protected-route"
import { Badge } from "@/components/ui/badge"

export default function AIInsightsPage() {
  return (
    <ProtectedRoute>
      <div className="min-h-screen tech-dark-bg">
        <Header />
        <main className="container mx-auto px-4 py-16">
          <div className="text-center mb-12">
            <Badge className="mb-4 bg-purple-500/10 text-purple-500 border-purple-500/20">AI Insights</Badge>
            <h1 className="text-4xl font-bold mb-4">AI-Powered Trading Insights</h1>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Machine learning insights for optimal entry timing, risk management, and strategy selection
            </p>
          </div>
          <AIInsightsPanel />
        </main>
      </div>
    </ProtectedRoute>
  )
}
