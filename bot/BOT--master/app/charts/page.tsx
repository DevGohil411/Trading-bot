import Header from "@/components/header"
import TradingChart from "@/components/trading-chart"
import ProtectedRoute from "@/components/protected-route"
import { Badge } from "@/components/ui/badge"

export default function ChartsPage() {
  return (
    <ProtectedRoute>
      <div className="min-h-screen bg-background">
        <Header />
        <main className="container mx-auto px-4 py-6">
          <div className="text-center mb-8">
            <Badge className="mb-4 bg-green-500/10 text-green-500 border-green-500/20">Live Charts</Badge>
            <h1 className="text-3xl font-bold mb-2">TradingView Charts</h1>
            <p className="text-muted-foreground">Real-time market analysis and charting tools</p>
          </div>

          <div className="space-y-6">
            <TradingChart />

            {/* Additional Chart Features */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="glassmorphism p-6 rounded-lg">
                <h3 className="text-lg font-semibold mb-4">Market Overview</h3>
                <div className="h-64 bg-muted/20 rounded-lg flex items-center justify-center">
                  <p className="text-muted-foreground">Market Overview Widget</p>
                </div>
              </div>

              <div className="glassmorphism p-6 rounded-lg">
                <h3 className="text-lg font-semibold mb-4">Economic Calendar</h3>
                <div className="h-64 bg-muted/20 rounded-lg flex items-center justify-center">
                  <p className="text-muted-foreground">Economic Calendar Widget</p>
                </div>
              </div>
            </div>
          </div>
        </main>
      </div>
    </ProtectedRoute>
  )
}
