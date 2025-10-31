import Header from "@/components/header"
import MultiBotManager from "@/components/multi-bot-manager"
import ProtectedRoute from "@/components/protected-route"
import { Badge } from "@/components/ui/badge"

export default function MultiBotPage() {
  return (
    <ProtectedRoute>
      <div className="min-h-screen tech-dark-bg">
        <Header />
        <main className="container mx-auto px-4 py-16">
          <div className="text-center mb-12">
            <Badge className="mb-4 bg-primary/10 text-primary border-primary/20">Multi-Bot Management</Badge>
            <h1 className="text-4xl font-bold mb-4">Multi-Bot Trading System</h1>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Run multiple trading bots simultaneously with different strategies, symbols, and risk parameters
            </p>
          </div>
          <MultiBotManager />
        </main>
      </div>
    </ProtectedRoute>
  )
}
