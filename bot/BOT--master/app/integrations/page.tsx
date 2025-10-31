import Header from "@/components/header"
import ExternalIntegrations from "@/components/external-integrations"
import ProtectedRoute from "@/components/protected-route"
import { Badge } from "@/components/ui/badge"

export default function IntegrationsPage() {
  return (
    <ProtectedRoute>
      <div className="min-h-screen tech-dark-bg">
        <Header />
        <main className="container mx-auto px-4 py-16">
          <div className="text-center mb-12">
            <Badge className="mb-4 bg-blue-500/10 text-blue-500 border-blue-500/20">Integrations</Badge>
            <h1 className="text-4xl font-bold mb-4">External Integrations</h1>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Connect TradingView, Telegram, multiple brokers, and third-party applications
            </p>
          </div>
          <ExternalIntegrations />
        </main>
      </div>
    </ProtectedRoute>
  )
}
