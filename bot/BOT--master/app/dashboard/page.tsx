import Header from "@/components/header"
import DashboardStats from "@/components/dashboard-stats"
import BotControls from "@/components/bot-controls"
import TradingChart from "@/components/trading-chart"
import PerformanceChart from "@/components/performance-chart"
import SmartSummary from "@/components/smart-summary"
import ActivityLog from "@/components/activity-log"
import MT5Connector from "@/components/mt5-connector"
import ProtectedRoute from "@/components/protected-route"
import ChatBot from "@/components/chatbot"
import OnboardingTour from "@/components/onboarding-tour"
import MT5TradePanel from "@/components/mt5-trade-panel"

export default function DashboardPage() {
  return (
    <ProtectedRoute>
      <div className="min-h-screen tech-dark-bg">
        <Header />
        <main className="container mx-auto px-4 py-6">
          {/* Smart Summary Banner */}
          <SmartSummary />

          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-8">
            <div className="lg:col-span-3" data-tour="dashboard-stats">
              <DashboardStats />
            </div>
            <div data-tour="bot-controls">
              <BotControls />
            </div>
          </div>

          <div className="grid grid-cols-1 xl:grid-cols-3 gap-6 mb-8">
            <div className="xl:col-span-2">
              <TradingChart />
            </div>
            <div data-tour="analytics">
              <PerformanceChart />
            </div>
          </div>

          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
            <ActivityLog />
            <div data-tour="mt5-connector">
              <MT5Connector />
            </div>
          </div>
          <div className="mt-8">
            <MT5TradePanel />
          </div>
        </main>
        <ChatBot />
        <OnboardingTour />
      </div>
    </ProtectedRoute>
  )
}
