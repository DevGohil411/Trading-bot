import Header from "@/components/header"
import BotControls from "@/components/bot-controls"
import ProtectedRoute from "@/components/protected-route"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Bot, Brain, Target, Activity } from "lucide-react"

export default function AIBotPage() {
  return (
    <ProtectedRoute>
      <div className="min-h-screen bg-background">
        <Header />
        <main className="container mx-auto px-4 py-6">
          <div className="text-center mb-8">
            <Badge className="mb-4 bg-purple-500/10 text-purple-500 border-purple-500/20">AI Trading Bot</Badge>
            <h1 className="text-3xl font-bold mb-2">AI Trading Bot</h1>
            <p className="text-muted-foreground">Advanced AI-powered trading automation</p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-1">
              <BotControls />
            </div>

            <div className="lg:col-span-2 space-y-6">
              {/* AI Bot Features */}
              <Card className="glassmorphism">
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <Brain className="w-5 h-5 mr-2 text-primary" />
                    AI Intelligence
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="p-4 border rounded-lg">
                      <div className="flex items-center mb-2">
                        <Target className="w-4 h-4 mr-2 text-green-500" />
                        <span className="font-medium">Pattern Recognition</span>
                      </div>
                      <p className="text-sm text-muted-foreground">AI identifies market patterns and entry points</p>
                    </div>

                    <div className="p-4 border rounded-lg">
                      <div className="flex items-center mb-2">
                        <Activity className="w-4 h-4 mr-2 text-blue-500" />
                        <span className="font-medium">Risk Management</span>
                      </div>
                      <p className="text-sm text-muted-foreground">Intelligent position sizing and stop losses</p>
                    </div>

                    <div className="p-4 border rounded-lg">
                      <div className="flex items-center mb-2">
                        <Bot className="w-4 h-4 mr-2 text-purple-500" />
                        <span className="font-medium">Market Analysis</span>
                      </div>
                      <p className="text-sm text-muted-foreground">Real-time market sentiment analysis</p>
                    </div>

                    <div className="p-4 border rounded-lg">
                      <div className="flex items-center mb-2">
                        <Brain className="w-4 h-4 mr-2 text-orange-500" />
                        <span className="font-medium">Learning Algorithm</span>
                      </div>
                      <p className="text-sm text-muted-foreground">Continuously improves trading decisions</p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Bot Performance */}
              <Card className="glassmorphism">
                <CardHeader>
                  <CardTitle>AI Bot Performance</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-64 bg-muted/20 rounded-lg flex items-center justify-center">
                    <p className="text-muted-foreground">AI Performance Analytics</p>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </main>
      </div>
    </ProtectedRoute>
  )
}
