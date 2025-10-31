import Header from "@/components/header"
import MT5Settings from "@/components/mt5-settings"
import TelegramBotManager from "@/components/telegram-bot-manager"
import ProtectedRoute from "@/components/protected-route"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Settings, User, Bell, Shield } from "lucide-react"

export default function SettingsPage() {
  return (
    <ProtectedRoute>
      <div className="min-h-screen tech-dark-bg">
        <Header />
        <main className="container mx-auto px-4 py-16">
          <div className="text-center mb-12">
            <Badge className="mb-4 bg-primary/10 text-primary border-primary/20">Settings</Badge>
            <h1 className="text-4xl font-bold mb-4">Customize Your Trading Experience</h1>
            <p className="text-xl text-muted-foreground">Fine-tune every aspect of your automated trading setup</p>
          </div>

          <div className="max-w-6xl mx-auto">
            <Tabs defaultValue="trading" className="space-y-8">
              <TabsList className="grid w-full grid-cols-4">
                <TabsTrigger value="trading" className="flex items-center">
                  <Settings className="w-4 h-4 mr-2" />
                  Trading
                </TabsTrigger>
                <TabsTrigger value="account" className="flex items-center">
                  <User className="w-4 h-4 mr-2" />
                  Account
                </TabsTrigger>
                <TabsTrigger value="notifications" className="flex items-center">
                  <Bell className="w-4 h-4 mr-2" />
                  Alerts
                </TabsTrigger>
                <TabsTrigger value="security" className="flex items-center">
                  <Shield className="w-4 h-4 mr-2" />
                  Security
                </TabsTrigger>
              </TabsList>

              <TabsContent value="trading">
                <MT5Settings />
              </TabsContent>

              <TabsContent value="account">
                <div className="text-center py-12">
                  <h3 className="text-lg font-semibold mb-2">Account Settings</h3>
                  <p className="text-muted-foreground">Profile and subscription management coming soon...</p>
                </div>
              </TabsContent>

              <TabsContent value="notifications">
                <div className="space-y-6">
                  <div className="text-center mb-8">
                    <h3 className="text-2xl font-semibold mb-2">Notification Settings</h3>
                    <p className="text-muted-foreground">Configure how you receive trading alerts and updates</p>
                  </div>
                  <TelegramBotManager />
                </div>
              </TabsContent>

              <TabsContent value="security">
                <div className="text-center py-12">
                  <h3 className="text-lg font-semibold mb-2">Security Settings</h3>
                  <p className="text-muted-foreground">Two-factor authentication and security options coming soon...</p>
                </div>
              </TabsContent>
            </Tabs>
          </div>
        </main>
      </div>
    </ProtectedRoute>
  )
}
