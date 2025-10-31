import Header from "@/components/header"
import FAQSection from "@/components/faq-section"
import ProtectedRoute from "@/components/protected-route"
import { Badge } from "@/components/ui/badge"

export default function FAQPage() {
  return (
    <ProtectedRoute>
      <div className="min-h-screen tech-dark-bg">
        <Header />
        <main className="container mx-auto px-4 py-16">
          <div className="text-center mb-12">
            <Badge className="mb-4 bg-primary/10 text-primary border-primary/20">FAQ</Badge>
            <h1 className="text-4xl font-bold mb-4">Frequently Asked Questions</h1>
            <p className="text-xl text-muted-foreground">
              Everything you need to know about SniprX premium trading platform
            </p>
          </div>
          <FAQSection />
        </main>
      </div>
    </ProtectedRoute>
  )
}
