import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { TrendingUp, Shield, Zap, Target } from "lucide-react"
import Link from "next/link"

export default function HeroSection() {
  return (
    <section className="relative py-20 lg:py-32 overflow-hidden">
      {/* Background Pattern */}
      <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-background to-primary/10" />
      <div className="absolute inset-0 bg-[url('/grid.svg')] bg-center [mask-image:linear-gradient(180deg,white,rgba(255,255,255,0))]" />

      <div className="container mx-auto px-4 relative">
        <div className="text-center max-w-4xl mx-auto">
          <Badge className="mb-6 bg-primary/10 text-primary border-primary/20 text-sm px-4 py-2">
            Built For Traders, By Traders
          </Badge>

          <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold mb-6 bg-gradient-to-r from-foreground to-foreground/70 bg-clip-text text-transparent">
            Sniper entries. <span className="gradient-primary bg-clip-text text-transparent">Professional</span>{" "}
            profits.
          </h1>

          <p className="text-xl md:text-2xl text-muted-foreground mb-8 max-w-2xl mx-auto">
            Let your bot hunt for you. Automate institutional-level strategies for XAUUSD and crypto through MT5.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center mb-12">
            <Link href="/login">
              <Button size="lg" className="gradient-primary text-white hover:opacity-90 text-lg px-8 py-6">
                Start Free Trial
              </Button>
            </Link>
            <Link href="/login">
              <Button size="lg" variant="outline" className="text-lg px-8 py-6">
                Connect to MT5
              </Button>
            </Link>
          </div>

          {/* Feature Icons */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8 max-w-3xl mx-auto">
            <div className="flex flex-col items-center">
              <div className="w-12 h-12 rounded-2xl bg-primary/10 flex items-center justify-center mb-3">
                <Target className="w-6 h-6 text-primary" />
              </div>
              <h3 className="font-semibold mb-1">Sniper Entries</h3>
              <p className="text-sm text-muted-foreground">Precision market timing</p>
            </div>
            <div className="flex flex-col items-center">
              <div className="w-12 h-12 rounded-2xl bg-primary/10 flex items-center justify-center mb-3">
                <TrendingUp className="w-6 h-6 text-primary" />
              </div>
              <h3 className="font-semibold mb-1">Institutional Logic</h3>
              <p className="text-sm text-muted-foreground">Smart money concepts</p>
            </div>
            <div className="flex flex-col items-center">
              <div className="w-12 h-12 rounded-2xl bg-primary/10 flex items-center justify-center mb-3">
                <Shield className="w-6 h-6 text-primary" />
              </div>
              <h3 className="font-semibold mb-1">Risk Management</h3>
              <p className="text-sm text-muted-foreground">Advanced position sizing</p>
            </div>
            <div className="flex flex-col items-center">
              <div className="w-12 h-12 rounded-2xl bg-primary/10 flex items-center justify-center mb-3">
                <Zap className="w-6 h-6 text-primary" />
              </div>
              <h3 className="font-semibold mb-1">24/7 Trading</h3>
              <p className="text-sm text-muted-foreground">Never miss opportunities</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
