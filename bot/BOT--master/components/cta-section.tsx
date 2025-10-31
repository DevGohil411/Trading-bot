import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import Link from "next/link"

export default function CTASection() {
  return (
    <section className="py-20">
      <div className="container mx-auto px-4">
        <div className="max-w-4xl mx-auto text-center">
          <Badge className="mb-6 bg-primary/10 text-primary border-primary/20">Ready to Start?</Badge>
          <h2 className="text-3xl md:text-5xl font-bold mb-6">Join 2,800+ Traders Using SniprX</h2>
          <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
            Start your 7-day free trial today. No credit card required. Connect your MT5 account and watch your
            strategies come to life.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link href="/login">
              <Button size="lg" className="gradient-primary text-white hover:opacity-90 text-lg px-8 py-6">
                Start Free Trial
              </Button>
            </Link>
            <Link href="/login">
              <Button size="lg" variant="outline" className="text-lg px-8 py-6">
                View Strategies
              </Button>
            </Link>
          </div>
        </div>
      </div>
    </section>
  )
}
