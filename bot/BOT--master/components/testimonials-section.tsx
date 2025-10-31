import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Star } from "lucide-react"

const testimonials = [
  {
    name: "Marcus Chen",
    role: "Professional Trader",
    content:
      "SniprX has completely transformed my trading. The MMXM strategy is incredibly accurate, and I can finally trade while focusing on my day job.",
    rating: 5,
    profit: "+$18,450",
  },
  {
    name: "Sarah Williams",
    role: "Forex Enthusiast",
    content:
      "As a beginner, SniprX gave me the confidence to trade gold. The risk management features saved me from major losses while I was learning.",
    rating: 5,
    profit: "+$5,280",
  },
  {
    name: "David Rodriguez",
    role: "Hedge Fund Manager",
    content:
      "The institutional-level strategies are impressive. We use SniprX for our gold allocation with excellent results.",
    rating: 5,
    profit: "+$67,600",
  },
]

export default function TestimonialsSection() {
  return (
    <section className="py-20 bg-muted/30">
      <div className="container mx-auto px-4">
        <div className="text-center mb-16">
          <Badge className="mb-4 bg-blue-500/10 text-blue-500 border-blue-500/20">Testimonials</Badge>
          <h2 className="text-3xl md:text-4xl font-bold mb-4">Trusted by Traders Worldwide</h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            See what our community of successful traders has to say about SniprX
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-6xl mx-auto">
          {testimonials.map((testimonial, index) => (
            <Card key={index} className="glassmorphism">
              <CardContent className="p-6">
                <div className="flex items-center mb-4">
                  {[...Array(testimonial.rating)].map((_, i) => (
                    <Star key={i} className="w-4 h-4 fill-primary text-primary" />
                  ))}
                </div>
                <p className="text-muted-foreground mb-4 italic">"{testimonial.content}"</p>
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-semibold">{testimonial.name}</div>
                    <div className="text-sm text-muted-foreground">{testimonial.role}</div>
                  </div>
                  <Badge className="bg-green-500/10 text-green-500 border-green-500/20">{testimonial.profit}</Badge>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  )
}
