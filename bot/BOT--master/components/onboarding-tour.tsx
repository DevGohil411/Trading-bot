"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { X, ArrowRight, ArrowLeft, Target, Zap } from "lucide-react"

interface TourStep {
  id: string
  title: string
  description: string
  target: string
  position: "top" | "bottom" | "left" | "right"
  action?: string
}

const tourSteps: TourStep[] = [
  {
    id: "welcome",
    title: "Welcome to SniprX! 🎯",
    description:
      "Let's take a quick tour to get you started with premium algorithmic trading. This will only take 2 minutes.",
    target: "body",
    position: "top",
  },
  {
    id: "dashboard",
    title: "Your Trading Dashboard",
    description: "This is your command center. Monitor all your bots, P&L, and trading performance in real-time.",
    target: "[data-tour='dashboard-stats']",
    position: "bottom",
  },
  {
    id: "bot-controls",
    title: "Bot Control Center",
    description: "Start, stop, and configure your trading bots. Each bot can run different strategies simultaneously.",
    target: "[data-tour='bot-controls']",
    position: "left",
  },
  {
    id: "mt5-connection",
    title: "MT5 Integration",
    description: "Connect your MT5 accounts here. Your funds stay in your broker account - we only execute trades.",
    target: "[data-tour='mt5-connector']",
    position: "top",
  },
  {
    id: "strategies",
    title: "Premium Strategies",
    description:
      "Access 7 battle-tested strategies like MMXM, OTE, and Judas Swing. Each optimized for different market conditions.",
    target: "[data-tour='strategies']",
    position: "right",
  },
  {
    id: "analytics",
    title: "Advanced Analytics",
    description: "Deep dive into your performance with strategy breakdowns, heat maps, and AI-powered insights.",
    target: "[data-tour='analytics']",
    position: "bottom",
  },
  {
    id: "complete",
    title: "You're All Set! 🚀",
    description:
      "Start by connecting your MT5 account, then activate your first bot. Our AI will guide you to optimal settings.",
    target: "body",
    position: "top",
    action: "Get Started",
  },
]

export default function OnboardingTour() {
  const [isActive, setIsActive] = useState(false)
  const [currentStep, setCurrentStep] = useState(0)
  const [hasSeenTour, setHasSeenTour] = useState(false)

  useEffect(() => {
    // Check if user has seen the tour
    const tourCompleted = localStorage.getItem("sniprx_tour_completed")
    if (!tourCompleted) {
      setTimeout(() => setIsActive(true), 2000) // Start tour after 2 seconds
    } else {
      setHasSeenTour(true)
    }
  }, [])

  const nextStep = () => {
    if (currentStep < tourSteps.length - 1) {
      setCurrentStep(currentStep + 1)
    } else {
      completeTour()
    }
  }

  const prevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1)
    }
  }

  const skipTour = () => {
    setIsActive(false)
    localStorage.setItem("sniprx_tour_completed", "true")
    setHasSeenTour(true)
  }

  const completeTour = () => {
    setIsActive(false)
    localStorage.setItem("sniprx_tour_completed", "true")
    setHasSeenTour(true)
  }

  const restartTour = () => {
    setCurrentStep(0)
    setIsActive(true)
  }

  if (!isActive && hasSeenTour) {
    return (
      <Button onClick={restartTour} variant="outline" size="sm" className="fixed bottom-4 left-4 z-50">
        <Target className="w-4 h-4 mr-2" />
        Restart Tour
      </Button>
    )
  }

  if (!isActive) return null

  const step = tourSteps[currentStep]

  return (
    <>
      {/* Overlay */}
      <div className="fixed inset-0 bg-black/50 z-40" />

      {/* Tour Card */}
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
        <Card className="glassmorphism border-primary/20 max-w-md w-full animate-in fade-in-0 zoom-in-95">
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-4">
              <Badge className="gradient-primary text-black">
                Step {currentStep + 1} of {tourSteps.length}
              </Badge>
              <Button size="sm" variant="ghost" onClick={skipTour}>
                <X className="w-4 h-4" />
              </Button>
            </div>

            <div className="space-y-4">
              <div>
                <h3 className="text-lg font-semibold mb-2">{step.title}</h3>
                <p className="text-muted-foreground">{step.description}</p>
              </div>

              {step.id === "welcome" && (
                <div className="flex items-center justify-center space-x-4 py-4">
                  <div className="text-center">
                    <div className="text-3xl mb-2">🎯</div>
                    <div className="text-sm font-medium">Precision Trading</div>
                  </div>
                  <div className="text-center">
                    <div className="text-3xl mb-2">🤖</div>
                    <div className="text-sm font-medium">AI Automation</div>
                  </div>
                  <div className="text-center">
                    <div className="text-3xl mb-2">📊</div>
                    <div className="text-sm font-medium">Live Analytics</div>
                  </div>
                </div>
              )}

              {step.id === "complete" && (
                <div className="text-center py-4">
                  <div className="w-16 h-16 gradient-primary rounded-full flex items-center justify-center mx-auto mb-4">
                    <Zap className="w-8 h-8 text-black" />
                  </div>
                  <div className="space-y-2">
                    <div className="text-sm text-muted-foreground">Quick Start Checklist:</div>
                    <div className="space-y-1 text-sm">
                      <div className="flex items-center space-x-2">
                        <span>1.</span>
                        <span>Connect MT5 account</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span>2.</span>
                        <span>Choose a strategy (MMXM recommended)</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span>3.</span>
                        <span>Set risk parameters</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span>4.</span>
                        <span>Start your first bot</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              <div className="flex items-center justify-between pt-4">
                <Button variant="outline" onClick={prevStep} disabled={currentStep === 0} className="flex items-center">
                  <ArrowLeft className="w-4 h-4 mr-2" />
                  Previous
                </Button>

                <div className="flex space-x-1">
                  {tourSteps.map((_, index) => (
                    <div
                      key={index}
                      className={`w-2 h-2 rounded-full ${index === currentStep ? "bg-primary" : "bg-muted"}`}
                    />
                  ))}
                </div>

                <Button onClick={nextStep} className="gradient-primary text-black flex items-center">
                  {step.action || (currentStep === tourSteps.length - 1 ? "Finish" : "Next")}
                  <ArrowRight className="w-4 h-4 ml-2" />
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </>
  )
}
