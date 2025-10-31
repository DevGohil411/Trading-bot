import Link from "next/link"
import { Target, Twitter, Github, MessageCircle, Mail } from "lucide-react"

export default function Footer() {
  return (
    <footer className="border-t border-border/40 bg-background/50 backdrop-blur">
      <div className="container mx-auto px-4 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div className="space-y-4">
            <Link href="/" className="flex items-center space-x-2">
              <div className="w-8 h-8 gradient-primary rounded-xl flex items-center justify-center">
                <Target className="w-5 h-5 text-black" />
              </div>
              <span className="text-xl font-bold">SniprX</span>
            </Link>
            <p className="text-muted-foreground">
              Premium algorithmic trading platform. Built for traders, by traders.
            </p>
            <div className="flex space-x-4">
              <Link href="https://t.me/sniprx" className="text-muted-foreground hover:text-primary">
                <MessageCircle className="w-5 h-5" />
              </Link>
              <Link href="https://tradingview.com/u/sniprx" className="text-muted-foreground hover:text-primary">
                <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M12 0C5.373 0 0 5.373 0 12s5.373 12 12 12 12-5.373 12-12S18.627 0 12 0zm5.568 7.432l-2.4 2.4c-.8.8-2.1.8-2.9 0l-2.4-2.4c-.8-.8-.8-2.1 0-2.9l2.4-2.4c.8-.8 2.1-.8 2.9 0l2.4 2.4c.8.8.8 2.1 0 2.9z" />
                </svg>
              </Link>
              <Link href="https://twitter.com/sniprx" className="text-muted-foreground hover:text-primary">
                <Twitter className="w-5 h-5" />
              </Link>
              <Link href="https://github.com/sniprx" className="text-muted-foreground hover:text-primary">
                <Github className="w-5 h-5" />
              </Link>
            </div>
          </div>

          <div>
            <h3 className="font-semibold mb-4">Platform</h3>
            <ul className="space-y-2">
              <li>
                <Link href="/strategies" className="text-muted-foreground hover:text-primary">
                  Strategies
                </Link>
              </li>
              <li>
                <Link href="/pricing" className="text-muted-foreground hover:text-primary">
                  Pricing
                </Link>
              </li>
              <li>
                <Link href="/about" className="text-muted-foreground hover:text-primary">
                  About
                </Link>
              </li>
              <li>
                <Link href="/faq" className="text-muted-foreground hover:text-primary">
                  FAQ
                </Link>
              </li>
            </ul>
          </div>

          <div>
            <h3 className="font-semibold mb-4">Support</h3>
            <ul className="space-y-2">
              <li>
                <Link
                  href="mailto:dev007gohil@gmail.com"
                  className="text-muted-foreground hover:text-primary flex items-center"
                >
                  <Mail className="w-4 h-4 mr-2" />
                  Contact
                </Link>
              </li>
              <li>
                <Link href="https://t.me/sniprx_support" className="text-muted-foreground hover:text-primary">
                  Telegram Support
                </Link>
              </li>
              <li>
                <Link href="https://tradingview.com/u/sniprx" className="text-muted-foreground hover:text-primary">
                  TradingView
                </Link>
              </li>
              <li>
                <Link href="/docs" className="text-muted-foreground hover:text-primary">
                  Documentation
                </Link>
              </li>
            </ul>
          </div>

          <div>
            <h3 className="font-semibold mb-4">Legal</h3>
            <ul className="space-y-2">
              <li>
                <Link href="/terms" className="text-muted-foreground hover:text-primary">
                  Terms of Service
                </Link>
              </li>
              <li>
                <Link href="/privacy" className="text-muted-foreground hover:text-primary">
                  Privacy Policy
                </Link>
              </li>
              <li>
                <Link href="/risk" className="text-muted-foreground hover:text-primary">
                  Risk Disclosure
                </Link>
              </li>
              <li>
                <Link href="/refund" className="text-muted-foreground hover:text-primary">
                  Refund Policy
                </Link>
              </li>
            </ul>
          </div>
        </div>

        <div className="border-t border-border/40 mt-8 pt-8 text-center text-muted-foreground">
          <p>&copy; 2024 SniprX. All rights reserved. Trading involves risk of loss.</p>
          <p className="text-xs mt-2">
            Contact:{" "}
            <a href="mailto:dev007gohil@gmail.com" className="text-primary hover:underline">
              dev007gohil@gmail.com
            </a>
          </p>
        </div>
      </div>
    </footer>
  )
}
