import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"

const faqs = [
  {
    question: "How does MT5 connection work with SniprX?",
    answer:
      "SniprX connects to your MT5 account through secure API integration. You provide your MT5 credentials, and our premium algorithms execute trades directly on your account. All connections are bank-level encrypted and your funds remain in your own brokerage account. We never have access to your money - only permission to execute trades based on our institutional strategies.",
  },
  {
    question: "What makes SniprX different from other trading bots?",
    answer:
      "SniprX offers institutional-level strategies developed by professional traders with hedge fund experience. Our algorithms use advanced smart money concepts like Order Blocks, CHOCH, and liquidity manipulation detection. We focus specifically on XAUUSD with strategies that have been backtested across multiple market conditions with consistent 80%+ win rates.",
  },
  {
    question: "Is my data and money safe with SniprX?",
    answer:
      "Absolutely. We use military-grade encryption for all data transmission. Your funds stay in your own MT5 account with your chosen broker - we never have access to your money. We only have permission to execute trades based on our algorithms. Additionally, our platform includes MT5 disconnect protection to prevent trades if connection is lost.",
  },
  {
    question: "What are the risks of using SniprX trading bot?",
    answer:
      "Trading involves risk of loss. While our strategies have strong historical performance with 80%+ win rates, past results don't guarantee future profits. We recommend starting with small position sizes and never risking more than you can afford to lose. Our built-in risk management includes daily drawdown limits, intelligent position sizing, and killzone filters to protect your capital.",
  },
  {
    question: "Can I customize the trading strategies?",
    answer:
      "Yes! Premium users can adjust risk parameters, position sizes, daily TP/SL limits, and enable/disable killzone filters. You can switch between our 7 battle-tested strategies, set custom trading hours, and adjust the bot's aggressiveness. Advanced users get access to strategy customization tools and can even request custom indicator development.",
  },
  {
    question: "What brokers work with SniprX?",
    answer:
      "SniprX works with any MT5 broker that supports API connections. Popular brokers include IC Markets, Pepperstone, FTMO, MyFundedFX, and many others. We provide a list of recommended brokers with optimal execution speeds and low spreads for XAUUSD trading.",
  },
  {
    question: "What is your refund policy?",
    answer:
      "We offer a 30-day money-back guarantee for all premium plans. If you're not satisfied with the performance or service, contact our support team for a full refund. The 3-day strategy trial allows you to test our algorithms risk-free before committing to a premium plan.",
  },
  {
    question: "Do you provide support and training?",
    answer:
      "Yes! We offer comprehensive support including MT5 setup assistance, strategy guidance, and ongoing technical support. Premium users get priority support and access to our private community. Annual subscribers get 1-on-1 strategy sessions and custom indicator development.",
  },
  {
    question: "How much capital do I need to start?",
    answer:
      "We recommend starting with at least $1,000 for live trading to allow proper risk management and position sizing. However, you can test everything with a demo account first using our 3-day strategy trial. Our algorithms work with accounts from $500 to $100,000+ with intelligent position sizing.",
  },
  {
    question: "Can I run multiple bots simultaneously?",
    answer:
      "Yes! Premium subscribers can run multiple bots with different strategies simultaneously. For example, you can run MMXM on 1M timeframe while running OTE on 15M timeframe. Each bot can have different risk settings and can be controlled independently through our dashboard.",
  },
]

export default function FAQSection() {
  return (
    <div className="max-w-4xl mx-auto">
      <Accordion type="single" collapsible className="space-y-4">
        {faqs.map((faq, index) => (
          <AccordionItem
            key={index}
            value={`item-${index}`}
            className="glassmorphism rounded-2xl px-6 border-primary/20"
          >
            <AccordionTrigger className="text-left hover:no-underline">{faq.question}</AccordionTrigger>
            <AccordionContent className="text-muted-foreground">{faq.answer}</AccordionContent>
          </AccordionItem>
        ))}
      </Accordion>
    </div>
  )
}
