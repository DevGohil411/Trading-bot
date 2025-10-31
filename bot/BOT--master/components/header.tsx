"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Button } from "@/components/ui/button";
import { ThemeToggle } from "@/components/theme-toggle";
import { Badge } from "@/components/ui/badge";
import {
  Menu,
  X,
  Target,
  BarChart3,
  Brain,
  Server,
  Settings,
  Bot,
  TrendingUp,
  LinkIcon,
  MessageSquare,
} from "lucide-react";
import { useAuth } from "@/hooks/use-auth";

export default function Header() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const { logout, user } = useAuth();
  const pathname = usePathname();

  const isActive = (path: string) => pathname === path;

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto px-4">
        <div className="flex h-16 items-center justify-between">
          <Link href="/dashboard" className="flex items-center space-x-2">
            <div className="w-8 h-8 gradient-primary rounded-xl flex items-center justify-center">
              <Target className="w-5 h-5 text-black" />
            </div>
            <span className="text-xl font-bold">SniprX</span>
            <Badge className="bg-primary/10 text-primary border-primary/20 text-xs">
              Premium
            </Badge>
          </Link>

          {/* Desktop Navigation */}
          <nav className="hidden lg:flex items-center space-x-1">
            <Link
              href="/dashboard"
              className={`text-sm font-medium transition-all px-3 py-2 rounded-lg flex items-center gap-2 whitespace-nowrap ${
                isActive("/dashboard")
                  ? "text-primary bg-primary/20 shadow-lg shadow-primary/50 border border-primary/30"
                  : "hover:text-primary hover:bg-primary/10"
              }`}
            >
              <BarChart3 className="w-4 h-4" />
              Dashboard
            </Link>
            <Link
              href="/multi-bot"
              className={`text-sm font-medium transition-all px-3 py-2 rounded-lg flex items-center gap-2 whitespace-nowrap ${
                isActive("/multi-bot")
                  ? "text-primary bg-primary/20 shadow-lg shadow-primary/50 border border-primary/30"
                  : "hover:text-primary hover:bg-primary/10"
              }`}
            >
              <Bot className="w-4 h-4" />
              Multi-Bot
            </Link>
            <Link
              href="/analytics"
              className={`text-sm font-medium transition-all px-3 py-2 rounded-lg flex items-center gap-2 whitespace-nowrap ${
                isActive("/analytics")
                  ? "text-primary bg-primary/20 shadow-lg shadow-primary/50 border border-primary/30"
                  : "hover:text-primary hover:bg-primary/10"
              }`}
            >
              <TrendingUp className="w-4 h-4" />
              Analytics
            </Link>
            <Link
              href="/ai-insights"
              className={`text-sm font-medium transition-all px-3 py-2 rounded-lg flex items-center gap-2 whitespace-nowrap ${
                isActive("/ai-insights")
                  ? "text-primary bg-primary/20 shadow-lg shadow-primary/50 border border-primary/30"
                  : "hover:text-primary hover:bg-primary/10"
              }`}
            >
              <Brain className="w-4 h-4" />
              AI Insights
            </Link>
            <Link
              href="/chatbot"
              className={`text-sm font-medium transition-all px-3 py-2 rounded-lg flex items-center gap-1.5 whitespace-nowrap ${
                isActive("/chatbot")
                  ? "text-primary bg-gradient-to-r from-primary/30 to-blue-500/30 shadow-xl shadow-primary/60 border-2 border-primary/50 animate-pulse"
                  : "text-primary bg-primary/10 hover:bg-primary/20 border border-primary/20 hover:shadow-lg hover:shadow-primary/30"
              }`}
            >
              <MessageSquare className="w-4 h-4" />
              <span className="inline-flex items-center gap-1">
                Snipr-X AI
                <Badge className="bg-blue-500 text-white text-xs px-1.5 py-0.5">
                  AI
                </Badge>
              </span>
            </Link>
            <Link
              href="/mt5"
              className={`text-sm font-medium transition-all px-3 py-2 rounded-lg flex items-center gap-2 whitespace-nowrap ${
                isActive("/mt5")
                  ? "text-primary bg-primary/20 shadow-lg shadow-primary/50 border border-primary/30"
                  : "hover:text-primary hover:bg-primary/10"
              }`}
            >
              <Server className="w-4 h-4" />
              MT5
            </Link>
            <Link
              href="/strategies"
              className={`text-sm font-medium transition-all px-3 py-2 rounded-lg flex items-center gap-2 whitespace-nowrap ${
                isActive("/strategies")
                  ? "text-primary bg-primary/20 shadow-lg shadow-primary/50 border border-primary/30"
                  : "hover:text-primary hover:bg-primary/10"
              }`}
            >
              <Target className="w-4 h-4" />
              Strategies
            </Link>
            <Link
              href="/integrations"
              className={`text-sm font-medium transition-all px-3 py-2 rounded-lg flex items-center gap-2 whitespace-nowrap ${
                isActive("/integrations")
                  ? "text-primary bg-primary/20 shadow-lg shadow-primary/50 border border-primary/30"
                  : "hover:text-primary hover:bg-primary/10"
              }`}
            >
              <LinkIcon className="w-4 h-4" />
              Integrations
            </Link>
            <Link
              href="/settings"
              className={`text-sm font-medium transition-all px-3 py-2 rounded-lg flex items-center gap-2 whitespace-nowrap ${
                isActive("/settings")
                  ? "text-primary bg-primary/20 shadow-lg shadow-primary/50 border border-primary/30"
                  : "hover:text-primary hover:bg-primary/10"
              }`}
            >
              <Settings className="w-4 h-4" />
              Settings
            </Link>
          </nav>

          <div className="hidden lg:flex items-center space-x-4">
            <Badge className="bg-green-500/10 text-green-500 border-green-500/20">
              {user?.displayName || user?.name || "Pro Trader"}
            </Badge>
            <ThemeToggle />
            <Button variant="ghost" onClick={logout}>
              Logout
            </Button>
          </div>

          {/* Mobile Menu Button */}
          <div className="lg:hidden flex items-center space-x-2">
            <ThemeToggle />
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setIsMenuOpen(!isMenuOpen)}
            >
              {isMenuOpen ? (
                <X className="h-5 w-5" />
              ) : (
                <Menu className="h-5 w-5" />
              )}
            </Button>
          </div>
        </div>

        {/* Mobile Navigation */}
        {isMenuOpen && (
          <div className="lg:hidden py-4 border-t border-border/40">
            <nav className="flex flex-col space-y-2">
              <Link
                href="/dashboard"
                className={`text-sm font-medium transition-all px-4 py-3 rounded-lg flex items-center gap-3 ${
                  isActive("/dashboard")
                    ? "text-primary bg-primary/20 shadow-lg shadow-primary/50 border border-primary/30"
                    : "hover:text-primary hover:bg-primary/10"
                }`}
              >
                <BarChart3 className="w-4 h-4" />
                Dashboard
              </Link>
              <Link
                href="/multi-bot"
                className={`text-sm font-medium transition-all px-4 py-3 rounded-lg flex items-center gap-3 ${
                  isActive("/multi-bot")
                    ? "text-primary bg-primary/20 shadow-lg shadow-primary/50 border border-primary/30"
                    : "hover:text-primary hover:bg-primary/10"
                }`}
              >
                <Bot className="w-4 h-4" />
                Multi-Bot
              </Link>
              <Link
                href="/analytics"
                className={`text-sm font-medium transition-all px-4 py-3 rounded-lg flex items-center gap-3 ${
                  isActive("/analytics")
                    ? "text-primary bg-primary/20 shadow-lg shadow-primary/50 border border-primary/30"
                    : "hover:text-primary hover:bg-primary/10"
                }`}
              >
                <TrendingUp className="w-4 h-4" />
                Analytics
              </Link>
              <Link
                href="/ai-insights"
                className={`text-sm font-medium transition-all px-4 py-3 rounded-lg flex items-center gap-3 ${
                  isActive("/ai-insights")
                    ? "text-primary bg-primary/20 shadow-lg shadow-primary/50 border border-primary/30"
                    : "hover:text-primary hover:bg-primary/10"
                }`}
              >
                <Brain className="w-4 h-4" />
                AI Insights
              </Link>
              <Link
                href="/chatbot"
                className={`text-sm font-medium transition-all px-4 py-3 rounded-lg flex items-center gap-2 whitespace-nowrap ${
                  isActive("/chatbot")
                    ? "text-primary bg-gradient-to-r from-primary/30 to-blue-500/30 shadow-xl shadow-primary/60 border-2 border-primary/50"
                    : "text-primary bg-primary/10 hover:bg-primary/20 border border-primary/20"
                }`}
              >
                <MessageSquare className="w-4 h-4 flex-shrink-0" />
                <span className="inline-flex items-center gap-1.5 flex-1">
                  Snipr-X AI
                  <Badge className="bg-blue-500 text-white text-xs px-1.5 py-0.5">
                    AI
                  </Badge>
                </span>
              </Link>
              <Link
                href="/mt5"
                className={`text-sm font-medium transition-all px-4 py-3 rounded-lg flex items-center gap-3 ${
                  isActive("/mt5")
                    ? "text-primary bg-primary/20 shadow-lg shadow-primary/50 border border-primary/30"
                    : "hover:text-primary hover:bg-primary/10"
                }`}
              >
                <Server className="w-4 h-4" />
                MT5
              </Link>
              <Link
                href="/strategies"
                className={`text-sm font-medium transition-all px-4 py-3 rounded-lg flex items-center gap-3 ${
                  isActive("/strategies")
                    ? "text-primary bg-primary/20 shadow-lg shadow-primary/50 border border-primary/30"
                    : "hover:text-primary hover:bg-primary/10"
                }`}
              >
                <Target className="w-4 h-4" />
                Strategies
              </Link>
              <Link
                href="/integrations"
                className={`text-sm font-medium transition-all px-4 py-3 rounded-lg flex items-center gap-3 ${
                  isActive("/integrations")
                    ? "text-primary bg-primary/20 shadow-lg shadow-primary/50 border border-primary/30"
                    : "hover:text-primary hover:bg-primary/10"
                }`}
              >
                <LinkIcon className="w-4 h-4" />
                Integrations
              </Link>
              <Link
                href="/settings"
                className={`text-sm font-medium transition-all px-4 py-3 rounded-lg flex items-center gap-3 ${
                  isActive("/settings")
                    ? "text-primary bg-primary/20 shadow-lg shadow-primary/50 border border-primary/30"
                    : "hover:text-primary hover:bg-primary/10"
                }`}
              >
                <Settings className="w-4 h-4" />
                Settings
              </Link>
              <div className="pt-4 border-t border-border/40 mt-2">
                <Button
                  variant="ghost"
                  className="w-full justify-start"
                  onClick={logout}
                >
                  Logout
                </Button>
              </div>
            </nav>
          </div>
        )}
      </div>
    </header>
  );
}
