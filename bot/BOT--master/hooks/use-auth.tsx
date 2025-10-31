"use client"

import type React from "react"
import { useState, useEffect, createContext, useContext } from "react"

interface AuthContextType {
  isAuthenticated: boolean
  user: any
  login: (email: string, password: string) => Promise<boolean>
  loginWithGoogle: () => void
  logout: () => void
  loading: boolean
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider")
  }
  return context
}

// Function to send Telegram notification
const sendTelegramNotification = async (type: string, data: any) => {
  try {
    await fetch('/api/telegram/notify', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ type, data }),
    });
  } catch (error) {
    console.error('Failed to send Telegram notification:', error);
  }
};

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [user, setUser] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // First check server session
    const checkServerSession = async () => {
      try {
        const res = await fetch('/auth/status', { credentials: 'include' })
        if (res.ok) {
          const data = await res.json()
          if (data.authenticated) {
            setIsAuthenticated(true)
            setUser(data.user)
            setLoading(false)
            return
          }
        }
      } catch (err) {
        // ignore and fallback to local storage
      }

      // Fallback: Check local storage mock
      const token = localStorage.getItem("sniprx_token")
      const userData = localStorage.getItem("sniprx_user")

      if (token && userData) {
        setIsAuthenticated(true)
        setUser(JSON.parse(userData))
      }
      setLoading(false)
    }

    checkServerSession()
  }, [])

  const login = async (email: string, password: string): Promise<boolean> => {
    try {
      // Simulate API call
      await new Promise((resolve) => setTimeout(resolve, 1500))

      // Mock successful login
      const mockUser = {
        id: "1",
        email: email,
        name: "Pro Trader",
        mt5Connected: true,
        plan: "Premium",
      }

      localStorage.setItem("sniprx_token", "premium_jwt_token")
      localStorage.setItem("sniprx_user", JSON.stringify(mockUser))

      setIsAuthenticated(true)
      setUser(mockUser)

      // Send Telegram notification
      await sendTelegramNotification('login', {
        userId: mockUser.id,
        userEmail: email
      });

      return true
    } catch (error) {
      return false
    }
  }

  const loginWithGoogle = () => {
    window.location.href = '/auth/google'
  }

  const logout = async () => {
    const currentUser = user;

    try {
      await fetch('/logout', { credentials: 'include' })
    } catch (e) {
      // ignore
    }

    localStorage.removeItem("sniprx_token")
    localStorage.removeItem("sniprx_user")
    setIsAuthenticated(false)
    setUser(null)

    // Send Telegram notification
    if (currentUser) {
      sendTelegramNotification('logout', {
        userId: currentUser.id,
        userEmail: currentUser.email
      });
    }

    window.location.href = "/"
  }

  return (
    <AuthContext.Provider value={{ isAuthenticated, user, login, loginWithGoogle, logout, loading }}>{children}</AuthContext.Provider>
  )
}
