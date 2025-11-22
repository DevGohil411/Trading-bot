"use client"

import Header from "@/components/header"
import LoginForm from "@/components/login-form"
import Footer from "@/components/footer"
import { Button } from "@/components/ui/button"
import { Chrome } from "lucide-react"
import { useAuth } from "@/hooks/use-auth"

function GoogleLoginButton() {
  const { loginWithGoogle } = useAuth()
  return (
    <Button onClick={loginWithGoogle} className="w-full mb-4" variant="outline">
      <Chrome className="w-4 h-4 mr-2" /> Continue with Google
    </Button>
  )
}

export default function LoginPage() {
  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main className="container mx-auto px-4 py-16">
        <div className="max-w-md mx-auto">
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold mb-2">Welcome Back</h1>
            <p className="text-muted-foreground">Sign in to your SniprX account</p>
          </div>
          <GoogleLoginButton />
          <LoginForm />
        </div>
      </main>
      <Footer />
    </div>
  )
}
