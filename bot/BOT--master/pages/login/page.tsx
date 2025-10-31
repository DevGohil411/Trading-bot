import { useRouter } from "next/router"
import { useEffect } from "react"
import { useAuth } from "@/hooks/use-auth"

export default function LoginPage() {
  const { isAuthenticated, loading } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (!loading && isAuthenticated) {
      router.push("/dashboard")
    }
  }, [isAuthenticated, loading, router])

  const handleGoogleSignIn = () => {
    window.location.href = '/auth/google'
  }

  return (
    <div style={{ minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", background: "#111" }}>
      <div style={{ background: "#fff", borderRadius: 12, boxShadow: "0 2px 16px #0002", padding: 32, width: 320, display: "flex", flexDirection: "column", gap: 16 }}>
        <h2 style={{ fontSize: 24, fontWeight: 700, textAlign: "center" }}>Sign in to your account</h2>
        <button
          type="button"
          onClick={handleGoogleSignIn}
          style={{
            width: "100%",
            padding: "16px",
            background: "#0a0a23",
            color: "#fff",
            border: "none",
            borderRadius: "8px",
            fontSize: "18px",
            cursor: "pointer",
            marginBottom: "16px",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: "8px"
          }}
        >
          <svg width="20" height="20" viewBox="0 0 48 48" style={{ marginRight: 8 }}><g><path fill="#4285F4" d="M24 9.5c3.54 0 6.7 1.22 9.19 3.23l6.85-6.85C35.64 2.39 30.18 0 24 0 14.82 0 6.73 5.48 2.69 13.44l7.98 6.2C12.13 13.16 17.62 9.5 24 9.5z"/><path fill="#34A853" d="M46.1 24.55c0-1.64-.15-3.22-.43-4.74H24v9.01h12.42c-.54 2.9-2.18 5.36-4.65 7.01l7.19 5.6C43.98 37.13 46.1 31.36 46.1 24.55z"/><path fill="#FBBC05" d="M9.67 28.09c-1.13-3.36-1.13-6.98 0-10.34l-7.98-6.2C-1.06 16.09-1.06 31.91 1.69 39.45l7.98-6.2z"/><path fill="#EA4335" d="M24 46c6.18 0 11.36-2.05 15.13-5.57l-7.19-5.6c-2.01 1.35-4.6 2.15-7.94 2.15-6.38 0-11.87-3.66-13.33-8.94l-7.98 6.2C6.73 42.52 14.82 48 24 48z"/><path fill="none" d="M0 0h48v48H0z"/></g></svg>
          Continue with Google
        </button>
        {/* TEST: Add a simple button to check click */}
        <button
          type="button"
          onClick={() => console.log("Test Button Clicked")}
          style={{
            width: "100%",
            padding: "8px 0",
            borderRadius: 6,
            background: "#0af",
            color: "#fff",
            fontWeight: 600,
            fontSize: 14,
            border: "none",
            cursor: "pointer"
          }}
        >
          Test Button
        </button>
      </div>
    </div>
  )
} 