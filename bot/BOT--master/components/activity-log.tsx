"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Activity, Bot, Settings, Wifi } from "lucide-react"

interface LogEntry {
  type: string;
  title: string;
  details: string;
  timestamp: string;
  tag: string;
}

export default function ActivityLog() {
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [error, setError] = useState("")

  const fetchLogs = async () => {
    setError("")
    try {
      const res = await fetch(`/api/activity-log`)
      if (!res.ok) throw new Error("Failed to fetch activity log")
      const data = await res.json()
      setLogs(data)
    } catch (err: any) {
      setError("Unable to load activity log.")
    }
  }

  useEffect(() => {
    fetchLogs()
    const interval = setInterval(fetchLogs, 15000)
    return () => clearInterval(interval)
  }, [])

  const getIcon = (type: string) => {
    switch (type) {
      case "bot":
        return <Bot className="w-4 h-4 text-primary" />
      case "strategy":
        return <Settings className="w-4 h-4 text-blue-500" />
      case "connection":
        return <Wifi className="w-4 h-4 text-green-500" />
      case "trade":
        return <Activity className="w-4 h-4 text-orange-500" />
      case "error":
        return <Activity className="w-4 h-4 text-red-500" />
      default:
        return <Activity className="w-4 h-4 text-muted-foreground" />
    }
  }

  const getBadgeColor = (type: string) => {
    switch (type) {
      case "bot":
        return "bg-primary/10 text-primary border-primary/20"
      case "strategy":
        return "bg-blue-500/10 text-blue-500 border-blue-500/20"
      case "connection":
        return "bg-green-500/10 text-green-500 border-green-500/20"
      case "trade":
        return "bg-orange-500/10 text-orange-500 border-orange-500/20"
      case "error":
        return "bg-red-500/10 text-red-500 border-red-500/20"
      default:
        return "bg-muted/10 text-muted-foreground border-muted/20"
    }
  }

  return (
    <Card className="glassmorphism border-primary/20">
      <CardHeader>
        <CardTitle className="flex items-center">
          <Activity className="w-5 h-5 mr-2 text-primary" />
          Activity Log
        </CardTitle>
      </CardHeader>
      <CardContent>
        {error && <div className="text-red-500 text-sm mb-2">{error}</div>}
        <div className="space-y-3 max-h-80 overflow-y-auto">
          {logs.length === 0 && !error && <div className="text-center text-muted-foreground">No activity yet.</div>}
          {logs.map((log, i) => (
            <div
              key={i}
              className="flex items-start space-x-3 p-3 rounded-lg bg-muted/20 hover:bg-muted/30 transition-colors"
            >
              <div className="flex-shrink-0 mt-1">{getIcon(log.type)}</div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between">
                  <p className="text-sm font-medium">{log.title}</p>
                  <Badge className={`text-xs ${getBadgeColor(log.type)}`}>{log.tag}</Badge>
                </div>
                {log.details && <p className="text-xs text-muted-foreground mt-1">{log.details}</p>}
                <p className="text-xs text-muted-foreground mt-1">{new Date(log.timestamp).toLocaleString()}</p>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
