"use client";

import { AssistantRuntimeProvider } from "@assistant-ui/react";
import {
  useChatRuntime,
  AssistantChatTransport,
} from "@assistant-ui/react-ai-sdk";
import { Thread } from "@/components/assistant-ui/thread";
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar";
import { ThreadListSidebar } from "@/components/assistant-ui/threadlist-sidebar";
import { Separator } from "@/components/ui/separator";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import ProtectedRoute from "@/components/protected-route";
import { Badge } from "@/components/ui/badge";
import Header from "@/components/header";
import { MessageSquare } from "lucide-react";

export default function ChatBotPage() {
  const runtime = useChatRuntime({
    transport: new AssistantChatTransport({
      api: "/api/chat",
    }),
  });

  return (
    <ProtectedRoute>
      <div className="min-h-screen bg-background">
        <Header />
        <div className="h-[calc(100vh-4rem)]">
          <AssistantRuntimeProvider runtime={runtime}>
            <SidebarProvider>
              <div className="flex h-full w-full">
                <ThreadListSidebar />
                <SidebarInset className="flex flex-col">
                  <header className="flex h-16 shrink-0 items-center gap-2 border-b px-4 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
                    <SidebarTrigger />
                    <Separator orientation="vertical" className="mr-2 h-4" />
                    <Breadcrumb>
                      <BreadcrumbList>
                        <BreadcrumbItem className="hidden md:block">
                          <BreadcrumbLink href="/dashboard">
                            Dashboard
                          </BreadcrumbLink>
                        </BreadcrumbItem>
                        <BreadcrumbSeparator className="hidden md:block" />
                        <BreadcrumbItem>
                          <BreadcrumbPage className="flex items-center gap-2">
                            <MessageSquare className="w-4 h-4" />
                            <span>AI ChatBot</span>
                            <Badge className="bg-blue-500/10 text-blue-500 border-blue-500/20 ml-2">
                              AI-Powered
                            </Badge>
                          </BreadcrumbPage>
                        </BreadcrumbItem>
                      </BreadcrumbList>
                    </Breadcrumb>
                  </header>
                  <div className="flex-1 overflow-hidden">
                    <Thread />
                  </div>
                </SidebarInset>
              </div>
            </SidebarProvider>
          </AssistantRuntimeProvider>
        </div>
      </div>
    </ProtectedRoute>
  );
}
