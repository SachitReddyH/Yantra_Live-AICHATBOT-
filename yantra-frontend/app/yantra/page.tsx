"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type Role = "user" | "assistant";

interface Message {
  id: string;
  role: Role;
  content: string;
  timestamp: string;
  brochure_urls?: string[] | null;
}

interface ChatApiResponse {
  answer: string;
  used_context: string[];
  from_fallback: boolean;
  brochure_urls?: string[] | null;
}

type Theme = "dark" | "light";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";

// Example uploaded image path (for developer/tooling use): /mnt/data/ee9aa334-6c83-4590-9d1e-f83dd8e775d3.png

const SUGGESTED_QUESTIONS: string[] = [
  "What is the chisel diameter for Hyundai R30?",
  "Which breaker is compatible with SANY SY20?",
  "Compare JCB 3DX and Hyundai R30.",
  "Which is the best breaker option for Hyundai R30?",
];

const FOLLOW_UP_LINES: string[] = [
  "Anything else I can help you with?",
  "Want to compare any other models?",
  "Need details on spare parts or dealers next?",
];

function getGreeting(): string {
  const hour = new Date().getHours();
  if (hour < 12) return "Good morning";
  if (hour < 17) return "Good afternoon";
  return "Good evening";
}

function randomFrom<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}

function formatTime(date: Date = new Date()): string {
  return date.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
}

/**
 * Helper to create a human-friendly label for a brochure URL.
 * Extracts filename and turns into uppercase model token if possible.
 */
function labelFromUrl(url: string): string {
  try {
    const parts = url.split("/");
    const fname = parts[parts.length - 1] || "";
    const nameNoExt = fname.replace(/\.[^/.]+$/, "");
    // try to uppercase VJ tokens
    return nameNoExt.toUpperCase();
  } catch {
    return "BROCHURE";
  }
}

/**
 * Simple Markdown table parser (same as previous).
 */
function parseMarkdownTable(md: string): { headers: string[]; rows: string[][] } | null {
  const lines = md
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l.length > 0);

  let tableStart = -1;
  for (let i = 0; i < lines.length - 1; i++) {
    if (lines[i].includes("|") && /^\s*\|?[-:\s|]+?\|?\s*$/.test(lines[i + 1])) {
      tableStart = i;
      break;
    }
  }
  if (tableStart === -1) return null;

  const headerLine = lines[tableStart];
  const separatorLine = lines[tableStart + 1];

  if (!headerLine.includes("|") || !separatorLine.includes("-")) return null;

  const splitRow = (line: string) =>
    line
      .replace(/^\|/, "")
      .replace(/\|$/, "")
      .split("|")
      .map((c) => c.trim());

  const headers = splitRow(headerLine);
  const rows: string[][] = [];

  for (let i = tableStart + 2; i < lines.length; i++) {
    if (!lines[i].includes("|")) break;
    rows.push(splitRow(lines[i]));
  }

  return { headers, rows };
}

function ComparisonTable({ headers, rows, theme }: { headers: string[]; rows: string[][]; theme: Theme }) {
  const isDark = theme === "dark";
  return (
    <div
      className={`rounded-2xl p-4 shadow-xl border ${isDark ? "bg-slate-800/80 border-slate-700" : "bg-white border-slate-200"}`}
      style={{ overflowX: "auto" }}
      role="table"
      aria-label="comparison table"
    >
      <table className="min-w-[600px] w-full table-fixed">
        <thead>
          <tr>
            {headers.map((h, idx) => (
              <th
                key={idx}
                className={`text-left align-top py-3 px-4 text-sm font-semibold ${isDark ? "text-slate-200" : "text-slate-700"}`}
                style={idx === 0 ? { width: "28%" } : { width: `${(72 / Math.max(1, headers.length - 1)).toFixed(0)}%` }}
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>

        <tbody>
          {rows.map((r, ri) => (
            <tr key={ri} className={`${ri % 2 === 0 ? (isDark ? "bg-slate-800/60" : "bg-slate-50") : ""}`}>
              {r.map((cell, ci) => (
                <td key={ci} className={`py-3 px-4 align-top text-sm ${isDark ? "text-slate-200" : "text-slate-800"}`}>
                  <div className={ci === 0 ? "text-left" : "text-left"}>
                    {cell.split("  ").join("\n").split("\\n").join("\n").split("\n").map((line, i) => (
                      <div key={i} className="leading-tight">
                        {line}
                      </div>
                    ))}
                  </div>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function YantraChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [greeting, setGreeting] = useState("Hello");
  const [theme, setTheme] = useState<Theme>("dark");
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  // refs to control scrolling
  const messagesContainerRef = useRef<HTMLDivElement | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    setGreeting(getGreeting());
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const stored = window.localStorage.getItem("yantra-theme");
    if (stored === "dark" || stored === "light") {
      setTheme(stored);
    }
  }, []);

  // restore chat history on mount
  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const raw = window.localStorage.getItem("yantra_chat_messages_v1");
      if (raw) {
        const parsed = JSON.parse(raw) as Message[];
        if (Array.isArray(parsed)) {
          setMessages(parsed);
        }
      }
    } catch (e) {
      console.warn("Failed to restore messages", e);
    }
  }, []);

  // persist messages to localStorage whenever messages change
  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      window.localStorage.setItem("yantra_chat_messages_v1", JSON.stringify(messages));
    } catch (e) {
      console.warn("Failed to save messages", e);
    }
  }, [messages]);

  // scroll to bottom whenever messages change
  useEffect(() => {
    if (messagesContainerRef.current) {
      setTimeout(() => {
        try {
          messagesContainerRef.current!.scrollTo({
            top: messagesContainerRef.current!.scrollHeight,
            behavior: "smooth",
          });
        } catch {
          messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
        }
      }, 50);
    } else {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  const quickActions = useMemo(() => SUGGESTED_QUESTIONS, []);

  const toggleTheme = () => {
    setTheme((prev) => {
      const next: Theme = prev === "dark" ? "light" : "dark";
      if (typeof window !== "undefined") {
        window.localStorage.setItem("yantra-theme", next);
      }
      return next;
    });
  };

  const handleClearChat = () => {
    setMessages([]);
    setInput("");
    textareaRef.current?.focus();
    try {
      if (typeof window !== "undefined") {
        window.localStorage.removeItem("yantra_chat_messages_v1");
      }
    } catch {}
  };

  const handleCopy = (content: string) => {
    if (typeof navigator === "undefined" || !navigator.clipboard) return;
    navigator.clipboard.writeText(content).catch((err) => {
      console.error("Failed to copy", err);
    });
  };

  const sendMessage = async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed) return;

    const now = formatTime();
    const lc = trimmed.toLowerCase();

    if (lc.includes("thank you") || lc.includes("thanks")) {
      const userMsg: Message = {
        id: crypto.randomUUID(),
        role: "user",
        content: trimmed,
        timestamp: now,
      };
      const botMsg: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: "You're welcome! Anything else I can help you with today?",
        timestamp: formatTime(),
      };
      setMessages((prev) => [...prev, userMsg, botMsg]);
      setInput("");
      textareaRef.current?.focus();
      return;
    }

    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: trimmed,
      timestamp: now,
    };

    const newMessages = [...messages, userMsg];
    setMessages(newMessages);
    setInput("");
    setLoading(true);

    try {
      const backendMessages = newMessages.map((m) => ({
        role: m.role,
        content: m.content,
      }));

      const res = await fetch(`${API_BASE_URL}/api/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ messages: backendMessages }),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || "Chat request failed");
      }

      const data: ChatApiResponse = await res.json();

      const followup = randomFrom(FOLLOW_UP_LINES);
      const combinedAnswer = `${data.answer.trim()}\n\n_${followup}_`;

      const botMsg: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: combinedAnswer,
        timestamp: formatTime(),
        brochure_urls: data.brochure_urls ?? null,
      };

      setMessages((prev) => [...prev, botMsg]);

      // DO NOT auto-open multiple tabs here (safer).
      // If you want auto-open behavior, change this to open each URL:
      // data.brochure_urls?.forEach(u => window.open(u, "_blank", "noopener,noreferrer"));

      textareaRef.current?.focus();
    } catch (err: any) {
      console.error(err);
      const errorMsg: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: `Error: ${err.message || "Something went wrong"}`,
        timestamp: formatTime(),
      };
      setMessages((prev) => [...prev, errorMsg]);
      textareaRef.current?.focus();
    } finally {
      setLoading(false);
    }
  };

  const handleSend = async () => {
    if (loading) return;
    await sendMessage(input);
  };

  const handleQuickQuestion = (q: string) => {
    if (loading) return;
    setInput("");
    void sendMessage(q);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!loading) {
        void handleSend();
      }
    }
  };

  const hasMessages = messages.length > 0;

  const rootClass =
    theme === "dark"
      ? "min-h-screen flex flex-col bg-slate-950 text-slate-100"
      : "min-h-screen flex flex-col bg-slate-50 text-slate-900";

  const headerBorder = theme === "dark" ? "border-slate-800" : "border-slate-200";
  const headerBg = theme === "dark" ? "bg-gradient-to-r from-slate-950 via-slate-900 to-slate-950" : "bg-white";
  const headerSubText = theme === "dark" ? "text-slate-400" : "text-slate-500";
  const cardClass =
    theme === "dark"
      ? "rounded-2xl border border-slate-800/80 bg-slate-900/60 shadow-[0_20px_60px_rgba(0,0,0,0.6)]"
      : "rounded-2xl border border-slate-200 bg-white shadow-[0_20px_40px_rgba(15,23,42,0.12)]";
  const dividerBorder = theme === "dark" ? "border-slate-800/80" : "border-slate-200";
  const assistantBubbleClass = theme === "dark" ? "bg-slate-800/90 text-slate-100 border border-slate-700/60" : "bg-slate-100 text-slate-900 border border-slate-200";
  const timestampClass = theme === "dark" ? "text-slate-500" : "text-slate-400";
  const inputBorder = theme === "dark" ? "border-slate-800" : "border-slate-300";
  const inputBg = theme === "dark" ? "bg-slate-900/80" : "bg-white";

  return (
    <div className={rootClass}>
      {/* Header */}
      <header className={`border-b px-4 py-3 flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between ${headerBorder} ${headerBg}`}>
        <div className="flex flex-col">
          <h1 className="text-xl font-semibold tracking-tight">YantraBuddy</h1>
          <p className={`text-xs ${headerSubText}`}>Your source for everything on rock breakers, spare parts, and dealer insights.</p>
        </div>

        <div className="flex items-center gap-3 mt-1 sm:mt-0 text-[11px]">
          <button type="button" onClick={toggleTheme} className="px-3 py-2 rounded-full border border-slate-500/40 bg-slate-900/10 text-xs hover:bg-slate-900/20 transition">
            {theme === "dark" ? "☀️ Light" : "🌙 Dark"}
          </button>
          <button type="button" onClick={handleClearChat} className="px-3 py-2 rounded-full border border-red-500/60 bg-red-500/10 text-xs text-red-400 hover:bg-red-500/20 transition">
            🧹 Clear
          </button>
        </div>
      </header>

      {/* Main */}
      <main className="flex-1 flex flex-col max-w-4xl mx-auto w-full p-4">
        <section className={`${cardClass} w-full flex flex-col h-[82vh]`}>
          {/* Greeting + quick actions */}
          <div className={`p-4 border-b ${dividerBorder}`}>
            <p className="text-sm font-medium">{greeting}, welcome to YantraLive.</p>
            <p className="text-xs mt-1 text-slate-400">{hasMessages ? "What else would you like to check? You can also try these questions:" : "How can I help you today? Here are some common questions:"}</p>

            <div className="mt-3 flex flex-wrap gap-2">
              {quickActions.map((q) => (
                <button key={q} type="button" onClick={() => handleQuickQuestion(q)} className="text-xs px-3 py-1 rounded-full border border-sky-500/60 bg-sky-900/40 hover:bg-sky-700/60 hover:border-sky-400 transition shadow-sm">
                  {q}
                </button>
              ))}
            </div>
          </div>

          {/* Messages area */}
          <div ref={messagesContainerRef} className="flex-1 min-h-0 overflow-y-auto p-4 space-y-3">
            {!hasMessages && (
              <div className="text-lg font-medium text-slate-300 text-center mt-20">Ask anything about breakers, machine compatibility, spare parts, or dealers.</div>
            )}

            {messages.map((m) => (
              <div key={m.id} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                <div className={`flex flex-col max-w-[80%] ${m.role === "user" ? "items-end" : "items-start"}`}>
                  <div className={`rounded-2xl px-3 py-2 text-sm whitespace-pre-wrap ${m.role === "user" ? "bg-sky-600 text-white shadow-md" : assistantBubbleClass}`}>
                    {m.role === "assistant" ? (
                      (() => {
                        const maybeTable = parseMarkdownTable(m.content);
                        if (maybeTable) {
                          return <ComparisonTable headers={maybeTable.headers} rows={maybeTable.rows} theme={theme} />;
                        }
                        return <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>;
                      })()
                    ) : (
                      m.content
                    )}
                  </div>

                  {/* Timestamp + tiny actions + brochure buttons (if present) */}
                  <div className="mt-1 flex items-center justify-between w-full gap-2">
                    <span className={`text-[10px] ${timestampClass}`}>{m.timestamp}</span>

                    <div className="flex items-center gap-1 text-[10px] text-slate-500">
                      {m.role === "assistant" && (
                        <>
                          <button type="button" onClick={() => handleCopy(m.content)} className="px-1 py-[1px] rounded-full border border-slate-600/60 hover:bg-slate-800/60" title="Copy reply">📋</button>
                          <button type="button" className="px-1 py-[1px] rounded-full border border-slate-600/60 hover:bg-slate-800/60" title="Like">👍</button>
                          <button type="button" className="px-1 py-[1px] rounded-full border border-slate-600/60 hover:bg-slate-800/60" title="Dislike">👎</button>
                        </>
                      )}

                      {/* If this assistant message included brochure_urls, render one button per brochure */}
                      {m.role === "assistant" && (m as any).brochure_urls && Array.isArray((m as any).brochure_urls) && (
                        <div className="flex items-center gap-1">
                          {((m as any).brochure_urls as string[]).map((u, i) => {
                            const label = labelFromUrl(u) || `Brochure ${i + 1}`;
                            return (
                              <button
                                key={i}
                                type="button"
                                onClick={() => {
                                  try {
                                    if (typeof window !== "undefined") {
                                      window.open(u, "_blank", "noopener,noreferrer");
                                    }
                                  } catch (e) {
                                    console.warn("Failed to open brochure", e);
                                  }
                                }}
                                className="text-[11px] px-2 py-1 rounded-md border border-slate-600/40 bg-slate-800/20 hover:bg-slate-800/30"
                                title={`Open ${label}`}
                              >
                                📘 Open Brochure — {label}
                              </button>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))}

            <div ref={messagesEndRef} />
            {loading && (
              <div className="flex justify-start">
                <div className="text-xs text-slate-400 px-3 py-2 bg-slate-800 rounded-xl border border-slate-700/60">Thinking based on the YantraLive datasets…</div>
              </div>
            )}
          </div>

          {/* Input */}
          <form className={`border-t p-3 flex gap-2 rounded-b-2xl flex-shrink-0 ${dividerBorder} ${theme === "dark" ? "bg-slate-950/70" : "bg-slate-50"}`} onSubmit={(e) => { e.preventDefault(); if (!loading) void handleSend(); }}>
            <textarea ref={textareaRef} className={`flex-1 text-sm rounded-xl border ${inputBorder} ${inputBg} p-2 resize-none outline-none focus:ring-1 focus:ring-sky-500 focus:border-sky-500`} rows={1} placeholder="Type your question about breakers, machines, spare parts, dealers, pricing..." value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={handleKeyDown} />
            <button type="submit" disabled={loading || !input.trim()} className="text-sm px-4 py-2 rounded-xl border border-sky-500 bg-sky-600 hover:bg-sky-500 disabled:opacity-50 disabled:cursor-not-allowed transition self-end shadow-md">{loading ? "Sending..." : "Send"}</button>
          </form>
        </section>
      </main>
    </div>
  );
}
