import type { AppStats, CompareResult, Health, HistoryItem, Length, Style, SummaryResult } from "./types"

/** Random id kept in localStorage; it is the history owner and the Mem0 user_id. No login. */
export function getSessionId(): string {
  let id = localStorage.getItem("saaraansh_session")
  if (!id) {
    id = "s-" + crypto.randomUUID()
    localStorage.setItem("saaraansh_session", id)
  }
  return id
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(url, init)
  } catch {
    throw new Error("Cannot reach the backend. Is it running on port 8000?")
  }
  if (!res.ok) {
    const body = await res.json().catch(() => null)
    throw new Error(body?.detail ?? `Request failed (${res.status})`)
  }
  return res.json()
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
})

export const api = {
  health: () => request<Health>("/api/health"),
  stats: () => request<AppStats>("/api/stats"),
  history: () => request<HistoryItem[]>(`/api/history?session_id=${getSessionId()}`),
  historyItem: (docId: string) => request<SummaryResult>(`/api/history/${docId}`),
  clearHistory: () => request<{ deleted: number }>(`/api/history?session_id=${getSessionId()}`, { method: "DELETE" }),
  correction: (docId: string, claimText: string, note: string) =>
    request<{ saved_to_memory: boolean }>("/api/corrections",
      json({ session_id: getSessionId(), doc_id: docId, claim_text: claimText, note })),
  feedback: (docId: string, rating: "up" | "down") => request("/api/feedback", json({ doc_id: docId, rating })),
}

export interface StreamHandlers {
  onStatus: (phase: string, data: { language_type?: string; warning?: boolean }) => void
  onToken: (text: string) => void
  onAttempt: (attempt: number, previousScore: number) => void
  onResult: (result: SummaryResult) => void
  onCompare: (compare: CompareResult) => void
  onError: (message: string) => void
}

/** POST /api/summarize and parse the Server-Sent Events stream (EventSource cannot POST). */
export async function summarizeStream(
  body: { text: string; length: Length; style: Style; compare: boolean },
  h: StreamHandlers,
  signal?: AbortSignal,
) {
  let res: Response
  try {
    res = await fetch("/api/summarize", { ...json({ ...body, session_id: getSessionId() }), signal })
  } catch (e) {
    if ((e as Error).name !== "AbortError") h.onError("Cannot reach the backend. Is it running on port 8000?")
    return
  }
  if (!res.ok || !res.body) {
    const err = await res.json().catch(() => null)
    h.onError(err?.detail ?? `Request failed (${res.status})`)
    return
  }

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader()
  let buffer = ""
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += value
    let sep: number
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, sep)
      buffer = buffer.slice(sep + 2)
      const event = block.match(/^event: (.*)$/m)?.[1]
      const data = block.match(/^data: (.*)$/m)?.[1]
      if (!event || !data) continue
      const d = JSON.parse(data)
      if (event === "status") h.onStatus(d.phase, d)
      else if (event === "token") h.onToken(d.text)
      else if (event === "attempt") h.onAttempt(d.attempt, d.previous_score)
      else if (event === "result") h.onResult(d)
      else if (event === "compare_result") h.onCompare(d)
      else if (event === "error") h.onError(d.message)
    }
  }
}
