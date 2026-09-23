import type { LanguageType, Verdict } from "./types"

// Same heuristic as backend/preprocessing.py detect_language (quick guess while typing;
// the backend's answer replaces it after summarizing).
const ROMAN_HINDI = new Set(("hai hain tha thi ki ka ke ko ne se mein aur ya par pe bhi nahi nahin kya kyun ye yeh " +
  "woh vo jo jab tab kiya kiye gaya gayi gaye raha rahi rahe hoga hogi karna karne liye wala wale wali abhi " +
  "sabhi unke unka iske uske apne diya diye saal logon jaisa sirf").split(" "))

export function guessLanguage(text: string): LanguageType | null {
  const dev = (text.match(/[ऀ-ॿ]/g) ?? []).length
  const lat = (text.match(/[A-Za-z]/g) ?? []).length
  if (dev + lat < 20) return null
  const ratio = dev / (dev + lat)
  if (ratio >= 0.85) return "hindi"
  if (ratio >= 0.15) return "code_mixed"
  const words = text.toLowerCase().match(/[a-z]+/g) ?? []
  const hits = words.filter((w) => ROMAN_HINDI.has(w)).length
  return hits / Math.max(words.length, 1) >= 0.08 ? "code_mixed" : "english"
}

export const LANGUAGE_LABEL: Record<LanguageType, string> = {
  hindi: "Hindi Input Detected",
  english: "English Input Detected",
  code_mixed: "Code-Mixed Input Detected",
}

export const VERDICT: Record<Verdict, { label: string; tone: Tone }> = {
  entailed: { label: "Supported", tone: "ok" },
  neutral: { label: "Not sure", tone: "warn" },
  contradicted: { label: "Contradicted", tone: "bad" },
  unsupported: { label: "Unsupported", tone: "bad" },
}

export type Tone = "ok" | "warn" | "bad"

export function scoreTone(score: number): Tone {
  if (score >= 8) return "ok"
  if (score >= 6) return "warn"
  return "bad"
}

export const TONE_CLASS: Record<Tone, string> = {
  ok: "text-[var(--ok)] bg-[var(--ok-bg)]",
  warn: "text-[var(--warn)] bg-[var(--warn-bg)]",
  bad: "text-[var(--bad)] bg-[var(--bad-bg)]",
}

export const seconds = (ms: number) => `${(ms / 1000).toFixed(1)}s`

export function timeAgo(iso: string): string {
  const diff = (Date.now() - new Date(iso).getTime()) / 1000
  if (diff < 60) return "just now"
  if (diff < 3600) return `${Math.floor(diff / 60)} min ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)} hr ago`
  if (diff < 172800) return "Yesterday"
  return `${Math.floor(diff / 86400)} days ago`
}
