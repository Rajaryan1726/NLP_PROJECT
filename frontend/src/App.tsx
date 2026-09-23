import { useCallback, useEffect, useRef, useState } from "react"
import { toast } from "sonner"
import { HistoryPanel } from "@/components/HistoryPanel"
import { InputPanel } from "@/components/InputPanel"
import { OutputPanel } from "@/components/OutputPanel"
import { StatCards } from "@/components/StatCards"
import { TopBar } from "@/components/TopBar"
import { Toaster } from "@/components/ui/sonner"
import { TooltipProvider } from "@/components/ui/tooltip"
import { api, summarizeStream } from "@/lib/api"
import { guessLanguage } from "@/lib/format"
import type { AppStats, CompareResult, Health, HistoryItem, LanguageType, Length, Style, SummaryResult } from "@/lib/types"

function useTheme() {
  const [dark, setDark] = useState(() => {
    try {
      return localStorage.getItem("saaraansh_theme") !== "light" // dark by default
    } catch {
      return true
    }
  })
  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark)
    try {
      localStorage.setItem("saaraansh_theme", dark ? "dark" : "light")
    } catch { /* storage unavailable */ }
  }, [dark])
  return [dark, () => setDark((d) => !d)] as const
}

export default function App() {
  const [dark, toggleTheme] = useTheme()
  const [health, setHealth] = useState<Health | null>(null)
  const [healthError, setHealthError] = useState(false)
  const [stats, setStats] = useState<AppStats | null>(null)
  const [history, setHistory] = useState<HistoryItem[]>([])

  const [text, setText] = useState("")
  const [length, setLength] = useState<Length>("medium")
  const [style, setStyle] = useState<Style>("paragraph")
  const [compareOn, setCompareOn] = useState(false)
  const [serverLanguage, setServerLanguage] = useState<LanguageType | null>(null)

  const [running, setRunning] = useState(false)
  const [phase, setPhase] = useState("")
  const [streamText, setStreamText] = useState("")
  const [attempt, setAttempt] = useState(1)
  const [result, setResult] = useState<SummaryResult | null>(null)
  const [compare, setCompare] = useState<CompareResult | null>(null)
  const [selectedClaim, setSelectedClaim] = useState<number | null>(null)
  const [feedback, setFeedback] = useState<"up" | "down" | null>(null)
  // the text that `result` belongs to (evidence highlighting is only valid for it)
  const [resultText, setResultText] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const refresh = useCallback(() => {
    api.stats().then(setStats).catch(() => {})
    api.history().then(setHistory).catch(() => {})
  }, [])

  useEffect(() => {
    api.health().then(setHealth).catch(() => {
      setHealthError(true)
      toast.error("Cannot reach the backend. Start it on port 8000.")
    })
    refresh()
  }, [refresh])

  function onTextChange(t: string) {
    setText(t)
    setServerLanguage(null) // fall back to the client-side guess until the next run
  }

  async function summarize() {
    if (!text.trim() || running) return
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    setRunning(true)
    setPhase("Starting")
    setStreamText("")
    setAttempt(1)
    setResult(null)
    setCompare(null)
    setSelectedClaim(null)
    setFeedback(null)
    setResultText(text)

    await summarizeStream({ text, length, style, compare: compareOn }, {
      onStatus: (ph, d) => {
        setPhase(ph)
        if (d.language_type) setServerLanguage(d.language_type as LanguageType)
        if (d.warning) toast.warning(ph)
      },
      onToken: (t) => setStreamText((s) => s + t),
      onAttempt: (n, prev) => {
        setAttempt(n)
        setStreamText("") // the retry rewrites the summary from scratch
        toast.info(`Score ${prev.toFixed(1)}/10 is below ${health?.factuality_threshold ?? 7}. Improving the summary (attempt ${n})…`)
      },
      onResult: (r) => {
        setResult(r)
        refresh()
      },
      onCompare: setCompare,
      onError: (msg) => toast.error(msg),
    }, controller.signal)

    setRunning(false)
    setPhase("")
  }

  async function openHistory(id: string) {
    if (running) return
    try {
      const r = await api.historyItem(id)
      setResult(r)
      setCompare(r.compare ?? null)
      setText(r.source_text)
      setResultText(r.source_text)
      setServerLanguage(r.language_type)
      setSelectedClaim(null)
      setFeedback(r.feedback ?? null)
      if (r.options) {
        setLength(r.options.length)
        setStyle(r.options.style)
        setCompareOn(r.options.compare)
      }
    } catch (e) {
      toast.error((e as Error).message)
    }
  }

  async function clearHistory() {
    try {
      const { deleted } = await api.clearHistory()
      setResult(null)
      setCompare(null)
      refresh()
      toast.success(`Cleared ${deleted} summaries`)
    } catch (e) {
      toast.error((e as Error).message)
    }
  }

  const claim = selectedClaim !== null ? result?.claims[selectedClaim] : null
  const evidence =
    result && claim && claim.evidence_start !== null && claim.evidence_end !== null && text === resultText
      ? { source: result.source_text, start: claim.evidence_start, end: claim.evidence_end }
      : null

  return (
    <TooltipProvider>
      <TopBar health={health} healthError={healthError} dark={dark} onToggleTheme={toggleTheme} />
      <main className="mx-auto flex max-w-[1440px] flex-col gap-6 px-4 py-6 sm:px-6 sm:py-8">
        <StatCards stats={stats} />
        <div className="grid grid-cols-1 items-start gap-6 lg:grid-cols-2 xl:grid-cols-[260px_minmax(0,1fr)_minmax(0,1fr)]">
          <div className="lg:col-span-2 xl:col-span-1">
            <HistoryPanel items={history} activeId={result?.doc_id ?? null} onSelect={openHistory} onClear={clearHistory} />
          </div>
          <InputPanel
            text={text}
            onTextChange={onTextChange}
            language={serverLanguage ?? guessLanguage(text)}
            length={length}
            onLength={setLength}
            style={style}
            onStyle={setStyle}
            compare={compareOn}
            onCompare={setCompareOn}
            running={running}
            phase={phase}
            onSummarize={summarize}
            evidence={evidence}
            onCloseEvidence={() => setSelectedClaim(null)}
          />
          <OutputPanel
            running={running}
            phase={phase}
            streamText={streamText}
            attempt={attempt}
            result={result}
            compareOn={compareOn}
            compare={compare}
            selectedClaim={selectedClaim}
            onSelectClaim={setSelectedClaim}
            onRegenerate={summarize}
            feedback={feedback}
            onFeedback={setFeedback}
            threshold={health?.factuality_threshold ?? 7}
          />
        </div>
      </main>
      <Toaster theme={dark ? "dark" : "light"} position="bottom-right" richColors />
    </TooltipProvider>
  )
}
