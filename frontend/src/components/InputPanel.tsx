import { Info, Loader2, Pencil, Sparkles, Upload } from "lucide-react"
import { useRef } from "react"
import { toast } from "sonner"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import { Textarea } from "@/components/ui/textarea"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { LANGUAGE_LABEL } from "@/lib/format"
import type { LanguageType, Length, Style } from "@/lib/types"

interface Props {
  text: string
  onTextChange: (t: string) => void
  language: LanguageType | null
  length: Length
  onLength: (l: Length) => void
  style: Style
  onStyle: (s: Style) => void
  compare: boolean
  onCompare: (c: boolean) => void
  running: boolean
  phase: string
  onSummarize: () => void
  /** Source text as the backend saw it + the span to highlight (from the selected claim). */
  evidence: { source: string; start: number; end: number } | null
  onCloseEvidence: () => void
}

const toggleItem =
  "h-8 rounded-lg px-3.5 text-sm font-medium whitespace-nowrap text-muted-foreground data-[state=on]:bg-card data-[state=on]:text-foreground data-[state=on]:shadow-sm"

export function InputPanel(p: Props) {
  const fileRef = useRef<HTMLInputElement>(null)
  const words = p.text.trim() ? p.text.trim().split(/\s+/).length : 0

  async function onFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    e.target.value = ""
    if (!file) return
    if (!file.name.toLowerCase().endsWith(".txt")) {
      toast.error("Please upload a .txt file.")
      return
    }
    p.onTextChange(await file.text())
    toast.success(`Loaded ${file.name}`)
  }

  return (
    <section className="panel flex flex-col gap-5 p-5 sm:p-6">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-base font-bold whitespace-nowrap">Input Text</h2>
        <div className="flex shrink-0 items-center gap-1 rounded-xl border bg-[var(--surface)] p-1">
          <span className="rounded-lg bg-card px-3 py-1.5 text-xs font-semibold whitespace-nowrap shadow-sm">Paste Text</span>
          <button
            onClick={() => fileRef.current?.click()}
            className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium whitespace-nowrap text-muted-foreground hover:text-foreground"
          >
            <Upload className="size-3.5" /> Upload .txt
          </button>
          <input ref={fileRef} type="file" accept=".txt,text/plain" hidden onChange={onFile} />
        </div>
      </div>

      {p.evidence ? (
        <div className="relative flex min-h-[260px] flex-1 flex-col rounded-2xl border bg-[var(--surface)]">
          <div className="flex items-center justify-between border-b px-4 py-2">
            <span className="section-label">Evidence in source</span>
            <Button variant="ghost" size="sm" className="h-7 gap-1.5 text-xs" onClick={p.onCloseEvidence}>
              <Pencil className="size-3" /> Edit text
            </Button>
          </div>
          <div className="max-h-[420px] flex-1 overflow-y-auto px-4 py-3 text-[15px] leading-7 whitespace-pre-wrap">
            {p.evidence.source.slice(0, p.evidence.start)}
            <mark
              ref={(el) => el?.scrollIntoView({ block: "nearest", behavior: "smooth" })}
              className="rounded-md bg-[var(--mark)] px-0.5 text-foreground ring-1 ring-primary/40"
            >
              {p.evidence.source.slice(p.evidence.start, p.evidence.end)}
            </mark>
            {p.evidence.source.slice(p.evidence.end)}
          </div>
        </div>
      ) : (
        <div className="relative flex min-h-[260px] flex-1 flex-col">
          <Textarea
            value={p.text}
            onChange={(e) => p.onTextChange(e.target.value)}
            placeholder="Paste Hindi, English or Hinglish (code-mixed) text here…"
            className="min-h-[260px] flex-1 resize-none rounded-2xl bg-[var(--surface)] px-4 py-3 text-[15px] leading-7"
          />
          <span className="pointer-events-none absolute right-3 bottom-2 text-[11px] text-muted-foreground tabular-nums">
            {words.toLocaleString()} / 5,000 words
          </span>
        </div>
      )}

      {p.language && (
        <div className="flex">
          <span className="inline-flex items-center gap-1.5 rounded-lg border border-primary/30 bg-accent px-2.5 py-1 text-xs font-semibold whitespace-nowrap text-[var(--brand-text)]">
            <Info className="size-3.5" />
            {LANGUAGE_LABEL[p.language]}
          </span>
        </div>
      )}

      <div className="flex flex-wrap gap-x-8 gap-y-4">
        <div className="flex flex-col gap-2">
          <span className="section-label">Summary Length</span>
          <ToggleGroup
            type="single"
            value={p.length}
            onValueChange={(v) => v && p.onLength(v as Length)}
            className="w-fit rounded-xl border bg-[var(--surface)] p-1"
          >
            <ToggleGroupItem value="short" className={toggleItem}>Short</ToggleGroupItem>
            <ToggleGroupItem value="medium" className={toggleItem}>Medium</ToggleGroupItem>
            <ToggleGroupItem value="long" className={toggleItem}>Long</ToggleGroupItem>
          </ToggleGroup>
        </div>
        <div className="flex flex-col gap-2">
          <span className="section-label">Style</span>
          <ToggleGroup
            type="single"
            value={p.style}
            onValueChange={(v) => v && p.onStyle(v as Style)}
            className="w-fit rounded-xl border bg-[var(--surface)] p-1"
          >
            <ToggleGroupItem value="paragraph" className={toggleItem}>Paragraph</ToggleGroupItem>
            <ToggleGroupItem value="bullets" className={toggleItem}>Bullet Points</ToggleGroupItem>
          </ToggleGroup>
        </div>
      </div>

      <label className="flex cursor-pointer items-center justify-between gap-4 rounded-xl border bg-[var(--surface)] px-4 py-3">
        <span className="min-w-0">
          <span className="block text-sm font-semibold">Compare Standard vs Factuality-Aware</span>
          <span className="block text-xs text-muted-foreground">Also runs the plain prompt and scores both</span>
        </span>
        <Switch checked={p.compare} onCheckedChange={p.onCompare} />
      </label>

      <Button
        onClick={p.onSummarize}
        disabled={p.running || words === 0}
        className="brand-gradient h-12 rounded-xl border-0 text-[15px] font-semibold text-white shadow-[0_0_32px_var(--glow)] hover:opacity-90 disabled:opacity-60 dark:text-[#1a0f33]"
      >
        {p.running ? (
          <>
            <Loader2 className="size-4 animate-spin" /> {p.phase || "Working"}…
          </>
        ) : (
          <>
            <Sparkles className="size-4" /> Summarize
          </>
        )}
      </Button>
    </section>
  )
}
