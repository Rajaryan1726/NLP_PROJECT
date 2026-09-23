import { AlertTriangle, Copy, Loader2, RefreshCw, Sparkles, ThumbsDown, ThumbsUp } from "lucide-react"
import { toast } from "sonner"
import { Message, MessageAction, MessageActions, MessageContent, MessageResponse } from "@/components/ai-elements/message"
import { Button } from "@/components/ui/button"
import { api } from "@/lib/api"
import { seconds } from "@/lib/format"
import type { CompareResult, SummaryResult } from "@/lib/types"
import { cn } from "@/lib/utils"
import { ClaimList } from "./ClaimList"
import { ScoreBadge } from "./ScoreBadge"

interface Props {
  running: boolean
  phase: string
  streamText: string
  attempt: number
  result: SummaryResult | null
  compareOn: boolean
  compare: CompareResult | null
  selectedClaim: number | null
  onSelectClaim: (i: number | null) => void
  onRegenerate: () => void
  feedback: "up" | "down" | null
  onFeedback: (f: "up" | "down") => void
  threshold: number
}

function StatsLine({ result }: { result: SummaryResult }) {
  const s = result.stats
  return (
    <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-muted-foreground tabular-nums">
      <span className="whitespace-nowrap">{s.words_in} → {s.words_out} words</span>
      <span>·</span>
      <span className="whitespace-nowrap">{Math.round(s.compression_ratio * 100)}% compressed</span>
      <span>·</span>
      <span className="whitespace-nowrap">{seconds(s.inference_time_ms)}</span>
    </div>
  )
}

type CompareState = "waiting" | "verifying" | "unavailable"

function CompareCards({ compare, state }: { compare: CompareResult | null; state: CompareState }) {
  return (
    <div className="flex flex-col gap-3">
      <span className="section-label">Prompt Comparison</span>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        <div className="rounded-xl border bg-[var(--surface)] p-4">
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
            <span className="section-label">Standard prompt</span>
            {compare && <ScoreBadge score={compare.standard.factuality_score} />}
          </div>
          {compare ? (
            <p className="text-sm leading-6 whitespace-pre-line">{compare.standard.summary}</p>
          ) : (
            <p className="flex items-center gap-2 text-sm text-muted-foreground">
              {state === "verifying" && <Loader2 className="size-3.5 animate-spin" />}
              {{ waiting: "Waiting for the main summary…", verifying: "Verifying standard summary…",
                 unavailable: "Comparison unavailable for this summary." }[state]}
            </p>
          )}
        </div>
        <div className="rounded-xl border border-primary/40 bg-accent/60 p-4">
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
            <span className="section-label text-[var(--brand-text)]">Factuality-aware</span>
            {compare && <ScoreBadge score={compare.factuality_aware.factuality_score} />}
          </div>
          {compare ? (
            <>
              <p className="text-sm leading-6 whitespace-pre-line">{compare.factuality_aware.summary}</p>
              {compare.factuality_aware.first_attempt_score !== compare.factuality_aware.factuality_score && (
                <p className="mt-2 text-xs text-muted-foreground">
                  First attempt scored {compare.factuality_aware.first_attempt_score.toFixed(1)}/10 before the retry loop.
                </p>
              )}
            </>
          ) : (
            <p className="text-sm text-muted-foreground">Shown above.</p>
          )}
        </div>
      </div>
    </div>
  )
}

export function OutputPanel(p: Props) {
  const r = p.result
  // The stream stays open after the result while the compare baseline is verified,
  // so "generating" (main summary still streaming) is not the same as "running".
  const generating = p.running && !r
  const text = generating ? p.streamText : r?.summary ?? ""
  const empty = !p.running && !r

  function copy() {
    if (!text) return
    navigator.clipboard.writeText(text).then(() => toast.success("Summary copied"))
  }

  function rate(f: "up" | "down") {
    if (!r) return
    p.onFeedback(f)
    api.feedback(r.doc_id, f).then(() => toast.success("Thanks for the feedback")).catch((e) => toast.error(e.message))
  }

  return (
    <section className="panel flex flex-col gap-5 p-5 sm:p-6">
      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1">
        <h2 className="text-base font-bold whitespace-nowrap">Hinglish Summary</h2>
        {r && <StatsLine result={r} />}
      </div>

      {empty ? (
        <div className="flex min-h-[260px] flex-1 flex-col items-center justify-center gap-3 rounded-2xl border border-dashed text-center">
          <div className="grid size-11 place-items-center rounded-xl bg-accent text-primary">
            <Sparkles className="size-5" />
          </div>
          <div>
            <p className="text-sm font-semibold">Your summary appears here</p>
            <p className="mt-1 max-w-xs text-xs text-muted-foreground">
              Every claim is checked against the source with retrieval, reranking and NLI.
            </p>
          </div>
        </div>
      ) : (
        <>
          <div className="rounded-2xl border border-primary/30 bg-accent/40 p-4">
            <div className="mb-3 flex items-center justify-between gap-3">
              <span className="flex items-center gap-2 text-xs font-semibold whitespace-nowrap text-[var(--brand-text)]">
                <span className="brand-gradient grid size-5 place-items-center rounded-full">
                  <Sparkles className="size-3 text-white dark:text-[#1a0f33]" />
                </span>
                AI Summary
                {p.attempt > 1 && generating && (
                  <span className="font-medium text-muted-foreground">· attempt {p.attempt}</span>
                )}
              </span>
              {r ? (
                <ScoreBadge score={r.factuality_score} />
              ) : (
                <span className="flex items-center gap-1.5 text-xs text-muted-foreground whitespace-nowrap">
                  <span className="size-1.5 animate-pulse rounded-full bg-primary" /> {p.phase || "Starting"}
                </span>
              )}
            </div>
            <Message from="assistant" className="max-w-full">
              <MessageContent className="w-full text-[15px] leading-7">
                {text ? (
                  <MessageResponse isAnimating={generating} caret="block" mode={generating ? "streaming" : "static"}>
                    {text}
                  </MessageResponse>
                ) : (
                  <p className="text-sm text-muted-foreground">
                    {p.phase === "Generating" ? "Listing key facts before writing…" : `${p.phase || "Starting"}…`}
                  </p>
                )}
              </MessageContent>
            </Message>
          </div>

          {r && r.low_confidence && (
            <div className="flex items-start gap-2 rounded-xl border border-[var(--warn)]/40 bg-[var(--warn-bg)] px-3.5 py-2.5 text-sm text-[var(--warn)]">
              <AlertTriangle className="mt-0.5 size-4 shrink-0" />
              <span>
                Low confidence: after {r.attempts} attempt{r.attempts > 1 ? "s" : ""} the score is still below{" "}
                {p.threshold}/10. Check the claims marked below before using this summary.
              </span>
            </div>
          )}
          {r && !r.low_confidence && r.attempts > 1 && (
            <p className="text-xs text-muted-foreground">
              Improved by the corrective retry loop ({r.attempts} attempts).
            </p>
          )}

          <MessageActions className="justify-between">
            <div className="flex items-center gap-2">
              <Button variant="outline" size="sm" className="gap-1.5 rounded-lg" onClick={copy} disabled={!text}>
                <Copy className="size-3.5" /> Copy
              </Button>
              <Button variant="outline" size="sm" className="gap-1.5 rounded-lg" onClick={p.onRegenerate} disabled={p.running}>
                <RefreshCw className="size-3.5" /> Regenerate
              </Button>
            </div>
            <div className="flex items-center gap-1">
              <MessageAction tooltip="Good summary" disabled={!r} onClick={() => rate("up")}
                             className={cn(p.feedback === "up" && "text-[var(--ok)]")}>
                <ThumbsUp className="size-4" />
              </MessageAction>
              <MessageAction tooltip="Bad summary" disabled={!r} onClick={() => rate("down")}
                             className={cn(p.feedback === "down" && "text-[var(--bad)]")}>
                <ThumbsDown className="size-4" />
              </MessageAction>
            </div>
          </MessageActions>

          {p.compareOn && (r || p.running) && (
            <>
              <div className="border-t" />
              <CompareCards compare={p.compare} state={!r ? "waiting" : p.running ? "verifying" : "unavailable"} />
            </>
          )}

          {r && (
            <>
              <div className="border-t" />
              <ClaimList claims={r.claims} mismatches={r.entity_mismatches} docId={r.doc_id}
                         selected={p.selectedClaim} onSelect={p.onSelectClaim} />
            </>
          )}
          {generating && (
            <p className="text-xs text-muted-foreground">Claim-level verification appears here when the summary is done.</p>
          )}
        </>
      )}
    </section>
  )
}
