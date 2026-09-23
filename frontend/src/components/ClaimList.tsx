import { ChevronDown, Flag, Loader2 } from "lucide-react"
import { useState } from "react"
import { toast } from "sonner"
import { Button } from "@/components/ui/button"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Input } from "@/components/ui/input"
import { api } from "@/lib/api"
import { TONE_CLASS, VERDICT } from "@/lib/format"
import type { Claim, Mismatch } from "@/lib/types"
import { cn } from "@/lib/utils"

function ClaimRow({ claim, docId, selected, onSelect }: {
  claim: Claim
  docId: string
  selected: boolean
  onSelect: () => void
}) {
  const [open, setOpen] = useState(false)
  const [marking, setMarking] = useState(false)
  const [note, setNote] = useState("")
  const [saving, setSaving] = useState(false)
  const [marked, setMarked] = useState(false)
  const v = VERDICT[claim.verdict]

  async function save() {
    setSaving(true)
    try {
      const res = await api.correction(docId, claim.claim_text, note)
      setMarked(true)
      setMarking(false)
      toast.success(res.saved_to_memory
        ? "Correction saved. Future summaries in this session will avoid this mistake."
        : "Correction saved to the database, but Mem0 was unavailable.")
    } catch (e) {
      toast.error((e as Error).message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <div className={cn("rounded-xl border transition-colors", selected ? "border-primary/50 bg-accent/50" : "bg-[var(--surface)]")}>
        <div className="flex items-start gap-3 px-3.5 py-3">
          <span className={cn("mt-0.5 shrink-0 rounded-full px-2 py-0.5 text-[11px] font-semibold whitespace-nowrap", TONE_CLASS[v.tone])}>
            {v.label}
          </span>
          <button onClick={onSelect} className="min-w-0 flex-1 text-left text-sm leading-6" title="Highlight the evidence in the input">
            {claim.claim_text}
          </button>
          <CollapsibleTrigger asChild>
            <Button variant="ghost" size="icon-sm" aria-label="Show evidence" className="shrink-0">
              <ChevronDown className={cn("size-4 transition-transform", open && "rotate-180")} />
            </Button>
          </CollapsibleTrigger>
        </div>

        <CollapsibleContent>
          <div className="space-y-3 border-t px-3.5 py-3 text-sm">
            {claim.evidence_text ? (
              <div>
                <div className="section-label mb-1">Evidence (chunk #{claim.evidence_chunk_id})</div>
                <p className="leading-6 text-muted-foreground">“{claim.evidence_text}”</p>
              </div>
            ) : (
              <p className="text-muted-foreground">No matching evidence found in the source.</p>
            )}
            <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground tabular-nums">
              <span>NLI confidence {(claim.nli_confidence * 100).toFixed(0)}%</span>
              <span>Rerank score {claim.rerank_score.toFixed(2)}</span>
              {claim.verdict === "unsupported" && <span>Abstained: evidence below relevance threshold</span>}
            </div>

            {marked ? (
              <p className="text-xs font-medium text-[var(--brand-text)]">Marked wrong ✓ Saved to memory.</p>
            ) : marking ? (
              <div className="flex gap-2">
                <Input
                  autoFocus
                  value={note}
                  onChange={(e) => setNote(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && save()}
                  placeholder="What is wrong? e.g. the source says 4,500 crore"
                  className="h-8 text-sm"
                />
                <Button size="sm" onClick={save} disabled={saving} className="h-8">
                  {saving ? <Loader2 className="size-3.5 animate-spin" /> : "Save"}
                </Button>
              </div>
            ) : (
              <Button variant="outline" size="sm" className="h-7 gap-1.5 text-xs" onClick={() => setMarking(true)}>
                <Flag className="size-3" /> Mark wrong
              </Button>
            )}
          </div>
        </CollapsibleContent>
      </div>
    </Collapsible>
  )
}

export function ClaimList({ claims, mismatches, docId, selected, onSelect }: {
  claims: Claim[]
  mismatches: Mismatch[]
  docId: string
  selected: number | null
  onSelect: (i: number | null) => void
}) {
  const supported = claims.filter((c) => c.verdict === "entailed").length
  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between gap-2">
        <span className="section-label">Claim Verification</span>
        <span className="text-xs text-muted-foreground whitespace-nowrap tabular-nums">
          {supported}/{claims.length} supported
        </span>
      </div>
      {claims.map((c, i) => (
        <ClaimRow key={`${docId}-${i}`} claim={c} docId={docId} selected={selected === i}
                  onSelect={() => onSelect(selected === i ? null : i)} />
      ))}
      {mismatches.length > 0 && (
        <div className="rounded-xl border border-[var(--bad)]/30 bg-[var(--bad-bg)] px-3.5 py-3">
          <div className="section-label mb-1.5 text-[var(--bad)]">Entity / number mismatches</div>
          <ul className="space-y-1 text-sm">
            {mismatches.map((m, i) => (
              <li key={i}><span className="font-semibold">{m.type}:</span> {m.reason}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
