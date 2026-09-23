import { TONE_CLASS, scoreTone } from "@/lib/format"
import { cn } from "@/lib/utils"

/** Factuality score (0-10) shown as "% factual", coloured green / amber / red. */
export function ScoreBadge({ score, className }: { score: number; className?: string }) {
  return (
    <span
      className={cn("inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-semibold whitespace-nowrap tabular-nums", TONE_CLASS[scoreTone(score)], className)}
      title={`Factuality score ${score.toFixed(2)} / 10`}
    >
      <span className="size-1.5 rounded-full bg-current" />
      {Math.round(score * 10)}% factual
    </span>
  )
}
