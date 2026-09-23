import { Clock, FileText, Languages, ShieldCheck } from "lucide-react"
import type { ReactNode } from "react"
import { seconds } from "@/lib/format"
import type { AppStats } from "@/lib/types"

function Card({ icon, value, label }: { icon: ReactNode; value: string; label: string }) {
  return (
    <div className="panel flex items-center gap-3 px-4 py-4 sm:gap-4 sm:px-5">
      <div className="hidden size-10 shrink-0 place-items-center rounded-xl border bg-accent text-primary sm:grid">{icon}</div>
      <div className="min-w-0">
        <div className="text-2xl font-bold tracking-tight tabular-nums whitespace-nowrap">{value}</div>
        <div className="truncate text-xs text-muted-foreground">{label}</div>
      </div>
    </div>
  )
}

/** Every number here comes from GET /api/stats. "—" means there is no data yet. */
export function StatCards({ stats }: { stats: AppStats | null }) {
  const total = stats?.total_documents ?? 0
  const pctMixed = total ? Math.round((100 * (stats?.code_mixed_documents ?? 0)) / total) : null
  return (
    <div className="grid grid-cols-2 gap-3 sm:gap-4 xl:grid-cols-4">
      <Card icon={<FileText className="size-5" />} value={stats ? total.toLocaleString() : "—"} label="Documents summarized" />
      <Card
        icon={<Languages className="size-5" />}
        value={pctMixed === null ? "—" : `${pctMixed}%`}
        label={stats ? `Code-mixed inputs (${stats.code_mixed_documents} of ${total})` : "Code-mixed inputs"}
      />
      <Card
        icon={<ShieldCheck className="size-5" />}
        value={stats?.avg_factuality_score != null ? `${stats.avg_factuality_score.toFixed(1)}/10` : "—"}
        label="Avg. factuality score"
      />
      <Card
        icon={<Clock className="size-5" />}
        value={stats?.avg_inference_time_ms != null ? seconds(stats.avg_inference_time_ms) : "—"}
        label="Avg. inference time"
      />
    </div>
  )
}
