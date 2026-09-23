import { History, Trash2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { TONE_CLASS, scoreTone, timeAgo } from "@/lib/format"
import type { HistoryItem } from "@/lib/types"
import { cn } from "@/lib/utils"

interface Props {
  items: HistoryItem[]
  activeId: string | null
  onSelect: (id: string) => void
  onClear: () => void
}

export function HistoryPanel({ items, activeId, onSelect, onClear }: Props) {
  return (
    <aside className="panel flex flex-col p-4 xl:sticky xl:top-6">
      <div className="mb-3 flex items-center gap-2 px-2">
        <History className="size-4 text-muted-foreground" />
        <span className="section-label">Recent</span>
      </div>

      {items.length === 0 ? (
        <p className="px-2 py-6 text-sm text-muted-foreground">No summaries yet. Your history appears here.</p>
      ) : (
        <ul className="flex max-h-[520px] flex-col gap-1 overflow-y-auto">
          {items.map((item) => (
            <li key={item.id}>
              <button
                onClick={() => onSelect(item.id)}
                className={cn(
                  "w-full rounded-xl px-3 py-2.5 text-left transition-colors hover:bg-accent/60",
                  activeId === item.id && "bg-accent",
                )}
              >
                <div className="truncate text-sm font-semibold">{item.title || "Untitled"}</div>
                <div className="mt-1 flex items-center gap-2 text-xs text-muted-foreground">
                  <span className="whitespace-nowrap">{timeAgo(item.created_at)}</span>
                  <span
                    className={cn(
                      "rounded-full px-1.5 py-px text-[10px] font-semibold whitespace-nowrap",
                      TONE_CLASS[scoreTone(item.final_score)],
                    )}
                  >
                    {item.final_score.toFixed(1)}
                  </span>
                  {item.language_type === "code_mixed" && <span className="whitespace-nowrap">code-mixed</span>}
                </div>
              </button>
            </li>
          ))}
        </ul>
      )}

      <div className="mt-3 border-t pt-3">
        <Button
          variant="ghost"
          size="sm"
          disabled={items.length === 0}
          onClick={onClear}
          className="w-full justify-start gap-2 text-muted-foreground"
        >
          <Trash2 className="size-4" />
          Clear history
        </Button>
      </div>
    </aside>
  )
}
