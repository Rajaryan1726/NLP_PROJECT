import { Moon, Sun } from "lucide-react"
import { Button } from "@/components/ui/button"
import type { Health } from "@/lib/types"

interface Props {
  health: Health | null
  healthError: boolean
  dark: boolean
  onToggleTheme: () => void
}

export function TopBar({ health, healthError, dark, onToggleTheme }: Props) {
  const status = healthError ? "offline" : health?.status ?? "loading"
  const dot = status === "ok" ? "bg-emerald-400" : status === "degraded" ? "bg-amber-400" : "bg-zinc-500"

  return (
    <header className="border-b border-border/80">
      <div className="mx-auto flex h-[76px] max-w-[1440px] items-center justify-between gap-4 px-4 sm:px-6">
        <div className="flex min-w-0 items-center gap-3">
          <div className="brand-gradient grid size-10 shrink-0 place-items-center rounded-xl text-lg font-bold text-white shadow-[0_0_24px_var(--glow)] dark:text-[#1a0f33]">
            सा
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-lg font-bold tracking-tight whitespace-nowrap">Saaraansh</span>
              <span className="brand-gradient rounded-md px-1.5 py-0.5 text-[10px] font-bold tracking-wide text-white dark:text-[#1a0f33]">
                PRO
              </span>
            </div>
            <p className="hidden truncate text-xs text-muted-foreground sm:block">
              Factuality-aware Hindi &amp; Hinglish summarization
            </p>
          </div>
        </div>

        <div className="flex shrink-0 items-center gap-2 sm:gap-3">
          <div
            className="hidden items-center gap-2 rounded-full border bg-card px-3 py-1.5 text-xs font-medium whitespace-nowrap sm:flex"
            title={healthError ? "Backend not reachable" : `Backend status: ${status}`}
          >
            <span className={`size-2 rounded-full ${dot}`} />
            {healthError ? "backend offline" : health?.llm.model ?? "connecting…"}
          </div>
          <Button variant="outline" size="icon" onClick={onToggleTheme} aria-label="Toggle theme" className="rounded-xl">
            {dark ? <Sun className="size-4" /> : <Moon className="size-4" />}
          </Button>
          {/* Visual only: there are no paid plans. */}
          <Button className="brand-gradient rounded-xl border-0 px-4 font-semibold whitespace-nowrap text-white shadow-[0_0_24px_var(--glow)] hover:opacity-90 dark:text-[#1a0f33]">
            Upgrade
          </Button>
        </div>
      </div>
    </header>
  )
}
