interface Stage {
  label: string;
  items: string[];
}

interface Props {
  stages: Stage[];
  accent?: string;
}

/**
 * A clean, animated "architecture" cover for a project card:
 * a left-to-right pipeline of stages with the tech used at each step.
 */
export default function ArchitectureCover({ stages, accent = 'text-primary' }: Props) {
  return (
    <div className="relative overflow-hidden rounded-xl border border-white/[0.07] bg-white/[0.015] p-4 sm:p-5">
      <div className="pointer-events-none absolute inset-0 opacity-[0.35] [background-image:linear-gradient(hsl(var(--foreground)/0.05)_1px,transparent_1px),linear-gradient(90deg,hsl(var(--foreground)/0.05)_1px,transparent_1px)] [background-size:22px_22px]" />

      <div className="relative flex flex-col sm:flex-row items-stretch gap-2 sm:gap-0">
        {stages.map((s, i) => (
          <div key={s.label} className="flex-1 flex flex-col sm:flex-row items-center gap-2 sm:gap-0 min-w-0">
            <div className="w-full rounded-lg border border-white/[0.08] bg-background/40 backdrop-blur-sm px-3 py-2.5 transition-colors duration-300 hover:border-primary/30">
              <div className={`text-[9px] uppercase tracking-[0.18em] ${accent} mb-1.5`}>{s.label}</div>
              <div className="flex flex-wrap gap-1">
                {s.items.map((it) => (
                  <span
                    key={it}
                    className="text-[10px] font-mono text-muted-foreground/90 rounded border border-white/[0.07] px-1.5 py-0.5"
                  >
                    {it}
                  </span>
                ))}
              </div>
            </div>

            {i < stages.length - 1 && (
              <div className="shrink-0 sm:w-6 sm:h-px h-5 w-px relative overflow-hidden bg-white/10 sm:mx-1">
                <span className="arch-flow absolute inset-0 bg-gradient-to-r from-transparent via-primary to-transparent" />
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
