import { useEffect, useState, useRef } from 'react';
import { GitCommit, Flame, Code2 } from 'lucide-react';

const stats = [
  { icon: GitCommit, label: 'Total Commits', value: 1247, color: 'text-primary' },
  { icon: Flame, label: 'Best Streak', value: 42, suffix: ' days', color: 'text-accent' },
  { icon: Code2, label: 'Repositories', value: 38, color: 'text-secondary' },
];

const languages = [
  { name: 'Python', pct: 35, color: 'bg-primary' },
  { name: 'Solidity', pct: 22, color: 'bg-secondary' },
  { name: 'JavaScript', pct: 18, color: 'bg-accent' },
  { name: 'C++', pct: 12, color: 'bg-primary' },
  { name: 'Rust', pct: 8, color: 'bg-secondary' },
  { name: 'Other', pct: 5, color: 'bg-muted-foreground' },
];

function AnimatedNumber({ target, suffix = '' }: { target: number; suffix?: string }) {
  const [count, setCount] = useState(0);
  const ref = useRef<HTMLSpanElement>(null);
  const hasAnimated = useRef(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !hasAnimated.current) {
          hasAnimated.current = true;
          const duration = 1500;
          const start = performance.now();
          const animate = (now: number) => {
            const progress = Math.min((now - start) / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            setCount(Math.floor(eased * target));
            if (progress < 1) requestAnimationFrame(animate);
          };
          requestAnimationFrame(animate);
        }
      },
      { threshold: 0.5 }
    );
    if (ref.current) observer.observe(ref.current);
    return () => observer.disconnect();
  }, [target]);

  return <span ref={ref}>{count}{suffix}</span>;
}

export default function GitHubStats() {
  const [barsVisible, setBarsVisible] = useState(false);
  const sectionRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) setBarsVisible(true); },
      { threshold: 0.3 }
    );
    if (sectionRef.current) observer.observe(sectionRef.current);
    return () => observer.disconnect();
  }, []);

  return (
    <section id="github" className="py-24 px-4 max-w-4xl mx-auto" ref={sectionRef}>
      <h2 className="font-display text-2xl md:text-3xl font-bold text-secondary neon-glow-purple mb-10 text-center">
        {'>'} GitHub Stats
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        {stats.map(({ icon: Icon, label, value, suffix, color }) => (
          <div key={label} className="glass-panel p-6 text-center neon-box-cyan">
            <Icon className={`w-8 h-8 mx-auto mb-2 ${color}`} />
            <div className={`font-display text-3xl font-bold ${color}`}>
              <AnimatedNumber target={value} suffix={suffix} />
            </div>
            <p className="text-xs font-mono text-muted-foreground mt-1">{label}</p>
          </div>
        ))}
      </div>

      <div className="glass-panel p-6 neon-box-purple">
        <h3 className="font-display text-sm font-semibold text-foreground mb-4">Top Languages</h3>
        <div className="space-y-3">
          {languages.map((lang) => (
            <div key={lang.name} className="flex items-center gap-3">
              <span className="text-xs font-mono text-muted-foreground w-20 text-right">{lang.name}</span>
              <div className="flex-1 h-3 bg-muted rounded-full overflow-hidden">
                <div
                  className={`h-full ${lang.color} rounded-full transition-all duration-1000 ease-out`}
                  style={{ width: barsVisible ? `${lang.pct}%` : '0%' }}
                />
              </div>
              <span className="text-xs font-mono text-muted-foreground w-10">{lang.pct}%</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
