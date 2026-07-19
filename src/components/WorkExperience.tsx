import { Shield } from 'lucide-react';
import SectionHeading from './SectionHeading';

const experiences = [
  {
    title: 'SOC Analyst L1',
    company: 'Teleperformance (TP)',
    location: 'Mumbai, India',
    period: 'May 2023 – Dec 2024',
    icon: Shield,
    color: 'primary',
    highlights: [
      'Triaged 30+ live security alerts per shift using enterprise SIEM tooling, cutting the team open queue by 40% in the first quarter through consistent severity classification and rapid resolution.',
      'Identified a coordinated credential-stuffing campaign across 3 client accounts that had bypassed manual review; escalated proactively and blocked the attack before any data was accessed.',
      'Authored internal runbooks covering 12 common incident types, cutting average analyst resolution time by approximately 20 minutes per ticket and reducing escalations from junior analysts.',
    ],
  },
  {
    title: 'IT Security Analyst',
    company: 'Teleperformance (TP)',
    location: 'Mumbai, India',
    period: 'Jul 2022 – Apr 2023',
    icon: Shield,
    color: 'accent',
    highlights: [
      'Executed monthly vulnerability scans across client infrastructure, tracked remediation progress and raised critical patch compliance from 67% to 91% over 6 months.',
      'Supported SIEM alert monitoring, log analysis and incident documentation, building the foundation that led to promotion to SOC Analyst L1 within the year.',
    ],
  },
];

export default function WorkExperience() {
  const colorMap: Record<string, { text: string; glow: string; dot: string }> = {
    primary: { text: 'text-primary', glow: 'neon-box-cyan', dot: 'bg-primary shadow-[0_0_10px_hsl(var(--primary))]' },
    accent: { text: 'text-accent', glow: 'neon-box-yellow', dot: 'bg-accent shadow-[0_0_10px_hsl(var(--accent))]' },
  };

  return (
    <section id="experience" className="py-24 px-4 max-w-4xl mx-auto">
      <SectionHeading label="Career" title="Work Experience" />

      <div className="relative">
        <div className="absolute left-6 md:left-8 top-0 bottom-0 w-px bg-gradient-to-b from-primary via-accent to-secondary" />

        <div className="space-y-12">
          {experiences.map((exp, i) => {
            const c = colorMap[exp.color];
            const Icon = exp.icon;
            return (
              <div key={exp.title + exp.company} className="relative pl-16 md:pl-20 animate-fade-up" style={{ animationDelay: `${i * 100}ms` }}>
                <div className={`absolute left-4 md:left-6 top-2 w-4 h-4 rounded-full ${c.dot} z-10`} />
                <div className={`glass-panel p-6 ${c.glow} hover:scale-[1.02] transition-transform duration-300`}>
                  <div className="flex items-start gap-3 mb-3">
                    <Icon className={`w-6 h-6 shrink-0 ${c.text}`} />
                    <div>
                      <h3 className="font-display text-base font-bold text-foreground hover-text-pop cursor-default">{exp.title}</h3>
                      <p className={`text-sm font-mono ${c.text}`}>{exp.company} | {exp.location}</p>
                      <p className="text-xs text-muted-foreground font-mono mt-1">{exp.period}</p>
                    </div>
                  </div>
                  <ul className="space-y-2 mt-4">
                    {exp.highlights.map((h, j) => (
                      <li key={j} className="text-xs text-muted-foreground leading-relaxed flex gap-2">
                        <span className={`mt-1.5 w-1.5 h-1.5 rounded-full shrink-0 ${c.dot}`} />
                        {h}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
