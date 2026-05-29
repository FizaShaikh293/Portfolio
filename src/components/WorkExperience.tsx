import { useEffect, useRef, useState } from 'react';
import { Shield } from 'lucide-react';
import SectionHeading from './SectionHeading';

const experiences = [
  {
    title: 'IT Security Support Analyst',
    company: 'Teleperformance',
    location: 'Mumbai, India',
    period: 'Jul 2023 – Aug 2024',
    icon: Shield,
    color: 'primary',
    highlights: [
      "Monitored live security events across the enterprise using SIEM tooling, resolving an average of 30+ alerts per shift and reducing the team's open queue by 40% within the first quarter through consistent triage and accurate severity classification.",
      'Identified a recurring pattern of failed authentication attempts across 3 client accounts that manual review had missed; escalated as a coordinated credential stuffing attempt which was blocked before any data was accessed.',
      'Ran monthly vulnerability scans across client infrastructure, documented findings in structured reports and tracked remediation progress, bringing the critical patch compliance rate from 67% to 91% over 6 months in collaboration with the infrastructure team.',
      'Built internal documentation covering 12 common incident types, cutting average analyst resolution time for those scenarios by approximately 20 minutes per ticket by giving junior team members a reliable reference rather than escalating every edge case.',
      'Skills applied: Threat & Vulnerability Management, Incident Response, SIEM, Security Monitoring and Log Analysis.',
    ],
  },
];

export default function WorkExperience() {
  const [visibleItems, setVisibleItems] = useState<number[]>([]);
  const itemRefs = useRef<(HTMLDivElement | null)[]>([]);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const idx = Number(entry.target.getAttribute('data-exp-index'));
            setVisibleItems((prev) => prev.includes(idx) ? prev : [...prev, idx]);
          }
        });
      },
      { threshold: 0.2 }
    );
    itemRefs.current.forEach((ref) => { if (ref) observer.observe(ref); });
    return () => observer.disconnect();
  }, []);

  const colorMap: Record<string, { text: string; border: string; glow: string; dot: string }> = {
    primary: { text: 'text-primary', border: 'border-primary/30', glow: 'neon-box-cyan', dot: 'bg-primary shadow-[0_0_10px_hsl(var(--primary))]' },
    accent: { text: 'text-accent', border: 'border-accent/30', glow: 'neon-box-yellow', dot: 'bg-accent shadow-[0_0_10px_hsl(var(--accent))]' },
  };

  return (
    <section id="experience" className="py-24 px-4 max-w-4xl mx-auto">
      <SectionHeading label="Career" title="Work Experience" />

      <div className="relative">
        {/* Timeline line */}
        <div className="absolute left-6 md:left-8 top-0 bottom-0 w-px bg-gradient-to-b from-primary via-accent to-secondary" />

        <div className="space-y-12">
          {experiences.map((exp, i) => {
            const c = colorMap[exp.color];
            const isVisible = visibleItems.includes(i);
            const Icon = exp.icon;

            return (
              <div
                key={exp.title + exp.company}
                ref={(el) => { itemRefs.current[i] = el; }}
                data-exp-index={i}
                className={`relative pl-16 md:pl-20 transition-all duration-700 ${
                  isVisible ? 'opacity-100 translate-x-0' : 'opacity-0 -translate-x-8'
                }`}
                style={{ transitionDelay: isVisible ? `${i * 200}ms` : '0ms' }}
              >
                {/* Timeline dot */}
                <div className={`absolute left-4 md:left-6 top-2 w-4 h-4 rounded-full ${c.dot} z-10`} />

                <div className={`glass-panel p-6 ${c.glow} hover:scale-[1.02] transition-transform duration-300`}>
                  <div className="flex items-start gap-3 mb-3">
                    <Icon className={`w-6 h-6 shrink-0 ${c.text}`} />
                    <div>
                      <h3 className="font-display text-base font-bold text-foreground">{exp.title}</h3>
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
