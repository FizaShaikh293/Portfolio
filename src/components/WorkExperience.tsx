import { useEffect, useRef, useState } from 'react';
import { Briefcase, Shield, GraduationCap } from 'lucide-react';
import SectionHeading from './SectionHeading';

const experiences = [
  {
    title: 'IT Security Support Analyst',
    company: 'Teleperformance',
    location: 'Mumbai, India',
    period: 'Jul 2023 – Sep 2024',
    icon: Shield,
    color: 'primary',
    highlights: [
      'Triaged 30+ security-related support escalations per week, identifying phishing attempts and social engineering incidents targeting enterprise customers across North America.',
      'Conducted first-line email threat analysis using Wireshark and manual header inspection, flagging ~15% of escalated cases as confirmed phishing or suspicious activity.',
      'Reduced average incident resolution time by 20% by developing structured triage checklists and escalation workflows aligned with SOC best practices.',
      'Documented 100+ security incidents end-to-end using ticketing systems, contributing to knowledge base articles that improved team response consistency.',
      'Supervised and mentored a team of 8 associates, monitoring KPIs and enforcing quality and security standards, maintaining CSAT scores above 92%.',
      'Used Burp Suite for basic web request inspection during escalated cases involving suspected session hijacking or credential theft.',
    ],
  },
  {
    title: 'Afterschool Educator',
    company: 'Cara Children\'s Center',
    location: 'Letterkenny, Ireland',
    period: 'May 2025 – Present',
    icon: GraduationCap,
    color: 'accent',
    highlights: [
      'Designed and delivered structured educational programs while maintaining compliance with child protection policies.',
      'Demonstrated adaptability and clear communication with parents and multidisciplinary staff.',
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
