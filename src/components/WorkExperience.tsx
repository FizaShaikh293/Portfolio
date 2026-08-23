import { Shield, ChevronUp } from 'lucide-react';
import SectionHeading from './SectionHeading';

const companies = [
  {
    company: 'Teleperformance (TP)',
    location: 'Mumbai, India',
    totalPeriod: 'Jul 2022 – Dec 2024',
    icon: Shield,
    roles: [
      {
        title: 'Junior SOC Analyst',
        period: 'May 2023 – Dec 2024',
        promotion: 'Promoted from IT Security Analyst',
        color: 'primary',
        highlights: [
          "Triaged 30+ live security alerts per shift using enterprise SIEM tooling, cutting the team's open queue by 40% in the first quarter through consistent severity classification and rapid resolution.",
          'Identified a coordinated credential-stuffing campaign across 3 client accounts that had bypassed manual review; escalated proactively and blocked the attack before any data was accessed.',
          'Authored internal runbooks covering 12 common incident types, cutting average analyst resolution time by approximately 20 minutes per ticket and reducing escalations from junior analysts.',
        ],
      },
      {
        title: 'IT Security Analyst',
        period: 'Jul 2022 – Apr 2023',
        color: 'accent',
        highlights: [
          'Executed monthly vulnerability scans across client infrastructure, tracked remediation progress and raised critical patch compliance from 67% to 91% over 6 months.',
          'Supported SIEM alert monitoring, log analysis and incident documentation, building the foundation that led to promotion to Junior SOC Analyst within the year.',
        ],
      },
    ],
  },
];

export default function WorkExperience() {
  const colorMap: Record<string, { text: string; glow: string; dot: string; subtle: string }> = {
    primary: {
      text: 'text-primary',
      glow: '',
      dot: 'bg-primary ',
      subtle: 'bg-primary/10',
    },
    accent: {
      text: 'text-accent',
      glow: '',
      dot: 'bg-accent ',
      subtle: 'bg-accent/10',
    },
  };

  return (
    <section id="experience" className="py-24 px-4 max-w-4xl mx-auto">
      <SectionHeading label="Career" title="Work Experience" />

      <div className="relative">
        <div className="absolute left-6 md:left-8 top-0 bottom-0 w-px bg-gradient-to-b from-primary via-accent to-secondary" />

        <div className="space-y-12">
          {companies.map((company, i) => {
            const Icon = company.icon;
            return (
              <div
                key={company.company}
                className="relative pl-16 md:pl-20 animate-fade-up"
                style={{ animationDelay: `${i * 100}ms` }}
              >
                <div className="absolute left-4 md:left-6 top-3 w-4 h-4 rounded-full bg-primary  z-10" />

                <div className="paper p-6 md:p-8  hover:scale-[1.01] transition-transform duration-300">
                  {/* Company header */}
                  <div className="flex items-start gap-4 mb-8 pb-6 border-b border-border/40">
                    <div className="p-3 rounded-xl bg-primary/10">
                      <Icon className="w-7 h-7 text-primary" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <h3 className="font-display text-lg font-bold text-foreground hover-text-pop cursor-default">
                        {company.company}
                      </h3>
                      <p className="text-sm font-mono text-muted-foreground">
                        {company.location} | {company.totalPeriod}
                      </p>
                    </div>
                  </div>

                  {/* Roles within the company */}
                  <div className="relative space-y-10 pl-1">
                    {company.roles.map((role, j) => {
                      const c = colorMap[role.color];
                      return (
                        <div key={role.title} className="relative">
                          {j > 0 && (
                            <div className="absolute -left-1 md:left-0 -top-10 bottom-0 w-px bg-gradient-to-b from-transparent via-accent/50 to-transparent" />
                          )}

                          <div className="flex items-start gap-3 mb-4">
                            <div className={`mt-1.5 w-2 h-2 rounded-full ${c.dot} shrink-0`} />
                            <div className="flex-1 min-w-0">
                              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-1">
                                <h4 className={`font-display text-base font-bold ${c.text} hover-text-pop cursor-default`}>
                                  {role.title}
                                </h4>
                                <span className="text-xs font-mono text-muted-foreground whitespace-nowrap">
                                  {role.period}
                                </span>
                              </div>
                              {role.promotion && (
                                <div className="mt-2 inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-semibold tracking-wide uppercase text-accent bg-accent/10 border border-accent/20">
                                  <ChevronUp className="w-3 h-3" />
                                  {role.promotion}
                                </div>
                              )}
                            </div>
                          </div>

                          <ul className="space-y-2.5">
                            {role.highlights.map((h, k) => (
                              <li key={k} className="text-sm text-muted-foreground leading-relaxed flex gap-3">
                                <span className={`mt-2 w-1 h-1 rounded-full shrink-0 ${c.dot}`} />
                                {h}
                              </li>
                            ))}
                          </ul>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}

