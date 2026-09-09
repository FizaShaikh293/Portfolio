import { ChevronUp } from 'lucide-react';
import SectionHeading from './SectionHeading';

const companies = [
  {
    company: 'Teleperformance (TP)',
    location: 'Mumbai, India',
    totalPeriod: 'Jul 2022 – Dec 2024',
    roles: [
      {
        title: 'Junior SOC Analyst',
        period: 'May 2023 – Dec 2024',
        promotion: 'Promoted from IT Security Analyst',
        highlights: [
          "Triaged 30+ live security alerts per shift using enterprise SIEM tooling, cutting the team's open queue by 40% in the first quarter through consistent severity classification and rapid resolution.",
          'Identified a coordinated credential-stuffing campaign across 3 client accounts that had bypassed manual review; escalated proactively and blocked the attack before any data was accessed.',
          'Authored internal runbooks covering 12 common incident types, cutting average analyst resolution time by approximately 20 minutes per ticket and reducing escalations from junior analysts.',
        ],
      },
      {
        title: 'IT Security Analyst',
        period: 'Jul 2022 – Apr 2023',
        highlights: [
          'Executed monthly vulnerability scans across client infrastructure, tracked remediation progress and raised critical patch compliance from 67% to 91% over 6 months.',
          'Supported SIEM alert monitoring, log analysis and incident documentation, building the foundation that led to promotion to Junior SOC Analyst within the year.',
        ],
      },
    ],
  },
];

export default function WorkExperience() {
  return (
    <section id="experience" className="py-24 md:py-32 px-4 sm:px-8 md:px-12 max-w-6xl mx-auto">
      <SectionHeading label="Career" title="Work Experience" />

      {companies.map((company) => (
        <div key={company.company} className="paper bold-panel p-6 md:p-10 animate-fade-up">
          <div className="flex flex-col sm:flex-row sm:items-baseline sm:justify-between gap-2 pb-7 border-b-[3px] border-foreground">
            <h3 className="text-3xl md:text-5xl text-foreground">{company.company}</h3>
            <p className="text-[11px] font-mono uppercase tracking-[0.16em] text-muted-foreground">
              {company.location} · {company.totalPeriod}
            </p>
          </div>

          <div className="mt-8 space-y-10">
            {company.roles.map((role) => (
              <div key={role.title} className="relative pl-6">
                <span className="absolute left-0 top-2 w-2 h-2 rounded-full bg-foreground/70" />
                <span className="absolute left-[3.5px] top-6 bottom-[-1.75rem] w-px bg-border last:hidden" />

                <div className="flex flex-col sm:flex-row sm:items-baseline sm:justify-between gap-1">
                  <h4 className="text-lg text-foreground">{role.title}</h4>
                  <span className="text-[11px] font-mono uppercase tracking-[0.16em] text-muted-foreground whitespace-nowrap">
                    {role.period}
                  </span>
                </div>

                {role.promotion && (
                  <span className="mt-2 inline-flex items-center gap-1.5 border border-foreground/20 px-2.5 py-1 text-[10px] font-mono uppercase tracking-[0.16em] text-muted-foreground">
                    <ChevronUp className="w-3 h-3" />
                    {role.promotion}
                  </span>
                )}

                <ul className="mt-4 space-y-3">
                  {role.highlights.map((h, k) => (
                    <li key={k} className="flex gap-3 text-sm leading-relaxed text-muted-foreground">
                      <span className="mt-2 h-px w-3 shrink-0 bg-foreground/30" />
                      {h}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
      ))}
    </section>
  );
}
