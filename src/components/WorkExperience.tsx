import SectionHeading from './SectionHeading';

const companies = [
  {
    company: 'Hewlett Packard (HP)',
    totalPeriod: 'Jul 2022 – Dec 2024',
    roles: [
      {
        title: 'Junior SOC Analyst',
        period: 'May 2023 – Dec 2024',
        highlights: [
          'Triaged more than 30 Splunk security alerts per shift, identifying false positives and suspicious activity before escalating incidents through established SOC procedures.',
          'Investigated Azure AD authentication and account anomalies by reviewing sign-in activity and security logs for suspicious access patterns.',
          'Recorded investigations and escalations in ServiceNow, maintained clear shift-handover notes, and developed 12 incident-response runbooks for common alert types.',
        ],
      },
      {
        title: 'IT Security Analyst',
        period: 'Jul 2022 – Apr 2023',
        highlights: [
          'Conducted monthly vulnerability scans across client infrastructure and tracked remediation actions through completion.',
          'Improved patch compliance from 67% to 91% in six months while supporting SIEM monitoring, log analysis, incident documentation, and follow-up investigations.',
        ],
      },
    ],
  },
];

export default function WorkExperience() {
  return (
    <section id="experience" className="py-24 md:py-32 px-4 sm:px-8 md:px-12 max-w-6xl mx-auto">
      <SectionHeading label="Career" title="Work Experience" />

      <div className="space-y-5">
      {companies.map((company) => (
        <div key={company.company} className="paper bold-panel p-6 md:p-10 animate-fade-up">
          <div className="flex flex-col sm:flex-row sm:items-baseline sm:justify-between gap-2 pb-7 border-b-[3px] border-foreground">
            <h3 className="text-3xl md:text-5xl text-foreground">{company.company}</h3>
            <p className="text-[11px] font-mono uppercase tracking-[0.16em] text-muted-foreground">
              {company.totalPeriod}
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
      </div>
    </section>
  );
}
