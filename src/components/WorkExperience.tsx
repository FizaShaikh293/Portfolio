import SectionHeading from './SectionHeading';

const companies = [
  {
    company: 'Teleperformance',
    client: 'Client: Hewlett Packard',
    roles: [
      {
        title: 'Junior SOC Analyst',
        period: 'Jul 2022 – Dec 2024',
        highlights: [
          'Triaged 30+ Splunk security alerts per shift, separating false positives from genuine threats and escalating confirmed or high-risk activity to senior analysts in line with SOC procedures.',
          'Investigated Azure AD / Entra ID sign-in anomalies, including brute-force patterns, unusual locations, and unfamiliar devices, escalating high-risk cases for further action such as account resets and access restrictions.',
          'Referenced MITRE ATT&CK techniques (e.g., T1078 Valid Accounts) when documenting escalations to provide consistent context for senior analysts reviewing cases.',
          'Documented findings, evidence, and escalations in ServiceNow, keeping case notes clear enough for seamless shift handovers.',
          'Authored 12 incident-response runbooks covering common alert types, standardizing investigation and escalation steps across the SOC team.',
          'Conducted monthly vulnerability scans across client infrastructure and tracked remediation actions through completion.',
          'Improved patch compliance from 67% to 91% over six months through vulnerability tracking and remediation follow-up.',
          'Monitored SIEM alerts and supported security log analysis, incident documentation and follow-up investigations.',
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
              {company.client}
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
