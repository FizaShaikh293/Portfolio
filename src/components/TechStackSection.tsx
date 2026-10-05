import SectionHeading from './SectionHeading';

const techCategories = [
  {
    title: 'Security Operations',
    items: ['Splunk', 'Microsoft Sentinel', 'SIEM Monitoring', 'Alert Triage', 'Incident Response', 'ServiceNow', 'Microsoft Defender', 'KQL'],
  },
  {
    title: 'Identity & Cloud',
    items: ['Azure AD', 'Entra ID', 'IAM', 'Identity Governance', 'GCP Cloud Security'],
  },
  {
    title: 'Security Analytics',
    items: ['Digital Forensics', 'Python', 'SQL', 'Anomaly Detection', 'Machine Learning', 'Isolation Forest', 'Autoencoders', 'SHAP'],
  },
  {
    title: 'Blockchain Security',
    items: ['Blockchain Forensics', 'Smart Contract Security', 'Solidity', 'Cryptography', 'ISO 27001'],
  },
];

export default function TechStackSection() {
  return (
    <section id="techstack" className="py-24 md:py-32 px-4 sm:px-8 md:px-12 max-w-6xl mx-auto">
      <SectionHeading label="Toolbox" title="Tech Stack" />

      <div className="mt-12 border-t-[3px] border-foreground">
        {techCategories.map((cat, idx) => (
          <div
            key={cat.title}
            className="grid grid-cols-1 md:grid-cols-[220px_1fr] lg:grid-cols-[260px_1fr] gap-x-12 gap-y-5 border-b border-border py-8 md:py-10 animate-fade-up"
            style={{ animationDelay: `${idx * 70}ms` }}
          >
            <div className="flex items-baseline gap-4">
              <span className="text-[11px] font-mono tracking-[0.2em] text-primary">
                {String(idx + 1).padStart(2, '0')}
              </span>
              <h3 className="font-display text-xl md:text-2xl leading-tight text-foreground">
                {cat.title}
              </h3>
            </div>

            <div className="flex flex-wrap gap-2.5 md:pt-1">
              {cat.items.map((item) => (
                <span
                  key={item}
                  className="inline-flex items-center border border-foreground/25 px-3.5 py-2 text-xs md:text-sm font-mono text-foreground/85 transition-colors duration-300 hover:border-primary hover:text-primary cursor-default"
                >
                  {item}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
