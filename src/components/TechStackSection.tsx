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
    title: 'Vulnerability & Network',
    items: ['Vulnerability Management', 'Patch Management', 'IDS/IPS', 'Network Segmentation', 'DNS/DHCP', 'NAT', 'NAC'],
  },
  {
    title: 'Security Analytics',
    items: ['Digital Forensics', 'Python', 'SQL', 'Anomaly Detection', 'Machine Learning', 'Isolation Forest', 'Autoencoders', 'SHAP'],
  },
  {
    title: 'Blockchain Security',
    items: ['Blockchain Forensics', 'Smart Contract Security', 'Solidity', 'Cryptography', 'ISO 27001'],
  },
  {
    title: 'Development',
    items: ['React', 'JavaScript', 'Vite', 'Streamlit', 'Git', 'Vercel'],
  },
];

export default function TechStackSection() {
  return (
    <section id="techstack" className="py-24 md:py-32 px-4 sm:px-8 md:px-12 max-w-6xl mx-auto">
      <SectionHeading label="Toolbox" title="Tech Stack" />

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {techCategories.map((cat, catIdx) => (
          <div
            key={cat.title}
            className="paper paper-lifted bold-panel p-6 min-h-48 animate-fade-up"
            style={{ animationDelay: `${catIdx * 70}ms` }}
          >
            <div className="flex items-baseline justify-between mb-4 pb-2 border-b border-border">
              <h3 className="text-2xl leading-none text-foreground">{cat.title}</h3>
              <span className="text-4xl font-display text-primary">
                {String(catIdx + 1).padStart(2, '0')}
              </span>
            </div>
            <div className="flex flex-wrap gap-x-3 gap-y-1.5">
              {cat.items.map((item) => (
                <span
                  key={item}
                  className="text-[11px] font-mono text-muted-foreground transition-colors duration-300 hover:text-primary cursor-default"
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
