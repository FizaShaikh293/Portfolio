import SectionHeading from './SectionHeading';

const certs = [
  { name: 'Microsoft Certified: Security Operations Analyst Associate', issuer: 'Microsoft', tags: ['Threat & Vulnerability Mgmt', 'Microsoft Defender'] },
  { name: 'Certified Online Fraud Prevention Specialist (COFPS)', issuer: 'Hack & Fix', tags: ['Fraud Prevention'] },
  { name: 'ISO/IEC 27001 Information Security Associate', issuer: 'SkillFront', tags: ['ISO 27001'] },
  { name: 'Junior Cybersecurity Analyst Career Path', issuer: 'Cisco Networking Academy', tags: ['Cybersecurity', 'Vulnerability Assessment'] },
  { name: 'Introduction to Digital Forensics', issuer: 'Security Blue Team', tags: ['Digital Forensics'] },
  { name: 'Encryption and Cryptography Essentials', issuer: 'IBM', tags: ['Cryptography'] },
  { name: 'Solidity Advanced: Secure Smart Contracts & DApp Development', issuer: 'Packt', tags: ['Smart Contracts'] },
  { name: 'Information Security Fundamentals', issuer: 'EC-Council', tags: [] },
  { name: 'Decentralized Finance (DeFi) Infrastructure', issuer: 'Duke University', tags: ['DeFi'] },
  { name: 'Web3 and Blockchain Fundamentals', issuer: 'INSEAD', tags: ['Web3'] },
  { name: 'Introduction to Cybersecurity Essentials', issuer: 'IBM', tags: ['Cybersecurity'] },
  { name: 'Blockchain Security', issuer: 'Infosec', tags: ['Blockchain'] },
  { name: 'Blockchain Basics', issuer: 'Coursera', tags: ['Smart Contracts', 'Blockchain'] },
  { name: 'Introduction to Cloud Identity', issuer: 'Google Cloud Security', tags: ['Cloud Security'] },
  { name: 'Cybersecurity Professional', issuer: 'Google', tags: ['Cybersecurity'] },
  { name: 'Ethical Hacking Essentials', issuer: 'EC-Council', tags: ['Ethical Hacking'] },
];

export default function CertificationsSection() {
  return (
    <section id="certs" className="py-24 px-4 max-w-4xl mx-auto">
      <SectionHeading label="Credentials" title="Certifications" />

      <div className="paper p-6 md:p-10">
        <ul>
          {certs.map((cert, i) => (
            <li
              key={cert.name + cert.issuer}
              className="group flex items-baseline gap-4 py-4 border-b border-border last:border-b-0 animate-fade-up"
              style={{ animationDelay: `${i * 40}ms` }}
            >
              <span className="text-[10px] font-mono text-muted-foreground w-6 shrink-0">
                {String(i + 1).padStart(2, '0')}
              </span>
              <div className="flex-1 min-w-0">
                <h3 className="text-base md:text-lg text-foreground transition-colors duration-300 group-hover:text-primary">
                  {cert.name}
                </h3>
                {cert.tags.length > 0 && (
                  <div className="mt-1.5 flex flex-wrap gap-1.5">
                    {cert.tags.map((tag) => (
                      <span
                        key={tag}
                        className="text-[9px] font-mono uppercase tracking-[0.1em] px-2 py-0.5 rounded-sm bg-secondary text-secondary-foreground border border-border"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                )}
              </div>
              <span className="text-[11px] font-mono uppercase tracking-[0.12em] text-muted-foreground whitespace-nowrap hidden sm:block">
                {cert.issuer}
              </span>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
