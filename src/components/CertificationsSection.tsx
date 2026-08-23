import SectionHeading from './SectionHeading';

const certs = [
  { name: 'Google Cybersecurity Professional Certificate', issuer: 'Google' },
  { name: 'Junior Cybersecurity Analyst Career Path', issuer: 'Cisco Networking Academy' },
  { name: 'ISO/IEC 27001 Information Security Associate', issuer: 'SkillFront' },
  { name: 'Introduction to Digital Forensics', issuer: 'Security Blue Team' },
  { name: 'Encryption & Cryptography Essentials', issuer: 'IBM' },
  { name: 'Ethical Hacking Essentials', issuer: 'EC-Council' },
  { name: 'Blockchain Security', issuer: 'Infosec' },
  { name: 'Solidity Advanced: Secure Smart Contracts & DApp Development', issuer: 'Packt' },
  { name: 'Introduction to Cloud Identity', issuer: 'Google Cloud Security' },
];

export default function CertificationsSection() {
  return (
    <section id="certs" className="py-24 px-4 max-w-4xl mx-auto">
      <SectionHeading label="Credentials" title="Certifications" />

      <div className="paper p-6 md:p-10">
        <ul>
          {certs.map((cert, i) => (
            <li
              key={cert.name}
              className="group flex items-baseline gap-4 py-4 border-b border-border last:border-b-0 animate-fade-up"
              style={{ animationDelay: `${i * 50}ms` }}
            >
              <span className="text-[10px] font-mono text-muted-foreground w-6 shrink-0">
                {String(i + 1).padStart(2, '0')}
              </span>
              <div className="flex-1 min-w-0">
                <h3 className="text-base md:text-lg text-foreground transition-colors duration-300 group-hover:text-primary">
                  {cert.name}
                </h3>
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
