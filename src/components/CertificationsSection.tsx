import { Award } from 'lucide-react';
import SectionHeading from './SectionHeading';

const certs = [
  { name: 'Google Cybersecurity Professional Certificate', issuer: 'Google', year: 'Jul 2025' },
  { name: 'Junior Cybersecurity Analyst Career Path', issuer: 'Cisco Networking Academy', year: 'Mar 2026' },
  { name: 'ISO/IEC 27001 Information Security Associate', issuer: 'SkillFront', year: 'Apr 2026' },
  { name: 'Introduction to Digital Forensics', issuer: 'Security Blue Team', year: 'Feb 2026' },
  { name: 'Encryption & Cryptography Essentials', issuer: 'IBM', year: 'Jul 2025' },
  { name: 'Ethical Hacking Essentials', issuer: 'EC-Council', year: 'Apr 2023' },
  { name: 'Blockchain Security', issuer: 'Infosec', year: '' },
  { name: 'Solidity Advanced: Secure Smart Contracts & DApp Development', issuer: 'Packt', year: '' },
  { name: 'Introduction to Cloud Identity', issuer: 'Google Cloud Security', year: '' },
];

const colors = ['neon-box-cyan', 'neon-box-purple', 'neon-box-yellow'];

export default function CertificationsSection() {
  return (
    <section id="certs" className="py-24 px-4 max-w-6xl mx-auto">
      <SectionHeading label="Credentials" title="Certifications" />
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {certs.map((cert, i) => (
          <div
            key={cert.name}
            className={`glass-panel p-5 group transition-all duration-500 cursor-default hover:scale-[1.04] hover:-translate-y-1 ${colors[i % 3]} animate-fade-up`}
            style={{ animationDelay: `${(i % 6) * 70}ms` }}
          >
            <div className="flex items-start gap-3">
              <Award className="w-6 h-6 shrink-0 text-muted-foreground group-hover:text-primary group-hover:scale-125 group-hover:drop-shadow-[0_0_10px_hsl(var(--primary)/0.7)] transition-all duration-300" />
              <div>
                <h3 className="font-display text-sm font-semibold text-foreground hover-text-pop cursor-default group-hover:text-primary transition-colors">
                  {cert.name}
                </h3>
                <p className="text-xs text-muted-foreground mt-1">
                  {cert.issuer}{cert.year ? ` · ${cert.year}` : ''}
                </p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
