import { Award } from 'lucide-react';

const certs = [
  { name: 'CompTIA Security+', issuer: 'CompTIA', color: 'neon-box-cyan' },
  { name: 'Certified Ethical Hacker', issuer: 'EC-Council', color: 'neon-box-purple' },
  { name: 'AWS Cloud Practitioner', issuer: 'Amazon', color: 'neon-box-yellow' },
  { name: 'Cisco CyberOps Associate', issuer: 'Cisco', color: 'neon-box-cyan' },
  { name: 'Google Cybersecurity Certificate', issuer: 'Google', color: 'neon-box-purple' },
  { name: 'Blockchain Fundamentals', issuer: 'UC Berkeley', color: 'neon-box-yellow' },
  { name: 'Certified Blockchain Developer', issuer: 'Blockchain Council', color: 'neon-box-cyan' },
  { name: 'OSCP Foundation', issuer: 'Offensive Security', color: 'neon-box-purple' },
  { name: 'Smart Contract Security', issuer: 'Consensys', color: 'neon-box-yellow' },
];

export default function CertificationsSection() {
  return (
    <section id="certs" className="py-24 px-4 max-w-6xl mx-auto">
      <h2 className="font-display text-2xl md:text-3xl font-bold text-secondary neon-glow-purple mb-10 text-center">
        {'>'} Certifications
      </h2>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {certs.map((cert) => (
          <div
            key={cert.name}
            className={`glass-panel p-5 group hover:scale-105 transition-all duration-300 cursor-default hover:${cert.color}`}
          >
            <div className="flex items-start gap-3">
              <Award className="w-6 h-6 text-accent shrink-0 group-hover:animate-pulse-glow" />
              <div>
                <h3 className="font-display text-sm font-semibold text-foreground group-hover:text-accent transition-colors">
                  {cert.name}
                </h3>
                <p className="text-xs text-muted-foreground mt-1 font-mono">{cert.issuer}</p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
