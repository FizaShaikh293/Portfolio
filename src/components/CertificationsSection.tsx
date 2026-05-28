import { Award } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import SectionHeading from './SectionHeading';

const certs = [
  { name: 'Junior Cybersecurity Analyst Career Path', issuer: 'Cisco', year: '2026' },
  { name: 'Introduction to Digital Forensics', issuer: 'Security Blue Team', year: '2026' },
  { name: 'Encryption and Cryptography Essentials', issuer: 'IBM', year: '2025' },
  { name: 'Introduction to Cybersecurity Essentials', issuer: 'IBM', year: '2025' },
  { name: 'Cybersecurity Professional', issuer: 'Google', year: '2025' },
  { name: 'Solidity Advanced: Secure Smart Contracts & DApp Development', issuer: 'Packt', year: '' },
  { name: 'Information Security Fundamentals', issuer: 'EC-Council', year: '' },
  { name: 'Decentralized Finance (DeFi) Infrastructure', issuer: 'Duke University', year: '' },
  { name: 'Web3 and Blockchain Fundamentals', issuer: 'INSEAD', year: '' },
  { name: 'Blockchain Security', issuer: 'Infosec', year: '' },
  { name: 'Blockchain Basics', issuer: 'Coursera', year: '' },
  { name: 'Introduction to Cloud Identity', issuer: 'Google Cloud Security', year: '' },
  { name: 'Ethical Hacking Essentials', issuer: 'EC-Council', year: '2023' },
];

const colors = ['neon-box-cyan', 'neon-box-purple', 'neon-box-yellow'];

export default function CertificationsSection() {
  const [visibleItems, setVisibleItems] = useState<number[]>([]);
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const itemRefs = useRef<(HTMLDivElement | null)[]>([]);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const idx = Number(entry.target.getAttribute('data-cert-index'));
            setVisibleItems((prev) => prev.includes(idx) ? prev : [...prev, idx]);
          }
        });
      },
      { threshold: 0.3 }
    );
    itemRefs.current.forEach((ref) => { if (ref) observer.observe(ref); });
    return () => observer.disconnect();
  }, []);

  return (
    <section id="certs" className="py-24 px-4 max-w-6xl mx-auto">
      <SectionHeading label="Credentials" title="Certifications" />
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {certs.map((cert, i) => {
          const isVisible = visibleItems.includes(i);
          const colorClass = colors[i % 3];
          const isHovered = hoveredIdx === i;

          return (
            <div
              key={cert.name}
              ref={(el) => { itemRefs.current[i] = el; }}
              data-cert-index={i}
              onMouseEnter={() => setHoveredIdx(i)}
              onMouseLeave={() => setHoveredIdx(null)}
              className={`glass-panel p-5 group transition-all duration-500 cursor-default ${colorClass} ${
                isVisible ? 'opacity-100 translate-y-0 rotate-0' : 'opacity-0 translate-y-6 rotate-1'
              }`}
              style={{
                transitionDelay: isVisible ? `${(i % 6) * 80}ms` : '0ms',
                transform: isHovered
                  ? `perspective(600px) rotateY(${(i % 2 === 0 ? 3 : -3)}deg) rotateX(2deg) scale(1.05)`
                  : undefined,
              }}
            >
              <div className="flex items-start gap-3">
                <Award className={`w-6 h-6 shrink-0 transition-all duration-300 ${
                  isHovered ? 'text-primary scale-125 drop-shadow-[0_0_10px_hsl(var(--primary)/0.7)]' : 'text-muted-foreground'
                }`} />
                <div>
                  <h3 className="font-display text-sm font-semibold text-foreground group-hover:text-primary transition-colors">
                    {cert.name}
                  </h3>
                  <p className="text-xs text-muted-foreground mt-1">
                    {cert.issuer}{cert.year ? ` · ${cert.year}` : ''}
                  </p>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
