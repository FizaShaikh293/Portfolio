import profilePhoto from '@/assets/profile-photo.png';
import SectionHeading from './SectionHeading';
import { Shield, Cpu, Boxes } from 'lucide-react';

const focus = [
  { icon: Shield, label: 'Security Research' },
  { icon: Boxes, label: 'Blockchain & Web3' },
  { icon: Cpu, label: 'AI & Machine Learning' },
];

export default function AboutSection() {
  return (
    <section id="about" className="py-24 px-4 max-w-5xl mx-auto">
      <SectionHeading label="Who I Am" title="About Me" />

      <div className="glass-panel p-8 md:p-12">
        <div className="flex flex-col md:flex-row gap-10 items-center md:items-start">
          <div className="shrink-0">
            <div className="relative group">
              <div className="absolute -inset-2 rounded-3xl bg-gradient-to-br from-primary/40 to-secondary/40 blur-xl opacity-50 group-hover:opacity-80 transition-opacity duration-500" />
              <div className="relative w-44 h-44 md:w-52 md:h-52 rounded-3xl overflow-hidden border border-white/10 shadow-[0_20px_60px_-20px_hsl(240_30%_2%/0.9)]">
                <img
                  src={profilePhoto}
                  alt="Fiza Shaikh"
                  loading="lazy"
                  className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-105"
                />
              </div>
            </div>
          </div>

          <div className="flex-1">
            <p className="text-foreground/90 text-base md:text-lg leading-relaxed mb-6">
              Hi! I'm a cybersecurity &amp; blockchain enthusiast with a passion for breaking things (ethically)
              and building them back stronger. I love diving deep into smart contract security, network defense,
              and decentralized systems. I'm also deeply interested in AI and Machine Learning, exploring how
              intelligent systems can enhance security and solve complex problems.
            </p>

            <div className="flex flex-wrap gap-3">
              {focus.map(({ icon: Icon, label }) => (
                <div
                  key={label}
                  className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.03] px-4 py-2 text-sm text-foreground/80 transition-all duration-300 hover:border-primary/40 hover:text-foreground"
                >
                  <Icon className="w-4 h-4 text-primary" />
                  {label}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
