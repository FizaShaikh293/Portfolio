import profilePhoto from '@/assets/profile-photo.png';
import SectionHeading from './SectionHeading';
import { Shield, Cpu, Boxes } from 'lucide-react';

const focus = [
  { icon: Shield, label: 'Security Expert' },
  { icon: Boxes, label: 'Blockchain & Web3' },
  { icon: Cpu, label: 'AI & Machine Learning' },
];

export default function AboutSection() {
  return (
    <section id="about" className="py-24 px-4 max-w-5xl mx-auto">
      <SectionHeading label="Who I Am" title="About Me" />

      <div className="paper p-8 md:p-12">
        <div className="flex flex-col md:flex-row gap-10 items-center md:items-start">
          <div className="shrink-0">
            <div className="relative w-44 h-52 md:w-52 md:h-64 border border-border bg-background p-2 shadow-[var(--shadow-paper)] rotate-[-1.5deg]">
              <img
                src={profilePhoto}
                alt="Fiza Shaikh"
                loading="lazy"
                className="w-full h-full object-cover grayscale contrast-[1.05] transition-all duration-700 hover:grayscale-0"
              />
            </div>
            <p className="mt-3 text-center text-[10px] font-mono uppercase tracking-[0.2em] text-muted-foreground">
              Donegal, Ireland
            </p>
          </div>

          <div className="flex-1">
            <p className="text-base md:text-lg leading-relaxed text-foreground/90 mb-6">
              Hi! I'm Fiza, a girl who couldn't pick between cybersecurity or blockchain so I did both.
              I started with a Bachelors in IT, then worked for more than a year as an IT Security Analyst,
              and eventually landed in Ireland, where I completed my MSc in Blockchain Technologies at ATU Donegal.
              I love diving deep into smart contract security, network defense, and decentralized systems.
              I'm also deeply interested in AI and Machine Learning, exploring how intelligent systems can
              enhance security and solve complex problems.
              <br /><br />
              Currently based in Donegal, open to roles across Ireland.
            </p>

            <div className="flex flex-wrap gap-2">
              {focus.map(({ icon: Icon, label }) => (
                <div
                  key={label}
                  className="inline-flex items-center gap-2 border border-border px-3 py-1.5 text-[11px] font-mono uppercase tracking-[0.12em] text-muted-foreground transition-colors duration-300 hover:text-foreground hover:border-foreground/30 cursor-default"
                >
                  <Icon className="w-3.5 h-3.5" />
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
