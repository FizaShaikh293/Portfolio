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
    <section id="about" className="py-20 px-4 sm:px-8 md:px-16 max-w-5xl mx-auto">
      <SectionHeading label="Who I Am" title="About Me" />

      <div className="flex flex-col md:flex-row gap-10 items-start">
        <div className="shrink-0 mx-auto md:mx-0">
          <div className="relative w-44 h-52 md:w-52 md:h-64 border border-border bg-card p-2 shadow-[var(--shadow-paper)] rotate-[-1.5deg] transition-transform duration-500 hover:rotate-0 tape">
            <img
              src={profilePhoto}
              alt="Fiza Shaikh"
              loading="lazy"
              className="w-full h-full object-cover transition-transform duration-700 hover:scale-[1.02]"
            />
          </div>
          <p className="mt-4 text-center text-[10px] font-mono uppercase tracking-[0.24em] text-muted-foreground">
            Donegal, Ireland
          </p>
          <p className="marginalia mt-3 text-center">msc blockchain, ATU '25</p>
        </div>

        <div className="flex-1">
          <p className="drop-cap text-base md:text-lg leading-relaxed text-foreground/90 mb-5">
            Hi! I'm Fiza, a girl who couldn't pick between cybersecurity or blockchain so I did both.
            I started with a Bachelors in IT, then worked for more than a year as an IT Security Analyst,
            and eventually landed in Ireland, where I completed my MSc in Blockchain Technologies at ATU Donegal.
            I love diving deep into smart contract security, network defense, and decentralized systems.
            I'm also deeply interested in AI and Machine Learning, exploring how intelligent systems can
            enhance security and solve complex problems.
          </p>
          <p className="mb-6 border-l-2 border-primary/60 pl-4 text-sm italic text-muted-foreground font-display">
            Currently based in Donegal, open to roles across Ireland.
          </p>
          <div className="flex flex-wrap gap-2">
            {focus.map(({ icon: Icon, label }) => (
              <div
                key={label}
                className="inline-flex items-center gap-2 border border-border bg-card px-3 py-1.5 text-[11px] font-mono uppercase tracking-[0.12em] text-muted-foreground transition-all duration-300 hover:text-primary hover:border-primary/50 cursor-default"
              >
                <Icon className="w-3.5 h-3.5 ink-teal" />
                {label}
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>

  );
}
