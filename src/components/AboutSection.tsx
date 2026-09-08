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

      <div className="grid gap-10 md:grid-cols-12 items-start">
        <div className="md:col-span-4">
          <div className="relative border border-foreground/15 bg-card p-2 shadow-[var(--shadow-paper)]">
            <img
              src={profilePhoto}
              alt="Fiza Shaikh"
              loading="lazy"
              className="w-full h-64 object-cover transition-transform duration-700 hover:scale-[1.02]"
            />
          </div>
          <p className="mt-3 text-[10px] font-mono uppercase tracking-[0.24em] text-muted-foreground">
            Donegal, Ireland — MSc Blockchain, ATU '25
          </p>
        </div>

        <div className="md:col-span-8 md:border-l md:border-border md:pl-8">
          <p className="text-base md:text-lg leading-relaxed text-foreground/90 mb-5 md:columns-2 md:gap-8">
            Hi! I'm Fiza, a girl who couldn't pick between cybersecurity or blockchain so I did both.
            I started with a Bachelors in IT, then worked for more than a year as an IT Security Analyst,
            and eventually landed in Ireland, where I completed my MSc in Blockchain Technologies at ATU Donegal.
            I love diving deep into smart contract security, network defense, and decentralized systems.
            I'm also deeply interested in AI and Machine Learning, exploring how intelligent systems can
            enhance security and solve complex problems.
          </p>
          <p className="mb-6 border-l-4 border-primary pl-4 text-lg font-display leading-snug text-foreground">
            Currently based in Donegal, open to roles across Ireland.
          </p>
          <div className="flex flex-wrap gap-2">
            {focus.map(({ icon: Icon, label }) => (
              <div
                key={label}
                className="inline-flex items-center gap-2 border border-foreground/25 px-3 py-1.5 text-[10px] font-mono uppercase tracking-[0.2em] text-muted-foreground transition-colors hover:border-primary hover:text-primary cursor-default"
              >
                <Icon className="w-3.5 h-3.5 text-primary" />
                {label}
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>

  );
}
