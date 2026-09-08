import { useEffect, useState } from 'react';
import { Download, MapPin, Shield, Boxes, Sparkles } from 'lucide-react';
import { CV_URL } from './Navbar';

const contents: [string, string, string][] = [
  ['01', 'About', '#about'],
  ['02', 'Experience', '#experience'],
  ['03', 'Tech Stack', '#techstack'],
  ['04', 'Certifications', '#certs'],
  ['05', 'Projects', '#projects'],
  ['06', 'Contact', '#socials'],
];

export default function HeroSection() {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const t = setTimeout(() => setVisible(true), 80);
    return () => clearTimeout(t);
  }, []);

  return (
    <section className="relative px-4 sm:px-8 md:px-12 pt-32 pb-20 overflow-hidden">
      <div
        className={`relative z-10 mx-auto w-full max-w-6xl transition-all duration-700 ${
          visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-6'
        }`}
      >
        {/* Masthead line */}
        <div className="flex items-center justify-between gap-4 pb-3 border-b-[3px] border-foreground">
          <span className="kicker">Portfolio — Issue No. 01</span>
          <span className="kicker hidden sm:inline text-muted-foreground">
            Donegal, Ireland
          </span>
        </div>

        {/* Big nameplate */}
        <h1 className="mt-6 text-[15vw] md:text-[8.5rem] leading-[0.82] uppercase">
          Fiza
          <span className="block text-primary">Shaikh</span>
        </h1>

        <div className="mt-6 grid gap-8 md:grid-cols-12 border-t border-border pt-6">
          {/* Standfirst */}
          <div className="md:col-span-7">
            <p className="text-xl md:text-2xl leading-snug font-display text-foreground">
              Security analyst &amp; blockchain builder — I break things carefully,
              then write down exactly how to fix them.
            </p>

            <div className="mt-6 flex flex-wrap gap-2">
              <span className="inline-flex items-center gap-2 block-rust px-3 py-1.5 text-[10px] font-mono uppercase tracking-[0.2em]">
                <Sparkles className="w-3 h-3" />
                Open to work
              </span>
              {[
                { icon: Shield, label: 'Cybersecurity' },
                { icon: Boxes, label: 'Blockchain' },
                { icon: MapPin, label: 'Donegal, IE' },
              ].map(({ icon: Icon, label }) => (
                <span
                  key={label}
                  className="inline-flex items-center gap-2 border border-foreground/25 px-3 py-1.5 text-[10px] font-mono uppercase tracking-[0.2em] text-muted-foreground transition-colors hover:border-primary hover:text-primary"
                >
                  <Icon className="w-3 h-3" />
                  {label}
                </span>
              ))}
            </div>

            <a
              href={CV_URL}
              download
              className="group mt-8 inline-flex items-center gap-3 border-2 border-foreground px-6 py-3 text-[11px] font-mono uppercase tracking-[0.22em] text-foreground transition-colors duration-300 hover:bg-foreground hover:text-background"
            >
              <Download className="w-3.5 h-3.5 transition-transform duration-300 group-hover:translate-y-0.5" />
              Download CV
            </a>
          </div>

          {/* Contents column */}
          <div className="md:col-span-5 md:border-l md:border-border md:pl-8">
            <p className="kicker mb-4">In this issue</p>
            <div className="flex flex-col">
              {contents.map(([n, t, href]) => (
                <a
                  key={t}
                  href={href}
                  className="group flex items-baseline gap-3 border-b border-border py-2.5 text-[11px] font-mono uppercase tracking-[0.16em] text-muted-foreground transition-colors hover:text-primary"
                >
                  <span className="text-primary w-6 shrink-0">{n}</span>
                  <span className="flex-1 truncate">{t}</span>
                  <span className="opacity-0 transition-opacity group-hover:opacity-100">→</span>
                </a>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
