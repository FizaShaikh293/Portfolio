import { useEffect, useState } from 'react';
import { Download, ArrowDown, MapPin } from 'lucide-react';
import { CV_URL } from './Navbar';

export default function HeroSection() {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const t = setTimeout(() => setVisible(true), 100);
    return () => clearTimeout(t);
  }, []);

  return (
    <section className="relative flex flex-col justify-center px-4 sm:px-8 md:px-14 pt-32 pb-24 overflow-hidden">
      {/* Masthead rule */}
      <div className="mx-auto w-full max-w-4xl">
        <div className="flex items-end justify-between border-b-2 border-foreground/80 pb-2 text-[9px] font-mono uppercase tracking-[0.24em] text-muted-foreground">
          <span>Portfolio · Volume One</span>
          <span className="hidden sm:inline ink-rust">Est. Mumbai — Now Donegal</span>
          <span>MMXXVI</span>
        </div>
        <div className="mt-[3px] h-px w-full bg-foreground/40" />
      </div>

      <div
        className={`relative z-10 mx-auto w-full max-w-4xl transition-all duration-700 ${
          visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-6'
        }`}
      >
        <div className="grid grid-cols-1 md:grid-cols-[1fr_auto] gap-8 items-end pt-10">
          <div>
            <p className="mb-4 text-[10px] font-mono uppercase tracking-[0.32em] text-muted-foreground">
              Cybersecurity · Blockchain · AI
            </p>

            <h1 className="animate-ink-in text-6xl md:text-7xl lg:text-[6.5rem] leading-[0.88] tracking-tight">
              Fiza
              <br />
              <span className="italic ink-rust">Shaikh</span>
            </h1>

            <div className="mt-6 flex items-center gap-2 text-[10px] font-mono uppercase tracking-[0.2em] text-muted-foreground">
              <MapPin className="w-3 h-3 ink-teal" />
              Donegal, Ireland
            </div>
          </div>

          <div className="md:text-right md:pb-3">
            <span className="stamp">Open to work</span>
          </div>
        </div>

        {/* Standfirst, set like a print deck */}
        <div className="mt-10 grid grid-cols-1 md:grid-cols-[1.6fr_1fr] gap-8 md:gap-12 border-t border-border pt-8">
          <p className="text-lg md:text-xl leading-relaxed text-foreground/90 font-display">
            Security analyst and blockchain builder. I break things carefully,
            then write down exactly how to fix them — SOC triage by day,
            smart-contract teardowns by night.
          </p>

          <div className="flex flex-col gap-5">
            <p className="marginalia">
              “read it like a notebook — chapters, margins and all.”
            </p>

            <a
              href={CV_URL}
              download
              className="group inline-flex w-fit items-center gap-2 bg-foreground px-6 py-3 text-xs font-mono uppercase tracking-[0.18em] text-background transition-all duration-300 hover:bg-primary"
            >
              <Download className="w-3.5 h-3.5 transition-transform duration-300 group-hover:translate-y-0.5" />
              Download CV
            </a>
          </div>
        </div>

        {/* Mini contents strip */}
        <div className="mt-12 grid grid-cols-2 sm:grid-cols-3 gap-x-8 gap-y-2 border-t border-border pt-5">
          {[
            ['I', 'About', '#about'],
            ['II', 'Experience', '#experience'],
            ['III', 'Tech Stack', '#techstack'],
            ['IV', 'Certifications', '#certs'],
            ['V', 'Projects', '#projects'],
            ['VI', 'Contact', '#socials'],
          ].map(([n, t, href]) => (
            <a
              key={t}
              href={href}
              className="group flex items-baseline gap-3 py-1 text-xs font-mono uppercase tracking-[0.14em] text-muted-foreground transition-colors hover:text-foreground"
            >
              <span className="ink-rust w-5 shrink-0">{n}</span>
              <span className="leader flex-1 truncate">{t}</span>
              <span className="opacity-0 transition-opacity group-hover:opacity-100 ink-teal">→</span>
            </a>
          ))}
        </div>
      </div>

      <a
        href="#about"
        className={`mt-14 self-center text-muted-foreground transition-all duration-1000 delay-700 ${
          visible ? 'opacity-100' : 'opacity-0'
        }`}
        aria-label="Scroll down"
      >
        <ArrowDown className="w-5 h-5 animate-bounce" />
      </a>
    </section>
  );
}
