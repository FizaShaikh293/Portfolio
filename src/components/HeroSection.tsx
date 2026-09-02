import { useEffect, useState } from 'react';
import { Download, ArrowDown, MapPin, Sparkles, Shield, Boxes } from 'lucide-react';
import { CV_URL } from './Navbar';

export default function HeroSection() {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const t = setTimeout(() => setVisible(true), 100);
    return () => clearTimeout(t);
  }, []);

  return (
    <section className="relative flex flex-col justify-center px-4 sm:px-8 md:px-14 pt-32 pb-24 overflow-hidden">
      <div
        className={`relative z-10 mx-auto w-full max-w-4xl text-center transition-all duration-700 ${
          visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-6'
        }`}
      >
        <div className="flex justify-center gap-2 mb-8 flex-wrap">
          <span className="stamp">
            <Sparkles className="w-3 h-3" />
            Open to work
          </span>
          <span className="chip">
            <MapPin className="w-3 h-3 ink-blue" />
            Donegal, Ireland
          </span>
        </div>

        <h1 className="animate-ink-in text-6xl md:text-7xl lg:text-8xl leading-[0.95] tracking-tight">
          hi, i'm <span className="italic text-gradient">Fiza</span>
          <span className="text-primary">!</span>
        </h1>

        <p className="mt-6 mx-auto max-w-xl text-lg md:text-xl leading-relaxed text-muted-foreground">
          Security analyst &amp; blockchain builder — I break things carefully,
          then write down exactly how to fix them.
        </p>

        <div className="mt-8 flex justify-center gap-2 flex-wrap">
          {[
            { icon: Shield, label: 'Cybersecurity' },
            { icon: Boxes, label: 'Blockchain' },
            { icon: Sparkles, label: 'AI' },
          ].map(({ icon: Icon, label }) => (
            <span key={label} className="chip hover-wiggle cursor-default">
              <Icon className="w-3.5 h-3.5 text-primary" />
              {label}
            </span>
          ))}
        </div>

        <div className="mt-10 flex justify-center">
          <a
            href={CV_URL}
            download
            className="group inline-flex items-center gap-2 rounded-full bg-primary px-7 py-3.5 text-xs font-mono uppercase tracking-[0.18em] text-primary-foreground shadow-[0_10px_24px_-10px_hsl(var(--primary)/0.6)] transition-all duration-300 hover:scale-[1.04] hover:shadow-[0_16px_32px_-10px_hsl(var(--primary)/0.7)]"
          >
            <Download className="w-3.5 h-3.5 transition-transform duration-300 group-hover:translate-y-0.5" />
            Download CV
          </a>
        </div>

        {/* Mini contents strip */}
        <div className="mt-14 grid grid-cols-2 sm:grid-cols-3 gap-x-8 gap-y-3 text-left max-w-2xl mx-auto">
          {[
            ['01', 'About', '#about'],
            ['02', 'Experience', '#experience'],
            ['03', 'Tech Stack', '#techstack'],
            ['04', 'Certifications', '#certs'],
            ['05', 'Projects', '#projects'],
            ['06', 'Contact', '#socials'],
          ].map(([n, t, href]) => (
            <a
              key={t}
              href={href}
              className="group flex items-baseline gap-3 py-1 text-[11px] font-mono uppercase tracking-[0.14em] text-muted-foreground transition-colors hover:text-foreground"
            >
              <span className="text-primary w-6 shrink-0">{n}</span>
              <span className="leader flex-1 truncate">{t}</span>
              <span className="opacity-0 transition-opacity group-hover:opacity-100 ink-blue">→</span>
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
