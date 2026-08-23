import { useEffect, useState } from 'react';
import { Download, ArrowDown } from 'lucide-react';
import { CV_URL } from './Navbar';

export default function HeroSection() {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const t = setTimeout(() => setVisible(true), 100);
    return () => clearTimeout(t);
  }, []);

  return (
    <section className="relative min-h-screen flex flex-col items-center justify-center px-4 overflow-hidden">
      <div className="relative z-10 w-full max-w-3xl mx-auto">
        <div
          className={`paper relative px-8 py-14 md:px-16 md:py-20 text-center transition-all duration-700 ${
            visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-6'
          }`}
        >
          <span className="absolute left-8 top-6 text-[10px] font-mono uppercase tracking-[0.3em] text-muted-foreground">
            Portfolio · 2026
          </span>

          <p className="mb-5 text-[10px] font-mono uppercase tracking-[0.32em] text-muted-foreground">
            Cybersecurity · Blockchain · AI
          </p>

          <h1 className="text-6xl md:text-7xl lg:text-8xl leading-[0.95] mb-6">
            Fiza Shaikh
          </h1>

          <div className="mx-auto mb-7 h-px w-24 bg-foreground/25" />

          <p className="mx-auto max-w-md text-sm md:text-base leading-relaxed text-muted-foreground">
            Security analyst and blockchain builder. I break things carefully,
            then write down exactly how to fix them.
          </p>

          <a
            href={CV_URL}
            download
            className="group mt-9 inline-flex items-center gap-2 border border-foreground/25 px-6 py-3 text-xs font-mono uppercase tracking-[0.18em] text-foreground transition-all duration-300 hover:bg-foreground hover:text-background"
          >
            <Download className="w-3.5 h-3.5 transition-transform duration-300 group-hover:translate-y-0.5" />
            Download CV
          </a>
        </div>
      </div>

      <a
        href="#about"
        className={`absolute bottom-10 left-1/2 -translate-x-1/2 text-muted-foreground transition-all duration-1000 delay-700 ${
          visible ? 'opacity-100' : 'opacity-0'
        }`}
        aria-label="Scroll down"
      >
        <ArrowDown className="w-5 h-5 animate-bounce" />
      </a>
    </section>
  );
}
