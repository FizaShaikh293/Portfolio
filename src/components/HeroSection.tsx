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
      {/* Soft gradient glow backdrop */}
      <div
        className="pointer-events-none absolute inset-0"
        style={{ background: 'var(--gradient-glow)' }}
      />
      <div
        className="pointer-events-none absolute left-1/2 top-1/3 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full blur-[120px] opacity-40"
        style={{ background: 'radial-gradient(circle, hsl(var(--primary)/0.3), transparent 70%)' }}
      />

      <div className="relative z-10 text-center w-full max-w-5xl mx-auto flex flex-col items-center">
        <span
          className={`inline-block mb-6 rounded-full border border-white/10 bg-white/[0.03] px-4 py-1.5 text-[11px] font-medium tracking-[0.2em] uppercase text-muted-foreground backdrop-blur-md transition-all duration-700 ${
            visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
          }`}
        >
          Cybersecurity · Blockchain · AI
        </span>

        <h1
          className={`font-display text-5xl sm:text-6xl md:text-7xl lg:text-8xl font-bold leading-[0.95] mb-6 text-gradient hover-text-pop cursor-default transition-all duration-700 delay-100 break-words ${
            visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-6'
          }`}
        >
          FIZA SHAIKH
        </h1>


        <div
          className={`flex flex-col sm:flex-row items-center justify-center gap-4 mt-8 transition-all duration-700 delay-500 ${
            visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-6'
          }`}
        >
          <a
            href={CV_URL}
            download
            className="group inline-flex items-center gap-2 rounded-full px-7 py-3.5 text-sm font-semibold text-primary-foreground bg-gradient-to-r from-primary to-secondary shadow-[0_10px_30px_-8px_hsl(var(--primary)/0.6)] transition-all duration-300 hover:shadow-[0_14px_40px_-6px_hsl(var(--primary)/0.85)] hover:-translate-y-0.5"
          >
            <Download className="w-4 h-4 transition-transform duration-300 group-hover:translate-y-0.5" />
            Download CV
          </a>
          <a
            href="#projects"
            className="inline-flex items-center gap-2 rounded-full px-7 py-3.5 text-sm font-semibold text-foreground border border-white/10 bg-white/[0.03] backdrop-blur-md transition-all duration-300 hover:bg-white/[0.06] hover:border-white/20 hover:-translate-y-0.5"
          >
            View My Work
          </a>
        </div>
      </div>

      {/* Scroll indicator */}
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
