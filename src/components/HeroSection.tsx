import { useEffect, useState } from 'react';
import { Download, MapPin, Shield, Boxes, Sparkles } from 'lucide-react';
import { CV_URL } from './Navbar';

export default function HeroSection() {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const t = setTimeout(() => setVisible(true), 80);
    return () => clearTimeout(t);
  }, []);

  return (
    <section className="relative px-4 sm:px-8 md:px-12 pt-28 md:pt-36 pb-20 md:pb-28 overflow-hidden">
      <div
        className={`relative z-10 mx-auto w-full max-w-6xl transition-all duration-700 ${
          visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-6'
        }`}
      >
        {/* Big nameplate */}
        <h1 className="text-[20vw] sm:text-[17vw] md:text-[10rem] lg:text-[12rem] leading-[0.72] uppercase">
          <span className="block">Fiza</span>
          <span className="block text-primary md:ml-[14%]">Shaikh</span>
        </h1>

        <p className="mt-8 border-t-[6px] border-foreground pt-4 text-sm sm:text-base font-mono uppercase tracking-[0.16em] text-foreground">
          Cybersecurity Analyst · SOC Analyst · SIEM &amp; Incident Response
        </p>

        <div className="mt-7 border-t-[3px] border-foreground pt-7">
            <div className="flex flex-wrap gap-2">
              <span className="inline-flex items-center gap-2 block-rust px-3 py-1.5 text-[10px] font-mono uppercase tracking-[0.2em]">
                <Sparkles className="w-3 h-3" />
                Open to work
              </span>
              {[
                { icon: Shield, label: 'Cybersecurity' },
                { icon: Boxes, label: 'Blockchain' },
                { icon: MapPin, label: 'Ireland' },
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
              className="group mt-8 mx-auto flex w-fit items-center gap-3 block-rust border-2 border-primary px-8 py-4 text-[11px] font-mono uppercase tracking-[0.22em] transition-colors duration-300 hover:bg-foreground hover:border-foreground"
            >
              <Download className="w-3.5 h-3.5 transition-transform duration-300 group-hover:translate-y-0.5" />
              Download CV
            </a>
        </div>
      </div>
    </section>
  );
}
