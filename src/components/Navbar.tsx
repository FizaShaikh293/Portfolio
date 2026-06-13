import { useState, useEffect } from 'react';
import { Menu, X, Download } from 'lucide-react';

const links = [
  { label: 'About', href: '#about' },
  { label: 'Experience', href: '#experience' },
  { label: 'Tech Stack', href: '#techstack' },
  { label: 'Certs', href: '#certs' },
  { label: 'Projects', href: '#projects' },
  { label: 'Connect', href: '#socials' },
];

export const CV_URL = '/Fiza-Shaikh-CV.pdf';

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 50);
    window.addEventListener('scroll', onScroll);
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  return (
    <nav
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-500 ${
        scrolled
          ? 'bg-background/70 backdrop-blur-xl border-b border-white/[0.06] shadow-[0_4px_30px_-12px_hsl(240_30%_2%/0.8)]'
          : 'bg-transparent'
      }`}
    >
      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-4 flex items-center justify-between">
        <a href="#hero" className="font-display text-base font-semibold tracking-tight text-gradient hover-text-pop">
          Fiza Shaikh
        </a>

        <div className="hidden md:flex items-center gap-7">
          {links.map(({ label, href }) => (
          <a
              key={label}
              href={href}
              className="relative text-[13px] font-medium text-muted-foreground hover:text-foreground hover-text-glow transition-all duration-300 after:absolute after:-bottom-1.5 after:left-0 after:h-px after:w-0 after:bg-primary after:transition-all after:duration-300 hover:after:w-full"
            >
              {label}
            </a>
          ))}
          <a
            href={CV_URL}
            download
            className="group inline-flex items-center gap-2 rounded-full px-4 py-2 text-[13px] font-medium text-primary-foreground bg-gradient-to-r from-primary to-secondary shadow-[0_8px_24px_-8px_hsl(var(--primary)/0.6)] transition-all duration-300 hover:shadow-[0_10px_32px_-6px_hsl(var(--primary)/0.8)] hover:-translate-y-0.5"
          >
            <Download className="w-3.5 h-3.5 transition-transform duration-300 group-hover:translate-y-0.5" />
            Download CV
          </a>
        </div>

        <button
          className="md:hidden text-muted-foreground hover:text-foreground transition-colors"
          onClick={() => setMenuOpen(!menuOpen)}
          aria-label="Toggle menu"
        >
          {menuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>
      </div>

      {menuOpen && (
        <div className="md:hidden glass-panel mx-4 mb-2 p-5 flex flex-col gap-4 animate-fade-in">
          {links.map(({ label, href }) => (
            <a
              key={label}
              href={href}
              onClick={() => setMenuOpen(false)}
              className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors"
            >
              {label}
            </a>
          ))}
          <a
            href={CV_URL}
            download
            onClick={() => setMenuOpen(false)}
            className="inline-flex items-center justify-center gap-2 rounded-full px-4 py-2.5 text-sm font-medium text-primary-foreground bg-gradient-to-r from-primary to-secondary"
          >
            <Download className="w-4 h-4" />
            Download CV
          </a>
        </div>
      )}
    </nav>
  );
}
