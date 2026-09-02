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
        scrolled ? 'bg-background/85 backdrop-blur-sm border-b border-border' : 'bg-transparent'
      }`}
    >
      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-4 flex items-center justify-between">
        <a href="#hero" className="font-display italic text-xl tracking-tight text-foreground hover:text-primary transition-colors">
          fiza shaikh <span className="text-primary not-italic">✿</span>
        </a>

        <div className="hidden md:flex items-center gap-7">
          {links.map(({ label, href }) => (
            <a
              key={label}
              href={href}
              className="text-[11px] font-mono uppercase tracking-[0.14em] text-muted-foreground hover:text-foreground transition-colors duration-300 ink-underline"
            >
              {label}
            </a>
          ))}
          <a
            href={CV_URL}
            download
            className="group inline-flex items-center gap-2 rounded-full bg-primary px-4 py-2 text-[11px] font-mono uppercase tracking-[0.14em] text-primary-foreground transition-all duration-300 hover:scale-105"
          >
            <Download className="w-3.5 h-3.5" />
            CV
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
        <div className="md:hidden paper mx-4 mb-2 p-5 flex flex-col gap-4 animate-fade-in">
          {links.map(({ label, href }) => (
            <a
              key={label}
              href={href}
              onClick={() => setMenuOpen(false)}
              className="text-xs font-mono uppercase tracking-[0.14em] text-muted-foreground hover:text-foreground transition-colors"
            >
              {label}
            </a>
          ))}
          <a
            href={CV_URL}
            download
            onClick={() => setMenuOpen(false)}
            className="inline-flex items-center justify-center gap-2 border border-foreground/25 px-4 py-2.5 text-xs font-mono uppercase tracking-[0.14em]"
          >
            <Download className="w-4 h-4" />
            Download CV
          </a>
        </div>
      )}
    </nav>
  );
}
