import { useState, useEffect } from 'react';
import { Menu, X } from 'lucide-react';

const links = [
  { label: 'About', href: '#about' },
  { label: 'Education', href: '#education' },
  { label: 'Tech Stack', href: '#techstack' },
  { label: 'Certs', href: '#certs' },
  { label: 'Projects', href: '#projects' },
  { label: 'Connect', href: '#socials' },
];

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 50);
    window.addEventListener('scroll', onScroll);
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  return (
    <nav className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${scrolled ? 'bg-background/80 backdrop-blur-md border-b border-primary/10' : ''}`}>
      <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
        <span className="font-display text-sm font-bold text-accent neon-glow-yellow">
          My Portfolio
        </span>
        <div className="hidden md:flex items-center gap-6">
          {links.map(({ label, href }) => (
            <a
              key={label}
              href={href}
              className="text-xs font-mono text-muted-foreground hover:text-primary transition-colors duration-200"
            >
              {label}
            </a>
          ))}
        </div>
        <button className="md:hidden text-muted-foreground hover:text-primary transition-colors" onClick={() => setMenuOpen(!menuOpen)}>
          {menuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>
      </div>
      {menuOpen && (
        <div className="md:hidden glass-panel mx-4 mb-2 p-4 flex flex-col gap-3 animate-fade-in">
          {links.map(({ label, href }) => (
            <a key={label} href={href} onClick={() => setMenuOpen(false)} className="text-sm font-mono text-muted-foreground hover:text-primary transition-colors">
              {label}
            </a>
          ))}
        </div>
      )}
    </nav>
  );
}
