import { Instagram, Linkedin, Mail, Github } from 'lucide-react';
import credlyLogo from '@/assets/credly-logo.png';

const socials = [
  { icon: Instagram, label: 'Instagram', href: 'https://instagram.com', color: 'hover:neon-box-purple hover:text-secondary' },
  { icon: Linkedin, label: 'LinkedIn', href: 'https://www.linkedin.com/in/fizashaikh293', color: 'hover:neon-box-cyan hover:text-primary' },
  { icon: Mail, label: 'Email', href: 'mailto:fiza.sk293@gmail.com', color: 'hover:neon-box-yellow hover:text-accent' },
  { icon: Github, label: 'GitHub', href: 'https://github.com/FizaShaikh293/Portfolio', color: 'hover:neon-box-cyan hover:text-primary' },
  { icon: null, label: 'Credly', href: 'https://www.credly.com/users/fizashaikh293', color: 'hover:neon-box-yellow hover:text-accent', isCredly: true },
];

export default function SocialsSection() {
  return (
    <section id="socials" className="py-24 px-4 max-w-3xl mx-auto">
      <h2 className="font-display text-2xl md:text-3xl font-bold text-primary neon-glow-cyan mb-10 text-center">
        {'>'} Connect
      </h2>
      <div className="flex flex-wrap justify-center gap-4">
        {socials.map(({ icon: Icon, label, href, color, isCredly }) => (
          <a
            key={label}
            href={href}
            target="_blank"
            rel="noopener noreferrer"
            className={`glass-panel p-6 flex flex-col items-center gap-3 transition-all duration-300 hover:scale-110 group ${color}`}
          >
            {isCredly ? (
              <img
                src={credlyLogo}
                alt="Credly"
                className="w-8 h-8 rounded transition-transform group-hover:scale-110 group-hover:animate-pulse-glow"
                style={{ filter: 'grayscale(0.6) brightness(0.8)', transition: 'filter 0.3s' }}
                onMouseEnter={(e) => { (e.target as HTMLImageElement).style.filter = 'grayscale(0) brightness(1) drop-shadow(0 0 8px hsl(51 100% 50% / 0.6))'; }}
                onMouseLeave={(e) => { (e.target as HTMLImageElement).style.filter = 'grayscale(0.6) brightness(0.8)'; }}
              />
            ) : Icon ? (
              <Icon className="w-8 h-8 text-muted-foreground group-hover:animate-pulse-glow transition-colors" />
            ) : null}
            <span className="text-xs font-mono text-muted-foreground group-hover:text-foreground transition-colors">{label}</span>
          </a>
        ))}
      </div>
    </section>
  );
}
