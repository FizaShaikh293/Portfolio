import { Linkedin, Mail, Github } from 'lucide-react';
import SectionHeading from './SectionHeading';

const socials = [
  { icon: Linkedin, label: 'LinkedIn', href: 'https://www.linkedin.com/in/fizashaikh293/', color: 'hover:neon-box-cyan hover:text-primary' },
  { icon: Mail, label: 'Email', href: 'mailto:fiza.sk293@gmail.com', color: 'hover:neon-box-yellow hover:text-accent' },
  { icon: Github, label: 'GitHub', href: 'https://github.com/FizaShaikh293', color: 'hover:neon-box-cyan hover:text-primary' },
  { icon: null, label: 'Credly', href: 'https://www.credly.com/users/fizashaikh293', color: 'hover:neon-box-yellow hover:text-accent', isCredly: true },
];

export default function SocialsSection() {
  return (
    <section id="socials" className="py-24 px-4 max-w-3xl mx-auto">
      <SectionHeading label="Get In Touch" title="Connect" />
      <div className="flex flex-wrap justify-center gap-4 mb-16">
        {socials.map(({ icon: Icon, label, href, color, isCredly }) => (
          <a
            key={label}
            href={href}
            target="_blank"
            rel="noopener noreferrer"
            className={`glass-panel p-6 flex flex-col items-center gap-3 transition-all duration-300 hover:scale-110 group ${color}`}
          >
            {isCredly ? (
              <svg className="w-8 h-8 text-muted-foreground group-hover:text-accent group-hover:animate-pulse-glow transition-colors" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <rect x="2" y="2" width="20" height="20" rx="4" stroke="currentColor" strokeWidth="1.5" />
                <path d="M7 13.5C7 13.5 8.5 8 12 8C15.5 8 17 13.5 17 13.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                <circle cx="12" cy="14" r="2.5" stroke="currentColor" strokeWidth="1.5" />
                <path d="M9.5 14L7 16.5M14.5 14L17 16.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
              </svg>
            ) : Icon ? (
              <Icon className="w-8 h-8 text-muted-foreground group-hover:animate-pulse-glow transition-colors" />
            ) : null}
            <span className="text-xs font-mono text-muted-foreground group-hover:text-foreground hover-text-glow transition-colors">{label}</span>
          </a>
        ))}
      </div>

      <div className="text-center">
        <p className="font-display text-3xl md:text-5xl font-bold text-gradient hover-text-pop cursor-default mb-3">
          Thank You!
        </p>
        <p className="text-sm text-muted-foreground">
          Thanks for scrolling through my world ✨
        </p>
      </div>
    </section>
  );
}
