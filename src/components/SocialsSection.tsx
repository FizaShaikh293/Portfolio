import { Instagram, Linkedin, Mail, Github } from 'lucide-react';

const socials = [
  { icon: Instagram, label: 'Instagram', href: '#', color: 'hover:neon-box-purple hover:text-secondary' },
  { icon: Linkedin, label: 'LinkedIn', href: '#', color: 'hover:neon-box-cyan hover:text-primary' },
  { icon: Mail, label: 'Email', href: 'mailto:hello@example.com', color: 'hover:neon-box-yellow hover:text-accent' },
  { icon: Github, label: 'GitHub', href: '#', color: 'hover:neon-box-cyan hover:text-primary' },
];

export default function SocialsSection() {
  return (
    <section id="socials" className="py-24 px-4 max-w-3xl mx-auto">
      <h2 className="font-display text-2xl md:text-3xl font-bold text-primary neon-glow-cyan mb-10 text-center">
        {'>'} Connect
      </h2>
      <div className="flex flex-wrap justify-center gap-4">
        {socials.map(({ icon: Icon, label, href, color }) => (
          <a
            key={label}
            href={href}
            target="_blank"
            rel="noopener noreferrer"
            className={`glass-panel p-6 flex flex-col items-center gap-3 transition-all duration-300 hover:scale-110 group ${color}`}
          >
            <Icon className="w-8 h-8 text-muted-foreground group-hover:animate-pulse-glow transition-colors" />
            <span className="text-xs font-mono text-muted-foreground group-hover:text-foreground transition-colors">{label}</span>
          </a>
        ))}
      </div>
    </section>
  );
}
