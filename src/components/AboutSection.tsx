import { Shield, Link, Search, FolderKanban } from 'lucide-react';

const interests = [
  { icon: Shield, label: 'Cybersecurity', color: 'text-primary' },
  { icon: Link, label: 'Blockchain Security', color: 'text-secondary' },
  { icon: Search, label: 'Threat Analysis', color: 'text-accent' },
  { icon: FolderKanban, label: 'Projects', color: 'text-primary' },
];

export default function AboutSection() {
  return (
    <section id="about" className="py-24 px-4 max-w-4xl mx-auto">
      <h2 className="font-display text-2xl md:text-3xl font-bold text-primary neon-glow-cyan mb-8 text-center">
        {'>'} About Me
      </h2>
      <div className="glass-panel p-6 md:p-10 neon-box-cyan">
        <p className="text-foreground leading-relaxed mb-4">
          Hi! I'm a cybersecurity &amp; blockchain enthusiast with a passion for breaking things (ethically) 
          and building them back stronger. I love diving deep into smart contract security, network defense, 
          and decentralized systems.
        </p>
        <p className="text-muted-foreground text-sm mb-6">
          <span className="text-accent">Fun fact:</span> I once debugged a smart contract vulnerability 
          while waiting for my coffee to brew. The coffee got cold — but the chain stayed secure. ☕🔐
        </p>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {interests.map(({ icon: Icon, label, color }) => (
            <div
              key={label}
              className="glass-panel-purple p-4 text-center group hover:neon-box-purple transition-all duration-300 cursor-default"
            >
              <Icon className={`w-8 h-8 mx-auto mb-2 ${color} group-hover:scale-110 transition-transform duration-300 animate-float`} />
              <span className="text-xs font-mono text-foreground">{label}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
