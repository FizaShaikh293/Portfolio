import SectionHeading from './SectionHeading';

const techCategories = [
  {
    title: 'Languages',
    color: 'primary',
    items: ['Python', 'JavaScript', 'Solidity', 'Rust', 'Java', 'C++', 'C#', 'PHP', 'R', 'XML'],
  },
  {
    title: 'Security',
    color: 'secondary',
    items: ['Burp Suite', 'Wireshark', 'Nmap', 'Metasploit', 'OWASP', 'Digital Forensics'],
  },
  {
    title: 'Blockchain',
    color: 'accent',
    items: ['Web3.js', 'Ethereum', 'Smart Contracts', 'DeFi', 'Hardhat', 'Truffle'],
  },
  {
    title: 'Frameworks & Tools',
    color: 'primary',
    items: ['React', 'Flask', 'FastAPI', 'ASP.NET', '.NET', 'Streamlit', 'MongoDB', 'MySQL', 'Hadoop', 'Hive'],
  },
  {
    title: 'Hardware & IoT',
    color: 'secondary',
    items: ['Arduino', 'Raspberry Pi', 'ESP8266', 'Cisco', 'Packet Tracer', 'Proteus'],
  },
  {
    title: 'Design & Other',
    color: 'accent',
    items: ['Unity', 'Figma', 'Canva', 'Framer', 'LaTeX', 'Android Studio', 'Visual Studio', 'NetBeans'],
  },
];

export default function TechStackSection() {
  const colorMap: Record<string, { text: string; border: string; bg: string; glow: string }> = {
    primary: { text: 'text-primary', border: 'border-primary/30', bg: 'bg-primary/10', glow: 'neon-box-cyan' },
    secondary: { text: 'text-secondary', border: 'border-secondary/30', bg: 'bg-secondary/10', glow: 'neon-box-purple' },
    accent: { text: 'text-accent', border: 'border-accent/30', bg: 'bg-accent/10', glow: 'neon-box-yellow' },
  };

  return (
    <section id="techstack" className="py-24 px-4 max-w-6xl mx-auto">
      <SectionHeading label="Toolbox" title="Tech Stack" />

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {techCategories.map((cat, catIdx) => {
          const c = colorMap[cat.color];
          return (
            <div
              key={cat.title}
              className={`glass-panel p-5 hover:scale-[1.03] transition-transform duration-300 group ${c.glow} animate-fade-up`}
              style={{ animationDelay: `${catIdx * 80}ms` }}
            >
              <h3 className={`font-display text-xs font-bold ${c.text} hover-text-pop cursor-default mb-4 tracking-wider uppercase`}>
                {cat.title}
              </h3>
              <div className="flex flex-wrap gap-2">
                {cat.items.map((item) => (
                  <span
                    key={item}
                    className={`text-[10px] font-mono px-2 py-1 rounded border ${c.border} ${c.bg} text-foreground/80 hover:scale-110 hover-text-glow transition-all duration-300 cursor-default`}
                  >
                    {item}
                  </span>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
