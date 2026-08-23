import SectionHeading from './SectionHeading';

const techCategories = [
  {
    title: 'Languages',
    items: ['Python', 'JavaScript', 'Solidity', 'Rust', 'Java', 'C++', 'C#', 'PHP', 'R', 'XML'],
  },
  {
    title: 'Security',
    items: ['Burp Suite', 'Wireshark', 'Nmap', 'Metasploit', 'OWASP', 'Digital Forensics'],
  },
  {
    title: 'Blockchain',
    items: ['Web3.js', 'Ethereum', 'Smart Contracts', 'DeFi', 'Hardhat', 'Truffle'],
  },
  {
    title: 'Frameworks & Tools',
    items: ['React', 'Flask', 'FastAPI', 'ASP.NET', '.NET', 'Streamlit', 'MongoDB', 'MySQL', 'Hadoop', 'Hive'],
  },
  {
    title: 'Hardware & IoT',
    items: ['Arduino', 'Raspberry Pi', 'ESP8266', 'Cisco', 'Packet Tracer', 'Proteus'],
  },
  {
    title: 'Design & Other',
    items: ['Unity', 'Figma', 'Canva', 'Framer', 'LaTeX', 'Android Studio', 'Visual Studio', 'NetBeans'],
  },
];

export default function TechStackSection() {
  return (
    <section id="techstack" className="py-24 px-4 max-w-6xl mx-auto">
      <SectionHeading label="Toolbox" title="Tech Stack" />

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
        {techCategories.map((cat, catIdx) => (
          <div
            key={cat.title}
            className="paper paper-lifted p-5 animate-fade-up"
            style={{ animationDelay: `${catIdx * 70}ms` }}
          >
            <div className="flex items-baseline justify-between mb-4 pb-2 border-b border-border">
              <h3 className="text-lg text-foreground">{cat.title}</h3>
              <span className="text-[10px] font-mono text-muted-foreground">
                {String(catIdx + 1).padStart(2, '0')}
              </span>
            </div>
            <div className="flex flex-wrap gap-x-3 gap-y-1.5">
              {cat.items.map((item) => (
                <span
                  key={item}
                  className="text-[11px] font-mono text-muted-foreground transition-colors duration-300 hover:text-primary cursor-default"
                >
                  {item}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
