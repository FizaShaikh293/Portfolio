import { useEffect, useRef, useState } from 'react';

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
    title: 'Cloud & DevOps',
    color: 'primary',
    items: ['AWS', 'Google Cloud', 'Docker', 'Linux', 'Firebase', 'GlassFish Server'],
  },
  {
    title: 'Data & AI',
    color: 'secondary',
    items: ['TensorFlow', 'PyTorch', 'Keras', 'scikit-learn', 'Pandas', 'NumPy', 'Matplotlib', 'ARIMA', 'MATLAB'],
  },
  {
    title: 'Frameworks & Tools',
    color: 'accent',
    items: ['React', 'Flask', 'FastAPI', 'ASP.NET', '.NET', 'Streamlit', 'MongoDB', 'MySQL', 'Hadoop', 'Hive'],
  },
  {
    title: 'Hardware & IoT',
    color: 'primary',
    items: ['Arduino', 'Raspberry Pi', 'ESP8266', 'Cisco', 'Packet Tracer', 'Proteus'],
  },
  {
    title: 'Design & Other',
    color: 'secondary',
    items: ['Unity', 'Figma', 'Canva', 'Framer', 'LaTeX', 'Android Studio', 'Visual Studio', 'NetBeans'],
  },
];

export default function TechStackSection() {
  const [visibleCats, setVisibleCats] = useState<number[]>([]);
  const catRefs = useRef<(HTMLDivElement | null)[]>([]);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const idx = Number(entry.target.getAttribute('data-cat-index'));
            setVisibleCats((prev) => prev.includes(idx) ? prev : [...prev, idx]);
          }
        });
      },
      { threshold: 0.2 }
    );
    catRefs.current.forEach((ref) => { if (ref) observer.observe(ref); });
    return () => observer.disconnect();
  }, []);

  const colorMap: Record<string, { text: string; border: string; bg: string; glow: string }> = {
    primary: { text: 'text-primary', border: 'border-primary/30', bg: 'bg-primary/10', glow: 'neon-box-cyan' },
    secondary: { text: 'text-secondary', border: 'border-secondary/30', bg: 'bg-secondary/10', glow: 'neon-box-purple' },
    accent: { text: 'text-accent', border: 'border-accent/30', bg: 'bg-accent/10', glow: 'neon-box-yellow' },
  };

  return (
    <section id="techstack" className="py-24 px-4 max-w-6xl mx-auto">
      <h2 className="font-display text-2xl md:text-3xl font-bold text-secondary neon-glow-purple mb-16 text-center">
        {'>'} Tech Arsenal
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {techCategories.map((cat, catIdx) => {
          const c = colorMap[cat.color];
          const isVisible = visibleCats.includes(catIdx);

          return (
            <div
              key={cat.title}
              ref={(el) => { catRefs.current[catIdx] = el; }}
              data-cat-index={catIdx}
              className={`glass-panel p-5 transition-all duration-700 hover:scale-[1.03] group ${c.glow} ${
                isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'
              }`}
              style={{ transitionDelay: isVisible ? `${catIdx * 100}ms` : '0ms' }}
            >
              <h3 className={`font-display text-xs font-bold ${c.text} mb-4 tracking-wider uppercase`}>
                {cat.title}
              </h3>
              <div className="flex flex-wrap gap-2">
                {cat.items.map((item, itemIdx) => (
                  <span
                    key={item}
                    className={`text-[10px] font-mono px-2 py-1 rounded border ${c.border} ${c.bg} text-foreground/80 
                      hover:scale-110 hover:${c.text} transition-all duration-300 cursor-default
                      tech-item-float`}
                    style={{
                      animationDelay: `${itemIdx * 0.15}s`,
                    }}
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
