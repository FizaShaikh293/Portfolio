import { useEffect, useState } from 'react';

const techStack = [
  'HTML', 'JavaScript', 'Python', 'Rust', 'Solidity', 'PHP', 'AWS', 'Google Cloud',
  '.NET', 'Hadoop', 'Hive', 'NPM', 'Streamlit', 'Web3.js', 'MongoDB', 'Firebase',
  'Canva', 'Figma', 'Framer', 'Keras', 'Matplotlib', 'scikit-learn', 'NumPy',
  'Pandas', 'Plotly', 'PyTorch', 'TensorFlow', 'GitHub', 'Arduino', 'Cisco',
  'Raspberry Pi', 'Unity', 'C++',
];

export default function HeroSection() {
  const [visible, setVisible] = useState(false);
  useEffect(() => { setVisible(true); }, []);

  const doubled = [...techStack, ...techStack];

  return (
    <section className="relative min-h-screen flex flex-col items-center justify-center px-4">
      <div className={`text-center transition-all duration-1000 ${visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>
        <p className="font-mono text-sm text-primary mb-4 tracking-[0.3em] uppercase">
          // cybersecurity &amp; blockchain
        </p>
        <h1 className="font-display text-4xl sm:text-5xl md:text-7xl font-bold text-accent neon-glow-yellow leading-tight mb-6">
          Despite everything,
          <br />
          it's still you
        </h1>
        <p className="font-mono text-muted-foreground text-sm md:text-base max-w-lg mx-auto">
          Security researcher · Blockchain builder · Eternal learner
        </p>
      </div>

      {/* Tech Stack Ticker */}
      <div className="absolute bottom-12 left-0 right-0 overflow-hidden">
        <div className="glass-panel mx-4 py-3 overflow-hidden">
          <div className="flex ticker-scroll whitespace-nowrap">
            {doubled.map((tech, i) => (
              <span key={i} className="inline-flex items-center mx-4 text-xs font-mono">
                <span className="w-1.5 h-1.5 rounded-full bg-accent mr-2 animate-pulse-glow" />
                <span className="text-primary">{tech}</span>
              </span>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
