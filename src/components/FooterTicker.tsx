const techStack = [
  'Python', 'JavaScript', 'Solidity', 'Rust', 'Java', 'C++', 'C#', 'PHP',
  'React', 'Web3.js', 'TensorFlow', 'Docker', 'AWS', 'Linux',
  'Burp Suite', 'Wireshark', 'Arduino', 'MongoDB', 'MySQL', 'Flask',
  'Unity', 'MATLAB', 'Hadoop', 'Firebase', 'Figma', 'R',
];

export default function FooterTicker() {
  const doubled = [...techStack, ...techStack];

  return (
    <footer className="fixed bottom-0 left-0 right-0 z-50 bg-background/80 backdrop-blur-md border-t border-primary/10">
      <div className="overflow-hidden py-2">
        <div className="flex ticker-scroll whitespace-nowrap">
          {doubled.map((tech, i) => (
            <span key={i} className="inline-flex items-center mx-6 text-xs font-mono text-muted-foreground">
              <span className="w-1.5 h-1.5 rounded-full bg-accent mr-2 animate-pulse-glow" />
              {tech}
            </span>
          ))}
        </div>
      </div>
    </footer>
  );
}
