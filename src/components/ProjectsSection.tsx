import { useState } from 'react';
import { Database, Cpu, Heart, Pickaxe, FolderSearch, TrafficCone, Network, Bot } from 'lucide-react';

const projects = [
  {
    title: 'Monero Dissertation',
    icon: Database,
    desc: 'In-depth research on Monero privacy mechanisms, ring signatures, and stealth addresses. Analyzed traceability attacks and proposed countermeasures.',
    tech: ['Monero', 'Cryptography', 'LaTeX', 'Python'],
    glow: 'neon-box-purple',
  },
  {
    title: 'AI Car Game',
    icon: Cpu,
    desc: 'Reinforcement learning agent that learns to navigate a 2D racetrack using neural networks and genetic algorithms.',
    tech: ['Python', 'PyTorch', 'Pygame', 'NumPy'],
    glow: 'neon-box-cyan',
  },
  {
    title: 'Heartbeat Monitor',
    icon: Heart,
    desc: 'IoT heartbeat monitoring system with real-time data visualization and anomaly detection alerts.',
    tech: ['Arduino', 'C++', 'Firebase', 'React'],
    glow: 'neon-box-yellow',
  },
  {
    title: 'Smart Mining',
    icon: Pickaxe,
    desc: 'Blockchain-based mining optimization platform with smart contract reward distribution.',
    tech: ['Solidity', 'Web3.js', 'Hardhat', 'React'],
    glow: 'neon-box-purple',
  },
  {
    title: 'Directory Traversal Sim',
    icon: FolderSearch,
    desc: 'Educational security tool simulating directory traversal attacks in a sandboxed environment.',
    tech: ['Python', 'Flask', 'Docker', 'Linux'],
    glow: 'neon-box-cyan',
  },
  {
    title: 'Fuzzy Logic Traffic Lights',
    icon: TrafficCone,
    desc: 'Intelligent traffic management system using fuzzy logic controllers for dynamic signal timing.',
    tech: ['Python', 'scikit-fuzzy', 'Matplotlib', 'Streamlit'],
    glow: 'neon-box-yellow',
  },
  {
    title: 'Office Network Simulation',
    icon: Network,
    desc: 'Complete enterprise network design with VLANs, firewalls, ACLs and monitoring.',
    tech: ['Cisco', 'Packet Tracer', 'Wireshark'],
    glow: 'neon-box-purple',
  },
  {
    title: 'EdgePaw LLM',
    icon: Bot,
    desc: 'Edge-deployed LLM for pet health diagnostics, optimized for Raspberry Pi inference.',
    tech: ['Python', 'TensorFlow Lite', 'Raspberry Pi', 'FastAPI'],
    glow: 'neon-box-cyan',
  },
];

export default function ProjectsSection() {
  const [expanded, setExpanded] = useState<number | null>(null);

  return (
    <section id="projects" className="py-24 px-4 max-w-6xl mx-auto">
      <h2 className="font-display text-2xl md:text-3xl font-bold text-accent neon-glow-yellow mb-10 text-center">
        {'>'} Projects
      </h2>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {projects.map((project, i) => {
          const Icon = project.icon;
          const isExpanded = expanded === i;
          return (
            <div
              key={project.title}
              onClick={() => setExpanded(isExpanded ? null : i)}
              className={`glass-panel p-5 cursor-pointer transition-all duration-500 hover:scale-[1.03] ${isExpanded ? project.glow : ''}`}
            >
              <Icon className={`w-8 h-8 mb-3 ${i % 3 === 0 ? 'text-primary' : i % 3 === 1 ? 'text-secondary' : 'text-accent'}`} />
              <h3 className="font-display text-sm font-semibold text-foreground mb-1">{project.title}</h3>
              <div className={`overflow-hidden transition-all duration-500 ${isExpanded ? 'max-h-48 opacity-100 mt-3' : 'max-h-0 opacity-0'}`}>
                <p className="text-xs text-muted-foreground leading-relaxed mb-3">{project.desc}</p>
                <div className="flex flex-wrap gap-1.5">
                  {project.tech.map((t) => (
                    <span key={t} className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-muted text-primary border border-primary/20">
                      {t}
                    </span>
                  ))}
                </div>
              </div>
              {!isExpanded && (
                <p className="text-[10px] text-muted-foreground mt-2 font-mono">Click to expand →</p>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}
