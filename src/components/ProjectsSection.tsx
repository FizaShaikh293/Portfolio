import { useState } from 'react';
import { Cpu, FolderSearch, Bot, ShieldCheck, ExternalLink } from 'lucide-react';
import SectionHeading from './SectionHeading';

const projects = [
  {
    title: "Privacy-Preserving Blockchain Forensics (Master's Dissertation)",
    icon: ShieldCheck,
    desc: 'End-to-end forensics pipeline extracting and analysing Monero transaction behaviour (timing, frequency, structural signals) via a locally synced node and unsupervised ML, without exposing user-identifying data. Used Isolation Forest and Autoencoders with SHAP for explainable anomaly detection, delivered as an interactive Streamlit dashboard.',
    tech: ['Python', 'Monero RPC', 'Isolation Forest', 'Autoencoders', 'SHAP', 'Streamlit'],
    glow: 'neon-box-cyan',
  },
  {
    title: 'AI-Powered Blockchain Mining Simulator',
    icon: Bot,
    desc: 'Real-time web application comparing traditional Proof-of-Work mining against a neural-network-driven approach, demonstrating a measurable reduction in computational steps needed to reach a valid hash without compromising blockchain validation or decentralisation.',
    tech: ['Python', 'TensorFlow', 'Keras', 'Streamlit'],
    glow: 'neon-box-purple',
  },
  {
    title: 'Directory Traversal Attack Simulation',
    icon: FolderSearch,
    desc: 'Structured security testing to identify and exploit directory traversal vulnerabilities by manipulating URL parameters to access restricted files such as /etc/passwd. Documented input validation failures and effective security header configurations to support remediation guidance.',
    tech: ['Burp Suite', 'PortSwigger', 'Linux', 'Security'],
    glow: 'neon-box-yellow',
  },
  {
    title: 'AI Car Game on Unity 3D',
    icon: Cpu,
    desc: 'Interactive 3D car racing game built in Unity featuring AI-controlled opponents with pathfinding, obstacle avoidance, and difficulty scaling for realistic and dynamic gameplay.',
    tech: ['Unity', 'C#', 'AI', '3D'],
    glow: 'neon-box-cyan',
  },
];

export default function ProjectsSection() {
  const [expanded, setExpanded] = useState<number | null>(null);

  return (
    <section id="projects" className="py-24 px-4 max-w-6xl mx-auto">
      <SectionHeading label="Selected Work" title="Projects" />
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {projects.map((project, i) => {
          const Icon = project.icon;
          const isExpanded = expanded === i;
          return (
            <div
              key={project.title}
              onClick={() => setExpanded(isExpanded ? null : i)}
              className={`glass-panel p-5 cursor-pointer transition-all duration-500 hover:scale-[1.02] ${isExpanded ? project.glow : ''}`}
            >
              <Icon className={`w-8 h-8 mb-3 ${i % 3 === 0 ? 'text-primary' : i % 3 === 1 ? 'text-secondary' : 'text-accent'}`} />
              <h3 className="font-display text-sm font-semibold text-foreground mb-1">{project.title}</h3>
              <div className={`overflow-hidden transition-all duration-500 ${isExpanded ? 'max-h-96 opacity-100 mt-3' : 'max-h-0 opacity-0'}`}>
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

        {/* TryHackMe Writeups external link card */}
        <a
          href="https://fizashaikh293.github.io/thm-writeups/"
          target="_blank"
          rel="noopener noreferrer"
          className="glass-panel p-5 group transition-all duration-500 hover:scale-[1.02] neon-box-purple flex flex-col"
        >
          <div className="flex items-center justify-between mb-3">
            <ShieldCheck className="w-8 h-8 text-secondary" />
            <ExternalLink className="w-4 h-4 text-muted-foreground group-hover:text-primary transition-colors" />
          </div>
          <h3 className="font-display text-sm font-semibold text-foreground mb-1">
            TryHackMe Writeups
          </h3>
          <p className="text-xs text-muted-foreground leading-relaxed mt-2">
            A growing collection of hands-on TryHackMe room writeups covering offensive security, networking, and forensics labs.
          </p>
          <p className="text-[10px] text-primary mt-3 font-mono">Visit site →</p>
        </a>
      </div>
    </section>
  );
}
