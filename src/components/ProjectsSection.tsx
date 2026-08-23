import { useState } from 'react';
import { Cpu, FolderSearch, Bot, ShieldCheck, Search, ExternalLink, ArrowUpRight } from 'lucide-react';
import SectionHeading from './SectionHeading';
import ArchitectureCover from './ArchitectureCover';

const projects = [
  {
    title: 'Privacy-Preserving Blockchain Forensics',
    subtitle: "Master's Dissertation · Monero Anomaly Detection",
    icon: ShieldCheck,
    tagline: 'Catching suspicious behaviour on a privacy coin without ever de-anonymising a single user.',
    desc: 'End-to-end forensics pipeline that extracts and analyses Monero transaction behaviour (timing, frequency, structural signals) via a locally synced node and unsupervised ML, without exposing any user-identifying data. Combines Isolation Forest and Autoencoders with SHAP for explainable anomaly detection, delivered as an interactive Streamlit dashboard for analysts.',
    architecture: [
      { label: 'Source', items: ['Monero node', 'RPC'] },
      { label: 'Pipeline', items: ['Python', 'Pandas', 'Feature eng.'] },
      { label: 'Models', items: ['Isolation Forest', 'Autoencoder'] },
      { label: 'Insight', items: ['SHAP', 'Streamlit'] },
    ],
    tech: ['Python', 'Monero RPC', 'Isolation Forest', 'Autoencoders', 'SHAP', 'Streamlit'],
    accent: 'text-primary',
    featured: true,
  },
  {
    title: 'AI-Powered Blockchain Mining Simulator',
    subtitle: 'Neural-guided Proof-of-Work Research',
    icon: Bot,
    tagline: 'A neural net that learns to mine: fewer hashes, same security guarantees.',
    desc: 'Real-time web application comparing traditional Proof-of-Work mining against a neural-network-driven approach. Demonstrates a measurable reduction in the computational steps needed to reach a valid hash, without compromising blockchain validation or decentralisation.',
    architecture: [
      { label: 'Chain layer', items: ['Block builder', 'SHA-256 PoW'] },
      { label: 'AI layer', items: ['TensorFlow', 'Keras'] },
      { label: 'Interface', items: ['Streamlit', 'Live metrics'] },
    ],
    tech: ['Python', 'TensorFlow', 'Keras', 'Streamlit'],
    accent: 'text-secondary',
  },
  {
    title: 'Directory Traversal Attack Simulation',
    subtitle: 'Offensive Security · Web Exploitation',
    icon: FolderSearch,
    tagline: 'Reading /etc/passwd through a URL, then writing the fix.',
    desc: 'Structured security testing to identify and exploit directory traversal vulnerabilities by manipulating URL parameters to access restricted server files. Documented input validation failures and effective security header configurations to support remediation guidance for developers.',
    architecture: [
      { label: 'Target', items: ['PortSwigger lab', 'Linux host'] },
      { label: 'Attack', items: ['Burp Suite', 'Payload fuzzing'] },
      { label: 'Fix', items: ['Input validation', 'Security headers'] },
    ],
    tech: ['Burp Suite', 'PortSwigger', 'Linux', 'Security'],
    accent: 'text-primary',
  },
  {
    title: 'AI Car Game on Unity 3D',
    subtitle: 'Game AI · Pathfinding & Difficulty Scaling',
    icon: Cpu,
    tagline: 'Opponents that actually drive like opponents.',
    desc: 'Interactive 3D car racing game built in Unity featuring AI-controlled opponents with pathfinding, obstacle avoidance, and dynamic difficulty scaling for realistic, replayable gameplay.',
    architecture: [
      { label: 'Engine', items: ['Unity 3D', 'Physics'] },
      { label: 'Logic', items: ['C#', 'NavMesh pathfinding'] },
      { label: 'Gameplay', items: ['Obstacle avoidance', 'Difficulty scaling'] },
    ],
    tech: ['Unity', 'C#', 'AI', '3D'],
    accent: 'text-secondary',
  },
  {
    title: 'Log Detective',
    subtitle: 'SOC Log Analysis · Live on Vercel',
    icon: Search,
    tagline: 'Turn noisy logs into clear incident signals.',
    desc: 'A practical cybersecurity log-analysis engine that parses system and application logs to surface suspicious activity, repeated failed logins, brute-force patterns, and anomalous IP behaviour. Built with Python, Pandas and regex-driven detection, it automates the repetitive parts of SOC investigation and produces actionable security insights.',
    architecture: [
      { label: 'Ingest', items: ['Raw auth/app logs'] },
      { label: 'Parse', items: ['Python', 'Regex'] },
      { label: 'Detect', items: ['Pandas', 'Brute-force rules'] },
      { label: 'Deploy', items: ['Vercel'] },
    ],
    tech: ['Python', 'Regular Expressions', 'Pandas', 'Log Analysis', 'SOC', 'Vercel'],
    accent: 'text-primary',
    link: 'https://log-detective.vercel.app/',
    linkLabel: 'Open live app',
  },
];

export default function ProjectsSection() {
  const [expanded, setExpanded] = useState<number | null>(0);

  return (
    <section id="projects" className="py-24 px-4 max-w-6xl mx-auto">
      <SectionHeading label="Selected Work" title="Projects" />

      <div className="flex flex-col gap-4 max-w-4xl mx-auto">
        {projects.map((p, i) => {
          const Icon = p.icon;
          const isExpanded = expanded === i;

          return (
            <article
              key={p.title}
              onClick={() => setExpanded(isExpanded ? null : i)}
              className={`group relative overflow-hidden rounded-2xl border bg-muted/40 p-6 md:p-7 cursor-pointer transition-all duration-500 hover:-translate-y-1 animate-fade-up ${
                isExpanded ? 'border-primary/25 bg-muted/40' : 'border-border hover:border-border'
              }`}
              style={{ animationDelay: `${i * 80}ms` }}
            >
              <div className="flex items-start justify-between gap-4 mb-4">
                <div className="flex items-start gap-4 min-w-0">
                  <div className={`shrink-0 w-10 h-10 rounded-xl border border-border bg-muted/40 flex items-center justify-center ${p.accent}`}>
                    <Icon className="w-4.5 h-4.5" />
                  </div>
                  <div className="min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <h3 className="font-display text-lg md:text-xl font-semibold text-foreground leading-tight">
                        {p.title}
                      </h3>
                      {p.featured && (
                        <span className="rounded-full border border-primary/25 px-2 py-0.5 text-[9px] uppercase tracking-[0.18em] text-primary">
                          Featured
                        </span>
                      )}
                    </div>
                    <p className="text-[11px] font-mono text-muted-foreground mt-1 tracking-wide uppercase">
                      {p.subtitle}
                    </p>
                  </div>
                </div>
                <ArrowUpRight
                  className={`w-5 h-5 shrink-0 transition-all duration-300 ${
                    isExpanded ? 'rotate-45 text-primary' : 'text-muted-foreground group-hover:text-primary'
                  }`}
                />
              </div>

              {/* Architecture cover */}
              <ArchitectureCover stages={p.architecture} accent={p.accent} />

              <p className="text-sm text-muted-foreground italic leading-snug mt-4">{p.tagline}</p>

              <div className={`grid transition-all duration-500 ${isExpanded ? 'grid-rows-[1fr] opacity-100 mt-4' : 'grid-rows-[0fr] opacity-0'}`}>
                <div className="overflow-hidden">
                  <p className="text-sm text-muted-foreground leading-relaxed mb-4">{p.desc}</p>

                  <div className="flex flex-wrap gap-1.5">
                    {p.tech.map((t) => (
                      <span key={t} className="text-[10px] font-mono px-2 py-0.5 rounded-full border border-border text-muted-foreground">
                        {t}
                      </span>
                    ))}
                  </div>

                  {p.link && (
                    <a
                      href={p.link}
                      target="_blank"
                      rel="noopener noreferrer"
                      onClick={(e) => e.stopPropagation()}
                      className="inline-flex items-center gap-2 mt-4 text-[11px] font-mono tracking-widest text-primary hover:opacity-80 transition-opacity"
                    >
                      <ExternalLink className="w-3.5 h-3.5" />
                      {p.linkLabel ?? 'Visit'}
                    </a>
                  )}
                </div>
              </div>
            </article>
          );
        })}

        {/* TryHackMe Writeups external link card */}
        <a
          href="https://fizashaikh293.github.io/thm-writeups/"
          target="_blank"
          rel="noopener noreferrer"
          className="group rounded-2xl border border-border bg-muted/40 p-6 md:p-7 transition-all duration-500 hover:-translate-y-1 hover:border-border animate-fade-up"
        >
          <div className="flex items-start justify-between gap-4">
            <div className="flex items-start gap-4">
              <div className="w-10 h-10 rounded-xl border border-border bg-muted/40 flex items-center justify-center text-secondary">
                <ShieldCheck className="w-4.5 h-4.5" />
              </div>
              <div>
                <h3 className="font-display text-lg md:text-xl font-semibold text-foreground">TryHackMe Writeups</h3>
                <p className="text-[11px] font-mono text-muted-foreground mt-1 tracking-wide uppercase">
                  Live · Hands-on Lab Notes
                </p>
              </div>
            </div>
            <ExternalLink className="w-5 h-5 text-muted-foreground group-hover:text-primary transition-colors" />
          </div>
          <p className="text-sm text-muted-foreground mt-4 leading-relaxed">
            A growing collection of hands-on TryHackMe room writeups covering offensive security, networking, and digital
            forensics. Each one a documented kill chain from recon to remediation.
          </p>
          <p className="text-[11px] text-primary mt-4 font-mono tracking-widest">VISIT SITE →</p>
        </a>
      </div>
    </section>
  );
}
