import { useState } from 'react';
import { Cpu, FolderSearch, Bot, ShieldCheck, ExternalLink, Sparkles, ArrowUpRight } from 'lucide-react';
import SectionHeading from './SectionHeading';

const projects = [
  {
    title: "Privacy-Preserving Blockchain Forensics",
    subtitle: "Master's Dissertation · Monero Anomaly Detection",
    icon: ShieldCheck,
    tagline: 'Catching suspicious behaviour on a privacy coin without ever de-anonymising a single user.',
    desc: 'End-to-end forensics pipeline that extracts and analyses Monero transaction behaviour (timing, frequency, structural signals) via a locally synced node and unsupervised ML, without exposing any user-identifying data. Combines Isolation Forest and Autoencoders with SHAP for explainable anomaly detection, delivered as an interactive Streamlit dashboard for analysts.',
    metrics: [
      { label: 'Local node', value: 'Monero RPC' },
      { label: 'Models', value: 'IF + Autoencoder' },
      { label: 'Explainability', value: 'SHAP' },
    ],
    tech: ['Python', 'Monero RPC', 'Isolation Forest', 'Autoencoders', 'SHAP', 'Streamlit'],
    glow: 'neon-box-cyan',
    accent: 'text-primary',
    featured: true,
  },
  {
    title: 'AI-Powered Blockchain Mining Simulator',
    subtitle: 'Neural-guided Proof-of-Work Research',
    icon: Bot,
    tagline: 'A neural net that learns to mine: fewer hashes, same security guarantees.',
    desc: 'Real-time web application comparing traditional Proof-of-Work mining against a neural-network-driven approach. Demonstrates a measurable reduction in the computational steps needed to reach a valid hash, without compromising blockchain validation or decentralisation.',
    metrics: [
      { label: 'Compare', value: 'PoW vs AI' },
      { label: 'Live', value: 'Streamlit UI' },
    ],
    tech: ['Python', 'TensorFlow', 'Keras', 'Streamlit'],
    glow: 'neon-box-purple',
    accent: 'text-secondary',
  },
  {
    title: 'Directory Traversal Attack Simulation',
    subtitle: 'Offensive Security · Web Exploitation',
    icon: FolderSearch,
    tagline: 'Reading /etc/passwd through a URL, then writing the fix.',
    desc: 'Structured security testing to identify and exploit directory traversal vulnerabilities by manipulating URL parameters to access restricted server files. Documented input validation failures and effective security header configurations to support remediation guidance for developers.',
    metrics: [
      { label: 'Tooling', value: 'Burp Suite' },
      { label: 'Lab', value: 'PortSwigger' },
    ],
    tech: ['Burp Suite', 'PortSwigger', 'Linux', 'Security'],
    glow: 'neon-box-yellow',
    accent: 'text-accent',
  },
  {
    title: 'AI Car Game on Unity 3D',
    subtitle: 'Game AI · Pathfinding & Difficulty Scaling',
    icon: Cpu,
    tagline: 'Opponents that actually drive like opponents.',
    desc: 'Interactive 3D car racing game built in Unity featuring AI-controlled opponents with pathfinding, obstacle avoidance, and dynamic difficulty scaling for realistic, replayable gameplay.',
    metrics: [
      { label: 'Engine', value: 'Unity 3D' },
      { label: 'Lang', value: 'C#' },
    ],
    tech: ['Unity', 'C#', 'AI', '3D'],
    glow: 'neon-box-cyan',
    accent: 'text-primary',
  },
];

export default function ProjectsSection() {
  const [expanded, setExpanded] = useState<number | null>(0);

  return (
    <section id="projects" className="py-24 px-4 max-w-6xl mx-auto">
      <SectionHeading label="Selected Work" title="Projects" />

      <div className="grid grid-cols-1 lg:grid-cols-6 gap-5">
        {projects.map((p, i) => {
          const Icon = p.icon;
          const isExpanded = expanded === i;
          const colSpan = p.featured ? 'lg:col-span-6' : 'lg:col-span-3';

          return (
            <article
              key={p.title}
              onClick={() => setExpanded(isExpanded ? null : i)}
              className={`glass-panel p-6 md:p-7 cursor-pointer transition-all duration-500 hover:scale-[1.01] hover:-translate-y-1 group relative overflow-hidden ${colSpan} ${isExpanded ? p.glow : ''} animate-fade-up`}
              style={{ animationDelay: `${i * 90}ms` }}
            >
              {/* gradient corner glow */}
              <div className="pointer-events-none absolute -top-24 -right-24 w-56 h-56 rounded-full bg-gradient-to-br from-primary/20 via-secondary/10 to-transparent blur-3xl opacity-60 group-hover:opacity-100 transition-opacity duration-700" />

              {p.featured && (
                <div className="inline-flex items-center gap-1.5 mb-4 rounded-full border border-primary/30 bg-primary/10 px-3 py-1 text-[10px] font-medium tracking-widest uppercase text-primary">
                  <Sparkles className="w-3 h-3" />
                  Featured
                </div>
              )}

              <div className="flex items-start justify-between gap-4 mb-3">
                <div className="flex items-start gap-4 min-w-0">
                  <div className={`shrink-0 w-11 h-11 rounded-xl border border-white/10 bg-white/[0.04] flex items-center justify-center ${p.accent}`}>
                    <Icon className="w-5 h-5" />
                  </div>
                  <div className="min-w-0">
                    <h3 className="font-display text-lg md:text-xl font-semibold text-foreground hover-text-pop cursor-default leading-tight">
                      {p.title}
                    </h3>
                    <p className="text-[11px] font-mono text-muted-foreground mt-1 tracking-wide uppercase">
                      {p.subtitle}
                    </p>
                  </div>
                </div>
                <ArrowUpRight className={`w-5 h-5 shrink-0 text-muted-foreground transition-all duration-300 ${isExpanded ? 'rotate-45 text-primary' : 'group-hover:text-primary group-hover:-translate-y-0.5 group-hover:translate-x-0.5'}`} />
              </div>

              <p className="text-sm text-foreground/80 italic leading-snug mb-3">
                "{p.tagline}"
              </p>

              <div className={`grid transition-all duration-500 ${isExpanded ? 'grid-rows-[1fr] opacity-100 mt-2' : 'grid-rows-[0fr] opacity-0'}`}>
                <div className="overflow-hidden">
                  <p className="text-sm text-muted-foreground leading-relaxed mb-4">{p.desc}</p>

                  <div className="flex flex-wrap gap-2 mb-4">
                    {p.metrics.map((m) => (
                      <div key={m.label} className="rounded-lg border border-white/10 bg-white/[0.03] px-3 py-1.5">
                        <div className="text-[9px] uppercase tracking-widest text-muted-foreground">{m.label}</div>
                        <div className={`text-xs font-mono ${p.accent}`}>{m.value}</div>
                      </div>
                    ))}
                  </div>

                  <div className="flex flex-wrap gap-1.5">
                    {p.tech.map((t) => (
                      <span key={t} className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-muted text-primary border border-primary/20">
                        {t}
                      </span>
                    ))}
                  </div>
                </div>
              </div>

              {!isExpanded && (
                <p className="text-[10px] text-muted-foreground mt-3 font-mono tracking-wider">
                  Click to expand →
                </p>
              )}
            </article>
          );
        })}

        {/* TryHackMe Writeups external link card */}
        <a
          href="https://fizashaikh293.github.io/thm-writeups/"
          target="_blank"
          rel="noopener noreferrer"
          className="glass-panel p-6 md:p-7 group transition-all duration-500 hover:scale-[1.01] hover:-translate-y-1 neon-box-purple flex flex-col relative overflow-hidden lg:col-span-6 animate-fade-up"
        >
          <div className="pointer-events-none absolute -bottom-24 -left-24 w-56 h-56 rounded-full bg-gradient-to-tr from-secondary/25 via-primary/10 to-transparent blur-3xl" />
          <div className="flex items-start justify-between gap-4">
            <div className="flex items-start gap-4">
              <div className="w-11 h-11 rounded-xl border border-white/10 bg-white/[0.04] flex items-center justify-center text-secondary">
                <ShieldCheck className="w-5 h-5" />
              </div>
              <div>
                <h3 className="font-display text-lg md:text-xl font-semibold text-foreground hover-text-pop cursor-default">
                  TryHackMe Writeups
                </h3>
                <p className="text-[11px] font-mono text-muted-foreground mt-1 tracking-wide uppercase">
                  Live · Hands-on Lab Notes
                </p>
              </div>
            </div>
            <ExternalLink className="w-5 h-5 text-muted-foreground group-hover:text-primary group-hover:-translate-y-0.5 group-hover:translate-x-0.5 transition-all" />
          </div>
          <p className="text-sm text-foreground/80 mt-4 leading-relaxed">
            A growing collection of hands-on TryHackMe room writeups covering offensive security, networking, and digital forensics — each one a documented kill chain from recon to remediation.
          </p>
          <p className="text-[10px] text-primary mt-4 font-mono tracking-widest">VISIT SITE →</p>
        </a>
      </div>
    </section>
  );
}
