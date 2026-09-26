import { useState } from 'react';
import { ExternalLink, ArrowUpRight } from 'lucide-react';
import SectionHeading from './SectionHeading';
import ArchitectureCover from './ArchitectureCover';

const GITHUB = 'https://github.com/FizaShaikh293';

const projects = [
  {
    title: 'Privacy-Preserving Blockchain Forensics',
    subtitle: "Master's Dissertation · Monero Anomaly Detection",
    summary: 'Privacy-preserving analysis of Monero transaction patterns using explainable anomaly detection.',
    desc: 'End-to-end forensics pipeline that extracts and analyses Monero transaction behaviour (timing, frequency, structural signals) via a locally synced node and unsupervised ML, without exposing any user-identifying data. Combines Isolation Forest and Autoencoders with SHAP for explainable anomaly detection, delivered as an interactive Streamlit dashboard for analysts.',
    architecture: [
      { label: 'Source', items: ['Monero node', 'RPC'] },
      { label: 'Pipeline', items: ['Python', 'Pandas', 'Feature eng.'] },
      { label: 'Models', items: ['Isolation Forest', 'Autoencoder'] },
      { label: 'Insight', items: ['SHAP', 'Streamlit'] },
    ],
    tech: ['Python', 'Monero RPC', 'Isolation Forest', 'Autoencoders', 'SHAP', 'Streamlit'],
    featured: true,
  },
  {
    title: 'AI-Powered Blockchain Mining Simulator',
    subtitle: 'Neural-guided Proof-of-Work Research',
    summary: 'A real-time comparison of conventional Proof-of-Work mining and a neural-guided approach.',
    desc: 'Real-time web application comparing traditional Proof-of-Work mining against a neural-network-driven approach. Demonstrates a measurable reduction in the computational steps needed to reach a valid hash, without compromising blockchain validation or decentralisation.',
    architecture: [
      { label: 'Chain layer', items: ['Block builder', 'SHA-256 PoW'] },
      { label: 'AI layer', items: ['TensorFlow', 'Keras'] },
      { label: 'Interface', items: ['Streamlit', 'Live metrics'] },
    ],
    tech: ['Python', 'TensorFlow', 'Keras', 'Streamlit'],
  },
  {
    title: 'Directory Traversal Attack Simulation',
    subtitle: 'Offensive Security · Web Exploitation',
    summary: 'A controlled web-security exercise covering directory traversal testing and remediation.',
    desc: 'Structured security testing to identify and exploit directory traversal vulnerabilities by manipulating URL parameters to access restricted server files. Documented input validation failures and effective security header configurations to support remediation guidance for developers.',
    architecture: [
      { label: 'Target', items: ['PortSwigger lab', 'Linux host'] },
      { label: 'Attack', items: ['Burp Suite', 'Payload fuzzing'] },
      { label: 'Fix', items: ['Input validation', 'Security headers'] },
    ],
    tech: ['Burp Suite', 'PortSwigger', 'Linux', 'Security'],
  },
  {
    title: 'AI Car Game on Unity 3D',
    subtitle: 'Game AI · Pathfinding & Difficulty Scaling',
    summary: 'A Unity racing game with AI pathfinding, obstacle avoidance, and adaptive difficulty.',
    desc: 'Interactive 3D car racing game built in Unity featuring AI-controlled opponents with pathfinding, obstacle avoidance, and dynamic difficulty scaling for realistic, replayable gameplay.',
    architecture: [
      { label: 'Engine', items: ['Unity 3D', 'Physics'] },
      { label: 'Logic', items: ['C#', 'NavMesh pathfinding'] },
      { label: 'Gameplay', items: ['Obstacle avoidance', 'Difficulty scaling'] },
    ],
    tech: ['Unity', 'C#', 'AI', '3D'],
  },
  {
    title: 'Log Detective',
    subtitle: 'SOC Log Analysis · Live on Vercel',
    summary: 'A browser-based SOC investigation game built around realistic security-log analysis.',
    desc: 'Players analyse simulated logs, classify events, and earn XP for accurate threat detection. Scenarios cover brute-force attacks, phishing, suspicious PowerShell activity, rogue administrator accounts, ransomware indicators, and legitimate behaviour to strengthen true-positive and false-positive judgement.',
    architecture: [
      { label: 'Scenarios', items: ['SOC event logs'] },
      { label: 'Interface', items: ['React', 'JavaScript'] },
      { label: 'Build', items: ['Vite', 'XP system'] },
      { label: 'Deploy', items: ['Vercel'] },
    ],
    tech: ['React', 'JavaScript', 'Vite', 'SOC Investigation', 'Log Analysis', 'Vercel'],
    link: 'https://log-detective.vercel.app/',
    linkLabel: 'Open live app',
  },
  {
    title: 'SOC Monitoring Lab',
    subtitle: 'Self-Hosted SIEM · Wazuh, Sysmon & Threat Hunting',
    summary: 'A self-hosted detection lab with Wazuh as the SIEM, Sysmon endpoint telemetry, and documented investigations.',
    desc: 'Built and configured a self-hosted SOC lab in VirtualBox: a Windows 11 monitored endpoint, a Kali Linux testing machine, and Wazuh as the central SIEM, connected over an isolated host-only network. Deployed Sysmon on the Windows endpoint to extend process, network, and file telemetry, integrated with the Wazuh agent for centralised log forwarding and correlation. Investigated Windows security events — including Event ID 5157 (Windows Filtering Platform blocked connections) and command-line activity — through Wazuh\'s Threat Hunting interface, documenting the full detection-to-investigation workflow.',
    architecture: [
      { label: 'Endpoint', items: ['Windows 11', 'Sysmon'] },
      { label: 'Adversary', items: ['Kali Linux'] },
      { label: 'Network', items: ['VirtualBox', 'Host-only net'] },
      { label: 'SIEM', items: ['Wazuh agent', 'Threat Hunting'] },
    ],
    tech: ['Wazuh', 'Sysmon', 'Windows 11', 'Kali Linux', 'VirtualBox', 'Threat Hunting', 'Windows Event Logs'],
    link: 'https://github.com/FizaShaikh293/SecurityandForensics-Projects/tree/main/soc-monitoring-lab',
    linkLabel: 'View lab repo',
    featured: true,
  },
];

export default function ProjectsSection() {
  const [expanded, setExpanded] = useState<number | null>(0);

  return (
    <section id="projects" className="py-24 md:py-32 px-4 sm:px-8 md:px-12 max-w-6xl mx-auto">
      <SectionHeading label="Selected Work" title="Projects" />

      <div className="flex flex-col gap-4">
        {projects.map((p, i) => {
          const isExpanded = expanded === i;

          return (
            <article
              key={p.title}
              onClick={() => setExpanded(isExpanded ? null : i)}
              className="paper paper-lifted bold-panel p-6 md:p-10 cursor-pointer animate-fade-up"
              style={{ animationDelay: `${i * 70}ms` }}
            >
              <div className="flex items-start justify-between gap-4">
                <div className="min-w-0">
                  <div className="flex items-baseline gap-3 flex-wrap">
                    <a
                      href={GITHUB}
                      target="_blank"
                      rel="noopener noreferrer"
                      onClick={(e) => e.stopPropagation()}
                      className="font-display text-2xl md:text-4xl leading-tight text-foreground ink-underline"
                    >
                      {p.title}
                    </a>
                    {p.featured && (
                      <span className="border border-foreground/20 px-2 py-0.5 text-[9px] font-mono uppercase tracking-[0.2em] text-muted-foreground">
                        Featured
                      </span>
                    )}
                  </div>
                  <p className="mt-2 text-[11px] font-mono uppercase tracking-[0.18em] text-muted-foreground">
                    {p.subtitle}
                  </p>
                </div>
                <ArrowUpRight
                  className={`w-5 h-5 shrink-0 text-muted-foreground transition-transform duration-300 ${
                    isExpanded ? 'rotate-45' : ''
                  }`}
                />
              </div>

              <div className="mt-5">
                <ArchitectureCover stages={p.architecture} accent="text-foreground" />
              </div>

              <p className="mt-6 text-sm md:text-base leading-relaxed text-muted-foreground">{p.summary}</p>

              <div className={`grid transition-all duration-500 ${isExpanded ? 'grid-rows-[1fr] opacity-100 mt-5' : 'grid-rows-[0fr] opacity-0'}`}>
                <div className="overflow-hidden">
                  <p className="text-sm leading-relaxed text-muted-foreground">{p.desc}</p>

                  <div className="mt-5 flex flex-wrap gap-x-4 gap-y-2">
                    {p.tech.map((t) => (
                      <span key={t} className="text-[10px] font-mono uppercase tracking-[0.16em] text-muted-foreground">
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
                      className="mt-6 inline-flex items-center gap-2 border border-foreground/25 px-4 py-2 text-[11px] font-mono uppercase tracking-[0.18em] text-foreground transition-colors duration-300 hover:bg-foreground hover:text-background"
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

        <a
          href="https://fizashaikh293.github.io/thm-writeups/"
          target="_blank"
          rel="noopener noreferrer"
          className="paper paper-lifted bold-panel p-6 md:p-10 animate-fade-up"
        >
          <div className="flex items-start justify-between gap-4">
            <div>
              <h3 className="text-2xl md:text-4xl text-foreground">TryHackMe Writeups</h3>
              <p className="mt-2 text-[11px] font-mono uppercase tracking-[0.18em] text-muted-foreground">
                Live · Hands-on Lab Notes
              </p>
            </div>
            <ExternalLink className="w-5 h-5 shrink-0 text-muted-foreground" />
          </div>
          <p className="mt-5 text-sm leading-relaxed text-muted-foreground">
            A growing collection of hands-on TryHackMe room writeups covering offensive security, networking, and digital
            forensics. Each one a documented kill chain from recon to remediation.
          </p>
          <span className="mt-6 inline-block text-[11px] font-mono uppercase tracking-[0.18em] text-foreground ink-underline">
            Visit site
          </span>
        </a>
      </div>
    </section>
  );
}
