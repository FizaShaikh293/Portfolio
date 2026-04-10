import { useState } from 'react';
import { Cpu, Heart, FolderSearch, TrafficCone, Network, Hotel, Smartphone, BookOpen, Bot, Radio } from 'lucide-react';

const projects = [
  {
    title: 'AI Car Game on Unity 3D',
    icon: Cpu,
    desc: 'Developed an interactive car racing game in Unity 3D, featuring AI-controlled opponents with pathfinding, obstacle avoidance, and difficulty scaling for realistic and dynamic gameplay.',
    tech: ['Unity', 'C#', 'AI', '3D'],
    glow: 'neon-box-cyan',
  },
  {
    title: 'Directory Traversal Attack Simulation',
    icon: FolderSearch,
    desc: 'Security testing project identifying directory traversal vulnerabilities in web apps using Burp Suite and PortSwigger Labs. Demonstrated how attackers exploit insecure input handling to retrieve sensitive files.',
    tech: ['Burp Suite', 'PortSwigger', 'Linux', 'Security'],
    glow: 'neon-box-purple',
  },
  {
    title: 'Hotel Management System',
    icon: Hotel,
    desc: 'Web-based PHP application managing hotel operations — room booking, check-in/check-out, staff management, and payment processing with admin and customer panels.',
    tech: ['PHP', 'MySQL', 'HTML', 'CSS'],
    glow: 'neon-box-yellow',
  },
  {
    title: 'Smart Mining with Neural Networks',
    icon: Bot,
    desc: 'Applied neural network models to optimize smart mining operations, leveraging deep learning for predictive analysis and resource optimization.',
    tech: ['Python', 'TensorFlow', 'Neural Networks'],
    glow: 'neon-box-cyan',
  },
  {
    title: 'Quiz App',
    icon: Smartphone,
    desc: 'Dynamic Android app with user authentication, interactive quizzes, real-time scoring, animated FABs, and personalized feedback through pop-up messages.',
    tech: ['Java', 'XML', 'Android Studio'],
    glow: 'neon-box-purple',
  },
  {
    title: 'Heartbeat Monitor Using Arduino',
    icon: Heart,
    desc: 'Real-time heartbeat monitoring system using Arduino Uno, Pulse Sensor, ESP8266 Wi-Fi module, and LCD display. IoT-based solution for remote patient monitoring.',
    tech: ['Arduino', 'C++', 'ESP8266', 'IoT'],
    glow: 'neon-box-yellow',
  },
  {
    title: 'Simulation of Office Network',
    icon: Network,
    desc: 'Designed and simulated a small office network using Cisco Packet Tracer with VLANs, routing protocols (RIP/OSPF), wireless security, NAT, and access control.',
    tech: ['Cisco', 'Packet Tracer', 'VLANs', 'OSPF'],
    glow: 'neon-box-cyan',
  },
  {
    title: 'Fuzzy Logic Traffic Lights',
    icon: TrafficCone,
    desc: 'Intelligent traffic light control system using Mamdani-type fuzzy inference in MATLAB. Dynamically adjusts signal timings based on traffic density and waiting time.',
    tech: ['MATLAB', 'Fuzzy Logic Toolbox', 'Simulation'],
    glow: 'neon-box-purple',
  },
  {
    title: 'Leave Management System',
    icon: BookOpen,
    desc: 'Web-based application for managing employee leave requests with role-based access (employee, manager, admin), leave tracking, and MongoDB storage.',
    tech: ['MongoDB', 'JavaScript', 'Node.js', 'Web'],
    glow: 'neon-box-yellow',
  },
  {
    title: 'Serial Communication Between Microcontrollers',
    icon: Radio,
    desc: 'UART-based serial communication between two AT90S8535 microcontrollers, simulated in Proteus 8. Sender transmits values; receiver activates corresponding LED patterns.',
    tech: ['Proteus', 'UART', 'Microcontrollers', 'C'],
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
