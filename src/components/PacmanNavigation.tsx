import { useEffect, useState, useCallback } from 'react';

const sections = [
  { id: 'hero', label: 'Home', y: 0 },
  { id: 'about', label: 'About' },
  { id: 'education', label: 'Edu' },
  { id: 'techstack', label: 'Tech' },
  { id: 'certs', label: 'Certs' },
  { id: 'projects', label: 'Projects' },
  { id: 'socials', label: 'Connect' },
];

// Ghost colors
const ghosts = [
  { color: '#FF0000', name: 'Blinky', offset: -60 },
  { color: '#FFB8FF', name: 'Pinky', offset: -110 },
  { color: '#00FFFF', name: 'Inky', offset: -160 },
];

export default function PacmanNavigation() {
  const [activeSection, setActiveSection] = useState(0);
  const [scrollProgress, setScrollProgress] = useState(0);
  const [dotsEaten, setDotsEaten] = useState<Set<number>>(new Set());

  const handleScroll = useCallback(() => {
    const scrollTop = window.scrollY;
    const docHeight = document.documentElement.scrollHeight - window.innerHeight;
    const progress = docHeight > 0 ? scrollTop / docHeight : 0;
    setScrollProgress(progress);

    // Determine active section
    const sectionEls = sections.map((s) => document.getElementById(s.id));
    let current = 0;
    sectionEls.forEach((el, i) => {
      if (el) {
        const rect = el.getBoundingClientRect();
        if (rect.top <= window.innerHeight / 2) current = i;
      }
    });
    setActiveSection(current);

    // Eat dots
    const newEaten = new Set(dotsEaten);
    for (let i = 0; i <= current; i++) newEaten.add(i);
    setDotsEaten(newEaten);
  }, [dotsEaten]);

  useEffect(() => {
    window.addEventListener('scroll', handleScroll, { passive: true });
    handleScroll();
    return () => window.removeEventListener('scroll', handleScroll);
  }, [handleScroll]);

  const navigateTo = (id: string) => {
    const el = document.getElementById(id);
    if (el) el.scrollIntoView({ behavior: 'smooth' });
  };

  // Calculate pac-man position along the track
  const trackHeight = (sections.length - 1) * 64; // px between dots
  const pacY = scrollProgress * trackHeight;

  return (
    <div className="fixed right-4 md:right-8 top-1/2 -translate-y-1/2 z-40 flex flex-col items-center">
      {/* Track */}
      <div className="relative" style={{ height: `${trackHeight + 40}px` }}>
        {/* Vertical line */}
        <div className="absolute left-1/2 -translate-x-1/2 top-0 bottom-0 w-0.5 bg-primary/10" />

        {/* Pac-Man */}
        <div
          className="absolute left-1/2 -translate-x-1/2 -translate-y-1/2 z-20 transition-all duration-300 ease-out"
          style={{ top: `${pacY + 20}px` }}
        >
          <div className="pacman-character-nav w-6 h-6" />
        </div>

        {/* Ghosts following behind */}
        {ghosts.map((ghost, gi) => {
          const ghostY = Math.max(0, pacY + ghost.offset * 0.4);
          return (
            <div
              key={ghost.name}
              className="absolute left-1/2 -translate-x-1/2 -translate-y-1/2 z-10 transition-all duration-500 ease-out"
              style={{ top: `${ghostY + 20}px`, opacity: scrollProgress > 0.02 ? 0.8 : 0 }}
            >
              <div className="ghost-character" style={{ '--ghost-color': ghost.color } as React.CSSProperties}>
                <div className="ghost-eyes">
                  <div className="ghost-eye" />
                  <div className="ghost-eye" />
                </div>
              </div>
            </div>
          );
        })}

        {/* Section dots */}
        {sections.map((section, i) => {
          const dotY = i * 64;
          const isEaten = dotsEaten.has(i);
          const isActive = activeSection === i;

          return (
            <button
              key={section.id}
              onClick={() => navigateTo(section.id)}
              className="absolute left-1/2 -translate-x-1/2 z-30 group pointer-events-auto flex items-center"
              style={{ top: `${dotY + 14}px` }}
              title={section.label}
            >
              <div
                className={`w-3 h-3 rounded-full transition-all duration-300 ${
                  isEaten
                    ? isActive
                      ? 'bg-accent scale-125 shadow-[0_0_12px_hsl(51,100%,50%)]'
                      : 'bg-accent/30 scale-75'
                    : 'bg-accent/60 animate-pulse-glow'
                }`}
              />
              <span
                className={`absolute right-6 text-[10px] font-mono whitespace-nowrap transition-all duration-200 ${
                  isActive ? 'text-accent opacity-100' : 'text-muted-foreground opacity-0 group-hover:opacity-100'
                }`}
              >
                {section.label}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
