import { useEffect, useRef, useState } from 'react';
import { GraduationCap } from 'lucide-react';

const education = [
  {
    year: '2017 – 2019',
    degree: 'High School (HSC)',
    institution: 'Junior College',
    description: 'Completed higher secondary education with a focus on Science and Mathematics.',
    color: 'neon-cyan',
  },
  {
    year: '2019 – 2022',
    degree: "Bachelor's in Computer Science",
    institution: 'University',
    description: 'Studied core CS fundamentals including data structures, algorithms, networking, and software engineering.',
    color: 'neon-purple',
  },
  {
    year: '2022 – 2023',
    degree: "Master's in Cybersecurity",
    institution: 'University',
    description: 'Specialized in network security, cryptography, blockchain security, and ethical hacking.',
    color: 'neon-yellow',
  },
];

export default function EducationTimeline() {
  const [visibleItems, setVisibleItems] = useState<number[]>([]);
  const [pacmanPos, setPacmanPos] = useState(0);
  const sectionRef = useRef<HTMLElement>(null);
  const itemRefs = useRef<(HTMLDivElement | null)[]>([]);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const idx = Number(entry.target.getAttribute('data-index'));
            setVisibleItems((prev) => {
              if (!prev.includes(idx)) {
                const next = [...prev, idx].sort((a, b) => a - b);
                setPacmanPos(idx);
                return next;
              }
              return prev;
            });
          }
        });
      },
      { threshold: 0.4 }
    );

    itemRefs.current.forEach((ref) => {
      if (ref) observer.observe(ref);
    });

    return () => observer.disconnect();
  }, []);

  const dotColors = ['bg-primary', 'bg-secondary', 'bg-accent'];

  return (
    <section id="education" ref={sectionRef} className="py-24 px-4 max-w-4xl mx-auto">
      <h2 className="font-display text-2xl md:text-3xl font-bold text-primary neon-glow-cyan mb-16 text-center">
        {'>'} Education
      </h2>

      <div className="relative">
        {/* Vertical line */}
        <div className="absolute left-6 md:left-1/2 top-0 bottom-0 w-0.5 bg-primary/20 md:-translate-x-px" />

        {/* Pac-Man character on the line */}
        <div
          className="absolute left-6 md:left-1/2 -translate-x-1/2 z-20 transition-all duration-700 ease-in-out"
          style={{ top: `${pacmanPos * 280 + 20}px` }}
        >
          <div className="relative w-10 h-10">
            <div className="pacman-character" />
          </div>
        </div>

        {/* Dots along the line (pac-man food) */}
        {education.map((_, i) => (
          <div
            key={`dot-${i}`}
            className={`absolute left-6 md:left-1/2 -translate-x-1/2 w-3 h-3 rounded-full transition-all duration-500 ${
              visibleItems.includes(i) ? 'opacity-0 scale-0' : `${dotColors[i]} opacity-80 animate-pulse-glow`
            }`}
            style={{ top: `${i * 280 + 28}px` }}
          />
        ))}

        {/* Timeline items */}
        {education.map((item, i) => {
          const isLeft = i % 2 === 0;
          const isVisible = visibleItems.includes(i);

          return (
            <div
              key={i}
              ref={(el) => { itemRefs.current[i] = el; }}
              data-index={i}
              className="relative mb-20"
              style={{ minHeight: '200px' }}
            >
              {/* Connector dot */}
              <div
                className={`absolute left-6 md:left-1/2 -translate-x-1/2 w-5 h-5 rounded-full border-2 z-10 transition-all duration-500 ${
                  isVisible
                    ? `border-${item.color === 'neon-cyan' ? 'primary' : item.color === 'neon-purple' ? 'secondary' : 'accent'} bg-background scale-100`
                    : 'border-muted bg-background scale-75 opacity-50'
                }`}
                style={{ top: '24px' }}
              >
                <GraduationCap className={`w-3 h-3 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 ${
                  item.color === 'neon-cyan' ? 'text-primary' : item.color === 'neon-purple' ? 'text-secondary' : 'text-accent'
                }`} />
              </div>

              {/* Card */}
              <div
                className={`ml-16 md:ml-0 md:w-5/12 ${
                  isLeft ? 'md:mr-auto md:pr-12' : 'md:ml-auto md:pl-12'
                } transition-all duration-700 ${
                  isVisible
                    ? 'opacity-100 translate-y-0'
                    : `opacity-0 ${isLeft ? '-translate-x-8' : 'translate-x-8'} translate-y-4`
                }`}
              >
                <div className={`glass-panel p-6 ${
                  item.color === 'neon-cyan' ? 'neon-box-cyan' : item.color === 'neon-purple' ? 'neon-box-purple' : 'neon-box-yellow'
                } hover:scale-[1.03] transition-transform duration-300 group`}>
                  <span className={`text-xs font-mono ${
                    item.color === 'neon-cyan' ? 'text-primary' : item.color === 'neon-purple' ? 'text-secondary' : 'text-accent'
                  } mb-2 block`}>
                    {item.year}
                  </span>
                  <h3 className="font-display text-lg font-bold text-foreground mb-1">
                    {item.degree}
                  </h3>
                  <p className="text-sm text-muted-foreground mb-2 italic">{item.institution}</p>
                  <p className="text-sm text-muted-foreground leading-relaxed">{item.description}</p>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
