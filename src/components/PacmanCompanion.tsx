import { useEffect, useState } from 'react';

export default function PacmanCompanion() {
  const [scrollY, setScrollY] = useState(0);
  const [dots, setDots] = useState<{ id: number; y: number; eaten: boolean }[]>([]);

  useEffect(() => {
    // Generate dots along the page
    const generateDots = () => {
      const pageHeight = document.documentElement.scrollHeight;
      const count = Math.floor(pageHeight / 200);
      return Array.from({ length: count }, (_, i) => ({
        id: i,
        y: (i + 1) * 200,
        eaten: false,
      }));
    };
    setDots(generateDots());

    const handleScroll = () => {
      const y = window.scrollY + window.innerHeight / 2;
      setScrollY(y);
      setDots((prev) =>
        prev.map((d) => ({ ...d, eaten: d.y < y }))
      );
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <div className="fixed left-4 md:left-8 top-0 bottom-0 z-30 pointer-events-none">
      {/* Track line */}
      <div className="absolute left-3 top-0 bottom-0 w-px bg-primary/10" />

      {/* Dots */}
      {dots.map((dot) => (
        <div
          key={dot.id}
          className={`absolute left-1 w-3 h-3 rounded-full transition-all duration-300 ${
            dot.eaten ? 'scale-0 opacity-0' : 'bg-accent/60 scale-100 opacity-100'
          }`}
          style={{ top: `${(dot.y / document.documentElement.scrollHeight) * 100}vh` }}
        />
      ))}

      {/* Pac-Man */}
      <div
        className="absolute left-0 -translate-y-1/2 transition-none"
        style={{
          top: `${(scrollY / document.documentElement.scrollHeight) * 100}%`,
        }}
      >
        <div className="pacman-character w-7 h-7" />
      </div>
    </div>
  );
}
