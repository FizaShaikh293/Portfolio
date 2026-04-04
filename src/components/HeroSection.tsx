import { useEffect, useState } from 'react';

export default function HeroSection() {
  const [visible, setVisible] = useState(false);
  const [neonOn, setNeonOn] = useState(false);
  const [flickerCount, setFlickerCount] = useState(0);

  useEffect(() => {
    setVisible(true);
    // Neon sign flicker effect - blinks a few times then stays on
    const flickerSequence = [300, 200, 150, 100, 200, 150, 100];
    let timeout: ReturnType<typeof setTimeout>;
    let i = 0;
    const flicker = () => {
      if (i < flickerSequence.length) {
        setNeonOn((prev) => !prev);
        setFlickerCount(i);
        timeout = setTimeout(() => { i++; flicker(); }, flickerSequence[i]);
      } else {
        setNeonOn(true);
      }
    };
    timeout = setTimeout(flicker, 800);
    return () => clearTimeout(timeout);
  }, []);

  return (
    <section className="relative min-h-screen flex flex-col items-center justify-center px-4">
      <div className={`text-center transition-all duration-1000 ${visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>
        <h1
          className={`font-display text-4xl sm:text-5xl md:text-7xl font-bold leading-tight mb-4 transition-all duration-100 ${
            neonOn
              ? 'text-accent neon-glow-yellow'
              : 'text-accent/20'
          }`}
        >
          FIZA SHAIKH
        </h1>
        <p
          className={`font-mono text-sm md:text-base mb-4 italic transition-all duration-100 ${
            neonOn
              ? 'text-primary neon-glow-cyan'
              : 'text-primary/20'
          }`}
        >
          "Despite everything, it's still you"
        </p>
        <p className="font-mono text-muted-foreground text-xs sm:text-sm md:text-base whitespace-nowrap">
          Security Researcher · Blockchain Builder · Eternal Learner
        </p>
      </div>
    </section>
  );
}
