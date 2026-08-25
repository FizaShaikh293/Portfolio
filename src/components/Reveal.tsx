import { useEffect, useRef, useState, type ReactNode } from 'react';

interface RevealProps {
  children: ReactNode;
  from?: 'left' | 'right' | 'up';
  delay?: number;
  className?: string;
}

export default function Reveal({ children, from = 'up', delay = 0, className = '' }: RevealProps) {
  const ref = useRef<HTMLDivElement>(null);
  const [shown, setShown] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (typeof IntersectionObserver === 'undefined') {
      setShown(true);
      return;
    }
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            setShown(true);
            io.disconnect();
          }
        });
      },
      { threshold: 0.08, rootMargin: '0px 0px -8% 0px' }
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);

  const hidden =
    from === 'left'
      ? 'opacity-0 -translate-x-10 rotate-[-1.2deg]'
      : from === 'right'
        ? 'opacity-0 translate-x-10 rotate-[1.2deg]'
        : 'opacity-0 translate-y-10';

  return (
    <div
      ref={ref}
      style={{ transitionDelay: `${delay}ms` }}
      className={`transition-all duration-[900ms] [transition-timing-function:cubic-bezier(0.22,1,0.36,1)] ${
        shown ? 'opacity-100 translate-x-0 translate-y-0 rotate-0' : hidden
      } ${className}`}
    >
      {children}
    </div>
  );
}
