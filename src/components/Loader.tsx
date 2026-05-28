import { useEffect, useState } from 'react';

export default function Loader() {
  const [done, setDone] = useState(false);
  const [hidden, setHidden] = useState(false);

  useEffect(() => {
    const t1 = setTimeout(() => setDone(true), 1100);
    const t2 = setTimeout(() => setHidden(true), 1700);
    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
    };
  }, []);

  if (hidden) return null;

  return (
    <div
      className={`fixed inset-0 z-[10000] flex flex-col items-center justify-center bg-background transition-opacity duration-500 ${
        done ? 'opacity-0 pointer-events-none' : 'opacity-100'
      }`}
    >
      <div className="relative w-16 h-16">
        <div className="absolute inset-0 rounded-full border-2 border-white/10" />
        <div
          className="absolute inset-0 rounded-full border-2 border-transparent border-t-primary border-r-secondary"
          style={{ animation: 'loader-spin 0.9s linear infinite' }}
        />
      </div>
      <p className="mt-6 font-display text-sm font-medium tracking-[0.3em] uppercase text-gradient">
        Fiza Shaikh
      </p>
    </div>
  );
}
