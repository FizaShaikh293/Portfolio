import { useEffect, useState } from 'react';

export default function Loader() {
  const [done, setDone] = useState(false);
  const [hidden, setHidden] = useState(false);

  useEffect(() => {
    const t1 = setTimeout(() => setDone(true), 900);
    const t2 = setTimeout(() => setHidden(true), 1500);
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
      <p className="text-3xl text-foreground animate-unfold">Fiza Shaikh</p>
      <span className="mt-4 block h-px w-16 bg-foreground/30" />
      <p className="mt-4 text-[10px] font-mono uppercase tracking-[0.3em] text-muted-foreground">
        Portfolio
      </p>
    </div>
  );
}
