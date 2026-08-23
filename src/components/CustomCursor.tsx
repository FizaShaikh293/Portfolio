import { useEffect, useState } from 'react';

export default function CustomCursor() {
  const [pos, setPos] = useState({ x: -100, y: -100 });
  const [trail, setTrail] = useState({ x: -100, y: -100 });
  const [clicking, setClicking] = useState(false);
  const [hovering, setHovering] = useState(false);

  useEffect(() => {
    const move = (e: MouseEvent) => setPos({ x: e.clientX, y: e.clientY });
    const down = () => setClicking(true);
    const up = () => setClicking(false);
    const checkHover = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      setHovering(!!target.closest('a, button, [role="button"], input, textarea, select'));
    };

    window.addEventListener('mousemove', move);
    window.addEventListener('mousemove', checkHover);
    window.addEventListener('mousedown', down);
    window.addEventListener('mouseup', up);
    return () => {
      window.removeEventListener('mousemove', move);
      window.removeEventListener('mousemove', checkHover);
      window.removeEventListener('mousedown', down);
      window.removeEventListener('mouseup', up);
    };
  }, []);

  useEffect(() => {
    let raf: number;
    const follow = () => {
      setTrail((prev) => ({
        x: prev.x + (pos.x - prev.x) * 0.16,
        y: prev.y + (pos.y - prev.y) * 0.16,
      }));
      raf = requestAnimationFrame(follow);
    };
    raf = requestAnimationFrame(follow);
    return () => cancelAnimationFrame(raf);
  }, [pos]);

  return (
    <>
      <div
        className="fixed top-0 left-0 pointer-events-none z-[9999]"
        style={{
          transform: `translate(${pos.x - 3}px, ${pos.y - 3}px) scale(${clicking ? 0.6 : hovering ? 1.5 : 1})`,
          transition: 'transform 0.12s ease-out',
        }}
      >
        <div className="w-1.5 h-1.5 rounded-full bg-foreground" />
      </div>

      <div
        className="fixed top-0 left-0 pointer-events-none z-[9998]"
        style={{
          transform: `translate(${trail.x - 16}px, ${trail.y - 16}px) scale(${clicking ? 0.7 : hovering ? 1.4 : 1})`,
          transition: 'transform 0.2s ease-out',
        }}
      >
        <div
          className="w-8 h-8 rounded-full border"
          style={{ borderColor: hovering ? 'hsl(var(--primary) / 0.6)' : 'hsl(var(--foreground) / 0.25)' }}
        />
      </div>
    </>
  );
}
