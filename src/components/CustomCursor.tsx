import { useEffect, useState } from 'react';

export default function CustomCursor() {
  const [pos, setPos] = useState({ x: -100, y: -100 });
  const [trail, setTrail] = useState({ x: -100, y: -100 });
  const [clicking, setClicking] = useState(false);
  const [hovering, setHovering] = useState(false);

  useEffect(() => {
    const move = (e: MouseEvent) => {
      setPos({ x: e.clientX, y: e.clientY });
    };
    const down = () => setClicking(true);
    const up = () => setClicking(false);

    const checkHover = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      const isHoverable = target.closest('a, button, [role="button"], input, textarea, select, .glass-panel');
      setHovering(!!isHoverable);
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
        x: prev.x + (pos.x - prev.x) * 0.15,
        y: prev.y + (pos.y - prev.y) * 0.15,
      }));
      raf = requestAnimationFrame(follow);
    };
    raf = requestAnimationFrame(follow);
    return () => cancelAnimationFrame(raf);
  }, [pos]);

  return (
    <>
      {/* Main cursor dot */}
      <div
        className="fixed top-0 left-0 pointer-events-none z-[9999]"
        style={{
          transform: `translate(${pos.x - 4}px, ${pos.y - 4}px) scale(${clicking ? 0.6 : hovering ? 1.6 : 1})`,
          transition: 'transform 0.15s ease-out',
        }}
      >
        <div
          className="w-2 h-2 rounded-full bg-primary"
          style={{
            boxShadow: '0 0 10px hsl(var(--primary) / 0.8), 0 0 20px hsl(var(--primary) / 0.4)',
          }}
        />
      </div>

      {/* Trail ring */}
      <div
        className="fixed top-0 left-0 pointer-events-none z-[9998]"
        style={{
          transform: `translate(${trail.x - 18}px, ${trail.y - 18}px) scale(${clicking ? 0.7 : hovering ? 1.6 : 1})`,
          transition: 'transform 0.2s ease-out',
        }}
      >
        <div
          className="w-9 h-9 rounded-full border"
          style={{
            borderColor: hovering ? 'hsl(var(--primary) / 0.7)' : 'hsl(var(--silver) / 0.4)',
            boxShadow: '0 0 18px hsl(var(--primary) / 0.15)',
            transition: 'border-color 0.2s ease-out',
          }}
        />
      </div>
    </>
  );
}

}
