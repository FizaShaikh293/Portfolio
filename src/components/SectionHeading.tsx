interface SectionHeadingProps {
  label?: string;
  title: string;
  className?: string;
}

export default function SectionHeading({ label, title, className = '' }: SectionHeadingProps) {
  return (
    <div className={`mb-12 md:mb-16 ${className}`}>
      {label && (
        <div className="mb-4 flex items-center gap-4">
          <span className="block-rust px-3 py-2 text-[10px] font-mono uppercase tracking-[0.24em]">{label}</span>
          <span className="h-[3px] flex-1 bg-foreground" />
        </div>
      )}

      <h2 className="max-w-5xl text-5xl sm:text-6xl md:text-8xl uppercase leading-[0.82] text-foreground">
        {title}
      </h2>
      <div className="mt-6 grid grid-cols-[minmax(5rem,1fr)_3fr] gap-3" aria-hidden="true">
        <span className="h-2 bg-primary" />
        <span className="h-2 bg-foreground" />
      </div>
    </div>
  );
}
