interface SectionHeadingProps {
  label?: string;
  title: string;
  className?: string;
}

export default function SectionHeading({ label, title, className = '' }: SectionHeadingProps) {
  return (
    <div className={`flex flex-col items-center text-center mb-14 ${className}`}>
      {label && (
        <span className="mb-3 rounded-full border border-white/10 bg-white/[0.03] px-3.5 py-1 text-[10px] font-medium tracking-[0.22em] uppercase text-muted-foreground backdrop-blur-md">
          {label}
        </span>
      )}
      <h2 className="font-display text-3xl md:text-4xl font-bold text-gradient hover-text-pop cursor-default">
        {title}
      </h2>
    </div>
  );
}
