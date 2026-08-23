interface SectionHeadingProps {
  label?: string;
  title: string;
  className?: string;
}

export default function SectionHeading({ label, title, className = '' }: SectionHeadingProps) {
  return (
    <div className={`flex flex-col items-center text-center mb-14 ${className}`}>
      {label && (
        <span className="mb-3 text-[10px] font-mono uppercase tracking-[0.3em] text-muted-foreground">
          {label}
        </span>
      )}
      <h2 className="text-3xl md:text-5xl text-foreground">{title}</h2>
      <span className="mt-4 h-px w-16 bg-foreground/25" />
    </div>
  );
}
