interface SectionHeadingProps {
  label?: string;
  title: string;
  className?: string;
}

export default function SectionHeading({ label, title, className = '' }: SectionHeadingProps) {
  return (
    <div className={`mb-14 ${className}`}>
      <div className="flex items-center gap-3">
        <span className="h-px w-8 bg-primary" />
        {label && (
          <span className="text-[10px] font-mono uppercase tracking-[0.3em] text-primary">
            {label}
          </span>
        )}
        <span className="h-px flex-1 bg-border" />
      </div>

      <h2 className="mt-3 text-4xl md:text-5xl text-foreground">
        {title}
      </h2>
    </div>
  );
}
