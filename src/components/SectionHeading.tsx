interface SectionHeadingProps {
  label?: string;
  title: string;
  className?: string;
}

export default function SectionHeading({ label, title, className = '' }: SectionHeadingProps) {
  return (
    <div className={`mb-14 ${className}`}>
      <div className="flex items-center gap-3">
        {label && (
          <span className="chip">
            <span className="w-1.5 h-1.5 rounded-full bg-primary" />
            {label}
          </span>
        )}
        <span className="h-px flex-1 bg-border" />
      </div>

      <h2 className="mt-4 text-4xl md:text-5xl text-foreground">
        {title}
      </h2>
    </div>
  );
}
