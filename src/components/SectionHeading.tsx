interface SectionHeadingProps {
  label?: string;
  title: string;
  className?: string;
}

export default function SectionHeading({ label, title, className = '' }: SectionHeadingProps) {
  return (
    <div className={`mb-12 ${className}`}>
      {label && (
        <div className="flex items-center gap-3 mb-3">
          <span className="kicker">{label}</span>
          <span className="h-px flex-1 bg-border" />
        </div>
      )}

      <h2 className="text-4xl md:text-6xl uppercase leading-[0.9] text-foreground">
        {title}
      </h2>
      <div className="mt-4 rule-thick w-24" />
    </div>
  );
}
