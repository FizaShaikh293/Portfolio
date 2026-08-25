interface SectionHeadingProps {
  label?: string;
  title: string;
  className?: string;
}

export default function SectionHeading({ label, title, className = '' }: SectionHeadingProps) {
  return (
    <div className={`flex flex-col items-center text-center mb-14 ${className}`}>
      {label && (
        <span className="mb-1 text-xl font-hand text-primary/80 rotate-[-1.5deg]">
          {label}
        </span>
      )}
      <h2 className="text-3xl md:text-5xl text-foreground">{title}</h2>
      <span className="mt-4 h-[2px] w-20 bg-primary/50 rounded-full" />
    </div>
  );
}
