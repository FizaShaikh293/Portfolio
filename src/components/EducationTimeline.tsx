import { GraduationCap } from 'lucide-react';
import SectionHeading from './SectionHeading';

const education = [
  {
    year: 'Jan 2025 – Apr 2026',
    degree: 'MSc in Blockchain Technologies and Application',
    institution: 'Atlantic Technological University',
    description: 'Grade: First Class Honours (1:1)',
    color: 'primary',
  },
  {
    year: '2020 – 2023',
    degree: 'Bachelor of Science in Information Technology',
    institution: 'Jai Hind College, Mumbai',
    description: '',
    color: 'secondary',
  },
  {
    year: '',
    degree: 'HSC, Business/Commerce, General',
    institution: 'Thakur College of Science & Commerce',
    description: '',
    color: 'accent',
  },
];

export default function EducationTimeline() {
  return (
    <section id="education" className="py-24 px-4 max-w-4xl mx-auto">
      <SectionHeading label="Academic Journey" title="Education" />

      <div className="relative">
        <div className="absolute left-6 top-0 bottom-0 w-px bg-gradient-to-b from-primary via-secondary to-accent" />

        <div className="space-y-8">
          {education.map((item, i) => {
            const colorText = item.color === 'primary' ? 'text-primary' : item.color === 'secondary' ? 'text-secondary' : 'text-accent';
            const glow = item.color === 'primary' ? 'neon-box-cyan' : item.color === 'secondary' ? 'neon-box-purple' : 'neon-box-yellow';
            return (
              <div key={i} className="relative pl-16 animate-fade-up" style={{ animationDelay: `${i * 120}ms` }}>
                <div className={`absolute left-3 top-2 w-6 h-6 rounded-full border-2 ${item.color === 'primary' ? 'border-primary' : item.color === 'secondary' ? 'border-secondary' : 'border-accent'} bg-background flex items-center justify-center z-10`}>
                  <GraduationCap className={`w-3 h-3 ${colorText}`} />
                </div>
                <div className={`glass-panel p-6 ${glow} hover:scale-[1.02] transition-transform duration-300`}>
                  {item.year && (
                    <span className={`text-xs font-mono ${colorText} mb-2 block`}>{item.year}</span>
                  )}
                  <h3 className="font-display text-lg font-bold text-foreground hover-text-pop cursor-default mb-1">{item.degree}</h3>
                  <p className="text-sm text-muted-foreground italic">{item.institution}</p>
                  {item.description && (
                    <p className="text-sm text-muted-foreground leading-relaxed mt-2">{item.description}</p>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
