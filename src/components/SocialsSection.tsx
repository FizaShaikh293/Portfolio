import { Linkedin, Mail, Github } from 'lucide-react';
import SectionHeading from './SectionHeading';
import ContactForm from './ContactForm';

const socials = [
  { icon: Linkedin, label: 'LinkedIn', href: 'https://www.linkedin.com/in/fizashaikh293/' },
  { icon: Github, label: 'GitHub', href: 'https://github.com/FizaShaikh293' },
  { icon: Mail, label: 'Email', href: 'mailto:shaikh.fiza13558@gmail.com' },
];

export default function SocialsSection() {
  return (
    <section id="socials" className="py-24 px-4 max-w-3xl mx-auto">
      <SectionHeading label="Contact" title="Get In Touch" />

      <ContactForm />

      <div className="flex flex-wrap justify-center gap-3 mt-8">
        {socials.map(({ icon: Icon, label, href }) => (
          <a
            key={label}
            href={href}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 rounded-full border border-border bg-muted/40 px-4 py-2 text-xs text-muted-foreground transition-all duration-300 hover:-translate-y-0.5 hover:text-foreground hover:border-primary/30"
          >
            <Icon className="w-4 h-4" />
            {label}
          </a>
        ))}
      </div>

      <div className="text-center mt-20">
        <p className="font-display text-3xl md:text-4xl font-semibold text-gradient cursor-default mb-2">
          Thank You
        </p>
        <p className="text-sm text-muted-foreground">Thanks for scrolling through my world.</p>
      </div>
    </section>
  );
}
