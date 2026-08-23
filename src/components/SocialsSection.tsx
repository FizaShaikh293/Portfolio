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

      <div className="flex flex-wrap justify-center gap-6 mt-10">
        {socials.map(({ icon: Icon, label, href }) => (
          <a
            key={label}
            href={href}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 text-[11px] font-mono uppercase tracking-[0.18em] text-muted-foreground transition-colors duration-300 hover:text-foreground ink-underline"
          >
            <Icon className="w-4 h-4" />
            {label}
          </a>
        ))}
      </div>

      <div className="text-center mt-24">
        <p className="text-4xl md:text-5xl italic text-foreground mb-3">Thank You</p>
        <span className="mx-auto block h-px w-16 bg-foreground/25" />
        <p className="mt-4 text-[11px] font-mono uppercase tracking-[0.2em] text-muted-foreground">
          Thanks for scrolling through my world
        </p>
      </div>
    </section>
  );
}
