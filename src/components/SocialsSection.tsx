import { Linkedin, Mail, Github } from 'lucide-react';
import SectionHeading from './SectionHeading';
import ContactForm from './ContactForm';

const socials = [
  { icon: Linkedin, label: 'LinkedIn', href: 'https://www.linkedin.com/in/fizashaikh293/' },
  { icon: Github, label: 'GitHub', href: 'https://github.com/FizaShaikh293' },
  { icon: Mail, label: 'Email', href: 'mailto:fiza.sk293@gmail.com' },
];

export default function SocialsSection() {
  return (
    <section id="socials" className="py-24 md:py-32 px-4 sm:px-8 md:px-12 max-w-6xl mx-auto">
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

      <div className="mt-24 border-t-[6px] border-foreground pt-8 overflow-hidden">
        <p className="text-[20vw] sm:text-[17vw] md:text-[10rem] lg:text-[12rem] leading-[0.72] uppercase text-foreground">
          <span className="block">Thank</span>
          <span className="block text-primary md:ml-[14%]">You</span>
        </p>
      </div>
    </section>
  );
}
