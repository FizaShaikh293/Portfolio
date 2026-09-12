import profilePhoto from '@/assets/profile-photo.png';
import SectionHeading from './SectionHeading';
import { Shield, Search, Boxes } from 'lucide-react';

const focus = [
  { icon: Shield, label: 'Security Operations' },
  { icon: Search, label: 'SIEM & Incident Response' },
  { icon: Boxes, label: 'Blockchain Security' },
];

export default function AboutSection() {
  return (
    <section id="about" className="py-24 md:py-32 px-4 sm:px-8 md:px-12 max-w-6xl mx-auto">
      <SectionHeading label="Who I Am" title="About Me" />

      <div className="grid gap-10 md:grid-cols-12 items-start">
        <div className="md:col-span-5">
          <div className="relative border-[3px] border-foreground bg-card p-2 shadow-[var(--shadow-bold)]">
            <img
              src={profilePhoto}
              alt="Fiza Shaikh"
              loading="lazy"
              className="w-full aspect-[4/5] object-cover transition-transform duration-700 hover:scale-[1.02]"
            />
          </div>
          <p className="mt-3 text-[10px] font-mono uppercase tracking-[0.24em] text-muted-foreground">
            Ireland · MSc Blockchain Technologies, First Class Honours
          </p>
        </div>

        <div className="md:col-span-7 md:border-l-[3px] md:border-foreground md:pl-10">
          <p className="text-lg md:text-xl leading-relaxed text-foreground/90 mb-7">
            I&apos;m a Cybersecurity Analyst with more than two years of experience across SOC operations,
            SIEM alert triage, vulnerability management, and identity investigations. I work with Splunk,
            Microsoft Sentinel, Microsoft Defender, Azure AD, KQL, and ServiceNow to investigate activity,
            document incidents, and support effective response.
          </p>
          <p className="mb-6 border-l-4 border-primary pl-4 text-lg font-display leading-snug text-foreground">
            My MSc in Blockchain Technologies &amp; Applications complements my security experience with
            blockchain forensics, smart contract security, cryptography, and anomaly detection.
          </p>
          <div className="flex flex-wrap gap-2">
            {focus.map(({ icon: Icon, label }) => (
              <div
                key={label}
                className="inline-flex items-center gap-2 border border-foreground/25 px-3 py-1.5 text-[10px] font-mono uppercase tracking-[0.2em] text-muted-foreground transition-colors hover:border-primary hover:text-primary cursor-default"
              >
                <Icon className="w-3.5 h-3.5 text-primary" />
                {label}
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>

  );
}
