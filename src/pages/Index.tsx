import Navbar from '@/components/Navbar';
import HeroSection from '@/components/HeroSection';
import AboutSection from '@/components/AboutSection';
import WorkExperience from '@/components/WorkExperience';
import TechStackSection from '@/components/TechStackSection';
import CertificationsSection from '@/components/CertificationsSection';
import ProjectsSection from '@/components/ProjectsSection';
import SocialsSection from '@/components/SocialsSection';
import CustomCursor from '@/components/CustomCursor';
import PacmanNavigation from '@/components/PacmanNavigation';
import Loader from '@/components/Loader';
import Reveal from '@/components/Reveal';

const chapters = [
  { n: 'I', title: 'About', node: <AboutSection /> },
  { n: 'II', title: 'Experience', node: <WorkExperience /> },
  { n: 'III', title: 'Tech Stack', node: <TechStackSection /> },
  { n: 'IV', title: 'Certifications', node: <CertificationsSection /> },
  { n: 'V', title: 'Projects', node: <ProjectsSection /> },
  { n: 'VI', title: 'Contact', node: <SocialsSection /> },
];

const Index = () => {
  return (
    <div className="relative min-h-screen cursor-none">
      <Loader />
      <CustomCursor />
      <PacmanNavigation />
      <Navbar />
      <main className="relative z-10 px-3 sm:px-6 lg:px-10 pb-16">
        <div className="book-shell relative mx-auto max-w-6xl">
          <span className="book-margin hidden md:block" />

          <div id="hero">
            <HeroSection />
          </div>

          {chapters.map((c, i) => (
            <Reveal key={c.title} from={i % 2 === 0 ? 'left' : 'right'}>
              <div className="relative border-t border-border/70">
                <div className="pt-10 pl-4 sm:pl-8 md:pl-16">
                  <span className="chapter-tab">
                    <span className="font-mono text-[9px] uppercase tracking-[0.28em] text-primary">
                      Chapter {c.n}
                    </span>
                    <span className="font-display text-sm text-foreground">{c.title}</span>
                  </span>
                </div>
                {c.node}
                <div className="flex items-center justify-center gap-3 pb-8">
                  <span className="h-px w-10 bg-border" />
                  <p className="folio text-sm">{i + 2}</p>
                  <span className="h-px w-10 bg-border" />
                </div>
              </div>
            </Reveal>
          ))}

        </div>
      </main>
    </div>
  );
};

export default Index;
