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
                {c.node}
              </div>
            </Reveal>
          ))}

        </div>
      </main>
    </div>
  );
};

export default Index;
