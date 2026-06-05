import Background3D from '@/components/Background3D';
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

const Index = () => {
  return (
    <div className="relative min-h-screen cursor-none">
      <Loader />
      <CustomCursor />
      <PacmanNavigation />
      <Background3D />
      <Navbar />
      <main className="relative z-10">
        <div id="hero">
          <HeroSection />
        </div>
        <AboutSection />
        <WorkExperience />
        
        <TechStackSection />
        <CertificationsSection />
        <ProjectsSection />
        <SocialsSection />
      </main>
    </div>
  );
};

export default Index;
