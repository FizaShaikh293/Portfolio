import Background3D from '@/components/Background3D';
import Navbar from '@/components/Navbar';
import HeroSection from '@/components/HeroSection';
import AboutSection from '@/components/AboutSection';
import EducationTimeline from '@/components/EducationTimeline';
import TechStackSection from '@/components/TechStackSection';
import CertificationsSection from '@/components/CertificationsSection';
import ProjectsSection from '@/components/ProjectsSection';
import SocialsSection from '@/components/SocialsSection';
import CustomCursor from '@/components/CustomCursor';
import PacmanNavigation from '@/components/PacmanNavigation';
import FooterTicker from '@/components/FooterTicker';

const Index = () => {
  return (
    <div className="relative min-h-screen cursor-none">
      <CustomCursor />
      <PacmanNavigation />
      <Background3D />
      <Navbar />
      <main className="relative z-10 pb-16">
        <div id="hero">
          <HeroSection />
        </div>
        <AboutSection />
        <EducationTimeline />
        <TechStackSection />
        <CertificationsSection />
        <ProjectsSection />
        <SocialsSection />
      </main>
      <FooterTicker />
    </div>
  );
};

export default Index;
