import Background3D from '@/components/Background3D';
import Navbar from '@/components/Navbar';
import HeroSection from '@/components/HeroSection';
import AboutSection from '@/components/AboutSection';
import EducationTimeline from '@/components/EducationTimeline';
import CertificationsSection from '@/components/CertificationsSection';
import ProjectsSection from '@/components/ProjectsSection';
import SocialsSection from '@/components/SocialsSection';
import CustomCursor from '@/components/CustomCursor';

const Index = () => {
  return (
    <div className="relative min-h-screen cursor-none">
      <CustomCursor />
      <Background3D />
      <Navbar />
      <main className="relative z-10 pb-12">
        <HeroSection />
        <AboutSection />
        <EducationTimeline />
        <CertificationsSection />
        <ProjectsSection />
        <SocialsSection />
      </main>
    </div>
  );
};

export default Index;
