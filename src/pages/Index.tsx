import Background3D from '@/components/Background3D';
import Navbar from '@/components/Navbar';
import HeroSection from '@/components/HeroSection';
import AboutSection from '@/components/AboutSection';
import CertificationsSection from '@/components/CertificationsSection';
import ProjectsSection from '@/components/ProjectsSection';
import SocialsSection from '@/components/SocialsSection';
import GitHubStats from '@/components/GitHubStats';


const Index = () => {
  return (
    <div className="relative min-h-screen">
      <Background3D />
      <Navbar />
      <main className="relative z-10 pb-12">
        <HeroSection />
        <AboutSection />
        <CertificationsSection />
        <ProjectsSection />
        <GitHubStats />
        <SocialsSection />
      </main>
      
    </div>
  );
};

export default Index;
