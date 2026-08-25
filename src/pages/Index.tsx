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

const Index = () => {
  return (
    <div className="relative min-h-screen cursor-none">
      <Loader />
      <CustomCursor />
      <PacmanNavigation />
      <Navbar />
      <main className="relative z-10">
        <div id="hero">
          <HeroSection />
        </div>
        <Reveal from="left"><AboutSection /></Reveal>
        <Reveal from="right"><WorkExperience /></Reveal>
        <Reveal from="left"><TechStackSection /></Reveal>
        <Reveal from="right"><CertificationsSection /></Reveal>
        <Reveal from="left"><ProjectsSection /></Reveal>
        <Reveal from="up"><SocialsSection /></Reveal>
      </main>
    </div>
  );
};

export default Index;
