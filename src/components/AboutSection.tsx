import profilePhoto from '@/assets/profile-photo.png';

export default function AboutSection() {
  return (
    <section id="about" className="py-24 px-4 max-w-4xl mx-auto">
      <h2 className="font-display text-2xl md:text-3xl font-bold text-primary neon-glow-cyan mb-8 text-center">
        {'>'} About Me
      </h2>
      <div className="glass-panel p-6 md:p-10 neon-box-cyan">
        <div className="flex flex-col md:flex-row gap-6 items-center md:items-start">
          <div className="shrink-0">
            <div className="w-36 h-36 rounded-full overflow-hidden border-2 border-primary/40 shadow-[0_0_20px_hsl(var(--primary)/0.3)]">
              <img
                src={profilePhoto}
                alt="Fiza Shaikh"
                className="w-full h-full object-cover"
              />
            </div>
          </div>
          <div>
            <p className="text-foreground leading-relaxed mb-4">
              Hi! I'm a cybersecurity &amp; blockchain enthusiast with a passion for breaking things (ethically) 
              and building them back stronger. I love diving deep into smart contract security, network defense, 
              and decentralized systems. I'm also deeply interested in AI and Machine Learning, exploring how 
              intelligent systems can enhance security and solve complex problems.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
