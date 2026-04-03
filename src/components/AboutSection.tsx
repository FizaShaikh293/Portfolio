export default function AboutSection() {
  return (
    <section id="about" className="py-24 px-4 max-w-4xl mx-auto">
      <h2 className="font-display text-2xl md:text-3xl font-bold text-primary neon-glow-cyan mb-8 text-center">
        {'>'} About Me
      </h2>
      <div className="glass-panel p-6 md:p-10 neon-box-cyan">
        <p className="text-foreground leading-relaxed mb-4">
          Hi! I'm a cybersecurity &amp; blockchain enthusiast with a passion for breaking things (ethically) 
          and building them back stronger. I love diving deep into smart contract security, network defense, 
          and decentralized systems.
        </p>
        <p className="text-muted-foreground text-sm">
          <span className="text-accent">Fun fact:</span> I once debugged a smart contract vulnerability 
          while waiting for my coffee to brew. The coffee got cold — but the chain stayed secure. ☕🔐
        </p>
      </div>
    </section>
  );
}
