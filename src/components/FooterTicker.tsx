const headlines = [
  '🏆 Completed Certified Ethical Hacker certification',
  '🔐 Published Monero privacy research dissertation',
  '🤖 EdgePaw LLM deployed on Raspberry Pi',
  '⚡ 42-day GitHub contribution streak',
  '🛡️ Smart contract audit completed for DeFi protocol',
  '🎮 AI Car Game achieved 99.2% track completion',
  '📡 Built enterprise network simulation with 50+ nodes',
  '💛 Always learning, always building',
];

export default function FooterTicker() {
  const doubled = [...headlines, ...headlines];

  return (
    <footer className="fixed bottom-0 left-0 right-0 z-50 bg-background/80 backdrop-blur-md border-t border-primary/10">
      <div className="overflow-hidden py-2">
        <div className="flex ticker-scroll-slow whitespace-nowrap">
          {doubled.map((headline, i) => (
            <span key={i} className="inline-flex items-center mx-8 text-xs font-mono text-muted-foreground">
              <span className="text-accent mr-2">▸</span>
              {headline}
            </span>
          ))}
        </div>
      </div>
    </footer>
  );
}
