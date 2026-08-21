import { useState } from 'react';
import { Send, Loader2, CheckCircle2 } from 'lucide-react';

const ENDPOINT = 'https://formsubmit.co/ajax/shaikh.fiza13558@gmail.com';

export default function ContactForm() {
  const [status, setStatus] = useState<'idle' | 'sending' | 'sent' | 'error'>('idle');
  const [error, setError] = useState('');

  const onSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const form = e.currentTarget;
    const data = Object.fromEntries(new FormData(form).entries());
    setStatus('sending');
    setError('');
    try {
      const res = await fetch(ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify({ ...data, _subject: `Portfolio message from ${data.name}` }),
      });
      if (!res.ok) throw new Error('Request failed');
      setStatus('sent');
      form.reset();
    } catch {
      setStatus('error');
      setError('Something went wrong. You can email me directly at shaikh.fiza13558@gmail.com');
    }
  };

  const field =
    'w-full rounded-xl border border-white/[0.08] bg-white/[0.02] px-4 py-3 text-sm text-foreground placeholder:text-muted-foreground/70 outline-none transition-all duration-300 focus:border-primary/40 focus:bg-white/[0.04] focus:shadow-[0_0_0_3px_hsl(var(--primary)/0.08)]';

  return (
    <form onSubmit={onSubmit} className="glass-panel p-6 md:p-8 flex flex-col gap-4">
      <div className="grid sm:grid-cols-2 gap-4">
        <input name="name" required placeholder="Your name" className={field} />
        <input name="email" type="email" required placeholder="Email address" className={field} />
      </div>
      <input name="subject" placeholder="Subject (optional)" className={field} />
      <textarea name="message" required rows={5} placeholder="Tell me about the role, project or idea…" className={`${field} resize-none`} />

      <button
        type="submit"
        disabled={status === 'sending'}
        className="group inline-flex items-center justify-center gap-2 rounded-xl px-6 py-3 text-sm font-medium text-primary-foreground bg-gradient-to-r from-primary to-secondary transition-all duration-300 hover:-translate-y-0.5 hover:shadow-[0_12px_36px_-10px_hsl(var(--primary)/0.7)] disabled:opacity-60 disabled:translate-y-0"
      >
        {status === 'sending' ? (
          <Loader2 className="w-4 h-4 animate-spin" />
        ) : status === 'sent' ? (
          <CheckCircle2 className="w-4 h-4" />
        ) : (
          <Send className="w-4 h-4 transition-transform duration-300 group-hover:translate-x-0.5" />
        )}
        {status === 'sending' ? 'Sending' : status === 'sent' ? 'Message sent' : 'Send message'}
      </button>

      {status === 'sent' && (
        <p className="text-xs text-primary text-center animate-fade-in">Thanks — I'll get back to you soon.</p>
      )}
      {status === 'error' && <p className="text-xs text-destructive text-center">{error}</p>}
    </form>
  );
}
