import React, { useState, useRef } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { useGSAP } from '@gsap/react';
import { CheckCircle2 } from 'lucide-react';

gsap.registerPlugin(ScrollTrigger, useGSAP);

export const ContactSection = () => {
  const containerRef = useRef(null);
  const [submitted, setSubmitted] = useState(false);
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [message, setMessage] = useState('');

  useGSAP(
    () => {
      gsap.fromTo(
        '.contact-rise',
        { opacity: 0, y: 30 },
        {
          opacity: 1,
          y: 0,
          duration: 0.8,
          stagger: 0.15,
          ease: 'power3.out',
          scrollTrigger: {
            trigger: containerRef.current,
            start: 'top 80%',
          },
        }
      );
    },
    { scope: containerRef }
  );

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!message.trim()) return;
    setSubmitted(true);
  };

  return (
    <section
      id="contact"
      ref={containerRef}
      className="scene scroll-mt-20 py-20 px-6 sm:px-10 lg:px-12 max-w-7xl mx-auto"
    >
      <div className="stage max-w-5xl">
        {/* Altnota Category Tag */}
        <p className="contact-rise font-mono text-xs uppercase tracking-[0.18em] text-black font-semibold mb-6 flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-[#FFE500]" />
          contact
        </p>

        {/* Section Headline */}
        <div className="contact-rise mb-12">
          <h2 className="poster-md max-w-3xl">
            Leave your mark
          </h2>
          <p className="mt-6 text-lg sm:text-xl text-soft font-display leading-relaxed max-w-2xl">
            You don’t need to know the ordinance section number. Tell us what you’re looking at, and we’ll cite what the university bylaws say.
          </p>
        </div>

        {/* Altnota Clean Contact Box */}
        <div className="contact-rise max-w-3xl border border-rule p-8 sm:p-12 rounded-2xl bg-white shadow-xs">
          {submitted ? (
            <div className="py-8 text-center space-y-4">
              <div className="w-12 h-12 rounded-full bg-[#FFE500] text-black flex items-center justify-center mx-auto">
                <CheckCircle2 className="w-6 h-6 text-black" />
              </div>
              <h3 className="font-display font-bold text-2xl text-black">
                Query received by NEXi team.
              </h3>
              <p className="text-soft font-display max-w-md mx-auto text-sm">
                If this involves an unindexed ordinance, our knowledge curation team will evaluate and ingest it into the dual-RAG vector store.
              </p>
              <button
                onClick={() => setSubmitted(false)}
                className="font-mono text-xs text-black u-ink pt-2 cursor-pointer"
              >
                Submit another inquiry
              </button>
            </div>
          ) : (
            <form onSubmit={handleSubmit} className="space-y-8">
              {/* Name */}
              <div>
                <label
                  htmlFor="contact-name"
                  className="block font-mono text-xs uppercase tracking-wider text-soft mb-2"
                >
                  Your name <span className="text-black">*</span>
                </label>
                <input
                  id="contact-name"
                  type="text"
                  required
                  placeholder="Student or Faculty name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="w-full bg-transparent border-b border-rule focus:border-black py-2.5 text-base font-display text-ink focus:outline-none transition-colors placeholder:text-dim"
                />
              </div>

              {/* Email */}
              <div>
                <label
                  htmlFor="contact-email"
                  className="block font-mono text-xs uppercase tracking-wider text-soft mb-2"
                >
                  Your SRM email <span className="text-black">*</span>
                </label>
                <input
                  id="contact-email"
                  type="email"
                  required
                  placeholder="e.g. rohan_sharma@srmap.edu.in"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full bg-transparent border-b border-rule focus:border-black py-2.5 text-base font-display text-ink focus:outline-none transition-colors placeholder:text-dim"
                />
              </div>

              {/* Message */}
              <div>
                <label
                  htmlFor="contact-message"
                  className="block font-mono text-xs uppercase tracking-wider text-soft mb-2"
                >
                  What policy or rule should we look at? <span className="text-black">*</span>
                </label>
                <textarea
                  id="contact-message"
                  rows={3}
                  required
                  placeholder="A question, a course policy, or an ordinance you want verified..."
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  className="w-full bg-transparent border-b border-rule focus:border-black py-2.5 text-base font-display text-ink focus:outline-none transition-colors placeholder:text-dim resize-none"
                />
              </div>

              {/* Bright Yellow Button */}
              <div>
                <button
                  type="submit"
                  className="rounded-full bg-[#FFE500] hover:bg-[#F2D800] px-10 py-3.5 font-display font-bold text-black transition-transform hover:-translate-y-0.5 active:translate-y-0 shadow-xs cursor-pointer text-sm"
                >
                  Send the query
                </button>
              </div>
            </form>
          )}
        </div>

        {/* Altnota Privacy Micro-copy */}
        <div className="contact-rise mt-6 max-w-3xl space-y-1 text-xs text-soft font-display leading-relaxed">
          <p>You’ll hear from the campus intelligence team usually within 24 hours.</p>
          <p>Whatever you send stays strictly confidential within SRM University-AP.</p>
        </div>
      </div>
    </section>
  );
};
