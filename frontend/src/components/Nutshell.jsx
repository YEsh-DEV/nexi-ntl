import React, { useRef } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { useGSAP } from '@gsap/react';

gsap.registerPlugin(ScrollTrigger, useGSAP);

export const Nutshell = () => {
  const containerRef = useRef(null);

  useGSAP(
    () => {
      gsap.fromTo(
        '.nutshell-rise',
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

  return (
    <section
      id="nutshell"
      ref={containerRef}
      className="scene scroll-mt-20 py-12 sm:py-20 px-4 sm:px-8 lg:px-12 max-w-7xl mx-auto"
    >
      <div className="stage max-w-5xl">
        {/* Altnota Category Tag */}
        <p className="nutshell-rise font-mono text-xs uppercase tracking-[0.18em] text-black font-semibold mb-6 flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-[#FFE500]" />
          Us in a nutshell
        </p>

        {/* Section Headline & Lead */}
        <div className="nutshell-rise mb-14">
          <h2 className="poster-md max-w-3xl">
            We shape institutional ordinances into instant, trustworthy voice intelligence.
          </h2>
          <p className="mt-6 text-lg sm:text-xl text-soft font-display leading-relaxed max-w-2xl">
            For students navigating academic life, and for faculty managing administrative guidelines.
          </p>
        </div>

        {/* Two-Column Grid: How we think / What we make */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-12 lg:gap-16 pt-8 border-t border-rule max-w-4xl">
          {/* Column 1: How we think */}
          <div className="nutshell-rise flex flex-col justify-between">
            <div>
              <h3 className="font-mono text-xs uppercase tracking-wider text-soft mb-4">
                How we think
              </h3>
              <p className="text-base text-ink font-display leading-relaxed">
                We’d rather find out from official ordinances than take a good guess. Every policy inquiry has a window where misinformation is costly: attendance condonation, hostel curfew, grading regulations. That’s where we put the work: indexing 817 chunks, calibrated cross-encoder gating, and testing every query while being wrong costs nothing.
              </p>
            </div>
            <p className="mt-6">
              <a
                href="#sandbox"
                className="u-ink font-display font-semibold text-base text-black inline-flex items-center gap-1 group"
              >
                <span>Explore the policy engine</span>
                <span className="group-hover:translate-x-1 transition-transform">→</span>
              </a>
            </p>
          </div>

          {/* Column 2: What we make */}
          <div className="nutshell-rise flex flex-col justify-between">
            <div>
              <h3 className="font-mono text-xs uppercase tracking-wider text-soft mb-4">
                What we make
              </h3>
              <p className="text-base text-ink font-display leading-relaxed">
                Conversational voice in three Indian languages: English, Telugu (తెలుగు), and Hindi (हिन्दी). Available through WebRTC in browser and telephone DTMF dial-in. Built from 11 official SRM-AP regulations with 100% citation transparency and zero hallucination.
              </p>
            </div>
            <p className="mt-6">
              <a
                href="#how-we-work"
                className="u-ink font-display font-semibold text-base text-black inline-flex items-center gap-1 group"
              >
                <span>See how it works</span>
                <span className="group-hover:translate-x-1 transition-transform">→</span>
              </a>
            </p>
          </div>
        </div>

        {/* Altnota Signature Punchline Statement */}
        <div className="nutshell-rise mt-20 pt-10 border-t border-rule">
          <p className="statement-thin">
            Fast is nice. Right is better.
          </p>
        </div>
      </div>
    </section>
  );
};
