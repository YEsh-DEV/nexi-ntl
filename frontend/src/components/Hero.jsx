import React, { useRef } from 'react';
import gsap from 'gsap';
import { useGSAP } from '@gsap/react';

gsap.registerPlugin(useGSAP);

export const Hero = () => {
  const containerRef = useRef(null);

  useGSAP(
    () => {
      const tl = gsap.timeline({ defaults: { ease: 'power3.out' } });

      tl.fromTo(
        '.hero-tag',
        { opacity: 0, y: 15 },
        { opacity: 1, y: 0, duration: 0.6 }
      )
        .fromTo(
          '.hero-poster-line',
          { opacity: 0, y: 35 },
          { opacity: 1, y: 0, duration: 0.8, stagger: 0.15 },
          '-=0.3'
        )
        .fromTo(
          '.hero-say',
          { opacity: 0, y: 20 },
          { opacity: 1, y: 0, duration: 0.7 },
          '-=0.3'
        )
        .fromTo(
          '.hero-links',
          { opacity: 0, y: 15 },
          { opacity: 1, y: 0, duration: 0.6 },
          '-=0.4'
        );
    },
    { scope: containerRef }
  );

  return (
    <section
      ref={containerRef}
      className="scene-hero pt-16 sm:pt-24 pb-16 px-6 sm:px-10 lg:px-12 max-w-7xl mx-auto"
    >
      <div className="stage max-w-5xl">
        {/* Altnota Small Mono Category Tag */}
        <p className="hero-tag font-mono text-xs uppercase tracking-[0.18em] text-soft mb-8">
          official campus intelligence &amp; dual-rag voice
        </p>

        {/* Altnota Massive Bold Poster Headline with Hand-Drawn SVG Annotations */}
        <h1
          className="poster select-none"
          aria-label="Backed by policy. Informed by data. Built with purpose."
        >
          <span className="hero-poster-line block">
            Backed by{' '}
            <span className="circled">
              policy
              <svg viewBox="0 0 200 60" preserveAspectRatio="none" aria-hidden="true">
                <path d="M10 34 C 12 10, 80 2, 135 5 C 188 8, 198 30, 178 45 C 150 59, 45 59, 20 48 C 4 40, 14 12, 52 6" />
              </svg>
            </span>
            .
          </span>
          <span className="hero-poster-line block mt-1 sm:mt-2">
            Informed by data.
          </span>
          <span className="hero-poster-line block mt-1 sm:mt-2">
            Built with{' '}
            <span className="hd-u">
              purpose
              <svg viewBox="0 0 200 10" preserveAspectRatio="none" aria-hidden="true">
                <path d="M2 6 C 60 2, 140 9, 198 5" />
              </svg>
            </span>
            .
          </span>
        </h1>

        {/* Subhead Statement */}
        <p className="hero-say mt-10 sm:mt-12 text-lg sm:text-xl md:text-2xl text-soft font-display font-normal max-w-2xl leading-relaxed">
          Every campus inquiry has an ordinance nobody’s cited yet. We go and find it.
        </p>

        {/* Altnota Clean Dual CTAs */}
        <div className="hero-links mt-10 sm:mt-12 flex flex-wrap items-baseline gap-x-10 gap-y-4 text-base font-display">
          <a
            href="#nutshell"
            className="u-ink font-semibold text-black inline-flex items-center gap-1 group"
          >
            <span>see for yourself</span>
            <span aria-hidden="true" className="group-hover:translate-y-0.5 transition-transform">
              ↓
            </span>
          </a>
          <a
            href="#sandbox"
            className="font-semibold text-black inline-flex items-center gap-1.5 group"
          >
            <span className="bg-[#FFE500] px-2 py-0.5 rounded-md text-xs font-mono font-bold">LIVE</span>
            <span className="u-ink">in a hurry? try live voice</span>
            <span aria-hidden="true" className="group-hover:translate-x-1 transition-transform">
              →
            </span>
          </a>
        </div>
      </div>
    </section>
  );
};
