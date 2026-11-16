import React, { useRef } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { useGSAP } from '@gsap/react';

gsap.registerPlugin(ScrollTrigger, useGSAP);

export const HowWeWork = () => {
  const containerRef = useRef(null);

  const questions = [
    {
      id: 1,
      quote: '“I missed classes due to viral fever. Can I write end-sem exams with 68% attendance?”',
      wrong: 'Ask your course teacher or HOD for a verbal waiver.',
      handNote: 'Section 4.2: Submit medical proof within 3 days for Dean condonation.',
      tag: 'Academic Regulations',
    },
    {
      id: 2,
      quote: '“Can I enter the campus hostel at 9:45 PM on a Saturday evening?”',
      wrong: 'Hostel timings are usually relaxed on weekends.',
      handNote: 'Clause 8.1: Biometric turnstiles lock strictly at 9:00 PM; gate pass required.',
      tag: 'Hostel Bylaws',
    },
    {
      id: 3,
      quote: '“Is there university funding for undergraduate students building prototypes?”',
      wrong: 'Research seed grants are reserved only for PhD scholars.',
      handNote: 'Section 3: Up to ₹50,000 seed grant per approved UG team via IRIS portal.',
      tag: 'Research Promotion',
    },
  ];

  useGSAP(
    () => {
      gsap.fromTo(
        '.hww-rise',
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
      id="how-we-work"
      ref={containerRef}
      className="scene scroll-mt-20 py-12 sm:py-20 px-4 sm:px-8 lg:px-12 max-w-7xl mx-auto"
    >
      <div className="stage max-w-5xl">
        {/* Altnota Category Tag */}
        <p className="hww-rise font-mono text-xs uppercase tracking-[0.18em] text-black font-semibold mb-6 flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-[#FFE500]" />
          how we work
        </p>

        {/* Section Headline */}
        <div className="hww-rise mb-12 sm:mb-16">
          <h2 className="poster-md max-w-3xl">
            The answer depends on the question. So we start there.
          </h2>
          <p className="mt-4 sm:mt-6 text-base sm:text-xl text-soft font-display leading-relaxed">
            It usually goes something like this:
          </p>
        </div>

        {/* 3 Altnota Signature Comparison Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 lg:gap-8">
          {questions.map((q) => (
            <div
              key={q.id}
              className="hww-rise p-6 sm:p-8 border border-rule rounded-xl bg-white flex flex-col justify-between hover:border-black/40 transition-colors"
            >
              <div>
                <span className="font-mono text-[10px] uppercase tracking-wider text-soft block mb-4">
                  {q.tag}
                </span>

                <p className="font-display font-medium text-ink text-base sm:text-lg mb-8 leading-snug">
                  {q.quote}
                </p>

                {/* Crossed out guess */}
                <div className="mb-6">
                  <span className="font-mono text-[11px] text-dim block mb-1 uppercase tracking-wider">
                    Standard AI Guess:
                  </span>
                  <p className="text-soft line-through text-sm font-display decoration-1 decoration-soft">
                    {q.wrong}
                  </p>
                </div>
              </div>

              {/* Verified Handwritten Note in Bright Yellow Box */}
              <div className="pt-6 border-t border-rule">
                <span className="font-mono text-[11px] text-black font-semibold uppercase tracking-wider block mb-1">
                  NEXi Grounded Citation:
                </span>
                <p className="font-hand text-xl text-black leading-tight bg-[#FFE500]/30 px-3 py-2 rounded-lg border border-[#FFE500]">
                  {q.handNote}
                </p>
              </div>
            </div>
          ))}
        </div>

        {/* Follow-up Note */}
        <div className="hww-rise mt-16 pt-10 border-t border-rule max-w-2xl">
          <p className="text-base sm:text-lg text-ink font-display leading-relaxed">
            Problems change shape when you find the right place to look from. Grounding queries in official ordinances is where every answer starts.
          </p>
          <p className="mt-4">
            <a
              href="#sandbox"
              className="u-ink font-display font-semibold text-base text-black inline-flex items-center gap-1 group"
            >
              <span>Curious what we’d find on yours? Test in the sandbox</span>
              <span className="group-hover:translate-x-1 transition-transform">→</span>
            </a>
          </p>
        </div>
      </div>
    </section>
  );
};
