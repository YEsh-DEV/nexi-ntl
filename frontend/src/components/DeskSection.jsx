import React, { useRef } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { useGSAP } from '@gsap/react';

gsap.registerPlugin(ScrollTrigger, useGSAP);

export const DeskSection = () => {
  const containerRef = useRef(null);

  const items = [
    {
      id: 1,
      title: 'An academic ordinance.',
      note: 'cheapest stage to be right',
      body: 'Grading credits, detention bylaws, attendance condonation, or makeup exams before submission deadlines. Ask in English, Telugu, or Hindi.',
    },
    {
      id: 2,
      title: 'A hostel or residential rule.',
      note: 'rarely where you think',
      body: 'Curfew timings, leave applications, warden permissions, and biometric turnstile compliance without standing in queues.',
    },
    {
      id: 3,
      title: 'A research grant application.',
      note: 'we like these',
      body: 'Undergraduate seed funding up to ₹50,000, conference travel grants, and publication support with cited clauses.',
    },
  ];

  useGSAP(
    () => {
      gsap.fromTo(
        '.desk-rise',
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
      id="desk"
      ref={containerRef}
      className="scene scroll-mt-20 py-12 sm:py-20 px-4 sm:px-8 lg:px-12 max-w-7xl mx-auto"
    >
      <div className="stage max-w-5xl">
        {/* Altnota Category Tag */}
        <p className="desk-rise font-mono text-xs uppercase tracking-[0.18em] text-black font-semibold mb-6 flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-[#FFE500]" />
          how can nexi help you
        </p>

        {/* Section Headline */}
        <div className="desk-rise mb-12 sm:mb-16">
          <h2 className="poster-md max-w-3xl">
            What’s on your desk right now?
          </h2>
        </div>

        {/* 3 Altnota Signature Lead Rows */}
        <div className="divide-y divide-rule border-y border-rule">
          {items.map((item) => (
            <div
              key={item.id}
              className="desk-rise py-8 sm:py-12 group transition-colors hover:bg-[#F8F8F6] px-3 sm:px-6 -mx-3 sm:-mx-6 rounded-lg"
            >
              <div className="flex flex-wrap items-baseline gap-x-4 gap-y-2 mb-3">
                <h3 className="font-display font-bold text-2xl sm:text-3xl text-ink flex items-center gap-2.5">
                  <span className="text-[#FFE500] font-black group-hover:translate-x-1 transition-transform">
                    &gt;
                  </span>
                  <span>{item.title}</span>
                </h3>
                <span className="flex items-center gap-2 text-black">
                  <span className="text-dim">–</span>
                  <span className="font-hand text-xl sm:text-2xl font-bold bg-[#FFE500] px-2.5 py-0.5 rounded-md">
                    {item.note}
                  </span>
                </span>
              </div>
              <p className="text-base sm:text-lg text-soft font-display leading-relaxed max-w-2xl pl-6">
                {item.body}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};
