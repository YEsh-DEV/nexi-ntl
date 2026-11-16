import React, { useRef } from 'react';
import { Star, CheckCircle2 } from 'lucide-react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { useGSAP } from '@gsap/react';

gsap.registerPlugin(ScrollTrigger, useGSAP);

export const Testimonials = () => {
  const containerRef = useRef(null);

  const reviews = [
    {
      id: 1,
      name: 'Rohan Sharma',
      department: 'B.Tech Computer Science & Engineering (3rd Year)',
      rating: 5,
      review:
        'During end-sem preparations, I needed to confirm the exact attendance condonation percentage for medical leave. Instead of waiting at the student affairs desk, NEXi gave me the exact section from the 2024 academic ordinances in less than 2 seconds.',
      badge: 'Attendance Condonation',
    },
    {
      id: 2,
      name: 'Pooja Reddy',
      department: 'B.Tech Electronics & Communication (2nd Year)',
      rating: 5,
      review:
        'మా అమ్మగారు హాస్టల్ ఫీజు మరియు కర్ఫ్యూ సమయాల గురించి తెలుసుకోవడానికి తెలుగులో మాట్లాడారు. నెక్సీ స్పష్టమైన తెలుగులో వివరించింది. Telugu voice synthesis feels natural, dialect-aware, and completely clear.',
      badge: 'Multilingual Telugu Call',
    },
    {
      id: 3,
      name: 'Aditya Verma',
      department: 'M.Tech Artificial Intelligence & Machine Learning',
      rating: 5,
      review:
        'Zero hallucination is what sets NEXi apart. When I asked about conference travel grant reimbursement procedures, it quoted the Directorate of Research guidelines word-for-word rather than guessing.',
      badge: 'Seed Grant & Research',
    },
  ];

  useGSAP(
    () => {
      gsap.fromTo(
        '.testimonials-rise',
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
      id="testimonials"
      ref={containerRef}
      className="scene scroll-mt-20 py-12 sm:py-20 px-4 sm:px-8 lg:px-12 max-w-7xl mx-auto"
    >
      <div className="stage max-w-5xl">
        {/* Altnota Category Tag */}
        <p className="testimonials-rise font-mono text-xs uppercase tracking-[0.18em] text-black font-semibold mb-6 flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-[#FFE500]" />
          working together
        </p>

        {/* Section Headline */}
        <div className="testimonials-rise mb-16">
          <h2 className="poster-md max-w-3xl">
            From the first question to the finish line.
          </h2>
          <p className="mt-6 text-lg sm:text-xl text-soft font-display leading-relaxed max-w-2xl">
            The students and scholars who rely on NEXi are the same peers who verify the accuracy of every policy response.
          </p>
        </div>

        {/* 3 Clean Quote Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {reviews.map((rev) => (
            <div
              key={rev.id}
              className="testimonials-rise p-8 border border-rule rounded-xl bg-white flex flex-col justify-between hover:border-black/40 transition-colors"
            >
              <div>
                <div className="flex items-center justify-between mb-5">
                  <div className="flex items-center space-x-1 text-[#FFE500]">
                    {[...Array(rev.rating)].map((_, i) => (
                      <Star key={i} className="w-4 h-4 fill-[#FFE500] text-[#FFE500]" />
                    ))}
                  </div>
                  <span className="font-mono text-[10px] uppercase tracking-wider text-black bg-[#FFE500] px-2 py-0.5 rounded-md font-bold">
                    {rev.badge}
                  </span>
                </div>

                <p className="text-ink font-display text-base leading-relaxed mb-6 italic">
                  "{rev.review}"
                </p>
              </div>

              <div className="pt-5 border-t border-rule flex items-center space-x-3">
                <div className="h-9 w-9 rounded-full bg-black text-[#FFE500] font-mono font-bold text-xs flex items-center justify-center shrink-0">
                  {rev.name.charAt(0)}
                </div>
                <div className="overflow-hidden">
                  <div className="flex items-center space-x-1.5">
                    <span className="font-display font-semibold text-black text-sm truncate">
                      {rev.name}
                    </span>
                    <CheckCircle2 className="w-3.5 h-3.5 text-black shrink-0" />
                  </div>
                  <div className="text-xs text-soft font-mono truncate">{rev.department}</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};
