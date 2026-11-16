import React, { useRef } from 'react';
import { Star, MessageSquareQuote, CheckCircle2 } from 'lucide-react';
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
        'During end-sem exam preparations, I needed to confirm the exact attendance condonation percentage for medical leave. Instead of waiting in line at the student affairs office, NEXi gave me the exact section from the 2024 academic ordinances in less than 2 seconds.',
      badge: 'Attendance Condonation',
      spec: 'TESTIMONIAL // 01',
    },
    {
      id: 2,
      name: 'Pooja Reddy',
      department: 'B.Tech Electronics & Communication (2nd Year)',
      rating: 5,
      review:
        'మా అమ్మగారు హాస్టల్ ఫీజు మరియు కర్ఫ్యూ సమయాల గురించి తెలుసుకోవడానికి తెలుగులో మాట్లాడారు. నెక్సీ స్పష్టమైన తెలుగులో వివరించింది. Telugu voice synthesis feels natural and accurate.',
      badge: 'Multilingual Telugu Call',
      spec: 'TESTIMONIAL // 02',
    },
    {
      id: 3,
      name: 'Aditya Verma',
      department: 'M.Tech Artificial Intelligence & Machine Learning',
      rating: 5,
      review:
        'Zero hallucination is what sets NEXi apart. When I asked about conference travel grant reimbursement procedures, it quoted the Directorate of Research guidelines word-for-word rather than guessing. Truly impressive engineering.',
      badge: 'Seed Grant & Research',
      spec: 'TESTIMONIAL // 03',
    },
  ];

  useGSAP(
    () => {
      gsap.fromTo(
        '.testimonials-header',
        { opacity: 0, y: 30 },
        {
          opacity: 1,
          y: 0,
          duration: 0.8,
          ease: 'power2.out',
          scrollTrigger: {
            trigger: '.testimonials-header',
            start: 'top 85%',
          },
        }
      );

      gsap.fromTo(
        '.testimonial-card',
        { opacity: 0, y: 40 },
        {
          opacity: 1,
          y: 0,
          duration: 0.85,
          stagger: 0.18,
          ease: 'power2.out',
          scrollTrigger: {
            trigger: '.testimonials-grid',
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
      className="relative py-28 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto border-t border-stone-200/80 overflow-hidden"
    >
      {/* Soft Ambient Background Glow */}
      <div
        className="absolute w-[440px] h-[440px] top-1/3 -right-24 rounded-full bg-amber-100/25 filter blur-[140px] opacity-25 pointer-events-none -z-10"
        aria-hidden="true"
      />

      <div className="testimonials-header max-w-3xl mx-auto text-center mb-20">
        <div className="inline-flex items-center space-x-2 bg-white/90 text-stone-700 px-3.5 py-1.5 rounded-full text-xs font-mono mb-4 border border-stone-200/90 shadow-xs">
          <MessageSquareQuote className="w-3.5 h-3.5 text-amber-600" />
          <span>VERIFIED PEER REVIEWS // [CAMPUS_EVAL]</span>
        </div>
        <h2 className="text-3xl sm:text-4xl lg:text-5xl font-editorial font-normal text-stone-900 tracking-tight">
          Trusted by SRM University-AP Students & Faculty
        </h2>
        <p className="mt-4 text-stone-600 text-base sm:text-lg leading-relaxed font-normal max-w-2xl mx-auto">
          See how real campus residents and scholars rely on NEXi for daily academic policies, hostel permissions, and administrative clarity.
        </p>
      </div>

      {/* Testimonials Grid */}
      <div className="testimonials-grid grid grid-cols-1 md:grid-cols-3 gap-8">
        {reviews.map((rev) => (
          <div
            key={rev.id}
            className="testimonial-card p-8 sm:p-9 rounded-2xl bg-white/85 border border-stone-200/90 shadow-xs hover:shadow-md hover:border-amber-400/80 hover:bg-white transition-all duration-300 flex flex-col justify-between"
          >
            <div>
              {/* Star Rating & Query Badge */}
              <div className="flex items-center justify-between mb-5">
                <div className="flex items-center space-x-1">
                  {[...Array(rev.rating)].map((_, i) => (
                    <Star key={i} className="w-4 h-4 fill-amber-400 text-amber-400" />
                  ))}
                </div>
                <div className="flex items-center space-x-2">
                  <span className="text-[10px] font-mono text-stone-400">
                    {rev.spec}
                  </span>
                  <span className="text-[10px] font-mono font-medium uppercase tracking-wider text-amber-900 bg-amber-100/80 px-2 py-0.5 rounded-md border border-amber-200/70">
                    {rev.badge}
                  </span>
                </div>
              </div>

              {/* Review Text */}
              <p className="text-stone-700 text-sm leading-relaxed mb-6 italic font-normal">
                "{rev.review}"
              </p>
            </div>

            {/* Student Info */}
            <div className="pt-5 border-t border-stone-200/70 flex items-center space-x-3.5">
              <div className="h-10 w-10 rounded-full bg-stone-900 text-amber-400 font-mono font-bold text-xs flex items-center justify-center shrink-0 shadow-xs">
                {rev.name.charAt(0)}
              </div>
              <div className="overflow-hidden">
                <div className="flex items-center space-x-1.5">
                  <span className="font-medium text-stone-900 text-sm truncate">{rev.name}</span>
                  <span title="Verified SRM-AP Student">
                    <CheckCircle2 className="w-3.5 h-3.5 text-amber-600 shrink-0" />
                  </span>
                </div>
                <div className="text-xs text-stone-500 font-mono truncate">{rev.department}</div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
};
