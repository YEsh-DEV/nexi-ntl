import React, { useRef } from 'react';
import { Star, MessageSquareQuote, CheckCircle2 } from 'lucide-react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { useGSAP } from '@gsap/react';

gsap.registerPlugin(ScrollTrigger, useGSAP);

export const Testimonials: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);

  const reviews = [
    {
      id: 1,
      name: 'Rohan Sharma',
      department: 'B.Tech Computer Science & Engineering (3rd Year)',
      rating: 5,
      review:
        'During end-sem exam preparations, I needed to confirm the exact attendance condonation percentage for medical leave. Instead of waiting in line at the student affairs office, NEXi gave me the exact section from the 2024 academic ordinances in less than 2 seconds.',
      badge: 'Attendance Condonation Query',
    },
    {
      id: 2,
      name: 'Pooja Reddy',
      department: 'B.Tech Electronics & Communication (2nd Year)',
      rating: 5,
      review:
        'మా అమ్మగారు హాస్టల్ ఫీజు మరియు కర్ఫ్యూ సమయాల గురించి తెలుసుకోవడానికి తెలుగులో మాట్లాడారు. నెక్సీ స్పష్టమైన తెలుగులో వివరించింది. Telugu voice synthesis feels natural and accurate.',
      badge: 'Multilingual Telugu Call',
    },
    {
      id: 3,
      name: 'Aditya Verma',
      department: 'M.Tech Artificial Intelligence & Machine Learning',
      rating: 5,
      review:
        'Zero hallucination is what sets NEXi apart. When I asked about conference travel grant reimbursement procedures, it quoted the Directorate of Research guidelines word-for-word rather than guessing. Truly impressive engineering.',
      badge: 'Seed Grant & Research Query',
    },
  ];

  useGSAP(
    () => {
      gsap.fromTo(
        '.testimonials-header',
        { opacity: 0, y: 35 },
        {
          opacity: 1,
          y: 0,
          duration: 0.8,
          ease: 'power3.out',
          scrollTrigger: {
            trigger: '.testimonials-header',
            start: 'top 85%',
          },
        }
      );

      gsap.fromTo(
        '.testimonial-card',
        { opacity: 0, y: 50, scale: 0.95 },
        {
          opacity: 1,
          y: 0,
          scale: 1,
          duration: 0.85,
          stagger: 0.18,
          ease: 'power3.out',
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
      className="relative py-28 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto border-t border-slate-200/60 overflow-hidden"
    >
      {/* Soft Ambient Background Glow */}
      <div
        className="absolute w-[460px] h-[460px] top-1/3 -right-28 rounded-full bg-gradient-to-bl from-indigo-200/25 via-purple-100/20 to-teal-100/15 filter blur-[130px] opacity-35 pointer-events-none -z-10"
        aria-hidden="true"
      />

      <div className="testimonials-header max-w-3xl mx-auto text-center mb-20">
        <div className="inline-flex items-center space-x-1.5 backdrop-blur-md bg-white/80 text-slate-700 px-3.5 py-1.5 rounded-full text-xs font-semibold mb-4 border border-slate-200 shadow-xs">
          <MessageSquareQuote className="w-3.5 h-3.5 text-indigo-600" />
          <span>Campus Community Feedback</span>
        </div>
        <h2 className="text-3xl sm:text-4xl lg:text-5xl font-extrabold text-slate-900 tracking-tight">
          Trusted by SRM University-AP Students & Faculty
        </h2>
        <p className="mt-4 text-slate-600 text-base sm:text-lg leading-relaxed font-normal">
          See how real campus residents and scholars rely on NEXi for daily academic policies, hostel permissions, and administrative clarity.
        </p>
      </div>

      {/* Testimonials Grid with Glassmorphism */}
      <div className="testimonials-grid grid grid-cols-1 md:grid-cols-3 gap-8">
        {reviews.map((rev) => (
          <div
            key={rev.id}
            className="testimonial-card p-8 sm:p-9 rounded-3xl backdrop-blur-md bg-white/70 border border-slate-200/60 shadow-xl shadow-slate-100/50 hover:shadow-2xl hover:border-slate-300/80 hover:bg-white/85 hover:-translate-y-1 transition-all duration-300 flex flex-col justify-between"
          >
            <div>
              {/* Star Rating & Query Badge */}
              <div className="flex items-center justify-between mb-5">
                <div className="flex items-center space-x-1 text-amber-400">
                  {[...Array(rev.rating)].map((_, i) => (
                    <Star key={i} className="w-4 h-4 fill-amber-400 text-amber-400" />
                  ))}
                </div>
                <span className="text-[10px] font-semibold uppercase tracking-wider text-indigo-700 bg-indigo-50/90 px-2.5 py-1 rounded-lg border border-indigo-100/80">
                  {rev.badge}
                </span>
              </div>

              {/* Review Text */}
              <p className="text-slate-700 text-sm leading-relaxed mb-6 italic font-normal">
                "{rev.review}"
              </p>
            </div>

            {/* Student Info */}
            <div className="pt-5 border-t border-slate-100/80 flex items-center space-x-3.5">
              <div className="h-11 w-11 rounded-full bg-gradient-to-tr from-slate-900 to-indigo-950 text-white font-bold text-sm flex items-center justify-center shrink-0 shadow-xs">
                {rev.name.charAt(0)}
              </div>
              <div className="overflow-hidden">
                <div className="flex items-center space-x-1.5">
                  <span className="font-bold text-slate-900 text-sm truncate">{rev.name}</span>
                  <span title="Verified SRM-AP Student">
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500 shrink-0" />
                  </span>
                </div>
                <div className="text-xs text-slate-500 truncate">{rev.department}</div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
};
