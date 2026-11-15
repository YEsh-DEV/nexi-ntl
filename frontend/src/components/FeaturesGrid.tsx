import React, { useRef } from 'react';
import { ShieldCheck, Languages, BookCheck, Cpu, CheckCircle2 } from 'lucide-react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { useGSAP } from '@gsap/react';

gsap.registerPlugin(ScrollTrigger, useGSAP);

export const FeaturesGrid: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);

  const features = [
    {
      icon: ShieldCheck,
      badge: 'Dual-Engine RAG',
      title: 'Zero Hallucinations via Cross-Encoder Gating',
      description:
        'Standard LLMs fabricate dates, grades, and fee rules. NEXi combines in-memory Okapi BM25 with ChromaDB vector search and reciprocal rank fusion. A cross-encoder reranker rejects queries below calibrated thresholds, ensuring answers are 100% grounded or safely declined.',
      points: [
        'Pure-Python Okapi BM25 for exact course codes & faculty names',
        'Dense semantic search with all-MiniLM-L6-v2',
        'Calibrated score thresholding eliminates speculation',
        'Automatic routing to admissions when information is unindexed',
      ],
    },
    {
      icon: Languages,
      badge: 'Multilingual Voice',
      title: 'Native English, Telugu & Hindi Conversational Voice',
      description:
        'SRM University-AP hosts a diverse student body from across India. NEXi speaks and listens in English, Telugu (తెలుగు), and Hindi (हिन्दी) with dialect awareness. Inbound phone callers can also select their language via DTMF keypad buttons.',
      points: [
        'Deepgram Nova-3 streaming speech recognition (<150ms)',
        'Sarvam AI Bulbul v3 neural voice synthesis',
        'Mid-call dynamic language handover with preserved history',
        'Twilio DTMF keypad menu (1=TE, 2=HI, 3=EN, 0=Transfer)',
      ],
    },
    {
      icon: BookCheck,
      badge: 'Institutional Knowledge',
      title: 'Complete SRM University-AP Policy Integration',
      description:
        'Indexed from 11 official university PDFs and 74 scanned pages extracted offline with zero runtime overhead. Covers every ordinance regarding grading schemes, hostel curfew, leave applications, attendance condonation, and seed research grants.',
      points: [
        'Academic Regulations & Ordinance 2024–2025',
        'Hostel Rules & Curfew Violation Guidelines',
        'Attendance Condonation (65%–75% Medical Limits)',
        'Conference Travel Funding & Student Seed Grant Rules',
      ],
    },
  ];

  useGSAP(
    () => {
      // 1. Header entrance
      gsap.fromTo(
        '.features-header',
        { opacity: 0, y: 35 },
        {
          opacity: 1,
          y: 0,
          duration: 0.8,
          ease: 'power3.out',
          scrollTrigger: {
            trigger: '.features-header',
            start: 'top 85%',
          },
        }
      );

      // 2. Feature Cards Staggered Scale-in
      gsap.fromTo(
        '.feature-card',
        { opacity: 0, y: 55, scale: 0.94 },
        {
          opacity: 1,
          y: 0,
          scale: 1,
          duration: 0.9,
          stagger: 0.2,
          ease: 'power3.out',
          scrollTrigger: {
            trigger: '.features-grid',
            start: 'top 80%',
          },
        }
      );
    },
    { scope: containerRef }
  );

  return (
    <section
      id="about"
      ref={containerRef}
      className="relative py-28 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto border-t border-slate-200/60 overflow-hidden"
    >
      {/* Soft Ambient Background Glows */}
      <div
        className="absolute w-[480px] h-[480px] top-1/4 -left-32 rounded-full bg-gradient-to-tr from-indigo-200/25 via-slate-100/30 to-teal-100/20 filter blur-[130px] opacity-40 pointer-events-none -z-10"
        aria-hidden="true"
      />
      <div
        className="absolute w-[440px] h-[440px] bottom-10 -right-24 rounded-full bg-gradient-to-bl from-purple-100/30 via-indigo-100/20 to-emerald-100/20 filter blur-[120px] opacity-35 pointer-events-none -z-10"
        aria-hidden="true"
      />

      {/* Header */}
      <div className="features-header max-w-3xl mx-auto text-center mb-20">
        <div className="inline-flex items-center space-x-1.5 backdrop-blur-md bg-white/80 text-slate-700 px-3.5 py-1.5 rounded-full text-xs font-semibold mb-4 border border-slate-200 shadow-xs">
          <Cpu className="w-3.5 h-3.5 text-indigo-600" />
          <span>Engineered for Academic Rigor</span>
        </div>
        <h2 className="text-3xl sm:text-4xl lg:text-5xl font-extrabold text-slate-900 tracking-tight">
          Enterprise Voice AI, Built for University Trust
        </h2>
        <p className="mt-4 text-slate-600 text-base sm:text-lg leading-relaxed font-normal">
          Unlike generic chatbot wrappers, NEXi is an end-to-end voice architecture designed to enforce institutional policy accuracy, low turn latency, and caller privacy.
        </p>
      </div>

      {/* 3-Column Feature Cards with Glassmorphism */}
      <div className="features-grid grid grid-cols-1 md:grid-cols-3 gap-8">
        {features.map((feature, idx) => (
          <div
            key={idx}
            className="feature-card flex flex-col justify-between p-8 sm:p-9 rounded-3xl backdrop-blur-md bg-white/70 border border-slate-200/60 shadow-xl shadow-slate-100/50 hover:shadow-2xl hover:border-indigo-200/80 hover:bg-white/90 hover:-translate-y-1 transition-all duration-300 group"
          >
            <div>
              <div className="flex items-center justify-between mb-7">
                <div className="h-13 w-13 rounded-2xl bg-indigo-50/80 border border-indigo-100 flex items-center justify-center text-indigo-600 group-hover:scale-108 group-hover:bg-indigo-600 group-hover:text-white transition-all duration-300 shadow-xs">
                  <feature.icon className="w-6 h-6" />
                </div>
                <span className="text-[11px] font-bold uppercase tracking-wider text-indigo-700 bg-indigo-50/90 px-3 py-1 rounded-xl border border-indigo-100">
                  {feature.badge}
                </span>
              </div>

              <h3 className="text-xl font-bold text-slate-900 tracking-tight mb-3.5 group-hover:text-indigo-600 transition-colors">
                {feature.title}
              </h3>

              <p className="text-sm text-slate-600 leading-relaxed mb-7 font-normal">
                {feature.description}
              </p>
            </div>

            <div className="pt-6 border-t border-slate-100/80">
              <ul className="space-y-3">
                {feature.points.map((pt, pIdx) => (
                  <li key={pIdx} className="flex items-start text-xs text-slate-600 space-x-2.5">
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500 shrink-0 mt-0.5" />
                    <span className="leading-snug">{pt}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
};
