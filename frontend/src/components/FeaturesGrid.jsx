import React, { useRef } from 'react';
import { ShieldCheck, Languages, BookCheck, Cpu, CheckCircle2 } from 'lucide-react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { useGSAP } from '@gsap/react';

gsap.registerPlugin(ScrollTrigger, useGSAP);

export const FeaturesGrid = () => {
  const containerRef = useRef(null);

  const features = [
    {
      icon: ShieldCheck,
      badge: 'Dual-Engine RAG',
      spec: 'SPEC // 01.1',
      title: 'Zero Hallucinations via Cross-Encoder Gating',
      description:
        'Standard LLMs fabricate dates, grades, and fee rules. NEXi combines in-memory Okapi BM25 with ChromaDB vector search and reciprocal rank fusion. A cross-encoder reranker rejects queries below calibrated thresholds, ensuring answers are 100% grounded or safely declined.',
      points: [
        'Pure-Python Okapi BM25 for exact course codes & faculty names',
        'Dense semantic search with all-MiniLM-L6-v2 embeddings',
        'Calibrated score thresholding eliminates speculation',
        'Automatic routing to admissions when information is unindexed',
      ],
    },
    {
      icon: Languages,
      badge: 'Multilingual Voice',
      spec: 'SPEC // 01.2',
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
      spec: 'SPEC // 01.3',
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
      // 1. Header entrance with ScrollTrigger
      gsap.fromTo(
        '.features-header',
        { opacity: 0, y: 30 },
        {
          opacity: 1,
          y: 0,
          duration: 0.8,
          ease: 'power2.out',
          scrollTrigger: {
            trigger: '.features-header',
            start: 'top 85%',
          },
        }
      );

      // 2. Feature Cards Staggered Reveal
      gsap.fromTo(
        '.feature-card',
        { opacity: 0, y: 40 },
        {
          opacity: 1,
          y: 0,
          duration: 0.85,
          stagger: 0.18,
          ease: 'power2.out',
          scrollTrigger: {
            trigger: '.features-grid',
            start: 'top 82%',
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
      className="relative py-28 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto border-t border-stone-200/80 overflow-hidden"
    >
      {/* Subtle Ambient Background Glows */}
      <div
        className="absolute w-[460px] h-[460px] top-1/4 -left-28 rounded-full bg-emerald-100/20 filter blur-[140px] opacity-25 pointer-events-none -z-10"
        aria-hidden="true"
      />
      <div
        className="absolute w-[440px] h-[440px] bottom-10 -right-24 rounded-full bg-amber-100/25 filter blur-[140px] opacity-25 pointer-events-none -z-10"
        aria-hidden="true"
      />

      {/* Header */}
      <div className="features-header max-w-3xl mx-auto text-center mb-20">
        <div className="inline-flex items-center space-x-2 bg-white/90 text-stone-700 px-3.5 py-1.5 rounded-full text-xs font-mono mb-4 border border-stone-200/90 shadow-xs">
          <Cpu className="w-3.5 h-3.5 text-amber-600" />
          <span>ENGINEERED FOR RIGOR // [SYSTEM_SPEC]</span>
        </div>
        <h2 className="text-3xl sm:text-4xl lg:text-5xl font-editorial font-normal text-stone-900 tracking-tight">
          Enterprise Voice AI, Built for University Trust
        </h2>
        <p className="mt-4 text-stone-600 text-base sm:text-lg leading-relaxed font-normal max-w-2xl mx-auto">
          Unlike generic chatbot wrappers, NEXi is an end-to-end voice architecture designed to enforce institutional policy accuracy, low turn latency, and caller privacy.
        </p>
      </div>

      {/* 3-Column Feature Cards */}
      <div className="features-grid grid grid-cols-1 md:grid-cols-3 gap-8">
        {features.map((feature, idx) => (
          <div
            key={idx}
            className="feature-card flex flex-col justify-between p-8 sm:p-9 rounded-2xl bg-white/85 border border-stone-200/90 shadow-xs hover:shadow-md hover:border-amber-400/80 hover:bg-white transition-all duration-300 group"
          >
            <div>
              <div className="flex items-center justify-between mb-6">
                <div className="h-12 w-12 rounded-xl bg-amber-500/10 border border-amber-500/20 flex items-center justify-center text-amber-700 group-hover:scale-105 group-hover:bg-amber-500 group-hover:text-stone-950 transition-all duration-300 shadow-xs">
                  <feature.icon className="w-6 h-6" />
                </div>
                <div className="flex items-center space-x-2">
                  <span className="text-[10px] font-mono text-stone-400">
                    {feature.spec}
                  </span>
                  <span className="text-[11px] font-medium uppercase tracking-wider text-amber-800 bg-amber-100/80 px-2.5 py-1 rounded-md border border-amber-200/70">
                    {feature.badge}
                  </span>
                </div>
              </div>

              <h3 className="text-xl font-editorial font-medium text-stone-900 tracking-tight mb-3 group-hover:text-amber-800 transition-colors">
                {feature.title}
              </h3>

              <p className="text-sm text-stone-600 leading-relaxed mb-6 font-normal">
                {feature.description}
              </p>
            </div>

            <div className="pt-6 border-t border-stone-200/70">
              <ul className="space-y-2.5">
                {feature.points.map((pt, pIdx) => (
                  <li key={pIdx} className="flex items-start text-xs text-stone-600 space-x-2.5">
                    <CheckCircle2 className="w-3.5 h-3.5 text-amber-600 shrink-0 mt-0.5" />
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
