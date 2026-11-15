import React, { useRef } from 'react';
import { Sparkles, ArrowRight, ShieldCheck, Zap, Globe, BookOpen } from 'lucide-react';
import gsap from 'gsap';
import { useGSAP } from '@gsap/react';

gsap.registerPlugin(useGSAP);

export const Hero: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);

  useGSAP(
    () => {
      // 1. Ambient Blob Floating Motion
      gsap.to('.hero-blob-1', {
        x: 40,
        y: 25,
        duration: 9,
        repeat: -1,
        yoyo: true,
        ease: 'sine.inOut',
      });
      gsap.to('.hero-blob-2', {
        x: -35,
        y: -30,
        duration: 11,
        repeat: -1,
        yoyo: true,
        ease: 'sine.inOut',
        delay: 1,
      });
      gsap.to('.hero-blob-3', {
        scale: 1.18,
        duration: 8,
        repeat: -1,
        yoyo: true,
        ease: 'sine.inOut',
      });

      // 2. Coordinated Staggered Entrance
      const tl = gsap.timeline({ defaults: { ease: 'power3.out' } });

      tl.fromTo(
        '.hero-badge',
        { y: -25, opacity: 0 },
        { y: 0, opacity: 1, duration: 0.7 }
      )
        .fromTo(
          '.hero-title',
          { y: 35, opacity: 0 },
          { y: 0, opacity: 1, duration: 0.9 },
          '-=0.4'
        )
        .fromTo(
          '.hero-subtitle',
          { y: 20, opacity: 0 },
          { y: 0, opacity: 1, duration: 0.7 },
          '-=0.5'
        )
        .fromTo(
          '.hero-cta-btn',
          { y: 20, opacity: 0, scale: 0.95 },
          { y: 0, opacity: 1, scale: 1, stagger: 0.12, duration: 0.6, ease: 'back.out(1.4)' },
          '-=0.4'
        )
        .fromTo(
          '.hero-metric-card',
          { y: 30, opacity: 0, scale: 0.96 },
          { y: 0, opacity: 1, scale: 1, stagger: 0.1, duration: 0.7 },
          '-=0.3'
        );
    },
    { scope: containerRef }
  );

  return (
    <section
      ref={containerRef}
      className="relative overflow-hidden pt-20 sm:pt-28 pb-20 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto text-center"
    >
      {/* Soft Ambient Blurred Gradient Blobs */}
      <div
        className="hero-blob-1 absolute w-[520px] h-[520px] -top-24 -left-20 rounded-full bg-gradient-to-tr from-indigo-300/40 via-indigo-200/30 to-purple-200/20 filter blur-[120px] opacity-40 pointer-events-none -z-10"
        aria-hidden="true"
      />
      <div
        className="hero-blob-2 absolute w-[460px] h-[460px] top-32 -right-16 rounded-full bg-gradient-to-bl from-emerald-200/35 via-teal-100/30 to-indigo-100/20 filter blur-[120px] opacity-40 pointer-events-none -z-10"
        aria-hidden="true"
      />
      <div
        className="hero-blob-3 absolute w-[420px] h-[420px] bottom-10 left-1/3 rounded-full bg-gradient-to-r from-purple-200/30 via-rose-100/20 to-amber-100/25 filter blur-[110px] opacity-35 pointer-events-none -z-10"
        aria-hidden="true"
      />

      {/* University Pill Badge */}
      <div className="hero-badge inline-flex items-center space-x-2 backdrop-blur-md bg-white/80 text-indigo-800 px-4 py-1.5 rounded-full text-xs font-semibold mb-6 border border-indigo-200/70 shadow-xs">
        <Sparkles className="w-3.5 h-3.5 text-indigo-600 animate-pulse" />
        <span>Official Campus AI • SRM University-AP</span>
        <span className="text-indigo-300">•</span>
        <span className="text-slate-500 font-normal">Academic Year 2024–2025</span>
      </div>

      {/* Main Headline */}
      <h1 className="hero-title text-4xl sm:text-6xl lg:text-7xl font-extrabold tracking-tight text-slate-900 leading-[1.08] max-w-4xl mx-auto">
        Your Limitless, Intelligent{' '}
        <span className="bg-gradient-to-r from-indigo-700 via-indigo-600 to-slate-800 bg-clip-text text-transparent inline-block">
          Campus Voice
        </span>
      </h1>

      {/* Subtitle */}
      <p className="hero-subtitle mt-6 text-base sm:text-lg text-slate-600 max-w-2xl mx-auto leading-relaxed font-normal">
        NEXi is a full-duplex conversational voice assistant grounded strictly in official SRM University-AP regulations.
        Get instant, verifiable answers regarding attendance condonation, hostel curfews, exam malpractice rules, and seed funding.
      </p>

      {/* Action Buttons */}
      <div className="mt-9 flex flex-col sm:flex-row items-center justify-center gap-3 sm:gap-4">
        <a
          href="#sandbox"
          className="hero-cta-btn w-full sm:w-auto inline-flex items-center justify-center space-x-2 bg-indigo-600 hover:bg-indigo-500 text-white font-semibold px-7 py-3.5 rounded-2xl transition-all duration-200 shadow-lg shadow-indigo-200/70 hover:shadow-indigo-300/80 hover:-translate-y-0.5"
        >
          <span>Open Interactive Sandbox</span>
          <ArrowRight className="w-4 h-4" />
        </a>
        <a
          href="#about"
          className="hero-cta-btn w-full sm:w-auto inline-flex items-center justify-center space-x-2 backdrop-blur-md bg-white/80 hover:bg-white text-slate-700 font-semibold px-7 py-3.5 rounded-2xl border border-slate-200/80 transition-all duration-200 shadow-xs hover:border-slate-300 hover:-translate-y-0.5"
        >
          <BookOpen className="w-4 h-4 text-slate-500" />
          <span>Explore Verified Ordinances</span>
        </a>
      </div>

      {/* Trust & Metric Highlights - Glassmorphism Cards */}
      <div className="mt-16 pt-10 border-t border-slate-200/60 grid grid-cols-2 md:grid-cols-4 gap-4 max-w-4xl mx-auto text-left">
        <div className="hero-metric-card p-5 rounded-2xl backdrop-blur-md bg-white/70 border border-slate-200/60 shadow-xl shadow-slate-100/50 hover:bg-white/85 hover:border-slate-300/70 transition-all duration-300">
          <div className="flex items-center space-x-2 text-indigo-600 mb-1.5">
            <ShieldCheck className="w-4 h-4" />
            <span className="text-[11px] font-bold uppercase tracking-wider text-slate-500">Grounding</span>
          </div>
          <div className="text-xl sm:text-2xl font-extrabold text-slate-900 tracking-tight">100% Policy Gated</div>
          <p className="text-xs text-slate-500 mt-1 leading-normal">Zero hallucinations; strict abstention over speculation</p>
        </div>

        <div className="hero-metric-card p-5 rounded-2xl backdrop-blur-md bg-white/70 border border-slate-200/60 shadow-xl shadow-slate-100/50 hover:bg-white/85 hover:border-slate-300/70 transition-all duration-300">
          <div className="flex items-center space-x-2 text-indigo-600 mb-1.5">
            <Zap className="w-4 h-4" />
            <span className="text-[11px] font-bold uppercase tracking-wider text-slate-500">Speed</span>
          </div>
          <div className="text-xl sm:text-2xl font-extrabold text-slate-900 tracking-tight">&lt; 1.1s Latency</div>
          <p className="text-xs text-slate-500 mt-1 leading-normal">Full-duplex WebRTC with live speech interruption</p>
        </div>

        <div className="hero-metric-card p-5 rounded-2xl backdrop-blur-md bg-white/70 border border-slate-200/60 shadow-xl shadow-slate-100/50 hover:bg-white/85 hover:border-slate-300/70 transition-all duration-300">
          <div className="flex items-center space-x-2 text-indigo-600 mb-1.5">
            <Globe className="w-4 h-4" />
            <span className="text-[11px] font-bold uppercase tracking-wider text-slate-500">Languages</span>
          </div>
          <div className="text-xl sm:text-2xl font-extrabold text-slate-900 tracking-tight">EN • తెలుగు • हिन्दी</div>
          <p className="text-xs text-slate-500 mt-1 leading-normal">Native pronunciation via Sarvam AI Bulbul TTS</p>
        </div>

        <div className="hero-metric-card p-5 rounded-2xl backdrop-blur-md bg-white/70 border border-slate-200/60 shadow-xl shadow-slate-100/50 hover:bg-white/85 hover:border-slate-300/70 transition-all duration-300">
          <div className="flex items-center space-x-2 text-indigo-600 mb-1.5">
            <BookOpen className="w-4 h-4" />
            <span className="text-[11px] font-bold uppercase tracking-wider text-slate-500">Corpus</span>
          </div>
          <div className="text-xl sm:text-2xl font-extrabold text-slate-900 tracking-tight">817 Chunks</div>
          <p className="text-xs text-slate-500 mt-1 leading-normal">Indexed across 11 official university handbooks</p>
        </div>
      </div>
    </section>
  );
};
