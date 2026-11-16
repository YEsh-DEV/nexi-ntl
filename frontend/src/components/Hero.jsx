import React, { useRef } from 'react';
import { ArrowRight, ShieldCheck, Zap, Globe, BookOpen } from 'lucide-react';
import gsap from 'gsap';
import { useGSAP } from '@gsap/react';

gsap.registerPlugin(useGSAP);

export const Hero = () => {
  const containerRef = useRef(null);

  useGSAP(
    () => {
      // 1. Ambient Blob Subtle Drift
      gsap.to('.hero-ambient-sage', {
        x: 35,
        y: 25,
        duration: 12,
        repeat: -1,
        yoyo: true,
        ease: 'sine.inOut',
      });
      gsap.to('.hero-ambient-champagne', {
        x: -30,
        y: -30,
        duration: 14,
        repeat: -1,
        yoyo: true,
        ease: 'sine.inOut',
        delay: 0.8,
      });
      gsap.to('.hero-ambient-peach', {
        scale: 1.15,
        duration: 10,
        repeat: -1,
        yoyo: true,
        ease: 'sine.inOut',
      });

      // 2. Coordinated Editorial Entrance
      const tl = gsap.timeline({ defaults: { ease: 'power2.out' } });

      tl.fromTo(
        '.hero-blueprint-badge',
        { y: -20, opacity: 0 },
        { y: 0, opacity: 1, duration: 0.7 }
      )
        .fromTo(
          '.hero-editorial-title',
          { y: 30, opacity: 0 },
          { y: 0, opacity: 1, duration: 0.85 },
          '-=0.35'
        )
        .fromTo(
          '.hero-editorial-sub',
          { y: 20, opacity: 0 },
          { y: 0, opacity: 1, duration: 0.75 },
          '-=0.45'
        )
        .fromTo(
          '.hero-cta-group',
          { y: 20, opacity: 0 },
          { y: 0, opacity: 1, duration: 0.65 },
          '-=0.4'
        )
        .fromTo(
          '.hero-blueprint-card',
          { y: 25, opacity: 0 },
          { y: 0, opacity: 1, stagger: 0.1, duration: 0.7 },
          '-=0.3'
        );
    },
    { scope: containerRef }
  );

  return (
    <section
      ref={containerRef}
      className="relative overflow-hidden pt-24 sm:pt-32 pb-24 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto text-center"
    >
      {/* 3 Low-Saturation Ambient Blurred Gradient Blobs */}
      <div
        className="hero-ambient-sage absolute w-[540px] h-[540px] -top-32 -left-20 rounded-full bg-emerald-200/25 filter blur-[140px] opacity-25 pointer-events-none -z-10"
        aria-hidden="true"
      />
      <div
        className="hero-ambient-champagne absolute w-[500px] h-[500px] top-28 -right-24 rounded-full bg-amber-200/30 filter blur-[140px] opacity-25 pointer-events-none -z-10"
        aria-hidden="true"
      />
      <div
        className="hero-ambient-peach absolute w-[460px] h-[460px] bottom-10 left-1/3 rounded-full bg-orange-200/20 filter blur-[140px] opacity-20 pointer-events-none -z-10"
        aria-hidden="true"
      />

      {/* Blueprint Index / Academic Year Badge */}
      <div className="hero-blueprint-badge inline-flex items-center space-x-2.5 bg-white/90 text-stone-800 px-4 py-1.5 rounded-full text-xs font-mono mb-8 border border-stone-200/90 shadow-xs">
        <span className="w-1.5 h-1.5 rounded-full bg-amber-500"></span>
        <span className="font-semibold uppercase tracking-wider text-stone-900">
          Official Institutional Intelligence
        </span>
        <span className="text-stone-300">/</span>
        <span className="text-stone-500">SRM University-AP (2024–2025)</span>
      </div>

      {/* High-Impact Editorial Headline */}
      <h1 className="hero-editorial-title text-4xl sm:text-6xl lg:text-7xl font-extrabold tracking-tight text-stone-950 leading-[1.08] max-w-4xl mx-auto">
        Your Limitless, Intelligent{' '}
        <span className="font-editorial italic font-normal text-amber-700 underline decoration-amber-300/80 decoration-wavy decoration-1 underline-offset-8">
          Campus Voice
        </span>
      </h1>

      {/* Subtitle with Generous Leading */}
      <p className="hero-editorial-sub mt-7 text-base sm:text-lg text-stone-600 max-w-2xl mx-auto leading-relaxed font-normal">
        NEXi is a sub-second conversational voice assistant grounded strictly in verified SRM University-AP statutes.
        Ask about attendance condonation, hostel curfew protocols, examination rules, and student seed funding.
      </p>

      {/* Action Buttons */}
      <div className="hero-cta-group mt-10 flex flex-col sm:flex-row items-center justify-center gap-3.5 sm:gap-4.5">
        <a
          href="#sandbox"
          className="w-full sm:w-auto inline-flex items-center justify-center space-x-2.5 bg-amber-500 hover:bg-amber-600 active:bg-amber-700 text-stone-950 font-semibold px-7 py-3.5 rounded-2xl transition-all shadow-md shadow-amber-500/20 hover:shadow-lg hover:-translate-y-0.5"
        >
          <span>Open Interactive Sandbox</span>
          <ArrowRight className="w-4 h-4 text-stone-950" />
        </a>
        <a
          href="#about"
          className="w-full sm:w-auto inline-flex items-center justify-center space-x-2 bg-white/90 hover:bg-white text-stone-700 font-semibold px-7 py-3.5 rounded-2xl border border-stone-200/90 transition-all shadow-xs hover:border-stone-300 hover:-translate-y-0.5"
        >
          <BookOpen className="w-4 h-4 text-stone-500" />
          <span>Explore Verified Ordinances</span>
        </a>
      </div>

      {/* Trust & Metric Highlights - Delicate Blueprint Cards */}
      <div className="mt-20 pt-12 border-t border-stone-200/80 grid grid-cols-2 md:grid-cols-4 gap-4.5 max-w-5xl mx-auto text-left">
        <div className="hero-blueprint-card p-6 rounded-2xl bg-white/80 border border-stone-200/90 shadow-xs hover:border-amber-300/80 hover:bg-white transition-all duration-300 relative group">
          <div className="flex items-center justify-between text-stone-400 font-mono text-[10px] uppercase tracking-wider mb-2">
            <span>[SPEC // 01]</span>
            <ShieldCheck className="w-4 h-4 text-amber-600" />
          </div>
          <div className="text-xl sm:text-2xl font-editorial font-bold text-stone-950 tracking-tight">
            100% Policy Gated
          </div>
          <p className="text-xs text-stone-500 mt-1.5 leading-normal">
            Zero hallucinations; calibrated abstention over speculation.
          </p>
        </div>

        <div className="hero-blueprint-card p-6 rounded-2xl bg-white/80 border border-stone-200/90 shadow-xs hover:border-amber-300/80 hover:bg-white transition-all duration-300 relative group">
          <div className="flex items-center justify-between text-stone-400 font-mono text-[10px] uppercase tracking-wider mb-2">
            <span>[SPEC // 02]</span>
            <Zap className="w-4 h-4 text-amber-600" />
          </div>
          <div className="text-xl sm:text-2xl font-editorial font-bold text-stone-950 tracking-tight">
            &lt; 1.1s Latency
          </div>
          <p className="text-xs text-stone-500 mt-1.5 leading-normal">
            Full-duplex WebRTC with live speech interruption.
          </p>
        </div>

        <div className="hero-blueprint-card p-6 rounded-2xl bg-white/80 border border-stone-200/90 shadow-xs hover:border-amber-300/80 hover:bg-white transition-all duration-300 relative group">
          <div className="flex items-center justify-between text-stone-400 font-mono text-[10px] uppercase tracking-wider mb-2">
            <span>[SPEC // 03]</span>
            <Globe className="w-4 h-4 text-amber-600" />
          </div>
          <div className="text-xl sm:text-2xl font-editorial font-bold text-stone-950 tracking-tight">
            EN • తెలుగు • हिन्दी
          </div>
          <p className="text-xs text-stone-500 mt-1.5 leading-normal">
            Native pronunciation via Sarvam AI Bulbul TTS.
          </p>
        </div>

        <div className="hero-blueprint-card p-6 rounded-2xl bg-white/80 border border-stone-200/90 shadow-xs hover:border-amber-300/80 hover:bg-white transition-all duration-300 relative group">
          <div className="flex items-center justify-between text-stone-400 font-mono text-[10px] uppercase tracking-wider mb-2">
            <span>[SPEC // 04]</span>
            <BookOpen className="w-4 h-4 text-amber-600" />
          </div>
          <div className="text-xl sm:text-2xl font-editorial font-bold text-stone-950 tracking-tight">
            817 Chunks
          </div>
          <p className="text-xs text-stone-500 mt-1.5 leading-normal">
            Indexed across 11 official university handbooks.
          </p>
        </div>
      </div>
    </section>
  );
};
