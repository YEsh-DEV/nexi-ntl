import React from 'react';
import { Sparkles, ArrowRight, ShieldCheck, Zap, Globe, BookOpen } from 'lucide-react';

export const Hero: React.FC = () => {
  return (
    <section className="relative overflow-hidden pt-16 sm:pt-24 pb-16 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto text-center">
      {/* Background Subtle Gradient Glow */}
      <div className="absolute inset-0 -top-24 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-indigo-50/80 via-transparent to-transparent -z-10" />

      {/* University Pill Badge */}
      <div className="inline-flex items-center space-x-2 bg-indigo-50/90 text-indigo-700 px-3.5 py-1.5 rounded-full text-xs font-semibold mb-6 border border-indigo-100/80 shadow-xs">
        <Sparkles className="w-3.5 h-3.5 text-indigo-600" />
        <span>Official Campus AI • SRM University-AP</span>
        <span className="text-indigo-300">•</span>
        <span className="text-slate-500 font-normal">Academic Year 2024–2025</span>
      </div>

      {/* Main Headline */}
      <h1 className="text-4xl sm:text-6xl lg:text-7xl font-extrabold tracking-tight text-slate-900 leading-[1.1] max-w-4xl mx-auto">
        Your Limitless, Intelligent <span className="text-indigo-600 inline-block">Campus Voice</span>
      </h1>

      {/* Subtitle */}
      <p className="mt-6 text-base sm:text-lg text-slate-600 max-w-2xl mx-auto leading-relaxed">
        NEXi is a full-duplex conversational voice assistant grounded strictly in official SRM University-AP regulations.
        Get instant, verifiable answers regarding attendance condonation, hostel curfews, exam malpractice rules, and seed funding.
      </p>

      {/* Action Buttons */}
      <div className="mt-8 flex flex-col sm:flex-row items-center justify-center gap-3 sm:gap-4">
        <a
          href="#sandbox"
          className="w-full sm:w-auto inline-flex items-center justify-center space-x-2 bg-indigo-600 hover:bg-indigo-500 text-white font-semibold px-6 py-3.5 rounded-xl transition-all shadow-md shadow-indigo-200 hover:shadow-lg"
        >
          <span>Open Interactive Sandbox</span>
          <ArrowRight className="w-4 h-4" />
        </a>
        <a
          href="#about"
          className="w-full sm:w-auto inline-flex items-center justify-center space-x-2 bg-white hover:bg-slate-50 text-slate-700 font-semibold px-6 py-3.5 rounded-xl border border-slate-200 transition-all shadow-xs"
        >
          <BookOpen className="w-4 h-4 text-slate-500" />
          <span>Explore Verified Ordinances</span>
        </a>
      </div>

      {/* Trust & Metric Highlights */}
      <div className="mt-14 pt-10 border-t border-slate-200/60 grid grid-cols-2 md:grid-cols-4 gap-4 max-w-4xl mx-auto text-left">
        <div className="p-4 rounded-2xl bg-white border border-slate-200/80 shadow-xs">
          <div className="flex items-center space-x-2 text-indigo-600 mb-1">
            <ShieldCheck className="w-4 h-4" />
            <span className="text-xs font-bold uppercase tracking-wider text-slate-500">Grounding</span>
          </div>
          <div className="text-xl sm:text-2xl font-extrabold text-slate-900">100% Policy Gated</div>
          <p className="text-xs text-slate-500 mt-1">Zero hallucinations; strict abstention over speculation</p>
        </div>

        <div className="p-4 rounded-2xl bg-white border border-slate-200/80 shadow-xs">
          <div className="flex items-center space-x-2 text-indigo-600 mb-1">
            <Zap className="w-4 h-4" />
            <span className="text-xs font-bold uppercase tracking-wider text-slate-500">Speed</span>
          </div>
          <div className="text-xl sm:text-2xl font-extrabold text-slate-900">&lt; 1.1s Latency</div>
          <p className="text-xs text-slate-500 mt-1">Full-duplex WebRTC with live speech interruption</p>
        </div>

        <div className="p-4 rounded-2xl bg-white border border-slate-200/80 shadow-xs">
          <div className="flex items-center space-x-2 text-indigo-600 mb-1">
            <Globe className="w-4 h-4" />
            <span className="text-xs font-bold uppercase tracking-wider text-slate-500">Languages</span>
          </div>
          <div className="text-xl sm:text-2xl font-extrabold text-slate-900">EN • తెలుగు • हिन्दी</div>
          <p className="text-xs text-slate-500 mt-1">Native pronunciation via Sarvam AI Bulbul TTS</p>
        </div>

        <div className="p-4 rounded-2xl bg-white border border-slate-200/80 shadow-xs">
          <div className="flex items-center space-x-2 text-indigo-600 mb-1">
            <BookOpen className="w-4 h-4" />
            <span className="text-xs font-bold uppercase tracking-wider text-slate-500">Corpus</span>
          </div>
          <div className="text-xl sm:text-2xl font-extrabold text-slate-900">817 Chunks</div>
          <p className="text-xs text-slate-500 mt-1">Indexed across 11 official university handbooks</p>
        </div>
      </div>
    </section>
  );
};
