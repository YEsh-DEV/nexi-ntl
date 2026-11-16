import React from 'react';
import { Sparkles, Activity } from 'lucide-react';

export const Navbar = () => {
  return (
    <header className="w-full border-b border-stone-200/80 bg-[#FAF9F6]/85 backdrop-blur-md sticky top-0 z-50 transition-all">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-18 flex items-center justify-between">
        {/* Brand Logo & Editorial Monogram */}
        <div className="flex items-center space-x-3.5">
          <div className="h-9 w-9 rounded-lg bg-stone-900 text-amber-400 font-editorial font-bold text-lg flex items-center justify-center border border-stone-800 shadow-xs">
            N
          </div>
          <div className="flex flex-col">
            <div className="flex items-center space-x-2">
              <span className="font-editorial text-xl font-bold tracking-tight text-stone-900">
                NEXi
              </span>
              <span className="text-[10px] font-mono font-semibold tracking-wider uppercase bg-amber-100/80 text-amber-900 px-2 py-0.5 rounded border border-amber-300/60">
                SRM-AP
              </span>
            </div>
            <span className="text-[11px] font-mono text-stone-500 hidden sm:inline tracking-tight">
              Campus Intelligence Layer
            </span>
          </div>
        </div>

        {/* Center Editorial Navigation Links */}
        <nav className="hidden md:flex items-center space-x-8 text-xs uppercase tracking-wider font-semibold text-stone-600">
          <a
            href="#about"
            className="hover:text-stone-950 transition-colors py-1 hover:border-b-2 hover:border-amber-500"
          >
            System Architecture
          </a>
          <a
            href="#sandbox"
            className="hover:text-stone-950 transition-colors flex items-center gap-1.5 py-1 hover:border-b-2 hover:border-amber-500"
          >
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-amber-500"></span>
            </span>
            Live Sandbox
          </a>
          <a
            href="#testimonials"
            className="hover:text-stone-950 transition-colors py-1 hover:border-b-2 hover:border-amber-500"
          >
            Campus Verdicts
          </a>
          <a
            href="#ordinances"
            className="hover:text-stone-950 transition-colors py-1 hover:border-b-2 hover:border-amber-500"
          >
            Ordinances
          </a>
        </nav>

        {/* Status Indicator & Golden CTA */}
        <div className="flex items-center space-x-3">
          <div className="hidden lg:flex items-center space-x-2 text-[11px] font-mono text-stone-600 bg-white/80 px-3 py-1.5 rounded-full border border-stone-200/80 shadow-xs">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
            </span>
            <span className="font-medium tracking-tight">Operational</span>
          </div>
          <a
            href="#sandbox"
            className="inline-flex items-center space-x-1.5 bg-amber-500 hover:bg-amber-600 active:bg-amber-700 text-stone-950 text-xs sm:text-sm font-semibold px-4.5 py-2.5 rounded-xl transition-all shadow-sm shadow-amber-500/20 hover:shadow-md hover:-translate-y-0.5"
          >
            <Sparkles className="w-3.5 h-3.5 text-stone-950" />
            <span>Try Live Agent</span>
          </a>
        </div>
      </div>
    </header>
  );
};
