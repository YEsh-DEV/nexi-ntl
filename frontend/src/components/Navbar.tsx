import React from 'react';
import { Sparkles, Activity } from 'lucide-react';

export const Navbar: React.FC = () => {
  return (
    <header className="w-full border-b border-slate-200/80 bg-white/80 backdrop-blur-md sticky top-0 z-50 transition-all">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand Logo & Name */}
        <div className="flex items-center space-x-3">
          <div className="h-9 w-9 rounded-xl bg-indigo-600 flex items-center justify-center text-white font-bold shadow-sm shadow-indigo-200">
            <span className="text-base tracking-tighter">N</span>
          </div>
          <div className="flex flex-col">
            <div className="flex items-center space-x-1.5">
              <span className="font-bold tracking-tight text-slate-900 text-lg">NEXi</span>
              <span className="text-[10px] font-semibold uppercase tracking-wider bg-indigo-50 text-indigo-700 px-1.5 py-0.5 rounded border border-indigo-100">
                SRM-AP
              </span>
            </div>
            <span className="text-[11px] text-slate-500 font-medium hidden sm:inline">Campus Voice AI</span>
          </div>
        </div>

        {/* Center Nav Links */}
        <nav className="hidden md:flex items-center space-x-8 text-sm font-medium text-slate-600">
          <a href="#about" className="hover:text-indigo-600 transition-colors">
            About & Architecture
          </a>
          <a href="#sandbox" className="hover:text-indigo-600 transition-colors flex items-center gap-1.5">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
            </span>
            Live Sandbox
          </a>
          <a href="#testimonials" className="hover:text-indigo-600 transition-colors">
            Student Reviews
          </a>
          <a href="#policies" className="hover:text-indigo-600 transition-colors">
            Ordinances
          </a>
        </nav>

        {/* Status Indicator & CTA Button */}
        <div className="flex items-center space-x-3">
          <div className="hidden lg:flex items-center space-x-1.5 text-xs text-slate-600 bg-slate-100/80 px-2.5 py-1 rounded-full border border-slate-200">
            <Activity className="w-3.5 h-3.5 text-emerald-500 animate-pulse" />
            <span className="font-medium">System Online</span>
          </div>
          <a
            href="#sandbox"
            className="inline-flex items-center space-x-1.5 bg-slate-900 hover:bg-slate-800 text-white text-xs sm:text-sm font-semibold px-4 py-2 rounded-xl transition-all shadow-sm hover:shadow"
          >
            <Sparkles className="w-3.5 h-3.5 text-indigo-300" />
            <span>Try Live Agent</span>
          </a>
        </div>
      </div>
    </header>
  );
};
