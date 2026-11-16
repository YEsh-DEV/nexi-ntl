import React from 'react';
import { ArrowRight, Radio } from 'lucide-react';

export const Navbar = () => {
  return (
    <header className="sticky top-0 z-50 w-full backdrop-blur-md bg-white/90 border-b border-rule transition-colors">
      <div className="max-w-7xl mx-auto px-6 sm:px-10 lg:px-12 h-16 flex items-center justify-between">
        {/* Altnota-style Logo: ~nexi */}
        <a
          href="#"
          className="flex items-center space-x-2 text-black hover:opacity-80 transition group"
          aria-label="NEXi SRM-AP Home"
        >
          <span className="font-display font-extrabold text-2xl tracking-tighter">
            ~nexi
          </span>
          <span className="font-mono text-[10px] uppercase tracking-widest text-soft border border-rule px-2 py-0.5 rounded-full">
            srm-ap
          </span>
        </a>

        {/* Navigation Links */}
        <nav className="hidden md:flex items-center space-x-8 text-sm font-display text-ink font-medium">
          <a href="#nutshell" className="hover:text-black u-ink">
            our expertise
          </a>
          <a href="#how-we-work" className="hover:text-black u-ink">
            how we work
          </a>
          <a href="#desk" className="hover:text-black u-ink">
            on your desk
          </a>
          <a href="#sandbox" className="hover:text-black u-ink flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-[#FFE500] animate-pulse"></span>
            live sandbox
          </a>
          <a href="#contact" className="hover:text-black u-ink">
            say hi <span className="inline-block transition-transform group-hover:translate-x-1">→</span>
          </a>
        </nav>

        {/* Right Action: Bright Yellow CTA Button */}
        <div className="flex items-center space-x-3">
          <a
            href="#sandbox"
            className="inline-flex items-center space-x-2 bg-electric hover:bg-[#F2D800] text-black font-display font-semibold text-xs sm:text-sm px-4 sm:px-5 py-2 sm:py-2.5 rounded-full transition-all duration-200 hover:-translate-y-0.5 active:translate-y-0 shadow-xs cursor-pointer"
          >
            <Radio className="w-3.5 h-3.5 text-black" />
            <span>try voice</span>
            <ArrowRight className="w-3.5 h-3.5 text-black" />
          </a>
        </div>
      </div>
    </header>
  );
};
