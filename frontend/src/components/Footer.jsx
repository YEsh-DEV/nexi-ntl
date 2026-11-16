import React from 'react';
import { ExternalLink } from 'lucide-react';

export const Footer = () => {
  return (
    <footer className="border-t border-rule bg-white py-12 sm:py-16 px-4 sm:px-8 lg:px-12 max-w-7xl mx-auto">
      <div className="max-w-5xl mx-auto">
        <div className="grid grid-cols-1 md:grid-cols-12 gap-12 mb-16">
          {/* Brand Signature */}
          <div className="md:col-span-6 space-y-4">
            <h3 className="font-display font-extrabold text-3xl sm:text-4xl text-black tracking-tight">
              yours, nexi
            </h3>
            <p className="text-sm font-display text-soft leading-relaxed max-w-md">
              For anything at all: <a href="mailto:nexi@srmap.edu.in" className="text-black u-ink font-semibold">nexi@srmap.edu.in</a>
              <br />
              Based at SRM University-AP, Amaravati, Andhra Pradesh.
              <br />
              Official voice intelligence grounded in university regulations.
            </p>
          </div>

          {/* Links Column 1 */}
          <div className="md:col-span-3 space-y-2.5 text-sm font-display">
            <div>
              <a href="#" className="u-ink text-black">home</a>
            </div>
            <div>
              <a href="#nutshell" className="u-ink text-black">our expertise</a>
            </div>
            <div>
              <a href="#how-we-work" className="u-ink text-black">how we work</a>
            </div>
            <div>
              <a href="#desk" className="u-ink text-black">on your desk</a>
            </div>
            <div>
              <a href="#sandbox" className="u-ink text-black font-semibold">live sandbox</a>
            </div>
            <div>
              <a href="#contact" className="u-ink text-black">contact</a>
            </div>
          </div>

          {/* Links Column 2: Ordinances */}
          <div className="md:col-span-3 space-y-2.5 text-sm font-display">
            <div>
              <a href="#sandbox" className="u-ink text-soft hover:text-black">academic regulations 2024–25</a>
            </div>
            <div>
              <a href="#sandbox" className="u-ink text-soft hover:text-black">hostel residence code</a>
            </div>
            <div>
              <a href="#sandbox" className="u-ink text-soft hover:text-black">attendance condonation</a>
            </div>
            <div>
              <a href="#sandbox" className="u-ink text-soft hover:text-black">research seed grants</a>
            </div>
            <div>
              <a
                href="https://srmap.edu.in"
                target="_blank"
                rel="noopener noreferrer"
                className="u-ink text-soft hover:text-black inline-flex items-center gap-1"
              >
                srmap.edu.in <ExternalLink className="w-3 h-3" />
              </a>
            </div>
          </div>
        </div>

        {/* Bottom Bar matching Altnota */}
        <div className="pt-8 border-t border-rule flex flex-col sm:flex-row items-center justify-between text-xs font-mono text-dim gap-4">
          <p>© {new Date().getFullYear()} SRM University-AP</p>
          <p className="hidden sm:block">press tab to navigate this page</p>
          <div className="flex items-center space-x-4">
            <a
              href="https://github.com/YEsh-DEV/nexi-ntl"
              target="_blank"
              rel="noopener noreferrer"
              className="u-ink text-black font-semibold"
            >
              github
            </a>
            <a
              href="https://srmap.edu.in"
              target="_blank"
              rel="noopener noreferrer"
              className="u-ink text-black font-semibold"
            >
              srmap
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
};
