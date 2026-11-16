import React from 'react';
import { ShieldCheck, Heart, ExternalLink } from 'lucide-react';

export const Footer = () => {
  return (
    <footer id="policies" className="border-t border-stone-200/80 bg-[#FAF9F6]">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-12">
          {/* Brand Info */}
          <div className="md:col-span-1 space-y-3">
            <div className="flex items-center space-x-2.5">
              <div className="h-8 w-8 rounded-lg bg-stone-900 flex items-center justify-center text-amber-400 font-mono font-bold text-sm">
                N
              </div>
              <span className="font-editorial text-xl text-stone-900 tracking-tight">NEXi</span>
            </div>
            <p className="text-xs text-stone-500 leading-relaxed font-sans">
              Official conversational campus intelligence layer for SRM University-AP, Andhra Pradesh, India.
            </p>
            <div className="flex items-center space-x-1.5 text-xs text-amber-800 font-mono font-medium pt-1">
              <ShieldCheck className="w-3.5 h-3.5 text-amber-600" />
              <span>100% Policy Grounded</span>
            </div>
          </div>

          {/* Core Ordinances Indexed */}
          <div>
            <h4 className="text-[11px] font-mono font-semibold uppercase tracking-wider text-stone-900 mb-3.5">
              Indexed Ordinances
            </h4>
            <ul className="space-y-2 text-xs text-stone-600 font-sans">
              <li>Academic Regulations 2024–25</li>
              <li>Attendance & Condonation Rules</li>
              <li>Hostel & Residence Code</li>
              <li>Exam Malpractice Bylaws</li>
              <li>Research Seed Grant Policies</li>
            </ul>
          </div>

          {/* Architecture & Tech Stack */}
          <div>
            <h4 className="text-[11px] font-mono font-semibold uppercase tracking-wider text-stone-900 mb-3.5">
              Underlying Stack
            </h4>
            <ul className="space-y-2 text-xs text-stone-600 font-mono">
              <li>LiveKit Agents WebRTC</li>
              <li>Google Gemini 3.5 Flash</li>
              <li>Deepgram Nova-3 ASR</li>
              <li>Sarvam AI Bulbul v3 TTS</li>
              <li>Okapi BM25 + ChromaDB RRF</li>
            </ul>
          </div>

          {/* Campus Resources */}
          <div>
            <h4 className="text-[11px] font-mono font-semibold uppercase tracking-wider text-stone-900 mb-3.5">
              Official Portals
            </h4>
            <ul className="space-y-2 text-xs text-stone-600 font-sans">
              <li>
                <a
                  href="https://srmap.edu.in"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-amber-700 inline-flex items-center gap-1 transition"
                >
                  SRM University-AP Portal <ExternalLink className="w-3 h-3 text-stone-400" />
                </a>
              </li>
              <li>
                <a
                  href="#sandbox"
                  className="hover:text-amber-700 inline-flex items-center gap-1 transition"
                >
                  Live Voice Sandbox
                </a>
              </li>
              <li>
                <a
                  href="https://github.com/YEsh-DEV/nexi-ntl"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-amber-700 inline-flex items-center gap-1 transition"
                >
                  Open Source Codebase <ExternalLink className="w-3 h-3 text-stone-400" />
                </a>
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="pt-8 border-t border-stone-200/70 flex flex-col sm:flex-row items-center justify-between text-xs text-stone-500 gap-3 font-mono">
          <p>© {new Date().getFullYear()} SRM University-AP. All rights reserved.</p>
          <p className="flex items-center gap-1">
            Engineered with <Heart className="w-3.5 h-3.5 text-amber-600 fill-amber-500 inline" /> for SRM AP Students & Faculty.
          </p>
        </div>
      </div>
    </footer>
  );
};
