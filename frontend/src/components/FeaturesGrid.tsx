import React from 'react';
import { ShieldCheck, Languages, BookCheck, Cpu, CheckCircle2 } from 'lucide-react';

export const FeaturesGrid: React.FC = () => {
  const features = [
    {
      icon: ShieldCheck,
      badge: 'Dual-Engine RAG',
      title: 'Zero Hallucinations via Cross-Encoder Gating',
      description:
        'Standard LLMs fabricate dates, grades, and fee rules. NEXi combines in-memory Okapi BM25 with ChromaDB vector search and reciprocal rank fusion. A cross-encoder reranker rejects any query below score 3.564, ensuring answers are 100% grounded or safely declined.',
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

  return (
    <section id="about" className="py-20 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto border-t border-slate-200/60">
      <div className="max-w-3xl mx-auto text-center mb-16">
        <div className="inline-flex items-center space-x-1.5 bg-slate-100 text-slate-700 px-3 py-1 rounded-full text-xs font-semibold mb-4 border border-slate-200">
          <Cpu className="w-3.5 h-3.5 text-indigo-600" />
          <span>Engineered for Academic Rigor</span>
        </div>
        <h2 className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight">
          Enterprise Voice AI, Built for University Trust
        </h2>
        <p className="mt-4 text-slate-600 text-base sm:text-lg leading-relaxed">
          Unlike generic chatbot wrappers, NEXi is an end-to-end voice architecture designed to enforce institutional policy accuracy, low turn latency, and caller privacy.
        </p>
      </div>

      {/* 3-Column Feature Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        {features.map((feature, idx) => (
          <div
            key={idx}
            className="flex flex-col justify-between p-8 rounded-3xl bg-white border border-slate-200 shadow-xs hover:shadow-md hover:border-slate-300 transition-all duration-200 group"
          >
            <div>
              <div className="flex items-center justify-between mb-6">
                <div className="h-12 w-12 rounded-2xl bg-indigo-50 border border-indigo-100 flex items-center justify-center text-indigo-600 group-hover:scale-105 transition-transform">
                  <feature.icon className="w-6 h-6" />
                </div>
                <span className="text-[11px] font-bold uppercase tracking-wider text-indigo-600 bg-indigo-50 px-2.5 py-1 rounded-lg border border-indigo-100/60">
                  {feature.badge}
                </span>
              </div>

              <h3 className="text-xl font-bold text-slate-900 tracking-tight mb-3 group-hover:text-indigo-600 transition-colors">
                {feature.title}
              </h3>

              <p className="text-sm text-slate-600 leading-relaxed mb-6">
                {feature.description}
              </p>
            </div>

            <div className="pt-6 border-t border-slate-100">
              <ul className="space-y-2.5">
                {feature.points.map((pt, pIdx) => (
                  <li key={pIdx} className="flex items-start text-xs text-slate-600 space-x-2">
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
