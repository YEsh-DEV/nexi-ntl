import React from 'react';
import { Star, MessageSquareQuote, CheckCircle2 } from 'lucide-react';

export const Testimonials: React.FC = () => {
  const reviews = [
    {
      id: 1,
      name: 'Rohan Sharma',
      department: 'B.Tech Computer Science & Engineering (3rd Year)',
      role: 'Day Scholar',
      rating: 5,
      review:
        'During end-sem exam preparations, I needed to confirm the exact attendance condonation percentage for medical leave. Instead of waiting in line at the student affairs office, NEXi gave me the exact section from the 2024 academic ordinances in less than 2 seconds.',
      badge: 'Attendance Condonation Query',
    },
    {
      id: 2,
      name: 'Pooja Reddy',
      department: 'B.Tech Electronics & Communication (2nd Year)',
      role: 'Hostel Resident',
      rating: 5,
      review:
        'మా అమ్మగారు హాస్టల్ ఫీజు మరియు కర్ఫ్యూ సమయాల గురించి తెలుసుకోవడానికి తెలుగులో మాట్లాడారు. నెక్సీ స్పష్టమైన తెలుగులో వివరించింది. Telugu voice synthesis feels natural and accurate.',
      badge: 'Multilingual Telugu Call',
    },
    {
      id: 3,
      name: 'Aditya Verma',
      department: 'M.Tech Artificial Intelligence & Machine Learning',
      role: 'Research Scholar',
      rating: 5,
      review:
        'Zero hallucination is what sets NEXi apart. When I asked about conference travel grant reimbursement procedures, it quoted the Directorate of Research guidelines word-for-word rather than guessing. Truly impressive engineering.',
      badge: 'Seed Grant & Research Query',
    },
  ];

  return (
    <section id="testimonials" className="py-20 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto border-t border-slate-200/60">
      <div className="max-w-3xl mx-auto text-center mb-16">
        <div className="inline-flex items-center space-x-1.5 bg-slate-100 text-slate-700 px-3 py-1 rounded-full text-xs font-semibold mb-4 border border-slate-200">
          <MessageSquareQuote className="w-3.5 h-3.5 text-indigo-600" />
          <span>Campus Community Feedback</span>
        </div>
        <h2 className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight">
          Trusted by SRM University-AP Students & Faculty
        </h2>
        <p className="mt-4 text-slate-600 text-base sm:text-lg leading-relaxed">
          See how real campus residents and scholars rely on NEXi for daily academic policies, hostel permissions, and administrative clarity.
        </p>
      </div>

      {/* Testimonials Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        {reviews.map((rev) => (
          <div
            key={rev.id}
            className="p-8 rounded-3xl bg-white border border-slate-200 shadow-xs hover:shadow-md transition-all duration-200 flex flex-col justify-between"
          >
            <div>
              {/* Star Rating & Query Badge */}
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-1 text-amber-400">
                  {[...Array(rev.rating)].map((_, i) => (
                    <Star key={i} className="w-4 h-4 fill-amber-400 text-amber-400" />
                  ))}
                </div>
                <span className="text-[10px] font-semibold uppercase tracking-wider text-indigo-700 bg-indigo-50 px-2 py-0.5 rounded border border-indigo-100/60">
                  {rev.badge}
                </span>
              </div>

              {/* Review Text */}
              <p className="text-slate-700 text-sm leading-relaxed mb-6 italic">
                "{rev.review}"
              </p>
            </div>

            {/* Student Info */}
            <div className="pt-4 border-t border-slate-100 flex items-center space-x-3">
              <div className="h-10 w-10 rounded-full bg-slate-900 text-white font-bold text-sm flex items-center justify-center shrink-0">
                {rev.name.charAt(0)}
              </div>
              <div className="overflow-hidden">
                <div className="flex items-center space-x-1.5">
                  <span className="font-bold text-slate-900 text-sm truncate">{rev.name}</span>
                  <span title="Verified SRM-AP Student">
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500 shrink-0" />
                  </span>
                </div>
                <div className="text-xs text-slate-500 truncate">{rev.department}</div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
};
