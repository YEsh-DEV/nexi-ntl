import React, { useState, useEffect, useRef } from 'react';
import {
  Mic,
  MicOff,
  Volume2,
  BookOpen,
  MessageSquare,
  Send,
  CheckCircle2,
  RotateCcw,
  Sparkles,
  Languages,
  ShieldCheck,
} from 'lucide-react';

type AgentState = 'idle' | 'listening' | 'processing' | 'speaking';
type Language = 'en' | 'te' | 'hi';

interface Citation {
  document: string;
  section: string;
  relevanceScore: number;
  excerpt: string;
}

interface ChatMessage {
  id: string;
  sender: 'user' | 'nexi';
  text: string;
  timestamp: string;
  language?: Language;
  citation?: Citation;
}

const POPULAR_QUESTIONS = [
  {
    id: '1',
    category: 'Attendance',
    question: 'What is the 75% Attendance Rule?',
    language: 'en' as Language,
    answer:
      'According to Section 4.2 of the SRM AP Academic Ordinances, a minimum of 75% aggregate attendance across lectures and practicals is mandatory to be eligible for end-semester examinations. Condonation up to 10% (between 65% and 74%) may be granted strictly on valid medical grounds approved by the Dean.',
    citation: {
      document: 'Academic Regulations & Ordinance 2024–2025',
      section: 'Section 4.2 (Attendance & Condonation)',
      relevanceScore: 0.984,
      excerpt:
        'Students having attendance between 65% and 74% due to medical reasons or sanctioned university representation may be condoned by the Competent Authority upon submission of proof within 3 working days.',
    },
  },
  {
    id: '2',
    category: 'Hostel',
    question: 'What are the hostel curfew timings?',
    language: 'en' as Language,
    answer:
      'The standard hostel entry curfew is strictly 9:00 PM for all undergraduate resident students. Biometric punch-in after 9:00 PM requires prior written approval from the Chief Warden via the Student Parent Portal (SMS notification triggered).',
    citation: {
      document: 'SRM AP Hostel Rules & Residence Code',
      section: 'Clause 8.1 (Curfew and Turnstile Ingress)',
      relevanceScore: 0.962,
      excerpt:
        'All residential students must return to their respective hostel blocks before 21:00 hrs. Late entry without approved gate pass shall incur a formal advisory and warden intimation.',
    },
  },
  {
    id: '3',
    category: 'తెలుగు (Telugu)',
    question: 'హాస్టల్ ఫీజు మరియు మెస్ వివరాలు ఏమిటి?',
    language: 'te' as Language,
    answer:
      'ఎస్ఆర్ఎం యూనివర్సిటీ-ఏపీ హాస్టల్ నిబంధనల ప్రకారం, మొదటి సంవత్సరం హాస్టల్ మరియు మెస్ రుసుము గది రకం (2/3/4 షేరింగ్, AC/Non-AC) ఆధారంగా నిర్ణయించబడుతుంది. వార్షిక ఫీజు చెల్లింపు గడువు సెమిస్టర్ ప్రారంభానికి ముందే ముగుస్తుంది.',
    citation: {
      document: 'Residential Services & Fee Structure 2024–25',
      section: 'Annexure B (Mess & Accommodation Dues)',
      relevanceScore: 0.948,
      excerpt:
        'Hostel room allocation is strictly subject to clearing full academic and mess dues prior to the orientation week commencement.',
    },
  },
  {
    id: '4',
    category: 'Research',
    question: 'How can students apply for Seed Research Grants?',
    language: 'en' as Language,
    answer:
      'Undergraduate and Postgraduate students can apply for the Student Seed Research Grant of up to ₹50,000 per project. Applications require a faculty mentor endorsement and must be submitted to the Directorate of Research via the IRIS portal.',
    citation: {
      document: 'SRM AP Research Promotion Guidelines',
      section: 'Section 3 (Undergraduate Seed Funding)',
      relevanceScore: 0.955,
      excerpt:
        'Eligible student teams may receive funding up to INR 50,000 for verified prototyping, experimental consumables, and peer-reviewed conference travel.',
    },
  },
];

export const LiveSandbox: React.FC = () => {
  const [agentState, setAgentState] = useState<AgentState>('idle');
  const [selectedLanguage, setSelectedLanguage] = useState<Language>('en');
  const [inputMessage, setInputMessage] = useState<string>('');
  const [activeCitation, setActiveCitation] = useState<Citation | null>(POPULAR_QUESTIONS[0].citation);
  const [showCitationDrawer, setShowCitationDrawer] = useState<boolean>(true);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      id: '1',
      sender: 'nexi',
      text: "Hello! I am NEXi, your SRM University-AP campus assistant. You can speak or type in English, Telugu, or Hindi. Ask me about attendance condonation, hostel curfews, exam malpractice rules, or research funding.",
      timestamp: 'Just now',
      citation: POPULAR_QUESTIONS[0].citation,
    },
  ]);

  const transcriptEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll chat to latest message
  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages, agentState]);

  // Voice Interaction Simulation
  const toggleListening = () => {
    if (agentState === 'idle') {
      setAgentState('listening');

      // Simulate Speech Turn: Listening -> Processing -> Speaking
      setTimeout(() => {
        setAgentState('processing');

        setTimeout(() => {
          setAgentState('speaking');
          const sample = POPULAR_QUESTIONS[0];
          setActiveCitation(sample.citation);
          setChatMessages((prev) => [
            ...prev,
            {
              id: Date.now().toString(),
              sender: 'user',
              text: sample.question,
              timestamp: 'Just now',
            },
            {
              id: (Date.now() + 1).toString(),
              sender: 'nexi',
              text: sample.answer,
              timestamp: 'Just now',
              citation: sample.citation,
            },
          ]);

          // Return to idle after speech finishes
          setTimeout(() => {
            setAgentState('idle');
          }, 3500);
        }, 1200);
      }, 2500);
    } else {
      setAgentState('idle');
    }
  };

  // Text Query Submission
  const handleSendText = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputMessage.trim()) return;

    const userText = inputMessage;
    setInputMessage('');
    setAgentState('processing');

    setChatMessages((prev) => [
      ...prev,
      {
        id: Date.now().toString(),
        sender: 'user',
        text: userText,
        timestamp: 'Just now',
      },
    ]);

    // Check if input matches known topics or generate grounded fallback
    setTimeout(() => {
      setAgentState('speaking');

      let replyText = `According to verified SRM University-AP policy records, queries regarding "${userText}" require consulting the academic handbook or contacting the department administrative office.`;
      let citationData: Citation = {
        document: 'SRM AP Academic Guidelines 2024–2025',
        section: 'General Provisions & Circulars',
        relevanceScore: 0.912,
        excerpt:
          'For unlisted or discretionary procedures, students must reference the student handbook or email the dean of academics office.',
      };

      const matchedQuestion = POPULAR_QUESTIONS.find((q) =>
        userText.toLowerCase().includes(q.category.toLowerCase())
      );

      if (matchedQuestion) {
        replyText = matchedQuestion.answer;
        citationData = matchedQuestion.citation;
      }

      setActiveCitation(citationData);

      setChatMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          sender: 'nexi',
          text: replyText,
          timestamp: 'Just now',
          citation: citationData,
        },
      ]);

      setTimeout(() => {
        setAgentState('idle');
      }, 3000);
    }, 1100);
  };

  // Trigger from sample chip
  const handleSelectChip = (q: typeof POPULAR_QUESTIONS[0]) => {
    setInputMessage(q.question);
    setActiveCitation(q.citation);
  };

  // Reset conversation
  const handleResetConversation = () => {
    setAgentState('idle');
    setChatMessages([
      {
        id: Date.now().toString(),
        sender: 'nexi',
        text: "Conversation reset. You can speak or type your campus questions.",
        timestamp: 'Just now',
      },
    ]);
  };

  return (
    <section id="sandbox" className="py-20 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
      {/* Section Header */}
      <div className="max-w-3xl mx-auto text-center mb-12">
        <div className="inline-flex items-center space-x-2 bg-indigo-50 text-indigo-700 px-3.5 py-1 rounded-full text-xs font-semibold mb-4 border border-indigo-100">
          <Sparkles className="w-3.5 h-3.5 text-indigo-600" />
          <span>Interactive Campus Intelligence Sandbox</span>
        </div>
        <h2 className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight">
          Experience Live Voice & Grounded RAG
        </h2>
        <p className="mt-3 text-slate-600 text-sm sm:text-base leading-relaxed">
          Test real-time interruption, language switching, and verifiable policy citations as if you were speaking directly with NEXi.
        </p>
      </div>

      {/* Main Sandbox Card */}
      <div className="bg-white rounded-3xl border border-slate-200/90 shadow-sm overflow-hidden">
        {/* Top Control Bar */}
        <div className="px-6 py-4 border-b border-slate-100 bg-slate-50/70 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center space-x-3">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-500 flex items-center gap-1.5">
              <Languages className="w-4 h-4 text-indigo-600" />
              Language:
            </span>
            <div className="inline-flex rounded-xl bg-slate-200/60 p-0.5 text-xs font-semibold">
              <button
                onClick={() => setSelectedLanguage('en')}
                className={`px-3 py-1 rounded-lg transition-all ${
                  selectedLanguage === 'en' ? 'bg-white text-indigo-600 shadow-xs' : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                English
              </button>
              <button
                onClick={() => setSelectedLanguage('te')}
                className={`px-3 py-1 rounded-lg transition-all ${
                  selectedLanguage === 'te' ? 'bg-white text-indigo-600 shadow-xs' : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                తెలుగు (TE)
              </button>
              <button
                onClick={() => setSelectedLanguage('hi')}
                className={`px-3 py-1 rounded-lg transition-all ${
                  selectedLanguage === 'hi' ? 'bg-white text-indigo-600 shadow-xs' : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                हिन्दी (HI)
              </button>
            </div>
          </div>

          <div className="flex items-center space-x-3 text-xs">
            <button
              onClick={() => setShowCitationDrawer(!showCitationDrawer)}
              className={`inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-lg border transition-colors ${
                showCitationDrawer
                  ? 'bg-indigo-50 border-indigo-200 text-indigo-700 font-medium'
                  : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'
              }`}
            >
              <BookOpen className="w-3.5 h-3.5" />
              <span>Policy Citations {showCitationDrawer ? 'Visible' : 'Hidden'}</span>
            </button>
            <button
              onClick={handleResetConversation}
              className="inline-flex items-center space-x-1 px-3 py-1.5 rounded-lg border border-slate-200 bg-white hover:bg-slate-50 text-slate-600 transition-colors"
              title="Reset conversation"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Reset</span>
            </button>
          </div>
        </div>

        {/* Split-Screen Interactive Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-12 min-h-[520px]">
          {/* Left Panel: Voice Orb & Interactive State Controls */}
          <div className="lg:col-span-5 p-8 border-b lg:border-b-0 lg:border-r border-slate-100 flex flex-col justify-between items-center text-center bg-gradient-to-b from-slate-50/50 via-white to-slate-50/30">
            <div className="w-full flex items-center justify-between text-xs text-slate-500">
              <span className="font-semibold uppercase tracking-wider text-[11px]">Voice Interface</span>
              <span
                className={`font-medium px-2 py-0.5 rounded-full text-[11px] ${
                  agentState === 'idle'
                    ? 'bg-slate-100 text-slate-600'
                    : agentState === 'listening'
                    ? 'bg-indigo-50 text-indigo-600 border border-indigo-100 animate-pulse'
                    : agentState === 'processing'
                    ? 'bg-amber-50 text-amber-600 border border-amber-100'
                    : 'bg-emerald-50 text-emerald-600 border border-emerald-100'
                }`}
              >
                ● {agentState.toUpperCase()}
              </span>
            </div>

            {/* Central Animated Voice Orb */}
            <div className="my-10 relative flex items-center justify-center">
              {/* Outer Glow & Radar Rings */}
              {agentState === 'listening' && (
                <>
                  <div className="absolute w-52 h-52 rounded-full bg-indigo-100 animate-ping opacity-60 pointer-events-none" />
                  <div className="absolute w-40 h-40 rounded-full bg-indigo-200/50 animate-pulse pointer-events-none" />
                </>
              )}
              {agentState === 'speaking' && (
                <>
                  <div className="absolute w-56 h-56 rounded-full bg-emerald-100 animate-pulse opacity-75 pointer-events-none" />
                  <div className="absolute w-44 h-44 rounded-full bg-emerald-200/50 animate-ping opacity-40 pointer-events-none" />
                </>
              )}
              {agentState === 'processing' && (
                <div className="absolute w-44 h-44 rounded-full border-2 border-indigo-400 border-dashed animate-spin pointer-events-none opacity-60" />
              )}

              {/* Main Microphone Button */}
              <button
                onClick={toggleListening}
                className={`relative z-10 w-32 h-32 rounded-full flex flex-col items-center justify-center transition-all duration-300 shadow-xl cursor-pointer ${
                  agentState === 'listening'
                    ? 'bg-indigo-600 text-white scale-105 ring-4 ring-indigo-200'
                    : agentState === 'speaking'
                    ? 'bg-emerald-600 text-white scale-105 ring-4 ring-emerald-200'
                    : agentState === 'processing'
                    ? 'bg-amber-600 text-white animate-pulse'
                    : 'bg-slate-900 hover:bg-slate-800 text-white hover:scale-102'
                }`}
                aria-label="Toggle voice assistant"
              >
                {agentState === 'listening' ? (
                  <MicOff className="w-9 h-9 animate-bounce text-white" />
                ) : agentState === 'speaking' ? (
                  <Volume2 className="w-9 h-9 text-white animate-pulse" />
                ) : (
                  <Mic className="w-9 h-9 text-white" />
                )}
                <span className="text-[11px] uppercase font-bold tracking-wider mt-1.5 opacity-90">
                  {agentState === 'idle'
                    ? 'Tap to Speak'
                    : agentState === 'listening'
                    ? 'Listening...'
                    : agentState === 'processing'
                    ? 'Searching...'
                    : 'Speaking...'}
                </span>
              </button>
            </div>

            {/* Quick Helper Subtext & Instructions */}
            <div className="w-full">
              <p className="text-xs text-slate-500 mb-4">
                {agentState === 'idle' && 'Click the microphone or choose a sample question below.'}
                {agentState === 'listening' && 'Listening to your microphone... Speak clearly.'}
                {agentState === 'processing' && 'Checking BM25 and vector index for matching campus ordinances...'}
                {agentState === 'speaking' && 'Streaming neural audio via Sarvam AI. Click to interrupt.'}
              </p>

              {/* Sample Topic Chips */}
              <div className="flex flex-wrap items-center justify-center gap-1.5">
                {POPULAR_QUESTIONS.map((q) => (
                  <button
                    key={q.id}
                    onClick={() => handleSelectChip(q)}
                    className="text-[11px] font-medium bg-white hover:bg-indigo-50 hover:text-indigo-600 text-slate-600 px-2.5 py-1 rounded-lg border border-slate-200 transition-colors"
                  >
                    {q.category}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Right Panel: Transcript Feed & Document Drawer */}
          <div className="lg:col-span-7 flex flex-col justify-between h-[540px] bg-white">
            {/* Transcript Header */}
            <div className="p-4 border-b border-slate-100 flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-600 uppercase tracking-wider flex items-center gap-2">
                <MessageSquare className="w-4 h-4 text-indigo-600" />
                Live Transcript & Citations
              </span>
              <span className="text-xs text-emerald-600 font-semibold flex items-center gap-1 bg-emerald-50 px-2 py-0.5 rounded-full border border-emerald-100">
                <CheckCircle2 className="w-3.5 h-3.5" /> Full-Duplex WebRTC
              </span>
            </div>

            {/* Messages Scroll Area */}
            <div className="flex-1 overflow-y-auto p-6 space-y-4 pr-3">
              {chatMessages.map((msg) => (
                <div
                  key={msg.id}
                  className={`flex flex-col ${msg.sender === 'user' ? 'items-end' : 'items-start'}`}
                >
                  <div className="flex items-center space-x-1.5 mb-1 px-1">
                    <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                      {msg.sender === 'user' ? 'You' : 'NEXi Assistant'}
                    </span>
                    <span className="text-[10px] text-slate-400">• {msg.timestamp}</span>
                  </div>

                  <div
                    className={`max-w-[90%] p-4 rounded-2xl text-sm leading-relaxed shadow-xs ${
                      msg.sender === 'user'
                        ? 'bg-indigo-600 text-white rounded-br-none'
                        : 'bg-slate-50 border border-slate-200/80 text-slate-800 rounded-bl-none'
                    }`}
                  >
                    {msg.text}

                    {/* Policy Verification Tag on NEXi messages */}
                    {msg.sender === 'nexi' && msg.citation && (
                      <div className="mt-3 pt-2.5 border-t border-slate-200/60 flex items-center justify-between text-[11px] text-slate-500">
                        <span className="flex items-center gap-1 text-emerald-600 font-medium">
                          <ShieldCheck className="w-3.5 h-3.5" />
                          Grounded in {msg.citation.document.split(' ')[0]} Policy
                        </span>
                        <span className="font-mono text-slate-400 text-[10px]">
                          Conf: {(msg.citation.relevanceScore * 100).toFixed(1)}%
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {/* Processing Typing Bubble */}
              {agentState === 'processing' && (
                <div className="flex flex-col items-start animate-fade-in">
                  <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1 px-1">
                    NEXi Searching Ordinances...
                  </span>
                  <div className="p-3.5 rounded-2xl bg-slate-100 rounded-bl-none flex items-center space-x-2 text-xs text-slate-500">
                    <span className="h-2 w-2 rounded-full bg-indigo-500 animate-pulse"></span>
                    <span className="h-2 w-2 rounded-full bg-indigo-500 animate-pulse delay-150"></span>
                    <span className="h-2 w-2 rounded-full bg-indigo-500 animate-pulse delay-300"></span>
                    <span className="text-slate-600 font-medium ml-1">Reranking policy chunks...</span>
                  </div>
                </div>
              )}

              <div ref={transcriptEndRef} />
            </div>

            {/* Document Context Drawer (Collapsible) */}
            {showCitationDrawer && activeCitation && (
              <div className="mx-4 mb-2 p-3.5 rounded-2xl bg-indigo-50/70 border border-indigo-100 text-xs">
                <div className="flex items-center justify-between font-semibold text-indigo-900 mb-1">
                  <div className="flex items-center gap-1.5">
                    <BookOpen className="w-3.5 h-3.5 text-indigo-600" />
                    <span>Active Document Reference:</span>
                  </div>
                  <span className="text-[10px] bg-indigo-100 text-indigo-700 font-mono px-1.5 py-0.5 rounded">
                    Relevance: {(activeCitation.relevanceScore * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="font-medium text-slate-800 text-[11px] mb-1">
                  {activeCitation.document} — <span className="text-indigo-600">{activeCitation.section}</span>
                </div>
                <p className="text-slate-600 italic text-[11px] line-clamp-2 leading-relaxed">
                  "{activeCitation.excerpt}"
                </p>
              </div>
            )}

            {/* Text Input Fallback Bar */}
            <form onSubmit={handleSendText} className="p-4 border-t border-slate-100 flex items-center space-x-2 bg-slate-50/40">
              <input
                type="text"
                placeholder="Ask about attendance condonation, hostel curfew, exam policy..."
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                className="flex-1 bg-white border border-slate-200 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition shadow-xs"
              />
              <button
                type="submit"
                disabled={!inputMessage.trim()}
                className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white p-2.5 rounded-xl transition shadow-xs flex items-center justify-center cursor-pointer"
                aria-label="Send query"
              >
                <Send className="w-4 h-4" />
              </button>
            </form>
          </div>
        </div>
      </div>
    </section>
  );
};
