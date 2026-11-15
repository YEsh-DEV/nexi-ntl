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
  AlertCircle,
  Radio,
  Loader2,
} from 'lucide-react';
import { useLiveKit } from '../hooks/useLiveKit';
import { queryKnowledge, Citation } from '../lib/api';

type Language = 'en' | 'te' | 'hi';

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
  const [selectedLanguage, setSelectedLanguage] = useState<Language>('en');
  const [inputMessage, setInputMessage] = useState<string>('');
  const [isSearchingText, setIsSearchingText] = useState<boolean>(false);
  const [activeCitation, setActiveCitation] = useState<Citation | null>(POPULAR_QUESTIONS[0].citation);
  const [showCitationDrawer, setShowCitationDrawer] = useState<boolean>(true);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      id: '1',
      sender: 'nexi',
      text: "Hello! I am NEXi, your SRM University-AP campus intelligence assistant. You can speak directly via WebRTC or query our verified policy database in English, Telugu, or Hindi.",
      timestamp: 'Just now',
      citation: POPULAR_QUESTIONS[0].citation,
    },
  ]);

  const transcriptEndRef = useRef<HTMLDivElement>(null);

  // LiveKit WebRTC Voice Hook
  const {
    status: voiceStatus,
    isMuted,
    audioLevel,
    isAgentSpeaking,
    errorMessage: voiceError,
    connect: connectVoice,
    disconnect: disconnectVoice,
    toggleMute,
  } = useLiveKit({
    language: selectedLanguage,
    onError: (err) => {
      console.warn('LiveKit voice session error:', err);
    },
  });

  // Effective agent state
  const isVoiceActive = voiceStatus === 'connected';
  const effectiveAgentState: 'idle' | 'connecting' | 'listening' | 'processing' | 'speaking' | 'error' =
    isSearchingText
      ? 'processing'
      : voiceStatus === 'connecting'
      ? 'connecting'
      : voiceStatus === 'error'
      ? 'error'
      : isVoiceActive
      ? isAgentSpeaking
        ? 'speaking'
        : 'listening'
      : 'idle';

  // Auto-scroll chat to latest message
  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages, effectiveAgentState]);

  // Voice Interaction Toggle
  const handleToggleVoice = () => {
    if (voiceStatus === 'connected') {
      disconnectVoice();
    } else if (voiceStatus === 'connecting') {
      disconnectVoice();
    } else {
      connectVoice();
    }
  };

  // Text Query Submission with Real Backend RAG API Wiring
  const executeQuery = async (queryText: string) => {
    if (!queryText.trim() || isSearchingText) return;

    const userText = queryText.trim();
    setInputMessage('');
    setIsSearchingText(true);

    const userMsgId = Date.now().toString();
    setChatMessages((prev) => [
      ...prev,
      {
        id: userMsgId,
        sender: 'user',
        text: userText,
        timestamp: 'Just now',
      },
    ]);

    try {
      // Real API request to backend POST /api/query
      const result = await queryKnowledge(userText, selectedLanguage);

      const topCitation: Citation | undefined =
        result.citations && result.citations.length > 0 ? result.citations[0] : undefined;

      if (topCitation) {
        setActiveCitation(topCitation);
      }

      setChatMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          sender: 'nexi',
          text: result.answer,
          timestamp: 'Just now',
          citation: topCitation,
        },
      ]);
    } catch (err: unknown) {
      const errString = err instanceof Error ? err.message : String(err);
      setChatMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          sender: 'nexi',
          text: `Unable to retrieve university policy records: ${errString}. Please verify backend status.`,
          timestamp: 'Just now',
        },
      ]);
    } finally {
      setIsSearchingText(false);
    }
  };

  const handleSendText = (e: React.FormEvent) => {
    e.preventDefault();
    executeQuery(inputMessage);
  };

  // Trigger from sample chip
  const handleSelectChip = (q: typeof POPULAR_QUESTIONS[0]) => {
    setInputMessage(q.question);
    executeQuery(q.question);
  };

  // Reset conversation
  const handleResetConversation = () => {
    disconnectVoice();
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
          Test real-time bi-directional WebRTC voice streaming, language switching, and verifiable policy citations as if you were speaking directly with NEXi.
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
        <div className="grid grid-cols-1 lg:grid-cols-12 min-h-[540px]">
          {/* Left Panel: Voice Orb & Interactive State Controls */}
          <div className="lg:col-span-5 p-8 border-b lg:border-b-0 lg:border-r border-slate-100 flex flex-col justify-between items-center text-center bg-gradient-to-b from-slate-50/50 via-white to-slate-50/30">
            {/* Connection Status Badge */}
            <div className="w-full flex items-center justify-between text-xs text-slate-500">
              <span className="font-semibold uppercase tracking-wider text-[11px] flex items-center gap-1.5">
                <Radio className="w-3.5 h-3.5 text-indigo-600" />
                LiveKit WebRTC
              </span>
              <span
                className={`font-medium px-2.5 py-0.5 rounded-full text-[11px] flex items-center gap-1 ${
                  voiceStatus === 'connected'
                    ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                    : voiceStatus === 'connecting'
                    ? 'bg-amber-50 text-amber-700 border border-amber-200 animate-pulse'
                    : voiceStatus === 'error'
                    ? 'bg-rose-50 text-rose-700 border border-rose-200'
                    : 'bg-slate-100 text-slate-600'
                }`}
              >
                <span
                  className={`w-1.5 h-1.5 rounded-full ${
                    voiceStatus === 'connected'
                      ? 'bg-emerald-500'
                      : voiceStatus === 'connecting'
                      ? 'bg-amber-500'
                      : voiceStatus === 'error'
                      ? 'bg-rose-500'
                      : 'bg-slate-400'
                  }`}
                />
                {voiceStatus === 'connected'
                  ? 'Connected'
                  : voiceStatus === 'connecting'
                  ? 'Connecting...'
                  : voiceStatus === 'error'
                  ? 'Connection Error'
                  : 'Ready to Connect'}
              </span>
            </div>

            {/* Central Animated Voice Orb */}
            <div className="my-8 relative flex items-center justify-center">
              {/* Outer Glow & Radar Rings reacting to real WebRTC audio level */}
              {voiceStatus === 'connected' && (
                <>
                  <div
                    className="absolute w-48 h-48 rounded-full bg-indigo-200/50 transition-transform duration-75 pointer-events-none"
                    style={{
                      transform: `scale(${1 + audioLevel * 0.5})`,
                      opacity: 0.35 + audioLevel * 0.55,
                    }}
                  />
                  <div
                    className="absolute w-40 h-40 rounded-full bg-indigo-300/40 transition-transform duration-75 pointer-events-none"
                    style={{
                      transform: `scale(${1 + audioLevel * 0.3})`,
                      opacity: 0.45 + audioLevel * 0.45,
                    }}
                  />
                </>
              )}

              {voiceStatus === 'connecting' && (
                <div className="absolute w-44 h-44 rounded-full border-2 border-indigo-400 border-dashed animate-spin pointer-events-none opacity-60" />
              )}

              {/* Main Voice Orb Button */}
              <button
                onClick={handleToggleVoice}
                className={`relative z-10 w-32 h-32 rounded-full flex flex-col items-center justify-center transition-all duration-300 shadow-xl cursor-pointer ${
                  voiceStatus === 'connected'
                    ? isMuted
                      ? 'bg-slate-700 text-white ring-4 ring-slate-200'
                      : 'bg-indigo-600 text-white scale-105 ring-4 ring-indigo-200'
                    : voiceStatus === 'connecting'
                    ? 'bg-amber-600 text-white animate-pulse'
                    : voiceStatus === 'error'
                    ? 'bg-rose-600 text-white ring-4 ring-rose-200'
                    : 'bg-slate-900 hover:bg-slate-800 text-white hover:scale-102'
                }`}
                aria-label="Toggle voice assistant"
              >
                {voiceStatus === 'connecting' ? (
                  <Loader2 className="w-9 h-9 text-white animate-spin" />
                ) : voiceStatus === 'connected' ? (
                  isMuted ? (
                    <MicOff className="w-9 h-9 text-rose-300" />
                  ) : isAgentSpeaking ? (
                    <Volume2 className="w-9 h-9 text-emerald-300 animate-pulse" />
                  ) : (
                    <Mic className="w-9 h-9 text-white animate-bounce" />
                  )
                ) : voiceStatus === 'error' ? (
                  <AlertCircle className="w-9 h-9 text-white" />
                ) : (
                  <Mic className="w-9 h-9 text-white" />
                )}
                <span className="text-[11px] uppercase font-bold tracking-wider mt-1.5 opacity-90">
                  {voiceStatus === 'connected'
                    ? isMuted
                      ? 'Muted'
                      : isAgentSpeaking
                      ? 'Speaking...'
                      : 'Listening...'
                    : voiceStatus === 'connecting'
                    ? 'Connecting...'
                    : voiceStatus === 'error'
                    ? 'Retry Voice'
                    : 'Tap to Speak'}
                </span>
              </button>
            </div>

            {/* Mic Mute & Controls when Connected */}
            {voiceStatus === 'connected' && (
              <div className="flex items-center space-x-2 mb-2">
                <button
                  onClick={toggleMute}
                  className={`inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-full text-xs font-semibold transition ${
                    isMuted
                      ? 'bg-rose-100 text-rose-700 border border-rose-200'
                      : 'bg-slate-100 text-slate-700 hover:bg-slate-200 border border-slate-200'
                  }`}
                >
                  {isMuted ? <MicOff className="w-3.5 h-3.5" /> : <Mic className="w-3.5 h-3.5" />}
                  <span>{isMuted ? 'Unmute Microphone' : 'Mute Microphone'}</span>
                </button>
                <button
                  onClick={disconnectVoice}
                  className="px-3 py-1.5 rounded-full text-xs font-semibold bg-rose-600 hover:bg-rose-700 text-white transition shadow-xs"
                >
                  End Call
                </button>
              </div>
            )}

            {/* Error Feedback Banner */}
            {voiceError && (
              <div className="w-full mb-3 p-2 bg-rose-50 border border-rose-200 rounded-xl text-rose-700 text-xs text-left flex items-start space-x-1.5">
                <AlertCircle className="w-4 h-4 shrink-0 text-rose-500 mt-0.5" />
                <span className="text-[11px] leading-tight">{voiceError}</span>
              </div>
            )}

            {/* Quick Helper Subtext & Instructions */}
            <div className="w-full">
              <p className="text-xs text-slate-500 mb-4">
                {voiceStatus === 'disconnected' && 'Click the microphone to start a live voice session or pick a question below.'}
                {voiceStatus === 'connecting' && 'Negotiating SDP & ICE candidates with LiveKit room...'}
                {voiceStatus === 'connected' &&
                  (isAgentSpeaking
                    ? 'Streaming neural audio from Sarvam Bulbul TTS.'
                    : 'Listening to your microphone with WebRTC noise cancellation.')}
                {voiceStatus === 'error' && 'Could not reach LiveKit room. Check server credentials or use text search below.'}
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
                <CheckCircle2 className="w-3.5 h-3.5" />
                {voiceStatus === 'connected' ? 'Active WebRTC Session' : 'Zero-Hallucination Dual RAG'}
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
                        <span className="font-mono text-slate-500 text-[10px] font-semibold">
                          Score: {(msg.citation.relevanceScore * 100).toFixed(1)}%
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {/* Processing Typing Bubble */}
              {isSearchingText && (
                <div className="flex flex-col items-start animate-fade-in">
                  <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1 px-1">
                    NEXi Searching Ordinances...
                  </span>
                  <div className="p-3.5 rounded-2xl bg-slate-100 rounded-bl-none flex items-center space-x-2 text-xs text-slate-500">
                    <span className="h-2 w-2 rounded-full bg-indigo-500 animate-pulse"></span>
                    <span className="h-2 w-2 rounded-full bg-indigo-500 animate-pulse delay-150"></span>
                    <span className="h-2 w-2 rounded-full bg-indigo-500 animate-pulse delay-300"></span>
                    <span className="text-slate-600 font-medium ml-1">Reranking policy chunks against Chroma DB...</span>
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
                    <span>Live Document Context:</span>
                  </div>
                  <span className="text-[10px] bg-indigo-100 text-indigo-700 font-mono px-1.5 py-0.5 rounded font-bold">
                    Confidence: {(activeCitation.relevanceScore * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="font-medium text-slate-800 text-[11px] mb-1">
                  <span className="text-slate-900 font-semibold">{activeCitation.document}</span> —{' '}
                  <span className="text-indigo-600">{activeCitation.section}</span>
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
                disabled={!inputMessage.trim() || isSearchingText}
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
