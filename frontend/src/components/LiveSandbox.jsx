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
  Languages,
  ShieldCheck,
  AlertCircle,
  Radio,
  Loader2,
} from 'lucide-react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { useGSAP } from '@gsap/react';
import { useLiveKit } from '../hooks/useLiveKit';
import { queryKnowledge } from '../lib/api';

gsap.registerPlugin(ScrollTrigger, useGSAP);

const POPULAR_QUESTIONS = [
  {
    id: '1',
    category: 'Attendance',
    question: 'What is the 75% Attendance Rule?',
    language: 'en',
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
    category: 'Hostel Curfew',
    question: 'What are the hostel curfew timings?',
    language: 'en',
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
    language: 'te',
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
    category: 'Seed Grants',
    question: 'How can students apply for Seed Research Grants?',
    language: 'en',
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

export const LiveSandbox = () => {
  const containerRef = useRef(null);
  const orbRef = useRef(null);
  const chatScrollRef = useRef(null);
  const hasUserInteracted = useRef(false);

  const [selectedLanguage, setSelectedLanguage] = useState('en');
  const [inputMessage, setInputMessage] = useState('');
  const [isSearchingText, setIsSearchingText] = useState(false);
  const [activeCitation, setActiveCitation] = useState(POPULAR_QUESTIONS[0].citation);
  const [showCitationDrawer, setShowCitationDrawer] = useState(true);
  const [chatMessages, setChatMessages] = useState([
    {
      id: '1',
      sender: 'nexi',
      text: 'Hello! I am NEXi, your SRM University-AP campus intelligence assistant. You can speak directly via WebRTC or query our verified policy database in English, Telugu, or Hindi.',
      timestamp: 'Just now',
      citation: POPULAR_QUESTIONS[0].citation,
    },
  ]);

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

  const isVoiceActive = voiceStatus === 'connected';

  // GSAP ScrollTrigger Entrance
  useGSAP(
    () => {
      gsap.fromTo(
        '.sandbox-rise',
        { opacity: 0, y: 30 },
        {
          opacity: 1,
          y: 0,
          duration: 0.8,
          stagger: 0.15,
          ease: 'power3.out',
          scrollTrigger: {
            trigger: containerRef.current,
            start: 'top 80%',
          },
        }
      );
    },
    { scope: containerRef }
  );

  // Organic Idle Breathing Animation for Voice Orb
  useEffect(() => {
    if (!orbRef.current) return;

    if (voiceStatus === 'disconnected') {
      const breathingTween = gsap.to(orbRef.current, {
        scale: 1.05,
        duration: 2.2,
        repeat: -1,
        yoyo: true,
        ease: 'sine.inOut',
      });
      return () => {
        breathingTween.kill();
        if (orbRef.current) gsap.set(orbRef.current, { scale: 1 });
      };
    }
  }, [voiceStatus]);

  // SAFE CHAT SCROLL: Only scrolls inner chat box, NEVER calls window.scrollIntoView
  useEffect(() => {
    if (hasUserInteracted.current && chatScrollRef.current) {
      chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight;
    }
  }, [chatMessages, isSearchingText]);

  // Voice Interaction Toggle
  const handleToggleVoice = () => {
    hasUserInteracted.current = true;
    if (voiceStatus === 'connected' || voiceStatus === 'connecting') {
      disconnectVoice();
    } else {
      connectVoice();
    }
  };

  // Text Query Submission with Real Backend RAG API Wiring
  const executeQuery = async (queryText) => {
    if (!queryText.trim() || isSearchingText) return;

    hasUserInteracted.current = true;
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
      const result = await queryKnowledge(userText, selectedLanguage);
      const topCitation = result.citations && result.citations.length > 0 ? result.citations[0] : null;

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
    } catch (err) {
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

  const handleSendText = (e) => {
    e.preventDefault();
    executeQuery(inputMessage);
  };

  // Trigger from sample chip
  const handleSelectChip = (q) => {
    hasUserInteracted.current = true;
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
        text: 'Conversation reset. You can speak or type your campus questions.',
        timestamp: 'Just now',
      },
    ]);
  };

  return (
    <section
      id="sandbox"
      ref={containerRef}
      className="scene scroll-mt-20 py-12 sm:py-20 px-4 sm:px-8 lg:px-12 max-w-7xl mx-auto"
    >
      <div className="stage max-w-5xl">
        {/* Altnota Category Tag */}
        <p className="sandbox-rise font-mono text-xs uppercase tracking-[0.18em] text-black font-semibold mb-6 flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-[#FFE500]" />
          live interactive sandbox
        </p>

        {/* Section Headline */}
        <div className="sandbox-rise mb-12 sm:mb-14">
          <h2 className="poster-md max-w-3xl">
            Experience real-time voice and zero-hallucination RAG.
          </h2>
          <p className="mt-4 sm:mt-6 text-base sm:text-xl text-soft font-display leading-relaxed max-w-2xl">
            Speak directly through WebRTC or test queries against indexed SRM University-AP regulations in English, Telugu, and Hindi.
          </p>
        </div>

        {/* Altnota-style Minimalist Sandbox Container */}
        <div className="sandbox-rise border border-rule rounded-2xl bg-white overflow-hidden shadow-xs">
          {/* Top Control Bar */}
          <div className="px-4 sm:px-6 py-3.5 sm:py-4 border-b border-rule flex flex-wrap items-center justify-between gap-4 bg-[#F8F8F6]">
            {/* Language Selector */}
            <div className="flex items-center space-x-3">
              <span className="font-mono text-xs uppercase tracking-wider text-soft flex items-center gap-1.5">
                <Languages className="w-3.5 h-3.5 text-black" />
                Language:
              </span>
              <div className="inline-flex rounded-full bg-white p-1 border border-rule text-xs">
                <button
                  onClick={() => setSelectedLanguage('en')}
                  className={`px-3 py-1 rounded-full transition-all font-display font-medium ${
                    selectedLanguage === 'en'
                      ? 'bg-[#FFE500] text-black font-bold'
                      : 'text-soft hover:text-black'
                  }`}
                >
                  English
                </button>
                <button
                  onClick={() => setSelectedLanguage('te')}
                  className={`px-3 py-1 rounded-full transition-all font-display font-medium ${
                    selectedLanguage === 'te'
                      ? 'bg-[#FFE500] text-black font-bold'
                      : 'text-soft hover:text-black'
                  }`}
                >
                  తెలుగు (TE)
                </button>
                <button
                  onClick={() => setSelectedLanguage('hi')}
                  className={`px-3 py-1 rounded-full transition-all font-display font-medium ${
                    selectedLanguage === 'hi'
                      ? 'bg-[#FFE500] text-black font-bold'
                      : 'text-soft hover:text-black'
                  }`}
                >
                  हिन्दी (HI)
                </button>
              </div>
            </div>

            {/* Toggle Citation & Reset */}
            <div className="flex items-center space-x-3 text-xs">
              <button
                onClick={() => setShowCitationDrawer(!showCitationDrawer)}
                className={`inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-full border transition-all font-mono text-[11px] ${
                  showCitationDrawer
                    ? 'bg-[#FFE500] border-black text-black font-bold'
                    : 'bg-white border-rule text-soft hover:text-black'
                }`}
              >
                <BookOpen className="w-3.5 h-3.5" />
                <span>Citations {showCitationDrawer ? 'Active' : 'Muted'}</span>
              </button>
              <button
                onClick={handleResetConversation}
                className="inline-flex items-center space-x-1 px-3 py-1.5 rounded-full border border-rule bg-white hover:bg-black hover:text-white text-soft transition-colors font-mono text-[11px]"
                title="Reset conversation"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                <span>Reset</span>
              </button>
            </div>
          </div>

          {/* Split Screen Layout */}
          <div className="grid grid-cols-1 lg:grid-cols-12 min-h-[540px]">
            {/* Left Panel: Voice Orb */}
            <div className="lg:col-span-5 p-8 sm:p-10 border-b lg:border-b-0 lg:border-r border-rule flex flex-col justify-between items-center text-center bg-[#FDFDFD]">
              {/* WebRTC State */}
              <div className="w-full flex items-center justify-between text-xs">
                <span className="font-mono text-[11px] uppercase tracking-wider text-soft flex items-center gap-1.5">
                  <Radio className="w-3.5 h-3.5 text-black" />
                  LiveKit WebRTC
                </span>
                <span
                  className={`font-mono text-[10px] px-2.5 py-1 rounded-full flex items-center gap-1.5 border transition-all ${
                    voiceStatus === 'connected'
                      ? 'bg-[#FFE500]/30 text-black border-[#FFE500] font-bold'
                      : voiceStatus === 'connecting'
                      ? 'bg-amber-50 text-amber-800 border-amber-300 animate-pulse'
                      : voiceStatus === 'error'
                      ? 'bg-rose-50 text-rose-800 border-rose-300'
                      : 'bg-[#F8F8F6] text-soft border-rule'
                  }`}
                >
                  <span
                    className={`w-1.5 h-1.5 rounded-full ${
                      voiceStatus === 'connected'
                        ? 'bg-black'
                        : voiceStatus === 'connecting'
                        ? 'bg-amber-500'
                        : voiceStatus === 'error'
                        ? 'bg-rose-500'
                        : 'bg-stone-400'
                    }`}
                  />
                  {voiceStatus === 'connected'
                    ? 'Connected'
                    : voiceStatus === 'connecting'
                    ? 'Connecting...'
                    : voiceStatus === 'error'
                    ? 'Error'
                    : 'Ready'}
                </span>
              </div>

              {/* Central BRIGHT ELECTRIC YELLOW Voice Orb */}
              <div className="my-10 relative flex items-center justify-center">
                {/* Audio Level Wave Rings */}
                {voiceStatus === 'connected' && (
                  <>
                    <div
                      className="absolute w-52 h-52 rounded-full bg-[#FFE500]/30 transition-transform duration-75 pointer-events-none"
                      style={{
                        transform: `scale(${1 + audioLevel * 0.6})`,
                        opacity: 0.4 + audioLevel * 0.5,
                      }}
                    />
                    <div
                      className="absolute w-44 h-44 rounded-full bg-[#FFE500]/40 transition-transform duration-75 pointer-events-none"
                      style={{
                        transform: `scale(${1 + audioLevel * 0.4})`,
                        opacity: 0.5 + audioLevel * 0.4,
                      }}
                    />
                    {isAgentSpeaking && (
                      <div className="absolute w-60 h-60 rounded-full border-2 border-[#FFE500] animate-ping opacity-40 pointer-events-none" />
                    )}
                  </>
                )}

                {voiceStatus === 'connecting' && (
                  <div className="absolute w-48 h-48 rounded-full border-2 border-dashed border-[#FFE500] animate-spin pointer-events-none" />
                )}

                {/* Main Orb Button in BRIGHT ELECTRIC YELLOW */}
                <button
                  ref={orbRef}
                  onClick={handleToggleVoice}
                  className={`relative z-10 w-36 h-36 rounded-full flex flex-col items-center justify-center transition-all duration-300 shadow-md cursor-pointer ${
                    voiceStatus === 'connected'
                      ? isMuted
                        ? 'bg-stone-800 text-white ring-4 ring-stone-300'
                        : 'bg-[#FFE500] text-black scale-105 ring-4 ring-black/10'
                      : voiceStatus === 'connecting'
                      ? 'bg-[#FFE500] text-black animate-pulse'
                      : voiceStatus === 'error'
                      ? 'bg-rose-600 text-white ring-4 ring-rose-200'
                      : 'bg-[#FFE500] hover:bg-[#F2D800] text-black hover:scale-105'
                  }`}
                  aria-label="Toggle live voice"
                >
                  {voiceStatus === 'connecting' ? (
                    <Loader2 className="w-10 h-10 text-black animate-spin" />
                  ) : voiceStatus === 'connected' ? (
                    isMuted ? (
                      <MicOff className="w-10 h-10 text-rose-300" />
                    ) : isAgentSpeaking ? (
                      <Volume2 className="w-10 h-10 text-black animate-pulse" />
                    ) : (
                      <Mic className="w-10 h-10 text-black animate-bounce" />
                    )
                  ) : voiceStatus === 'error' ? (
                    <AlertCircle className="w-10 h-10 text-white" />
                  ) : (
                    <Mic className="w-10 h-10 text-black" />
                  )}
                  <span className="font-mono text-[11px] font-bold uppercase tracking-wider mt-2">
                    {voiceStatus === 'connected'
                      ? isMuted
                        ? 'Muted'
                        : isAgentSpeaking
                        ? 'Speaking...'
                        : 'Listening...'
                      : voiceStatus === 'connecting'
                      ? 'Connecting...'
                      : voiceStatus === 'error'
                      ? 'Retry'
                      : 'Tap to Speak'}
                  </span>
                </button>
              </div>

              {/* Connected Action Buttons */}
              {voiceStatus === 'connected' && (
                <div className="flex items-center space-x-2 mb-2">
                  <button
                    onClick={toggleMute}
                    className={`inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-full text-xs font-mono transition-all border ${
                      isMuted
                        ? 'bg-rose-100 text-rose-700 border-rose-200'
                        : 'bg-white text-black border-rule hover:bg-stone-50'
                    }`}
                  >
                    {isMuted ? <MicOff className="w-3.5 h-3.5" /> : <Mic className="w-3.5 h-3.5" />}
                    <span>{isMuted ? 'Unmute' : 'Mute'}</span>
                  </button>
                  <button
                    onClick={disconnectVoice}
                    className="px-4 py-1.5 rounded-full text-xs font-mono font-medium bg-black text-white hover:bg-stone-800 transition-all"
                  >
                    End Call
                  </button>
                </div>
              )}

              {/* Voice Error Banner */}
              {voiceError && (
                <div className="w-full mb-3 p-3 bg-rose-50 border border-rose-200 rounded-lg text-rose-700 text-xs text-left flex items-start space-x-2">
                  <AlertCircle className="w-4 h-4 shrink-0 text-rose-500 mt-0.5" />
                  <span className="text-[11px] leading-tight font-medium">{voiceError}</span>
                </div>
              )}

              {/* Chips */}
              <div className="w-full">
                <p className="text-xs text-soft mb-3 font-display">
                  {voiceStatus === 'disconnected' && 'Click the microphone or choose a sample ordinance inquiry below:'}
                  {voiceStatus === 'connected' && 'LiveKit WebRTC active. Audio is streamed bi-directionally.'}
                </p>
                <div className="flex flex-wrap items-center justify-center gap-1.5">
                  {POPULAR_QUESTIONS.map((q) => (
                    <button
                      key={q.id}
                      onClick={() => handleSelectChip(q)}
                      className="text-[11px] font-mono bg-white hover:bg-[#FFE500] hover:text-black hover:border-black text-soft px-3 py-1 rounded-full border border-rule transition-colors"
                    >
                      {q.category}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Right Panel: Chat Feed & Document Drawer */}
            <div className="lg:col-span-7 flex flex-col justify-between h-[540px] bg-white">
              {/* Transcript Header */}
              <div className="p-4 border-b border-rule flex items-center justify-between bg-[#FAFAFA]">
                <span className="font-mono text-xs uppercase tracking-wider text-black font-semibold flex items-center gap-2">
                  <MessageSquare className="w-3.5 h-3.5 text-black" />
                  Live Transcript &amp; Citations
                </span>
                <span className="font-mono text-[10px] text-black bg-[#FFE500] px-2.5 py-0.5 rounded-full font-bold">
                  {isVoiceActive ? 'WebRTC Active' : 'Zero-Hallucination Dual RAG'}
                </span>
              </div>

              {/* Messages Scroll Box (Pure internal scroll) */}
              <div
                ref={chatScrollRef}
                className="flex-1 overflow-y-auto p-6 space-y-4 pr-3 select-text"
              >
                {chatMessages.map((msg) => (
                  <div
                    key={msg.id}
                    className={`flex flex-col ${msg.sender === 'user' ? 'items-end' : 'items-start'}`}
                  >
                    <div className="flex items-center space-x-1.5 mb-1 px-1">
                      <span className="font-mono text-[10px] uppercase tracking-wider text-soft">
                        {msg.sender === 'user' ? 'You' : 'NEXi Assistant'}
                      </span>
                      <span className="font-mono text-[10px] text-dim">• {msg.timestamp}</span>
                    </div>

                    <div
                      className={`max-w-[88%] p-4 rounded-xl text-sm leading-relaxed ${
                        msg.sender === 'user'
                          ? 'bg-black text-white rounded-br-none'
                          : 'bg-[#F8F8F6] border border-rule text-ink rounded-bl-none'
                      }`}
                    >
                      {msg.text}

                      {/* Policy Verification Tag */}
                      {msg.sender === 'nexi' && msg.citation && (
                        <div className="mt-3 pt-2.5 border-t border-rule flex items-center justify-between text-[11px] font-mono">
                          <span className="flex items-center gap-1 text-black font-semibold">
                            <ShieldCheck className="w-3.5 h-3.5 text-black" />
                            Grounded in {msg.citation.document.split(' ')[0]}
                          </span>
                          <span className="bg-[#FFE500] text-black px-1.5 py-0.5 rounded font-bold text-[10px]">
                            {(msg.citation.relevanceScore * 100).toFixed(1)}% match
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                ))}

                {isSearchingText && (
                  <div className="flex flex-col items-start animate-fade-in">
                    <span className="font-mono text-[10px] uppercase tracking-wider mb-1 text-soft">
                      Searching University Ordinances...
                    </span>
                    <div className="p-3 rounded-xl bg-[#F8F8F6] border border-rule flex items-center space-x-2 text-xs text-soft">
                      <span className="h-1.5 w-1.5 rounded-full bg-black animate-pulse"></span>
                      <span className="h-1.5 w-1.5 rounded-full bg-black animate-pulse delay-150"></span>
                      <span className="h-1.5 w-1.5 rounded-full bg-black animate-pulse delay-300"></span>
                      <span className="font-mono text-[11px] ml-1">Reranking with Cross-Encoder...</span>
                    </div>
                  </div>
                )}
              </div>

              {/* Document Context Drawer */}
              {showCitationDrawer && activeCitation && (
                <div className="mx-4 mb-2 p-3.5 rounded-xl bg-[#FFE500]/15 border border-[#FFE500] text-xs font-mono">
                  <div className="flex items-center justify-between text-black mb-1">
                    <div className="flex items-center gap-1.5">
                      <BookOpen className="w-3.5 h-3.5 text-black" />
                      <span className="font-bold uppercase tracking-wider text-[11px]">
                        Live Document Context:
                      </span>
                    </div>
                    <span className="bg-[#FFE500] text-black px-2 py-0.5 rounded text-[10px] font-bold">
                      {(activeCitation.relevanceScore * 100).toFixed(1)}% Grounded
                    </span>
                  </div>
                  <div className="font-display font-semibold text-black text-[12px] mb-1">
                    {activeCitation.document} — <span className="font-mono text-soft">{activeCitation.section}</span>
                  </div>
                  <p className="text-soft font-display italic text-[11px] line-clamp-2 leading-relaxed">
                    "{activeCitation.excerpt}"
                  </p>
                </div>
              )}

              {/* Text Input Fallback */}
              <form
                onSubmit={handleSendText}
                className="p-3.5 border-t border-rule flex items-center space-x-2 bg-[#FAFAFA]"
              >
                <input
                  type="text"
                  placeholder="Ask about attendance condonation, hostel curfew, exam policy..."
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  className="flex-1 bg-white border border-rule rounded-full px-4 py-2.5 text-sm focus:outline-none focus:border-black transition font-display text-ink placeholder:text-dim"
                />
                <button
                  type="submit"
                  disabled={!inputMessage.trim() || isSearchingText}
                  className="bg-[#FFE500] hover:bg-[#F2D800] disabled:opacity-50 text-black p-2.5 rounded-full transition-all flex items-center justify-center cursor-pointer shadow-xs"
                  aria-label="Send query"
                >
                  <Send className="w-4 h-4 text-black" />
                </button>
              </form>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
