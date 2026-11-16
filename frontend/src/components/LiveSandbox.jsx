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

  const transcriptEndRef = useRef(null);

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

  // GSAP Animations: ScrollTrigger Entrance & Ambient Glow Drift
  useGSAP(
    () => {
      // 1. Background Blob Drift
      gsap.to('.sandbox-blob-1', {
        x: -25,
        y: 30,
        duration: 11,
        repeat: -1,
        yoyo: true,
        ease: 'sine.inOut',
      });

      gsap.to('.sandbox-blob-2', {
        x: 30,
        y: -20,
        duration: 13,
        repeat: -1,
        yoyo: true,
        ease: 'sine.inOut',
        delay: 0.5,
      });

      // 2. Section ScrollTrigger Entrance
      gsap.fromTo(
        '.sandbox-header',
        { opacity: 0, y: 30 },
        {
          opacity: 1,
          y: 0,
          duration: 0.8,
          ease: 'power2.out',
          scrollTrigger: {
            trigger: '.sandbox-header',
            start: 'top 85%',
          },
        }
      );

      gsap.fromTo(
        '.sandbox-card-wrapper',
        { opacity: 0, y: 40 },
        {
          opacity: 1,
          y: 0,
          duration: 0.9,
          ease: 'power2.out',
          scrollTrigger: {
            trigger: '.sandbox-card-wrapper',
            start: 'top 80%',
          },
        }
      );
    },
    { scope: containerRef }
  );

  // Organic Idle Breathing Animation for the Central Golden Voice Orb
  useEffect(() => {
    if (!orbRef.current) return;

    if (voiceStatus === 'disconnected') {
      const breathingTween = gsap.to(orbRef.current, {
        scale: 1.04,
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

  // Auto-scroll chat to latest message
  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages, isSearchingText, voiceStatus]);

  // Voice Interaction Toggle
  const handleToggleVoice = () => {
    if (voiceStatus === 'connected' || voiceStatus === 'connecting') {
      disconnectVoice();
    } else {
      connectVoice();
    }
  };

  // Text Query Submission with Real Backend RAG API Wiring
  const executeQuery = async (queryText) => {
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
      className="relative py-28 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto overflow-hidden border-t border-stone-200/80"
    >
      {/* Soft Ambient Blurred Gradient Background Blobs */}
      <div
        className="sandbox-blob-1 absolute w-[520px] h-[520px] -top-32 -right-28 rounded-full bg-amber-200/25 filter blur-[140px] opacity-25 pointer-events-none -z-10"
        aria-hidden="true"
      />
      <div
        className="sandbox-blob-2 absolute w-[480px] h-[480px] -bottom-24 -left-20 rounded-full bg-emerald-100/20 filter blur-[140px] opacity-20 pointer-events-none -z-10"
        aria-hidden="true"
      />

      {/* Section Header */}
      <div className="sandbox-header max-w-3xl mx-auto text-center mb-16">
        <div className="inline-flex items-center space-x-2 bg-white/90 text-stone-700 px-4 py-1.5 rounded-full text-xs font-mono mb-4 border border-stone-200/90 shadow-xs">
          <Sparkles className="w-3.5 h-3.5 text-amber-600" />
          <span>INTERACTIVE SANDBOX // [VOICE_RAG]</span>
        </div>
        <h2 className="text-3xl sm:text-4xl lg:text-5xl font-editorial font-normal text-stone-900 tracking-tight">
          Experience Live Voice & Grounded RAG
        </h2>
        <p className="mt-4 text-stone-600 text-base sm:text-lg leading-relaxed font-normal max-w-2xl mx-auto">
          Test real-time bi-directional WebRTC voice streaming, language switching, and verifiable policy citations as if you were speaking directly with NEXi.
        </p>
      </div>

      {/* Main Sandbox Editorial Blueprint Card Container */}
      <div className="sandbox-card-wrapper bg-white/85 rounded-2xl border border-stone-200/90 shadow-xs overflow-hidden">
        {/* Top Control Bar */}
        <div className="px-6 py-4 border-b border-stone-200/80 bg-stone-50/70 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center space-x-3">
            <span className="text-xs font-mono uppercase tracking-wider text-stone-500 flex items-center gap-1.5">
              <Languages className="w-3.5 h-3.5 text-amber-700" />
              Dialect:
            </span>
            <div className="inline-flex rounded-lg bg-stone-200/60 p-1 text-xs">
              <button
                onClick={() => setSelectedLanguage('en')}
                className={`px-3 py-1 rounded-md transition-all font-medium ${
                  selectedLanguage === 'en'
                    ? 'bg-white text-stone-900 shadow-xs'
                    : 'text-stone-600 hover:text-stone-900'
                }`}
              >
                English
              </button>
              <button
                onClick={() => setSelectedLanguage('te')}
                className={`px-3 py-1 rounded-md transition-all font-medium ${
                  selectedLanguage === 'te'
                    ? 'bg-white text-stone-900 shadow-xs'
                    : 'text-stone-600 hover:text-stone-900'
                }`}
              >
                తెలుగు (TE)
              </button>
              <button
                onClick={() => setSelectedLanguage('hi')}
                className={`px-3 py-1 rounded-md transition-all font-medium ${
                  selectedLanguage === 'hi'
                    ? 'bg-white text-stone-900 shadow-xs'
                    : 'text-stone-600 hover:text-stone-900'
                }`}
              >
                हिन्दी (HI)
              </button>
            </div>
          </div>

          <div className="flex items-center space-x-3 text-xs">
            <button
              onClick={() => setShowCitationDrawer(!showCitationDrawer)}
              className={`inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-lg border transition-all font-mono text-[11px] ${
                showCitationDrawer
                  ? 'bg-amber-50 border-amber-300 text-amber-900 font-semibold shadow-xs'
                  : 'bg-white border-stone-200 text-stone-600 hover:bg-stone-50'
              }`}
            >
              <BookOpen className="w-3.5 h-3.5" />
              <span>Citations {showCitationDrawer ? 'Active' : 'Muted'}</span>
            </button>
            <button
              onClick={handleResetConversation}
              className="inline-flex items-center space-x-1 px-3 py-1.5 rounded-lg border border-stone-200 bg-white hover:bg-stone-50 text-stone-600 transition-colors shadow-xs font-mono text-[11px]"
              title="Reset conversation"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Reset</span>
            </button>
          </div>
        </div>

        {/* Split-Screen Interactive Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-12 min-h-[560px]">
          {/* Left Panel: Voice Orb & Interactive State Controls */}
          <div className="lg:col-span-5 p-8 sm:p-10 border-b lg:border-b-0 lg:border-r border-stone-200/80 flex flex-col justify-between items-center text-center bg-stone-50/40">
            {/* Connection Status Badge */}
            <div className="w-full flex items-center justify-between text-xs text-stone-500">
              <span className="font-mono text-[11px] uppercase tracking-wider flex items-center gap-1.5">
                <Radio className="w-3.5 h-3.5 text-amber-600" />
                LiveKit WebRTC
              </span>
              <span
                className={`font-mono text-[10px] px-2.5 py-1 rounded-full flex items-center gap-1.5 border transition-all ${
                  voiceStatus === 'connected'
                    ? 'bg-emerald-50 text-emerald-800 border-emerald-300 shadow-xs'
                    : voiceStatus === 'connecting'
                    ? 'bg-amber-50 text-amber-800 border-amber-300 animate-pulse'
                    : voiceStatus === 'error'
                    ? 'bg-rose-50 text-rose-800 border-rose-300'
                    : 'bg-stone-100 text-stone-600 border-stone-200'
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
                      : 'bg-stone-400'
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

            {/* Central Animated Golden Voice Orb */}
            <div className="my-10 relative flex items-center justify-center">
              {/* Dynamic Wave-Ripple Rings Reacting to Real WebRTC Audio Level */}
              {voiceStatus === 'connected' && (
                <>
                  <div
                    className="absolute w-52 h-52 rounded-full bg-amber-400/20 transition-transform duration-75 pointer-events-none"
                    style={{
                      transform: `scale(${1 + audioLevel * 0.55})`,
                      opacity: 0.35 + audioLevel * 0.55,
                    }}
                  />
                  <div
                    className="absolute w-42 h-42 rounded-full bg-amber-500/20 transition-transform duration-75 pointer-events-none"
                    style={{
                      transform: `scale(${1 + audioLevel * 0.35})`,
                      opacity: 0.45 + audioLevel * 0.45,
                    }}
                  />
                  {isAgentSpeaking && (
                    <div className="absolute w-60 h-60 rounded-full bg-emerald-200/40 animate-ping opacity-35 pointer-events-none" />
                  )}
                </>
              )}

              {voiceStatus === 'connecting' && (
                <div className="absolute w-48 h-48 rounded-full border-2 border-amber-500 border-dashed animate-spin pointer-events-none opacity-60" />
              )}

              {/* Main Golden Voice Orb Button */}
              <button
                ref={orbRef}
                onClick={handleToggleVoice}
                className={`relative z-10 w-36 h-36 rounded-full flex flex-col items-center justify-center transition-all duration-300 shadow-lg cursor-pointer ${
                  voiceStatus === 'connected'
                    ? isMuted
                      ? 'bg-stone-700 text-stone-100 ring-4 ring-stone-300'
                      : 'bg-gradient-to-tr from-amber-600 via-amber-500 to-yellow-500 text-stone-950 scale-105 ring-4 ring-amber-300 shadow-amber-400/40'
                    : voiceStatus === 'connecting'
                    ? 'bg-gradient-to-tr from-amber-600 to-amber-500 text-white animate-pulse'
                    : voiceStatus === 'error'
                    ? 'bg-gradient-to-tr from-rose-600 to-rose-500 text-white ring-4 ring-rose-200'
                    : 'bg-gradient-to-tr from-amber-500 via-amber-600 to-yellow-600 hover:from-amber-400 hover:to-amber-500 text-stone-950 hover:scale-103 shadow-amber-500/20'
                }`}
                aria-label="Toggle voice assistant"
              >
                {voiceStatus === 'connecting' ? (
                  <Loader2 className="w-10 h-10 text-white animate-spin" />
                ) : voiceStatus === 'connected' ? (
                  isMuted ? (
                    <MicOff className="w-10 h-10 text-rose-200" />
                  ) : isAgentSpeaking ? (
                    <Volume2 className="w-10 h-10 text-stone-950 animate-pulse" />
                  ) : (
                    <Mic className="w-10 h-10 text-stone-950 animate-bounce" />
                  )
                ) : voiceStatus === 'error' ? (
                  <AlertCircle className="w-10 h-10 text-white" />
                ) : (
                  <Mic className="w-10 h-10 text-stone-950" />
                )}
                <span className="text-[11px] uppercase font-mono font-bold tracking-wider mt-2 opacity-95">
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
              <div className="flex items-center space-x-2.5 mb-2">
                <button
                  onClick={toggleMute}
                  className={`inline-flex items-center space-x-1.5 px-3.5 py-1.5 rounded-lg text-xs font-mono transition-all shadow-xs ${
                    isMuted
                      ? 'bg-rose-100 text-rose-700 border border-rose-200'
                      : 'bg-white text-stone-700 hover:bg-stone-100 border border-stone-200'
                  }`}
                >
                  {isMuted ? <MicOff className="w-3.5 h-3.5" /> : <Mic className="w-3.5 h-3.5" />}
                  <span>{isMuted ? 'Unmute Mic' : 'Mute Mic'}</span>
                </button>
                <button
                  onClick={disconnectVoice}
                  className="px-4 py-1.5 rounded-lg text-xs font-mono font-medium bg-rose-600 hover:bg-rose-700 text-white transition-all shadow-xs"
                >
                  End Session
                </button>
              </div>
            )}

            {/* Error Feedback Banner */}
            {voiceError && (
              <div className="w-full mb-3 p-3 bg-rose-50/90 border border-rose-200 rounded-xl text-rose-700 text-xs text-left flex items-start space-x-2">
                <AlertCircle className="w-4 h-4 shrink-0 text-rose-500 mt-0.5" />
                <span className="text-[11px] leading-tight font-medium">{voiceError}</span>
              </div>
            )}

            {/* Subtext Instructions */}
            <div className="w-full">
              <p className="text-xs text-stone-500 mb-4 font-normal">
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
                    className="text-[11px] font-mono bg-white hover:bg-amber-50 hover:text-amber-900 hover:border-amber-300 text-stone-600 px-3 py-1 rounded-md border border-stone-200/90 transition-all shadow-xs"
                  >
                    {q.category}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Right Panel: Transcript Feed & Document Drawer */}
          <div className="lg:col-span-7 flex flex-col justify-between h-[560px] bg-white/70">
            {/* Transcript Header */}
            <div className="p-4 border-b border-stone-200/80 flex items-center justify-between">
              <span className="text-xs font-mono text-stone-700 uppercase tracking-wider flex items-center gap-2">
                <MessageSquare className="w-3.5 h-3.5 text-amber-700" />
                Live Transcript & Citations
              </span>
              <span className="text-[11px] text-amber-900 font-mono flex items-center gap-1 bg-amber-50 px-2.5 py-0.5 rounded-md border border-amber-200 shadow-xs">
                <CheckCircle2 className="w-3 h-3 text-amber-700" />
                {isVoiceActive ? 'WebRTC Active' : 'Zero-Hallucination Dual RAG'}
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
                    <span className="text-[10px] font-mono uppercase tracking-wider text-stone-400">
                      {msg.sender === 'user' ? 'You' : 'NEXi Assistant'}
                    </span>
                    <span className="text-[10px] text-stone-400 font-mono">• {msg.timestamp}</span>
                  </div>

                  <div
                    className={`max-w-[88%] p-4 rounded-xl text-sm leading-relaxed shadow-xs ${
                      msg.sender === 'user'
                        ? 'bg-stone-900 text-stone-50 rounded-br-none'
                        : 'bg-white border border-stone-200/90 text-stone-800 rounded-bl-none'
                    }`}
                  >
                    {msg.text}

                    {/* Policy Verification Tag on NEXi messages */}
                    {msg.sender === 'nexi' && msg.citation && (
                      <div className="mt-3 pt-2.5 border-t border-stone-200/70 flex items-center justify-between text-[11px] text-stone-500 font-mono">
                        <span className="flex items-center gap-1 text-amber-800 font-medium">
                          <ShieldCheck className="w-3.5 h-3.5" />
                          Grounded in {msg.citation.document.split(' ')[0]} Policy
                        </span>
                        <span className="text-stone-500 text-[10px]">
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
                  <span className="text-[10px] font-mono uppercase tracking-wider mb-1 px-1 text-stone-400">
                    NEXi Searching Ordinances...
                  </span>
                  <div className="p-3 rounded-xl bg-white border border-stone-200/90 rounded-bl-none flex items-center space-x-2 text-xs text-stone-500 shadow-xs">
                    <span className="h-1.5 w-1.5 rounded-full bg-amber-500 animate-pulse"></span>
                    <span className="h-1.5 w-1.5 rounded-full bg-amber-500 animate-pulse delay-150"></span>
                    <span className="h-1.5 w-1.5 rounded-full bg-amber-500 animate-pulse delay-300"></span>
                    <span className="text-stone-600 font-mono text-[11px] ml-1">Reranking policy chunks against Chroma DB...</span>
                  </div>
                </div>
              )}

              <div ref={transcriptEndRef} />
            </div>

            {/* Document Context Drawer */}
            {showCitationDrawer && activeCitation && (
              <div className="mx-4 mb-2 p-3.5 rounded-xl bg-amber-50/70 border border-amber-200/80 text-xs shadow-xs font-mono">
                <div className="flex items-center justify-between text-amber-950 mb-1">
                  <div className="flex items-center gap-1.5">
                    <BookOpen className="w-3.5 h-3.5 text-amber-700" />
                    <span className="font-semibold uppercase tracking-wider text-[11px]">Live Document Context:</span>
                  </div>
                  <span className="text-[10px] bg-white text-amber-800 px-2 py-0.5 rounded-md font-bold border border-amber-200">
                    Score: {(activeCitation.relevanceScore * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="font-sans font-medium text-stone-800 text-[11px] mb-1">
                  <span className="text-stone-900 font-semibold">{activeCitation.document}</span> —{' '}
                  <span className="text-amber-800">{activeCitation.section}</span>
                </div>
                <p className="text-stone-600 font-sans italic text-[11px] line-clamp-2 leading-relaxed">
                  "{activeCitation.excerpt}"
                </p>
              </div>
            )}

            {/* Text Input Fallback Bar */}
            <form onSubmit={handleSendText} className="p-3.5 border-t border-stone-200/80 flex items-center space-x-2 bg-stone-50/50">
              <input
                type="text"
                placeholder="Ask about attendance condonation, hostel curfew, exam policy..."
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                className="flex-1 bg-white border border-stone-200/90 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:ring-1 focus:ring-amber-500 focus:border-amber-500 transition shadow-xs text-stone-900 placeholder:text-stone-400"
              />
              <button
                type="submit"
                disabled={!inputMessage.trim() || isSearchingText}
                className="bg-amber-500 hover:bg-amber-400 disabled:opacity-50 text-stone-950 p-2.5 rounded-lg transition-all shadow-xs flex items-center justify-center cursor-pointer font-medium"
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
