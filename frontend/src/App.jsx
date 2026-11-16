import React, { useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { Hero } from './components/Hero';
import { InkLine } from './components/InkLine';
import { Nutshell } from './components/Nutshell';
import { HowWeWork } from './components/HowWeWork';
import { DeskSection } from './components/DeskSection';
import { LiveSandbox } from './components/LiveSandbox';
import { Testimonials } from './components/Testimonials';
import { ContactSection } from './components/ContactSection';
import { Footer } from './components/Footer';

export const App = () => {
  // Ensure that on page reload, the user starts at the top of the page
  useEffect(() => {
    if ('scrollRestoration' in window.history) {
      window.history.scrollRestoration = 'manual';
    }
    // If there is no specific hash target requested, reset to top
    if (!window.location.hash) {
      window.scrollTo(0, 0);
    }
  }, []);

  return (
    <div className="min-h-screen bg-white text-[#111111] flex flex-col font-sans selection:bg-[#FFE500] selection:text-black">
      {/* 1. Sticky Editorial Navigation Header */}
      <Navbar />

      {/* Main Page Content Flow Matching Altnota */}
      <main className="flex-1">
        {/* 2. Hero Section: Poster Headline + Yellow Drawn Scribble */}
        <Hero />

        {/* Ink Line 1: Animated Flowing SVG Stroke */}
        <InkLine variant={1} />

        {/* 3. Us in a Nutshell: How we think / What we make + "Fast is nice. Right is better." */}
        <Nutshell />

        {/* Ink Line 2: Animated Flowing SVG Stroke */}
        <InkLine variant={2} />

        {/* 4. How We Work: The answer depends on the question (3 Prompt/Correction Cards) */}
        <HowWeWork />

        {/* 5. What's on your desk right now? */}
        <DeskSection />

        {/* 6. Live Interactive Voice & Dual-RAG Sandbox (Core Requirement) */}
        <LiveSandbox />

        {/* Ink Line 3: Animated Swirling SVG Stroke */}
        <InkLine variant={3} />

        {/* 7. Working Together: From the first question to the finish line (Reviews) */}
        <Testimonials />

        {/* 8. Leave Your Mark: Minimalist Contact & Inquiry Form */}
        <ContactSection />
      </main>

      {/* 9. Institutional Signature Footer: yours, nexi */}
      <Footer />
    </div>
  );
};

export default App;
