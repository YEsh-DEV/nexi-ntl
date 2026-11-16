import React from 'react';
import { Navbar } from './components/Navbar';
import { Hero } from './components/Hero';
import { FeaturesGrid } from './components/FeaturesGrid';
import { LiveSandbox } from './components/LiveSandbox';
import { Testimonials } from './components/Testimonials';
import { Footer } from './components/Footer';

export const App = () => {
  return (
    <div className="min-h-screen bg-[#FAF9F6] text-stone-900 flex flex-col font-sans selection:bg-amber-300 selection:text-stone-950">
      {/* 1. Sticky Editorial Navigation Bar */}
      <Navbar />

      {/* Main Page Flow */}
      <main className="flex-1">
        {/* 2. Hero & Academic Authority Intro */}
        <Hero />

        {/* 3. Features Breakdown & Architecture */}
        <FeaturesGrid />

        {/* 4. Live Interactive Voice Sandbox (Core Experience) */}
        <LiveSandbox />

        {/* 5. Student Community Testimonials */}
        <Testimonials />
      </main>

      {/* 6. Institutional Footer */}
      <Footer />
    </div>
  );
};

export default App;
