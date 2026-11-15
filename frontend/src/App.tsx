import React from 'react';
import { Navbar } from './components/Navbar';
import { Hero } from './components/Hero';
import { FeaturesGrid } from './components/FeaturesGrid';
import { LiveSandbox } from './components/LiveSandbox';
import { Testimonials } from './components/Testimonials';
import { Footer } from './components/Footer';

export const App: React.FC = () => {
  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col font-sans selection:bg-indigo-500 selection:text-white">
      {/* 1. Sticky Navigation Bar */}
      <Navbar />

      {/* Main Page Flow */}
      <main className="flex-1">
        {/* 2. Hero & Executive Intro */}
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
