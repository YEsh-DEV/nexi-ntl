import React, { useRef, useEffect } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

const PATHS = {
  // Line 1: Hero to Nutshell (Big sweeping undulating wave)
  1: {
    d: 'M -20,120 C 180,240, 360,20, 540,140 C 720,260, 900,30, 1080,150 C 1220,230, 1340,60, 1460,130',
    viewBox: '0 0 1440 260',
  },
  // Line 2: Nutshell to How We Work (Deep flowing crest and trough)
  2: {
    d: 'M -20,190 C 220,30, 420,30, 600,160 C 740,250, 920,230, 1060,110 C 1200,10, 1340,60, 1460,150',
    viewBox: '0 0 1440 260',
  },
  // Line 3: Signature Altnota Loop-de-Loop Swirl (Before Testimonials & Contact)
  3: {
    d: 'M -20,130 C 180,130, 340,70, 500,120 C 660,170, 780,180, 880,110 C 940,65, 960,15, 910,10 C 850,5, 830,95, 900,125 C 980,160, 1100,170, 1220,100 C 1320,40, 1400,90, 1460,140',
    viewBox: '0 0 1440 260',
  },
};

export const InkLine = ({ variant = 1 }) => {
  const containerRef = useRef(null);
  const pathRef = useRef(null);

  const config = PATHS[variant] || PATHS[1];

  useEffect(() => {
    const path = pathRef.current;
    if (!path) return;

    const length = path.getTotalLength();
    gsap.set(path, {
      strokeDasharray: length,
      strokeDashoffset: length,
    });

    const trigger = ScrollTrigger.create({
      trigger: containerRef.current,
      start: 'top 90%',
      end: 'bottom 35%',
      scrub: 0.9,
      onUpdate: (self) => {
        const offset = length * (1 - self.progress);
        gsap.set(path, { strokeDashoffset: offset });
      },
    });

    return () => {
      trigger.kill();
    };
  }, [variant]);

  return (
    <div
      ref={containerRef}
      className="inkline relative w-full overflow-hidden my-8 sm:my-14 md:my-20 select-none pointer-events-none"
      aria-hidden="true"
    >
      <svg
        viewBox={config.viewBox}
        preserveAspectRatio="none"
        className="w-full h-28 sm:h-40 md:h-52 lg:h-64 block overflow-visible"
      >
        <path
          ref={pathRef}
          d={config.d}
          fill="none"
          stroke="#FFE500"
          strokeWidth="6"
          strokeLinecap="round"
          strokeLinejoin="round"
          vectorEffect="non-scaling-stroke"
        />
      </svg>
    </div>
  );
};
