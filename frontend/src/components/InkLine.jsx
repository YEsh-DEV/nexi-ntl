import React, { useRef, useEffect } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

const PATHS = {
  1: {
    d: 'M-5 60 C 170 190, 330 40, 480 110 C 545 142, 610 170, 655 130 C 690 98, 660 48, 615 62 C 570 78, 590 150, 665 158 C 760 165, 880 120, 1005 92',
    viewBox: '0 30 1000 180',
    height: '110px',
  },
  2: {
    d: 'M-5 150 C 105 40, 205 40, 300 130 C 360 190, 430 190, 490 120 C 545 55, 610 55, 665 125 C 720 190, 800 185, 1005 78',
    viewBox: '0 30 1000 180',
    height: '110px',
  },
  3: {
    d: 'M-5 140 C 44 133, 98 122, 152 124 C 200 127, 232 158, 264 170 C 302 180, 352 173, 400 142 C 414 132, 440 130, 452 144 C 462 156, 458 173, 443 179 C 429 184, 419 173, 423 162 C 427 151, 440 150, 456 158 C 490 172, 524 200, 533 250',
    viewBox: '0 30 1000 250',
    height: '140px',
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
      start: 'top 85%',
      end: 'bottom 45%',
      scrub: 0.8,
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
      className="inkline relative w-full overflow-hidden my-6 select-none pointer-events-none"
      aria-hidden="true"
    >
      <svg
        viewBox={config.viewBox}
        preserveAspectRatio="xMidYMid meet"
        style={{ width: '100%', height: config.height }}
      >
        <path
          ref={pathRef}
          d={config.d}
          fill="none"
          stroke="#FFE500"
          strokeWidth="4.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    </div>
  );
};
