---
title: About
icon: fas fa-user-ninja
order: 4
---

<style>
  @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Audiowide&display=swap');


  /* Synthwave Background with Parallax */
  .about-container {
    position: relative;
    overflow: hidden;
    padding: 2rem;
    min-height: 100vh;
    background: #000;
  }

  /* Multiple background layers for depth */
  .bg-layer {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
  }

  .bg-layer.stars {
    background-image: 
      radial-gradient(2px 2px at 20px 30px, #eee, transparent),
      radial-gradient(2px 2px at 40px 70px, #eee, transparent),
      radial-gradient(1px 1px at 50px 90px, #eee, transparent),
      radial-gradient(1px 1px at 130px 80px, #eee, transparent),
      radial-gradient(2px 2px at 200px 10px, #eee, transparent);
    background-repeat: repeat;
    background-size: 250px 250px;
    animation: stars-move 200s linear infinite;
    opacity: 0.5;
  }

  @keyframes stars-move {
    from { transform: translateY(0); }
    to { transform: translateY(-2000px); }
  }

  .bg-layer.gradient {
    background: linear-gradient(to bottom, 
      rgba(10, 10, 10, 0.8) 0%, 
      rgba(26, 0, 51, 0.8) 50%, 
      rgba(45, 27, 105, 0.8) 100%);
    z-index: 1;
  }

  .bg-layer.grid {
    background-image: 
      linear-gradient(rgba(174, 129, 255, 0.1) 1px, transparent 1px),
      linear-gradient(90deg, rgba(174, 129, 255, 0.1) 1px, transparent 1px);
    background-size: 50px 50px;
    transform: perspective(500px) rotateX(60deg) translateY(50%);
    animation: grid-move 20s linear infinite;
    z-index: 2;
  }

  @keyframes grid-move {
    from { transform: perspective(500px) rotateX(60deg) translateY(50%) translateZ(0); }
    to { transform: perspective(500px) rotateX(60deg) translateY(50%) translateZ(100px); }
  }

  /* VHS Distortion Effect */
  @keyframes vhs-distortion {
    0%, 100% { 
      transform: translateX(0);
      filter: hue-rotate(0deg);
    }
    20% {
      transform: translateX(-2px);
      filter: hue-rotate(90deg);
    }
    40% {
      transform: translateX(2px);
      filter: hue-rotate(180deg);
    }
    60% {
      transform: translateX(-1px);
      filter: hue-rotate(270deg);
    }
    80% {
      transform: translateX(1px);
      filter: hue-rotate(360deg);
    }
  }

  /* Glitch Effect Enhanced */
  @keyframes glitch {
    0%, 100% { 
      text-shadow: 
        0.05em 0 0 rgba(255, 0, 0, .75),
        -0.025em -0.05em 0 rgba(0, 255, 0, .75),
        0.025em 0.05em 0 rgba(0, 0, 255, .75);
    }
    14% {
      text-shadow: 
        0.05em 0 0 rgba(255, 0, 0, .75),
        -0.05em -0.025em 0 rgba(0, 255, 0, .75),
        0.025em 0.05em 0 rgba(0, 0, 255, .75);
    }
    15% {
      text-shadow: 
        -0.05em -0.025em 0 rgba(255, 0, 0, .75),
        0.025em 0.025em 0 rgba(0, 255, 0, .75),
        -0.05em -0.05em 0 rgba(0, 0, 255, .75);
    }
    49% {
      text-shadow: 
        -0.05em -0.025em 0 rgba(255, 0, 0, .75),
        0.025em 0.025em 0 rgba(0, 255, 0, .75),
        -0.05em -0.05em 0 rgba(0, 0, 255, .75);
    }
    50% {
      text-shadow: 
        0.025em 0.05em 0 rgba(255, 0, 0, .75),
        0.05em 0 0 rgba(0, 255, 0, .75),
        0 -0.05em 0 rgba(0, 0, 255, .75);
    }
    99% {
      text-shadow: 
        0.025em 0.05em 0 rgba(255, 0, 0, .75),
        0.05em 0 0 rgba(0, 255, 0, .75),
        0 -0.05em 0 rgba(0, 0, 255, .75);
    }
  }

  /* Main Title with Typing Effect */
  .main-title {
    font-family: 'Audiowide', 'Orbitron', monospace !important;
    font-weight: 900 !important;
    font-size: 4rem !important;
    text-align: center !important;
    color: transparent !important;
    background: linear-gradient(45deg, #ff00ff, #00ffff, #ff00ff);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    background-clip: text;
    text-transform: uppercase !important;
    letter-spacing: 0.2em !important;
    position: relative !important;
    z-index: 10 !important;
    margin: 2rem 0 !important;
    animation: 
      gradient-move 3s ease infinite,
      glitch 2.5s infinite,
      vhs-distortion 0.3s infinite;
    overflow: hidden;
  }

  .main-title::before {
    content: attr(data-text);
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    color: #ff00ff;
    z-index: -1;
    filter: blur(3px);
    animation: neon-flicker 3s ease-in-out infinite;
  }

  .main-title::after {
    content: attr(data-text);
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    color: #00ffff;
    z-index: -2;
    filter: blur(5px);
    animation: neon-flicker 3s ease-in-out infinite reverse;
  }

  @keyframes neon-flicker {
    0%, 100% { opacity: 1; transform: translateX(0); }
    33% { opacity: 0.8; transform: translateX(2px); }
    66% { opacity: 0.9; transform: translateX(-2px); }
  }

  /* Subtitle with Wave Effect */
  .subtitle {
    display: block !important;
    text-align: center !important;
    font-family: 'Share Tech Mono', monospace !important;
    color: #62ea00 !important;
    font-size: 1.4rem !important;
    text-shadow: 0 0 20px #62ea00 !important;
    margin-bottom: 3rem !important;
    position: relative !important;
    z-index: 10 !important;
    font-style: normal !important;
  }

  .subtitle span {
    display: inline-block;
    animation: wave 2s ease-in-out infinite;
    animation-delay: calc(var(--i) * 0.1s);
  }

  @keyframes wave {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-10px); }
  }

  /* Profile Image with Simple Glow */
  .about-img {
    width: 100%;
    max-width: 300px;
    border-radius: 12px;
    margin: 1rem auto;
    display: block;
    position: relative;
    z-index: 10;
    border: 4px solid #ae81ff;
    box-shadow: 
      0 0 30px #ae81ff,
      0 0 60px #ae81ff,
      inset 0 0 30px rgba(174, 129, 255, 0.2);
    animation: float 6s ease-in-out infinite;
  }

  @keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-20px); }
  }

  /* Interactive Laser Lines */
  .laser-line {
    position: absolute;
    height: 2px;
    background: linear-gradient(90deg, transparent, #ff00ff, transparent);
    opacity: 0;
    animation: laser-scan 4s ease-in-out infinite;
    pointer-events: none;
  }

  @keyframes laser-scan {
    0%, 100% { 
      opacity: 0;
      width: 0;
      left: 50%;
    }
    50% {
      opacity: 1;
      width: 100%;
      left: 0;
    }
  }

  /* Image Gallery with 3D Cards */
  .image-row {
    display: flex;
    flex-wrap: wrap;
    gap: 2rem;
    justify-content: center;
    margin: 3rem 0;
    position: relative;
    z-index: 10;
    perspective: 1000px;
  }

  .image-row img {
    flex: 1 1 300px;
    max-width: 400px;
    border-radius: 15px;
    border: 3px solid transparent;
    position: relative;
    transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    transform-style: preserve-3d;
    box-shadow: 
      0 10px 30px rgba(174, 129, 255, 0.3),
      inset 0 0 20px rgba(174, 129, 255, 0.1);
  }

  .image-row img::before {
    content: '';
    position: absolute;
    inset: -3px;
    border-radius: 15px;
    padding: 3px;
    background: linear-gradient(45deg, #ff00ff, #00ffff, #ff00ff);
    mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    mask-composite: exclude;
    animation: gradient-border 3s ease infinite;
    opacity: 0;
    transition: opacity 0.3s;
  }

  .image-row img:hover::before {
    opacity: 1;
  }

  .image-row img:hover {
    transform: translateY(-20px) rotateY(15deg) rotateX(5deg) scale(1.1);
    box-shadow: 
      0 30px 60px rgba(255, 0, 255, 0.4),
      0 0 100px rgba(0, 255, 255, 0.3),
      inset 0 0 30px rgba(174, 129, 255, 0.2);
    filter: brightness(1.2) contrast(1.1);
  }

  /* Interactive Badges - Simple Bright Green */
  .badge {
    display: inline-block;
    background: linear-gradient(45deg, #1a1a1a, #2d2d2d);
    color: #62ea00;
    padding: 0.5em 1em;
    border-radius: 25px;
    font-size: 0.9em;
    margin: 0.3em;
    font-family: 'Share Tech Mono', monospace;
    position: relative;
    overflow: hidden;
    border: 1px solid #62ea00;
    text-shadow: 0 0 5px #62ea00;
    transition: all 0.3s ease;
  }

  .badge::before {
    content: '';
    position: absolute;
    top: 50%;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, #62ea00, transparent);
    transition: left 0.5s ease;
  }

  .badge:hover {
    color: #000;
    background: #62ea00;
    box-shadow: 0 0 20px #62ea00;
    transform: scale(1.1);
  }

  .badge:hover::before {
    left: 100%;
  }

  /* Highlight Text with Neon Glow */
  .highlight {
    color: #ff00ff;
    font-weight: bold;
    text-shadow: 
      0 0 10px #ff00ff,
      0 0 20px #ff00ff,
      0 0 30px #ff00ff;
    position: relative;
    display: inline-block;
    transition: all 0.3s;
  }

  .highlight:hover {
    color: #00ffff;
    text-shadow: 
      0 0 10px #00ffff,
      0 0 20px #00ffff,
      0 0 30px #00ffff,
      0 0 40px #00ffff;
    animation: pulse 0.5s ease-in-out;
  }

  @keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.1); }
  }

  .highlight::after {
    content: '';
    position: absolute;
    bottom: -2px;
    left: 0;
    width: 0;
    height: 3px;
    background: linear-gradient(90deg, #ff00ff, #00ffff, #ff00ff);
    background-size: 200% 100%;
    animation: gradient-move 2s linear infinite;
    transition: width 0.3s;
  }

  .highlight:hover::after {
    width: 100%;
  }

  /* Section Titles with Cyberpunk Style */
  .section-title,
  .about-container h3,
  .about-container h4 {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    color: transparent !important;
    background: linear-gradient(90deg, #ae81ff, #ff00ff, #ae81ff);
    background-size: 200% 100%;
    -webkit-background-clip: text;
    background-clip: text;
    margin-top: 4rem !important;
    margin-bottom: 1.5rem !important;
    font-family: 'Audiowide', 'Orbitron', monospace !important;
    font-weight: 700 !important;
    font-size: 2rem !important;
    position: relative !important;
    padding-left: 40px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    animation: gradient-move 4s linear infinite;
    z-index: 10 !important;
  }

  h4 {
    font-size: 1.6rem !important;
    margin-top: 3rem !important;
  }

  .section-title::before,
  .about-container h3::before,
  .about-container h4::before {
    content: '▶';
    position: absolute;
    left: 0;
    top: 50%;
    transform: translateY(-50%);
    color: #62ea00;
    font-size: 1.5rem;
    animation: blink 1s ease-in-out infinite, rotate 2s linear infinite;
    text-shadow: 0 0 20px #62ea00;
  }

  @keyframes rotate {
    from { transform: translateY(-50%) rotate(0deg); }
    to { transform: translateY(-50%) rotate(360deg); }
  }

  @keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }

  /* HR Styles with Energy Beam */
  hr {
    border: none;
    height: 4px;
    background: #1a1a1a;
    margin: 4rem 0;
    position: relative;
    overflow: visible;
  }

  hr::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 100px;
    height: 100px;
    background: radial-gradient(circle, #ae81ff 0%, transparent 70%);
    opacity: 0.5;
    animation: pulse 2s ease-in-out infinite;
  }

  hr::after {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, 
      transparent, 
      #ff00ff 20%, 
      #00ffff 50%, 
      #ff00ff 80%, 
      transparent);
    animation: beam-scan 3s linear infinite;
  }

  @keyframes beam-scan {
    to { left: 100%; }
  }

  /* Paragraphs with better readability */
  p {
    position: relative;
    z-index: 10;
    line-height: 1.9;
    color: #f0f0f0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 1.1rem;
    text-shadow: 0 0 5px rgba(255, 255, 255, 0.1);
  }

  /* Terminal Effect for Code/Tech Terms */
  code {
    font-family: 'Share Tech Mono', monospace;
    background: rgba(98, 234, 0, 0.2);
    color: #62ea00;
    padding: 0.3em 0.6em;
    border-radius: 6px;
    border: 1px solid #62ea00;
    box-shadow: 
      0 0 5px rgba(98, 234, 0, 0.5),
      inset 0 0 5px rgba(98, 234, 0, 0.2);
    transition: all 0.3s;
  }

  code:hover {
    background: rgba(98, 234, 0, 0.3);
    box-shadow: 
      0 0 10px rgba(98, 234, 0, 0.7),
      inset 0 0 10px rgba(98, 234, 0, 0.3);
    transform: scale(1.05);
  }

  /* CRT Scan Line Effect */
  .about-container::after {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: 
      repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(255, 255, 255, 0.03) 2px,
        rgba(255, 255, 255, 0.03) 4px
      );
    animation: scan-lines 8s linear infinite;
    pointer-events: none;
    z-index: 100;
  }

  @keyframes scan-lines {
    0% { transform: translateY(0); }
    100% { transform: translateY(10px); }
  }

  /* Static Noise Overlay */
  .about-container::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    opacity: 0.03;
    z-index: 99;
    pointer-events: none;
    background-image: url('data:image/svg+xml;utf8,<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg"><filter id="noise"><feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="4" /></filter><rect width="100%" height="100%" filter="url(%23noise)" /></svg>');
  }

  /* Mobile Responsive with Style */
  @media (max-width: 768px) {
    .main-title {
      font-size: 2.5rem !important;
    }
    .section-title,
    .about-container h3 {
      font-size: 1.5rem !important;
    }
    .about-container h4 {
      font-size: 1.3rem !important;
    }
    .image-row {
      flex-direction: column;
      align-items: center;
    }
    .badge {
      font-size: 0.9rem;
      padding: 0.5em 1em;
    }
  }

  /* Easter Egg - Konami Code */
  .konami-activated {
    animation: rainbow-bg 2s linear infinite;
  }

  @keyframes rainbow-bg {
    0% { filter: hue-rotate(0deg); }
    100% { filter: hue-rotate(360deg); }
  }

  /* Loading Animation */
  @keyframes fade-in {
    from {
      opacity: 0;
      transform: translateY(30px) scale(0.9);
    }
    to {
      opacity: 1;
      transform: translateY(0) scale(1);
    }
  }

  .about-container > * {
    animation: fade-in 0.8s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
    opacity: 0;
  }


  /* Performance optimizations */
  * {
    will-change: auto;
  }

  .main-title,
  .about-img,
  .badge,
  .image-row img {
    will-change: transform;
  }

  /* Smooth scrolling */
  html {
    scroll-behavior: smooth;
  }
</style>

<div class="about-container">
  <!-- Background Layers -->
  <div class="bg-layer stars"></div>
  <div class="bg-layer gradient"></div>
  <div class="bg-layer grid"></div>
  


  <!-- Laser Lines -->
  <div class="laser-line" style="top: 20%; animation-delay: 0s;"></div>
  <div class="laser-line" style="top: 50%; animation-delay: 1s;"></div>
  <div class="laser-line" style="top: 80%; animation-delay: 2s;"></div>

  <div style="text-align: center; margin-bottom: 2rem;">
    <img src="/assets/img/mev3.jpg" alt="Jeremy Montes" class="about-img" />
  </div>

  <h1 class="main-title" data-text="Jeremy Montes">Jeremy Montes</h1>

  <div class="subtitle">
    <span style="--i: 0">Security</span>
    <span style="--i: 1">Enthusiast</span>
    <span style="--i: 2">·</span>
    <span style="--i: 3">Cool</span>
    <span style="--i: 4">Guy</span>
    <span style="--i: 5">·</span>
    <span style="--i: 6">Obsessed</span>
    <span style="--i: 7">with</span>
    <span style="--i: 8">the</span>
    <span style="--i: 9">Unknown</span>
  </div>

  <p>I'm Jeremy, but online I go by <span class="highlight">qewave</span>. This website is my personal playground and documentation hub. If you're looking for my more corporate side, click the LinkedIn icon in the bottom left on this site. Here, I post my CTF writeups, hacking stuff, notes, random tech rants, and other things I find useful or worth sharing.</p>

  <p>You'll notice the retro 80s vibes throughout the site. There's something about that neon/purple aesthetic that just works when you're staring at terminal windows at 2 AM. When I'm not breaking things, I'm either reading while vibing to Japanese jazz and neo-soul or actually going outside to hike and play tennis. You know, touching grass and all that.</p>

  <hr>

  <h3 class="section-title">Early Days</h3>

  <p>I've been into hacking since I was 16. I started out by modifying games to unlock unreleased skins just to flex. In high school, I never really knew what career path I wanted. The funny part is I was always talking about tech, coding, or software projects without realizing that this was what I was meant to do. It finally clicked senior year.</p>

  <hr>

  <h3 class="section-title">Breaking into Cyber</h3>

  <p>Since then, I've earned my associate's degree in Computer Science with a focus on Networking & Security, and I'm continuing coursework in the same field. I'm planning to pursue a bachelor's degree in either Cloud or Cybersecurity.</p>

  <p>I've attended nearly every major security event and conference in California, helped found the first Cybersecurity Club at my college (and became its representative), secured a SysOps/Security internship, won a few CTFs, picked up some certs, and built tons of personal labs and projects.</p>

  <div class="image-row">
    <img src="https://i.postimg.cc/2ys8f3HQ/PXL-20250324-231930647.jpg"
         alt="Workshop photo" />
    <img src="https://i.postimg.cc/xTQ46M39/rush.webp"
         alt="Workshop photo" />
  </div>

  <hr>

  <h3 class="section-title">What Keeps Me Going</h3>

  <p>I'm not a genius, but I love figuring things out. Especially in security, that constant mystery combined with a cat and mouse game is what keeps me on my toes. That's why I've been obsessed with **malware analysis** and **reverse engineering**. Do I know what I'm looking at right away? Hell no. But do I love digging into it until it clicks and I find what I need? Absolutely.</p>

  <h4>Electronics and IoT</h4>
  <p>I was always into electronics, but things really clicked after finishing my Arduino internship at college. Once that wrapped up, I was hungry for more, so I grabbed a big electronics kit and dove into some books. At that point I was torn between going the electrical engineering/hardware route or sticking with computer science. But seeing how I could build practically anything with just copper, modules, and code - that really opened my eyes. I love building stuff, and electronics is where I get to show that off the most. After getting comfortable with schematics and working on my own projects, I naturally drifted into IoT security, radio frequency work, and started making my own hacking gadgets and PCBs.</p>

  <div class="image-row">
    <img src="https://i.postimg.cc/fLgrHrCF/IMG-4875.webp"
         alt="Workshop photo" />
    <img src="https://i.postimg.cc/HWrYpGqr/image.webp"
         alt="Workshop photo" />
  </div>

  <div class="image-row">
    <img src="https://i.postimg.cc/Y9hHqgy5/IMG-5415.jpg" alt="Workshop photo" style="max-width: 300px;" />
    <img src="https://i.postimg.cc/RhK2NT1D/books.webp" alt="Workshop photo" style="transform: rotate(-90deg); transform-origin: center; width: 250px;" />
  </div>

  <h4>CTFs and reverse engineering</h4>
  <p>As far as CTFs go, I'm kicking myself for not getting into these way earlier. I just started with CTFs and King of the Hill stuff a little over a year ago, but I only really got serious about it this year. I find myself gravitating toward Pwn and reverse engineering challenges, plus hardware ones. While a lot of CTFs are pretty gamified and not always that useful for real-world application, some actually make you think and help sharpen your technical and problem-solving skills. Sometimes I'll burn a whole day on one challenge and come up empty, other times I'll dominate. What I love most about CTFs is the team communication and problem-solving, then reading writeups afterward and talking with other teams about their approaches once the competition wraps up. It's just cool because you don't see many other fields where you can learn and practice new stuff from random competitions that pop up basically every week.</p>

  <div class="image-row">
    <img src="https://i.postimg.cc/CxL6g02t/1746214331685.jpg"
         alt="Workshop photo" style="max-width: 400px;" />
    <img src="https://i.postimg.cc/L8CQ47Ks/rev.jpg"
         alt="Workshop photo"
         style="height: 350px; object-fit: cover;" />
  </div>

  <hr>

  <h3 class="section-title">Stuff I Like Doing</h3>

  <div style="text-align: center; margin: 2rem 0;">
    <div class="badge">Malware Analysis</div>
    <div class="badge">Reverse Engineering</div>
    <div class="badge">Electronics & Microcontrollers</div>
    <div class="badge">IoT Hacking</div>
    <div class="badge">Touching Grass</div>
  </div>

</div>

<script>
// Parallax Effect
document.addEventListener('scroll', () => {
  const scrolled = window.pageYOffset;
  const parallax = document.querySelector('.bg-layer.grid');
  const stars = document.querySelector('.bg-layer.stars');
  
  if (parallax) {
    parallax.style.transform = `perspective(500px) rotateX(60deg) translateY(${50 + scrolled * 0.2}%) translateZ(${scrolled * 0.1}px)`;
  }
  if (stars) {
    stars.style.transform = `translateY(${scrolled * 0.5}px)`;
  }
});

// Interactive Hover Effects for Images
document.querySelectorAll('.image-row img').forEach(img => {
  img.addEventListener('mousemove', (e) => {
    const rect = img.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;
    const rotateX = (y - centerY) / 10;
    const rotateY = (centerX - x) / 10;
    
    img.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale(1.1)`;
  });
  
  img.addEventListener('mouseleave', () => {
    img.style.transform = '';
  });
});

// Konami Code Easter Egg
let konamiCode = [];
const konamiPattern = ['ArrowUp', 'ArrowUp', 'ArrowDown', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'ArrowLeft', 'ArrowRight', 'b', 'a'];

document.addEventListener('keydown', (e) => {
  konamiCode.push(e.key);
  konamiCode = konamiCode.slice(-10);
  
  if (konamiCode.join('') === konamiPattern.join('')) {
    document.body.classList.add('konami-activated');
    setTimeout(() => {
      alert('🎮 ACHIEVEMENT UNLOCKED: Retro Gamer! 🎮');
    }, 100);
  }
});

// Initialize on load
window.addEventListener('load', () => {
  // Add random glitch effect
  setInterval(() => {
    if (Math.random() > 0.95) {
      document.body.style.transform = `translate(${Math.random() * 4 - 2}px, ${Math.random() * 4 - 2}px)`;
      setTimeout(() => {
        document.body.style.transform = '';
      }, 50);
    }
  }, 3000);
});
</script>