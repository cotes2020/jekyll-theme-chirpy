---
title: About
icon: fas fa-user-ninja
order: 4
---

<style>
  @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');

  /* Synthwave Background */
  .about-container {
    position: relative;
    overflow: hidden;
    padding: 2rem;
    background: linear-gradient(to bottom, #0a0a0a 0%, #1a0033 50%, #2d1b69 100%);
  }

  .about-container::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 200%;
    height: 200%;
    background: 
      repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(174, 129, 255, 0.03) 2px,
        rgba(174, 129, 255, 0.03) 4px
      ),
      repeating-linear-gradient(
        90deg,
        transparent,
        transparent 2px,
        rgba(174, 129, 255, 0.03) 2px,
        rgba(174, 129, 255, 0.03) 4px
      );
    transform: translate(-50%, -50%) perspective(500px) rotateX(60deg);
    animation: grid-move 20s linear infinite;
  }

  @keyframes grid-move {
    0% { transform: translate(-50%, -50%) perspective(500px) rotateX(60deg) translateZ(0); }
    100% { transform: translate(-50%, -50%) perspective(500px) rotateX(60deg) translateZ(100px); }
  }

  /* Glitch Effect */
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

  /* Main Title Styles - Target multiple possible elements */
  .main-title, 
  .about-container h1, 
  .about-container h2, 
  .about-container > h3:first-of-type {
    font-family: 'Orbitron', monospace !important;
    font-weight: 900 !important;
    font-size: 3rem !important;
    text-align: center !important;
    color: #fff !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    position: relative !important;
    z-index: 2 !important;
    margin: 2rem 0 !important;
    animation: glitch 2.5s infinite !important;
  }

  .main-title::before {
    content: 'Jeremy Montes';
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

  @keyframes neon-flicker {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.8; }
  }

  /* Subtitle */
  .subtitle,
  em {
    display: block !important;
    text-align: center !important;
    font-family: 'Share Tech Mono', monospace !important;
    color: #62ea00 !important;
    font-size: 1.2rem !important;
    text-shadow: 0 0 10px #62ea00 !important;
    margin-bottom: 2rem !important;
    position: relative !important;
    z-index: 2 !important;
    animation: pulse-neon 2s ease-in-out infinite !important;
    font-style: normal !important;
  }

  @keyframes pulse-neon {
    0%, 100% { text-shadow: 0 0 10px #62ea00, 0 0 20px #62ea00; }
    50% { text-shadow: 0 0 20px #62ea00, 0 0 40px #62ea00, 0 0 60px #62ea00; }
  }

  /* Profile Image */
  .about-img {
    width: 100%;
    max-width: 300px;
    border-radius: 12px;
    margin: 1rem auto;
    display: block;
    position: relative;
    z-index: 2;
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

  /* Image Gallery */
  .image-row {
    display: flex;
    flex-wrap: wrap;
    gap: 1.5rem;
    justify-content: center;
    margin: 2rem 0;
    position: relative;
    z-index: 2;
  }

  .image-row img {
    flex: 1 1 250px;
    max-width: 350px;
    border-radius: 12px;
    border: 2px solid transparent;
    background: linear-gradient(45deg, #ff00ff, #00ffff, #ff00ff);
    background-size: 200% 200%;
    animation: gradient-border 3s ease infinite;
    position: relative;
    transition: all 0.3s ease;
  }

  .image-row img::before {
    content: '';
    position: absolute;
    inset: -2px;
    border-radius: 12px;
    padding: 2px;
    background: linear-gradient(45deg, #ff00ff, #00ffff, #ff00ff);
    mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    mask-composite: exclude;
    animation: gradient-border 3s ease infinite;
  }

  @keyframes gradient-border {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
  }

  .image-row img:hover {
    transform: scale(1.05) rotateY(5deg);
    box-shadow: 
      0 0 30px #ff00ff,
      0 0 60px #00ffff;
  }

  /* Badges */
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

  /* Highlight Text */
  .highlight {
    color: #ff00ff;
    font-weight: bold;
    text-shadow: 0 0 10px #ff00ff;
    position: relative;
    display: inline-block;
  }

  .highlight::after {
    content: '';
    position: absolute;
    bottom: -2px;
    left: 0;
    width: 100%;
    height: 2px;
    background: linear-gradient(90deg, #ff00ff, #00ffff, #ff00ff);
    background-size: 200% 100%;
    animation: gradient-move 2s linear infinite;
  }

  @keyframes gradient-move {
    0% { background-position: 0% 0%; }
    100% { background-position: 200% 0%; }
  }

  /* Section Titles - Target all possible heading elements */
  .section-title, 
  .about-container h3,
  .about-container h4,
  .about-container h2:not(.main-title) {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    color: #ae81ff !important;
    margin-top: 3rem !important;
    margin-bottom: 1rem !important;
    font-family: 'Orbitron', monospace !important;
    font-weight: 700 !important;
    font-size: 1.8rem !important;
    position: relative !important;
    padding-left: 30px !important;
    text-shadow: 0 0 20px #ae81ff !important;
    text-transform: none !important;
    line-height: 1.2 !important;
    background: transparent !important;
  }

  /* Extra specificity for h3 and h4 elements */
  h3.section-title,
  .about-container h3,
  .about-container h4 {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
  }

  .section-title::before, 
  .about-container h3::before,
  .about-container h4::before,
  .about-container h2:not(.main-title)::before {
    content: '▶';
    position: absolute;
    left: 0;
    top: 50%;
    transform: translateY(-50%);
    color: #62ea00;
    font-size: 1.2rem;
    animation: blink 1s ease-in-out infinite;
  }

  @keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }

  /* HR Styles */
  hr {
    border: none;
    height: 2px;
    background: linear-gradient(90deg, transparent, #ae81ff, transparent);
    margin: 3rem 0;
    position: relative;
    overflow: hidden;
  }

  hr::after {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, #fff, transparent);
    animation: scan 3s linear infinite;
  }

  @keyframes scan {
    0% { left: -100%; }
    100% { left: 100%; }
  }

  /* Paragraphs */
  p {
    position: relative;
    z-index: 2;
    line-height: 1.8;
    color: #e0e0e0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }

  /* Terminal Effect for Code/Tech Terms */
  code {
    font-family: 'Share Tech Mono', monospace;
    background: rgba(98, 234, 0, 0.1);
    color: #62ea00;
    padding: 0.2em 0.4em;
    border-radius: 4px;
    border: 1px solid #62ea00;
  }

  /* Scan Line Effect */
  .about-container::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: linear-gradient(
      0deg,
      transparent 0%,
      rgba(255, 255, 255, 0.03) 50%,
      transparent 100%
    );
    background-size: 100% 4px;
    animation: scan-lines 8s linear infinite;
    pointer-events: none;
    z-index: 1;
  }

  @keyframes scan-lines {
    0% { transform: translateY(-100%); }
    100% { transform: translateY(100%); }
  }

  /* Special hover effect for images */
  img {
    transition: all 0.3s ease;
    cursor: pointer;
  }

  img:hover {
    filter: brightness(1.2) saturate(1.5);
  }

  /* Mobile Responsive */
  @media (max-width: 768px) {
    .main-title, 
    .about-container h1, 
    .about-container h2, 
    .about-container > h3:first-of-type {
      font-size: 2rem !important;
    }
    .section-title, 
    .about-container h3,
    .about-container h4 {
      font-size: 1.4rem !important;
    }
    .image-row {
      flex-direction: column;
      align-items: center;
    }
  }

  /* Debug - Force all headers to be visible */
  .about-container h1,
  .about-container h2,
  .about-container h3,
  .about-container h4,
  .about-container h5,
  .about-container h6 {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    position: relative !important;
    z-index: 10 !important;
  }
  @keyframes fade-in {
    from {
      opacity: 0;
      transform: translateY(20px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  .about-container > * {
    animation: fade-in 0.8s ease-out forwards;
    animation-delay: 0.2s;
    opacity: 0;
  }

  /* Sequential animation delay */
  .about-container > *:nth-child(1) { animation-delay: 0.1s; }
  .about-container > *:nth-child(2) { animation-delay: 0.2s; }
  .about-container > *:nth-child(3) { animation-delay: 0.3s; }
  .about-container > *:nth-child(4) { animation-delay: 0.4s; }
  .about-container > *:nth-child(5) { animation-delay: 0.5s; }
  .about-container > *:nth-child(6) { animation-delay: 0.6s; }
  .about-container > *:nth-child(7) { animation-delay: 0.7s; }
  .about-container > *:nth-child(8) { animation-delay: 0.8s; }
  .about-container > *:nth-child(9) { animation-delay: 0.9s; }
  .about-container > *:nth-child(10) { animation-delay: 1.0s; }
</style>

<div class="about-container">

<div style="text-align: center; margin-bottom: 2rem;">
  <img src="/assets/img/mev3.jpg" alt="Jeremy Montes" class="about-img" />
</div>

<h1 class="main-title">Jeremy Montes</h1>

<em class="subtitle">Security Enthusiast  ·  Cool Guy  ·  Obsessed with the Unknown</em>

I'm Jeremy, but online I go by <span class="highlight">qewave</span>. This website is my personal playground and documentation hub. If you're looking for my more corporate side, click the LinkedIn icon in the bottom left on this site. Here, I post my CTF writeups, hacking stuff, notes, random tech rants, and other things I find useful or worth sharing.

You'll notice the retro 80s vibes throughout the site. There's something about that neon/purple aesthetic that just works when you're staring at terminal windows at 2 AM. When I'm not breaking things, I'm either reading while vibing to Japanese jazz and neo-soul or actually going outside to hike and play tennis. You know, touching grass and all that.

<hr>

<h3 class="section-title" style="display: block !important; color: #ae81ff !important;">Early Days</h3>

I've been into hacking since I was 16. I started out by modifying games to unlock unreleased skins just to flex. In high school, I never really knew what career path I wanted. The funny part is I was always talking about tech, coding, or software projects without realizing that this was what I was meant to do. It finally clicked senior year.

<hr>

<h3 class="section-title" style="display: block !important; color: #ae81ff !important;">Breaking into Cyber</h3>

Since then, I've earned my associate's degree in Computer Science with a focus on Networking & Security, and I'm continuing coursework in the same field. I'm planning to pursue a bachelor's degree in either Cloud or Cybersecurity.

I've attended nearly every major security event and conference in California, helped found the first Cybersecurity Club at my college (and became its representative), secured a SysOps/Security internship, won a few CTFs, picked up some certs like <span class="badge">Security+</span> and <span class="badge">CCNA</span>, and built tons of personal labs and projects.

<div class="image-row">
  <img src="https://i.postimg.cc/2ys8f3HQ/PXL-20250324-231930647.jpg"
       alt="Workshop photo" />

  <img src="https://i.postimg.cc/xTQ46M39/rush.webp"
       alt="Workshop photo" />
</div>

<hr>

<h3 class="section-title" style="display: block !important; color: #ae81ff !important;">What Keeps Me Going</h3>

I'm not a genius, but I love figuring things out. Especially in security, that constant mystery combined with a cat and mouse game is what keeps me on my toes. That's why I've been obsessed with <span class="highlight">malware analysis</span> and <span class="highlight">reverse engineering</span>. Do I know what I'm looking at right away? Hell no. But do I love digging into it until it clicks and I find what I need? Absolutely.

<h4>Electronics and IoT</h4>
I was always into electronics, but things really clicked after finishing my Arduino internship at college. Once that wrapped up, I was hungry for more, so I grabbed a big electronics kit and dove into some books. At that point I was torn between going the electrical engineering/hardware route or sticking with computer science. But seeing how I could build practically anything with just copper, modules, and code - that really opened my eyes. I love building stuff, and electronics is where I get to show that off the most. After getting comfortable with schematics and working on my own projects, I naturally drifted into IoT security, radio frequency work, and started making my own hacking gadgets and PCBs.

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
As far as CTFs go, I'm kicking myself for not getting into these way earlier. I just started with CTFs and King of the Hill stuff a little over a year ago, but I only really got serious about it this year. I find myself gravitating toward Pwn and reverse engineering challenges, plus hardware ones. While a lot of CTFs are pretty gamified and not always that useful for real-world application, some actually make you think and help sharpen your technical and problem-solving skills. Sometimes I'll burn a whole day on one challenge and come up empty, other times I'll dominate. What I love most about CTFs is the team communication and problem-solving, then reading writeups afterward and talking with other teams about their approaches once the competition wraps up. It's just cool because you don't see many other fields where you can learn and practice new stuff from random competitions that pop up basically every week.

<div class="image-row">
  <img src="https://i.postimg.cc/CxL6g02t/1746214331685.jpg"
       alt="Workshop photo" style="max-width: 400px;" />

  <img src="https://i.postimg.cc/L8CQ47Ks/rev.jpg"
     alt="Workshop photo"
     style="height: 350px; object-fit: cover;" />
</div>

<hr>

<h3 class="section-title" style="display: block !important; color: #ae81ff !important;">Stuff I Like Doing</h3>

<div style="text-align: center; margin: 2rem 0;">
  <div class="badge">Malware Analysis</div>
  <div class="badge">Reverse Engineering</div>
  <div class="badge">Electronics & Microcontrollers</div>
  <div class="badge">IoT Hacking</div>
  <div class="badge">Touching Grass</div>
</div>

</div>