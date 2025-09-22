---
title: "Book Review: Fancy Bear Goes Phishing: The Dark History of the Information Age, in Five Extraordinary Hacks - Scott J. Shapiro"
date: 2025-09-19 01:17:48 -0700
categories: [BookReview]
tags: [cybersecurity, Philosophy]
media_subpath: 
image: 
  path: "https://i.postimg.cc/sgHKhCcX/5242ee21-2f9a-4d00-a148-20a37e684683.png"
---
**TL;DR**: A Yale law and philosophy professor's journey back into cybersecurity delivers the exploration of why we keep failing at security over the years. Through five perfectly chosen historical attacks - Morris worm (1988), Bulgarian virus factory (1980s-90s), Paris Hilton Sidekick hack (2005), Russian DNC interference (2016), and Mirai botnet (2016) - Shapiro shows that cybersecurity is fundamentally about human psychology, economics, and power rather than just technology. The book combines technical depth with philosophical insights.

<div style="text-align: center; margin: 20px 0;">
  <!-- Red Alert Variant -->
  <div style="
    display: inline-block;
    background: linear-gradient(135deg, #0a0a0a 0%, #1a0000 100%);
    border: 1px solid #ff0040;
    border-radius: 4px;
    padding: 8px 12px;
    position: relative;
    overflow: hidden;
    font-family: 'Courier New', Consolas, 'Monaco', monospace;
    font-size: 13px;
    box-shadow: 
      0 0 20px rgba(255, 0, 64, 0.2),
      inset 0 0 20px rgba(255, 0, 64, 0.05);
    animation: cyber-pulse-red 3s ease-in-out infinite;
    margin: 10px 0;
  ">
    
    <!-- Scan line effect -->
    <div style="
      position: absolute;
      top: -100%;
      left: 0;
      right: 0;
      height: 2px;
      background: linear-gradient(90deg, 
        transparent 0%, 
        rgba(255, 0, 64, 0.8) 50%, 
        transparent 100%);
      animation: scan-line 4s linear infinite;
    "></div>
    
    <!-- Glitch effect overlay -->
    <div style="
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(255, 0, 64, 0.03) 2px,
        rgba(255, 0, 64, 0.03) 4px
      );
      pointer-events: none;
    "></div>
    
    <!-- Terminal prompt -->
    <span style="
      color: #ff0040;
      text-shadow: 0 0 5px rgba(255, 0, 64, 0.5);
      font-weight: bold;
    ">root@review:~$</span>
    
    <!-- Rating text -->
    <span style="
      color: #fff;
      margin-left: 8px;
      text-shadow: 0 0 3px rgba(255, 255, 255, 0.3);
    ">SCORE://</span>
    
    <!-- Score -->
    <span style="
      color: #ff0040;
      font-weight: bold;
      margin-left: 4px;
      text-shadow: 
        0 0 10px rgba(255, 0, 64, 0.8),
        0 0 20px rgba(255, 0, 64, 0.4);
      animation: glow-text-red 2s ease-in-out infinite alternate;
    ">9.8</span>
    
    <span style="
      color: #666;
      margin-left: 2px;
    ">/10.0</span>
    
    <!-- Status indicator -->
    <span style="
      display: inline-block;
      width: 8px;
      height: 8px;
      background: #ff0040;
      border-radius: 50%;
      margin-left: 8px;
      box-shadow: 0 0 10px rgba(255, 0, 64, 0.9);
      animation: status-blink-red 2s ease-in-out infinite;
      vertical-align: middle;
    "></span>
    
    <!-- Mini progress bar -->
    <div style="
      position: absolute;
      bottom: 0;
      left: 0;
      right: 0;
      height: 2px;
      background: rgba(255, 0, 64, 0.1);
    ">
      <div style="
        width: 98%;
        height: 100%;
        background: linear-gradient(90deg, #ff0040 0%, #cc0030 100%);
        box-shadow: 0 0 5px rgba(255, 0, 64, 0.5);
        animation: progress-glow-red 2s ease-in-out infinite;
      "></div>
    </div>
  </div>
</div>

<style>
  @keyframes cyber-pulse-red {
    0%, 100% { 
      box-shadow: 
        0 0 20px rgba(255, 0, 64, 0.2),
        inset 0 0 20px rgba(255, 0, 64, 0.05);
    }
    50% { 
      box-shadow: 
        0 0 30px rgba(255, 0, 64, 0.4),
        inset 0 0 25px rgba(255, 0, 64, 0.1);
    }
  }
  
  @keyframes glow-text-red {
    0% { 
      text-shadow: 
        0 0 10px rgba(255, 0, 64, 0.8),
        0 0 20px rgba(255, 0, 64, 0.4);
    }
    100% { 
      text-shadow: 
        0 0 15px rgba(255, 0, 64, 1),
        0 0 30px rgba(255, 0, 64, 0.6);
    }
  }
  
  @keyframes status-blink-red {
    0%, 100% { 
      opacity: 0.8;
      transform: scale(1);
    }
    50% { 
      opacity: 1;
      transform: scale(1.2);
    }
  }
  
  @keyframes progress-glow-red {
    0%, 100% { opacity: 0.9; }
    50% { opacity: 1; }
  }
  
  @keyframes scan-line {
    0% { top: -100%; }
    100% { top: 200%; }
  }
</style>

<img src="https://i.postimg.cc/440TPNNr/bear.webp"
     alt="Book Cover"
     style="width: 200px; border-radius: 12px; box-shadow: 0 0 10px rgba(0,0,0,0.4);" />
<p style="margin-top: 0.5rem;">
   <a href="https://www.amazon.com/Fancy-Bear-Goes-Phishing-Extraordinary/dp/0374601178" target="_blank" rel="noopener noreferrer">
    View on Amazon
  </a>
</p>
---


## Who Is Scott Shapiro and Why His Background Matters

Scott Shapiro is a professor of law and philosophy at Yale Law School, which might seem like weird credentials for writing about cybersecurity. But here's what makes him perfect for this: he was originally a computer science guy who grew up in the same Bell Labs culture as Robert Morris Jr. (the kid who accidentally crashed the early internet).

Shapiro was obsessed with UNIX as a teenager, studied CS in college, and even ran a tech consultancy in the early 90s. Then he completely walked away from technology for almost three decades to focus on law and philosophy. When he decided to research cyberwar, he figured his old CS background would make it easy to catch up. He was completely wrong.

He describes feeling like Rip Van Winkle, waking up to a totally different technological world. So he went back to school, relearned C and assembly, audited graduate CS courses, started going to hacker conventions, and even hacked the Yale Law website (which his dean was not happy about).

**Shapiro approached this project with three fundamental questions:** 

1. Why is the internet so vulnerable? 
2. How do hackers exploit its vulnerabilities? 
3. What can companies, states, and the rest of us do in response? 

His goal is to help readers understand these questions through the lens of both technical analysis and human behavior, while adding some philosophy in the mix.

This unique journey gives him a perspective that pure technologists or pure academics miss. He understands the deep technical concepts but also see the bigger philosophical and legal patterns that I personally haven't seen in any other book. 

## The Five Attacks That Tell the Whole Story

Shapiro picked five historical attacks that perfectly illustrate different aspects of cyber-insecurity:

1. **The Morris Worm (1988)** - The first internet worm that accidentally crashed everything, showing how early systems prioritized convenience over security
2. **The Bulgarian Virus Factory (1980s-90s)** - How economic displacement and intellectual curiosity created the first virus-writing community
3. **Paris Hilton Sidekick Hack (2005)** - A teenager's social engineering attack that exposed how companies prioritize growth and features over basic security training and practices
4. **Russian DNC Hacks (2016)** - Fancy Bear's psychological manipulation campaign 
5. **Mirai Botnet (2016)** -College students whose IoT botnet escalated into a criminal enterprise that disrupted infrastructure.

Each story works on multiple levels - as technical case studies, as human dramas, and as illustrations of bigger systemic problems. The selection is brilliant because it spans different eras, attack types, and motivations while building toward a unified theory of why cybersecurity keeps failing.

## His Three-Layer Framework

Shapiro's main insight is breaking cybersecurity into three interconnected layers:

**Downcode**: The technical stuff - software, operating systems, network protocols

**Upcode**: The human systems that shape how downcode gets built - laws, corporate policies, economic incentives, human psychology

**Metacode**: The fundamental philosophical principles that make computation possible (discovered by Alan Turing)

What I took from this is that purely technical solutions always fall short. You can patch vulnerabilities all day, but if the underlying incentives reward speed over security, you'll just get new vulnerabilities. The book covers these examples with the stories. 

## The Human Element: Why We Keep Making the Same Mistakes

Reading through these stories, what really stood out was how human all the failures are. Humans want convenience over security. Humans take mental shortcuts when under pressure. Humans don't want to spend money on invisible security features that customers can't see.

**The pattern is everywhere:**

- UNIX was designed for convenience and collaboration, not security, because that's what the research community wanted
- Microsoft ignored security for decades because winner-takes-all markets don't reward it and they felt pressured to make a good product.
- IoT devices shipped with default passwords and the buyer doesnt change the password(because customers care more about price and features than security)
- The DNC refused basic two-factor authentication because it was "annoying"
- T-Mobile store managers gave up credentials to anyone claiming to be corporate because they weren't trained properly

What I learned is that these aren't isolated technical failures. They're predictable human responses to systemic incentives. Companies optimize for what gets rewarded (speed, features, cost), not what's actually secure(thankfully this is changing though).

## The Psychology Behind Social Engineering

The Fancy Bear phishing analysis was interesting  because Shapiro breaks down the actual cognitive mechanisms. 

Fancy Bear exploited three psychological heuristics to fool Clinton campaign staffers:

1. **Availability heuristic**: "Ukraine = hacking" felt believable because Ukraine was constantly in cyber news
2. **Affect heuristic**: Fear and urgency ("suspicious login detected!") triggered emotional rather than rational decision-making
3. **Loss aversion**: The threat of account compromise felt worse than the hassle of changing passwords manually

Shapiro's distinction between "nudges" (using psychology for good) and "mudges" (using psychology maliciously) is brilliant. Security teams should be nudging people toward secure behaviors rather than just telling them what not to do.

## The Cat-and-Mouse Game: Inevitable and Eternal

 These stories show the eternal cat-and-mouse dynamic between attackers and defenders. Some teenager finds a clever way to exploit a system, forces everyone to patch it, then someone else finds the next weakness.

- Morris exploited buffer overflows, so everyone added input validation
- Virus writers created polymorphic engines, so antivirus companies improved detection
- Phishers got blocked by email filters, so they moved to social media and messaging apps
- DDoS-for-hire services got their PayPal accounts seized, so they moved to Bitcoin

The game never ends - it just shifts to new terrain. This helped me understand why Shapiro argues against "solutionism" (the belief that technology can solve social problems). There's no final technical solution to cybersecurity because the fundamental tension between accessibility and security is built into the nature of computation itself.

## From von Neumann to Modern Malware: The Philosophical Depth

This was my first time really learning about John von Neumann's contributions to computer science, and Shapiro weaves his insights beautifully into the virus stories. Von Neumann was trying to understand how biological organisms achieve resilience through self-replication. His "Universal Constructor" became the theoretical foundation for self-replicating computer programs.

The Bulgarian virus writers were essentially exploring the same questions von Neumann asked - how can code reproduce itself and survive in hostile environments? Dark Avenger's polymorphic engine was a sophisticated answer to this challenge, creating viruses that could change their appearance while maintaining their function.

Connecting this philosophical foundation to modern malware gives the technical stories much deeper meaning. 

## Bulgarian Viruses: Economic Displacement Meets Intellectual Curiosity

The Bulgarian virus factory chapter was fascinating because it shows how economic and social conditions create cybercrime ecosystems. After communism collapsed, Bulgaria had lots of technically skilled people with no legitimate opportunities. Writing viruses became a way to gain intellectual challenge and social status.

Vesselin Bontchev started as a legitimate researcher trying to understand malware, but his reports accidentally taught more people how to write viruses. The Virus Exchange (vX) became a community where you had to contribute a new virus to gain access - essentially gamifying malware development.

Dark Avenger emerges from this community as this mysterious figure writing some of the most sophisticated malware of the era. His relationship with researcher Sarah Gordon shows the weird social dynamics - he dedicated viruses to her, they corresponded for months, and she helped him develop some remorse for his actions.

This humanizes virus writers in a way that pure technical analysis misses. They're not evil masterminds - they're often smart people in economically desperate situations finding community and purpose through technical challenges.

## Why "Just Build Better Security" Doesn't Work

Shapiro's argument against "solutionism" is backed by both practical examples and theoretical proof. On the practical level, every technical security measure creates new attack surfaces or shifts the problem elsewhere.

But the theoretical argument is what makes this book special. Shapiro walks through Turing's proof that bug detection is fundamentally undecidable. Here's why this matters in simple terms: Turing proved that you can't build a program that can look at any other program and perfectly predict whether it will work correctly or have bugs.

Turing's proof shows why there will never be a perfect security scanner that catches every vulnerability. But that doesn't mean current scanners are useless,  they're actually really valuable for finding common vulnerability patterns. The point is we need to stop chasing the fantasy of a magic tool that will solve all security problems and focus on the human/organizational issues that create vulnerabilities in the first place.

 We need to focus on making attacks economically unviable(once again, fix the upcode to make better downcode)

## Policy Ideas That Address Root Causes

**Go after the money**: Shapiro shows how disrupting payment systems can be more effective than technical countermeasures. When academic researchers traced DDoS-for-hire services PayPal accounts and got them seized, many services went out of business despite still having working botnets. Same thing happened when researchers tracked spam payments and got Visa/Mastercard to fine the banks processing them, counterfeit pharmaceutical sales plummeted overnight.

**Create legitimate pathways for technical talent**: Many hackers turn to crime because they can't find legitimate opportunities that match their skills. Jonathan Lusthaus's research shows Eastern European cybercriminals are often older, formally trained programmers who resort to crime because tech jobs don't pay enough. Recruiting these people for legitimate cybersecurity work removes a major pipeline into cybercrime.

**Make companies bear the cost of insecurity**: The banking fraud analogy is interesting,  in countries where banks are liable for fraudulent transactions, fraud rates are much lower than where customers bear the risk. If software companies faced real financial consequences for shipping insecure products, they'd invest more in security. The California IoT law requiring unique passwords worked because it changed manufacturer incentives across the entire market.

What makes these approaches practical is they target the underlying economics and incentives rather than trying to patch individual technical problems. They recognize that cybersecurity is fundamentally about human behavior and organizational priorities, not just better technology.

## What I Actually Learned About Human Nature

The biggest takeaway for me was understanding how predictably human all these failures are. We want the easy way out. We optimize for convenience over security. We take mental shortcuts under pressure. We don't want to pay for invisible benefits.

This applies at every level:

- **Individual users** click suspicious links because loss aversion makes the "reset password" option feel safer than doing nothing
- **Companies** ship insecure products because customers can't see security features
- **Governments** shelter friendly hackers while prosecuting foreign ones

Reading these stories, it really does make sense why these attacks happen. They're inevitable results of human psychology meeting technological systems designed without security in mind.

But here's what's weirdly optimistic about this: if the problems are fundamentally human, then we can address them with better incentives, training, and policies. We can't engineer our way out of human nature, but we can design systems that work with it instead of against it.

### Patterns that this book helped me understand

1. Bad upcode creates bad downcode, which creates attack vectors
2. Many countries shelter cybercriminals or refuse to cooperate on prosecutions
3. Crime-as-a-service has made hacking accessible to non-technical people
4. Most serious hackers are young males seeking social status and intellectual challenge
5. Many hackers see themselves as moral agents with a sense of justice
6. Most hackers don't think they'll get caught
7. Early systems prioritized convenience and collaboration over security
8. People take mental shortcuts that attackers exploit
9. Legal systems impose few penalties for data breaches
10. States will never give up their right to spy
11. Early internet systems were mostly the same, so one attack could hit everything
12. For weak states, cyberweapons often ARE the arsenal

## Why This Book Stands Out

After reading Nicole Perlroth's apocalyptic "This is How They Tell Me the World Ends," Shapiro's measured approach was refreshing. He acknowledges real risks without exaggerating it.

The combination of technical depth, historical context, psychological insights, and philosophical grounding makes this unique. Shapiro shows how economic incentives, legal frameworks, cognitive biases, and technical capabilities all interact.

The humor helps too. Shapiro includes enough wit and personality to keep things engaging without undermining the actual analysis.

## Minor Criticism

Some of the philosophical tangents might be too abstract for readers who just want practical insights. For example, I loved the Alan Turing material and how Shapiro weaves it throughout different stories, but I can see how it might slow things down for readers looking for more direct security lessons. I personally enjoy the historical context and philosophical depth, but others may find it gets in the way.

Additionally, some of Shapiro's analogies can actually make concepts more confusing rather than clearer. For example, his explanation of code vs. data through Lewis Carroll's 'Achilles and the Tortoise' parable might leave readers more puzzled than enlightened. While these analogies are well-intentioned attempts to make technical concepts accessible, they sometimes add unnecessary complexity to ideas that could be explained more directly. I know it took me a few tries to really understand what he was getting at. 

## Bottom Line

This book taught me to see cybersecurity incidents as symptoms of deeper systemic problems rather than isolated technical failures. Every major breach has both a downcode story (what vulnerability was exploited) and an upcode story (why that vulnerability existed).

Shapiro succeeds because he understands that perfect security is impossible both practically and theoretically. The goal should be making attacks economically unviable rather than technically impossible.

The upcode/downcode/metacode framework gives you a way to think systematically about why we keep failing at security and what it might take to fail less catastrophically. The historical perspective shows how current problems have deep roots in decisions made decades ago.

If you want to learn a good batch of attacks that have ultimately shaped our security over time, while still being entertained, then I recommend this book. 

**Who should read this**: People who want to understand the root causes behind major cyberattacks and the human psychology that drives both attackers and defenders. While the book does get technical at points, Shapiro explains concepts clearly with visual aids, so motivated readers without deep technical backgrounds can follow along.

**Who shouldn't**: People looking for a quick how-to security guide, those who want pure technical analysis without a lot  of historical context, or readers who prefer straightforward narratives without philosophical tangents.





