---
title: "Book Review: This Is How They Tell Me the World Ends by Nicole Perlroth"
date: 2025-08-29 21:10:26 -0700
categories: [BookReview]
tags: [cybersecurity, zero-day, cyberwar]
media_subpath: 
image: 
  path: "https://i.postimg.cc/QN4WPfV4/Chat-GPT-Image-Aug-29-2025-09-37-08-PM.png"
---

### Short Summary

This book follows Nicole Perlroth investigating the zero-day market, cyber weapons, and the cyber arms race. She was originally covering Silicon Valley stuff when suddenly the New York Times asked her to cover cybersecurity - even though she knew nothing about it. This actually works in the book's favor since her learning curve mirrors that of general readers, making her discoveries and reactions feel more relatable.

The book is heavily U.S.-focused, tracing how American government demand for cyber weapons transformed a small bug bounty market into a dangerous global arms trade. Through interviews with NSA hackers, exploit brokers, and foreign intelligence contractors, she shows how America's offensive cyber strategy became insane - the same weapons we developed (like Stuxnet) ended up in the hands of adversaries and authoritarian governments, paving the way for future chaos. She occasionally adds political commentary that doesn't really connect to the cybersecurity story, which might distract some readers. She also tends to present every development in the most alarming way possible, which might make readers feel more panicked about cyber threats than necessary.
<img src="https://m.media-amazon.com/images/I/51cRIu2us8L._SY445_SX342_.jpg"
     alt="This Is How They Tell Me the World Ends Book Cover"
     style="width: 200px; border-radius: 12px; box-shadow: 0 0 10px rgba(0,0,0,0.4);" />
<p style="margin-top: 0.5rem;">
   <a href="https://www.amazon.com/This-They-Tell-World-Ends/dp/1635576059" target="_blank" rel="noopener noreferrer">
    View on Amazon
  </a>
</p>

### **Key Takeaways**

This book is NOT a technical cybersecurity book. If you're planning to read this to gain technical knowledge, this book is NOT for you. This is the type of book you give to your siblings/parents/friends if you want them to understand cyber threats and cyberwarfare. The book doesn't get too technical but covers the main points just enough for readers to grasp the importance and implications.

**The cyber arms race is a cat-and-mouse game where everyone loses.** The NSA hoarded zero-days until adversaries stole them (the Shadow Brokers leak led to WannaCry), Stuxnet escaped from Iran's nuclear facility and taught the world what cyber weapons could do, and backdoors in global systems became vulnerabilities that everyone wanted to exploit. The Aurora attacks on Google showed how China turned the tables, using zero-days to target American companies.

**The zero-day market evolution reveals how government demand corrupts everything.** It started with iDefense paying hackers a few hundred dollars to report bugs, giving vendors time to patch while providing early warnings to subscribers. But government agencies paying 1000x more created perverse incentives. Companies like VRL (founded by ex-NSA hackers) weaponized raw exploits for millions. NSO Group democratized surveillance with Pegasus spyware, selling to any government with $1 million. The "NOBUS" (Nobody But Us) strategy assumed only America could exploit certain vulnerabilities - a dangerous miscalculation.

**Major case studies demonstrate the book's central thesis.** NotPetya showed how Russian hackers used stolen NSA tools to attack Ukraine but caused over $10 billion in global damage to Western companies like Merck and Maersk. Stuxnet/Olympic Games used sophisticated zero-days to physically destroy Iran's nuclear centrifuges while sending fake "normal" data to operators, but the worm escaped and revealed what cyber weapons could accomplish. Project Gunman (1983) discovered Soviet magnetometer bugs hidden in embassy typewriters that recorded keystrokes -  which was basically an early example of supply chain attacks.

**How it connects to cybersecurity/cyberwar:** The book demonstrates why offensive cyber strategies can backfire, how bug bounty programs try to compete with gray markets, and why the democratization of surveillance technology threatens democracies worldwide. 

### **Some Interesting Parts**

#### **The "Race to Bare Metal" - Kernel-Level Exploitation**

The NSA's Tailored Access Operations (TAO) - their elite hacking unit -  pursued what they called the "race to bare metal" ,  achieving kernel-level access on target systems. The kernel is the core component of any operating system that manages all communication between hardware and software.

TAO wanted kernel access because it provided the deepest possible foothold in a target's system. Most malware operates at the application level, but kernel exploits give you control over the machine's fundamental operations. You can hide from antivirus software, manipulate what users see on their screens, and maintain access even after reboots or updates.

This was part of the post-9/11 expansion where the NSA wanted total surveillance capabilities. As one former TAO analyst put it:

> "we were collecting crazy intelligence. You couldn't believe the shit you were looking at. Our work was landing directly in presidential briefings."


#### **The Apple vs. FBI Encryption Battle**

The San Bernardino case perfectly demonstrates the book's central argument about how government demands for backdoors ultimately make everyone less secure. After the 2015 mass shooting in San Bernardino, the FBI found the gunman's locked iPhone on the ground but couldn't access its contents due to Apple's encryption. FBI Director James Comey essentially asked Apple to create what critics dubbed "Government OS", new software that would bypass the phone's security features for this one case.

The government framed this as a simple law enforcement request, but as Perlroth notes:

> "If Apple refused, the government could make the case that the company was providing a terrorist safe haven."


This put Apple in a hard position - comply and weaken everyone's security, or refuse and be painted as helping the bad guys.

Apple CEO Tim Cook refused, making a crucial point that connects directly to the book's themes:

> "Even if Apple did what the government was asking, even if it wrote a backdoor just for this one case, that backdoor would become a target for every hacker, cybercriminal, terrorist, and nation-state under the sun. How could the U.S. government ever guarantee it could keep Apple's backdoor safe, when it could not even manage to protect its own data?"


Cook's argument proved prophetic. The Shadow Brokers leak of EternalBlue - one of the NSA's most powerful cyberweapons - showed that even elite intelligence agencies can't keep their tools secure. EternalBlue ended up powering WannaCry and other devastating attacks against the very infrastructure it was meant to protect.

The FBI eventually dropped the case after hiring outside hackers to break into the phone, reportedly paying over $1 million for the exploit. But they never disclosed the vulnerability to Apple, meaning millions of iPhones potentially remained exposed to the same attack method.  This perfectly encapsulates the book's argument about how the zero-day market creates perverse incentives.

As Apple stated in their public letter:

> "Building a version of iOS that bypasses security in this way would undeniably create a backdoor. And while the government may argue that its use would be limited to this case, there is no way to guarantee such control."


### **Strengths**

The book excels when Perlroth acts as an active investigative reporter, gaining access to key players who shaped the cyber arms trade. Her storytelling approach makes complex topics accessible to general readers without requiring technical background, and her role as an active character learning alongside the reader keeps the narrative engaging.

The clarity is strong throughout - readers can easily follow the progression from early bug bounty programs to sophisticated state-sponsored cyber weapons. She effectively uses the zero-day market as a lens to understand broader geopolitical cyber dynamics, explaining technical concepts  in terms anyone can grasp.

Her source access is awesome. She interviewed key figures like Charlie Miller (the ex-NSA hacker who sparked the "no free bugs" movement), Jim Gosler (the "godfather of American cyberwar"), Ralph Langner (who reverse-engineered Stuxnet), and numerous government officials, exploit brokers, and foreign contractors. Much of this was on-the-record, giving readers direct insights into this secretive world.

The case studies effectively support her central thesis. NotPetya demonstrated how cyberweapons can't be contained - Russian attacks on Ukraine caused more damage to Western companies than intended targets. The Aurora attacks revealed China's sophisticated targeting of American intellectual property and even their own people. 

### **Weaknesses / Gaps**

The book's U.S. focused perspective, while understandable, limits deeper analysis of how adversaries view their own cyber strategies. While she does cover attacks and their outcomes in other countries, more insight into Russian, Chinese, and Iranian strategic thinking would strengthen the analysis beyond seeing them primarily through an American lens.

Perlroth frequently uses hyperbolic language that undermines her credibility as an objective reporter. Phrases like "invisible armies," "ticking time bomb," and "digital hellscape"(there are just some of the many) make legitimate threats sound like thriller fiction. This dramatic style sometimes undermines the main point.

While Perlroth's investigative reporting is solid, she frequently injects political commentary that goes beyond what's necessary to understand policy implications. Readers seeking objective analysis may find  it a bit distracting from the core cybersecurity narrative.

Perlroth tends to present every issue in the most alarming and exaggerated way possible, which can undermine her credibility and make readers feel unnecessarily panicked about cyber threats as if our infrastructure is one click away from blowing up by Russia…lol. 

### **Final Verdict**

Essential reading for understanding how America's cyber weapons strategy blew up  and contributed to the dangerous global cyber arms market we face today. Perlroth's investigative access and storytelling ability make complex geopolitical cyber issues accessible to general audiences. However, readers should be prepared for dramatic commentary that detracts from otherwise solid reporting. Recommended for anyone wanting to grasp how cyber threats became a national security issue without getting lost in technical details.