<!-- ---
title: AI Security
date: 2023-04-24 11:11:11 -0400
description:
categories: [51AI, AIML]
# img: /assets/img/sample/rabbit.png
tags: [AI, ML]
--- -->

# AI Security
- [AI Security](#ai-security)
  - [Overview](#overview)
    - [AI tools](#ai-tools)
    - [AI Security](#ai-security-1)
    - [AI pipeline risk](#ai-pipeline-risk)
    - [AI Observability 可观察性](#ai-observability-可观察性)
    - [Secure AI Framework](#secure-ai-framework)
      - [Google Secure AI Framework](#google-secure-ai-framework)
        - [6 core elements:](#6-core-elements)
        - [5 steps to support and advance a framework](#5-steps-to-support-and-advance-a-framework)

---

## Overview

---

### AI tools

**AI Model**
- LLaMA,
- ChatGPT
- MPT
- Falcon
- Hugging Face model

LLM Engine
- https://github.com/scaleapi/llm-engine

LLM Security
- https://github.com/greshake/llm-security

**AI Platform**: Build, compare, deploy LLM Apps
- [ScaleAI](https://scale.com/spellbook)

**AI Observability**
- [whylabs](https://docs.whylabs.ai/docs/integrations-llm-whylogs-container)

---

### AI Security

- Artificial Intelligence is very good at finding vulnerabilities, and with the help of humans, it can exploit them even better.

- In computing, debuggers use AI software to look for bugs in source code, autocompletion, autocorrection, and handwriting software. AI can also find vulnerabilities in systems of finance, law, and even politics. AI is used to look for loopholes in contracts, datasets about people, and improve literature gaps.

This brings about two problems:

- AI can be **created to hack** a system.

  - it can be good or bad depending on how people use it.
  - A cybercriminal may create an advanced chatbot to obtain information from a wide range of people across vast platforms and perhaps even languages.
  - companies can use AI to look for the vulnerabilities they have and patch them up so an attacker cannot exploit them.

- AI might **unintentionally hack** the system.

  - Computers have a very different logic from humans. This means that almost all the time, they accept data, process it, and produce output in a completely different manner in contrast to humans.
  - Take an example of the classic game of chess:
  - Chess is an abstract strategy game that is played on a board with 64 squares arranged in an 8-by-8 grid. At the start, each player controls sixteen pieces. The aim is to checkmate the opponent's king with the condition that the king is in check and there is no escape.
  - A human and a classic chess engine look at this game in two very different ways. A human may play the value game (measuring winning by the value and number of pieces on the board), whereas a computer looks at a finite number of possibilities that can occur with each move the opponent makes via a search algorithm.
  - By having this limited ability to see into the future, the computer has the advantage almost every time to win the game. This is a very preliminary example and quite basic to the other systems that can be ‘hacked’ by Artificial intelligence.

- humans are programmed by implicit and explicit knowledge. Computers are programmed by a set of instructions and logic that never change unless told to. Therefore, computers and humans will have different approaches, solutions, and hacks for the same problem.

- But systems are built around humans and not computers. So, when the chips are down, computers can do a lot more vulnerability finding and exploitation to many more systems, both virtual and physical.

---

### AI pipeline risk

- AI's potential is limitless, but data security is paramount.
- as AI evolves, developers and researchers who rely on data sharing must prioritize securing sensitive information.

Breakdown of security risks in the AI pipeline:

![F6T3t15XMAAK23x](/assets/img/post/F6T3t15XMAAK23x.jpeg)

---

### AI Observability 可观察性

observability platform
- to control the behavior of ML & data applications.

- `Monitor and observe model performance` for predictive ML models, supporting delayed ground truth and custom performance metrics

- `Monitor and observe data quality` in ML model inputs, Feature Stores, batch and streaming pipelines

- Detect and root cause common ML issues such as drift, data quality, model performance degradation, and model bias

- Explain the cause of model performance degradation using tracing and feature importance

- Detect and root cause common LLM issues such as `toxicity, PII leakage, malicious activity, and indications of hallucinations`

---

### Secure AI Framework

> a summary of SAIF, click this [PDF](https://services.google.com/fh/files/blogs/google_secure_ai_framework_summary.pdf).
> how practitioners can implement SAIF, click this [PDF](https://services.google.com/fh/files/blogs/google_secure_ai_framework_approach.pdf).

The potential of AI, especially generative AI, is immense.

in the pursuit of progress within these new frontiers of innovation, there needs to be `clear industry security standards for building and deploying this technology in a responsible manner`

**Secure AI Framework (SAIF)**
- a conceptual framework for secure AI systems

- inspired by the security best practices (like reviewing, testing and controlling the supply chain) that we’ve applied to software development + incorporating understanding of security mega-trends and risks specific to AI systems

- A framework across the public and private sectors is essential for making sure that responsible actors safeguard the technology that supports AI advancements, so that when AI models are implemented, they’re secure-by-default.

#### Google Secure AI Framework

> google have long advocated for, and often developed, industry frameworks to raise the security bar and reduce overall risk.
> - `Supply-chain Levels for Software Artifacts (SLSA)` framework: improve software supply chain integrity
> - `BeyondCorp access model`: zero trust principles which are industry standard today.
> - `Google Secure AI Framework (SAIF)`:

- embraced an open and collaborative approach to cybersecurity. This includes combining frontline intelligence, expertise, and innovation with a commitment to share threat information with others to help respond to — and prevent — cyber attacks.

- Building on that approach, SAIF is designed to help `mitigate risks specific to AI systems` like stealing the model, data poisoning of the training data, injecting malicious inputs through prompt injection, and extracting confidential information in the training data.

##### 6 core elements:

1. Expand **strong security foundations** to the AI ecosystem

   - leveraging secure-by-default infrastructure protections and expertise built over the last two decades to protect AI systems, applications and users.
   - develop organizational expertise to keep pace with advances in AI and start to scale and adapt infrastructure protections in the context of AI and evolving threat models.
   - For example
     - injection techniques like SQL injection have existed for some time, and organizations can adapt mitigations, such as input sanitization and limiting, to help better defend against prompt injection style attacks.

2. Extend **detection and response** to bring AI into an organization’s threat universe

   - Timeliness is critical in detecting and responding to AI-related cyber incidents, and extending threat intelligence and other capabilities to an organization improves both.
   - For organizations, this includes `monitoring inputs and outputs of generative AI systems to detect anomalies` and `using threat intelligence to anticipate attacks`.
   - This effort typically requires collaboration with trust and safety, threat intelligence, and counter abuse teams.

3. **Automate defenses** to keep pace with existing and new threats

   - The latest AI innovations can `improve the scale and speed of response efforts` to security incidents.

   - Adversaries 对手 will likely use AI to scale their impact, so it is important to use AI and its current and emerging capabilities to stay nimble 灵活的 and cost effective in protecting against them.

4. **Harmonize 和声 platform level controls** to ensure consistent security across the organization

   - Consistency across control frameworks can support AI risk mitigation and scale protections across different platforms and tools to ensure that the best protections are available to all AI applications in a scalable and cost efficient manner.

   - At Google, this includes extending secure-by-default protections to AI platforms like Vertex AI and Security AI Workbench, and building controls and protections into the software development lifecycle.

   - Capabilities that address general use cases, like Perspective API, can help the entire organization benefit from state of the art protections.

5. Adapt controls to **adjust mitigations and create faster feedback loops** for AI deployment

   - Constant testing of implementations through continuous learning can ensure detection and protection capabilities address the changing threat environment.
   - This includes
     - techniques like reinforcement learning based on incidents and user feedback
     - steps such as updating training data sets, fine-tuning models to respond strategically to attacks and allowing the software that is used to build models to embed further security in context (e.g. detecting anomalous behavior).
   - Organizations can also conduct regular red team exercises to improve safety assurance for AI-powered products and capabilities.

6. Contextualize 置于上下文中 **AI system risks in surrounding business processes**
   - Lastly, conducting end-to-end risk assessments related to how organizations will deploy AI can help inform decisions.
   - This includes an `assessment of the end-to-end business risk`, such as data lineage, validation and operational behavior monitoring for certain types of applications.
   - In addition, organizations should `construct automated checks` to validate AI performance.

##### 5 steps to support and advance a framework

1. Fostering industry support for SAIF with the announcement of key partners and contributors in the coming months and continued industry engagement to help develop the `NIST AI Risk Management Framework` and `ISO/IEC 42001 AI Management System Standard` (the industry's first AI certification standard).
   1. These standards rely heavily on the security tenets in the `NIST Cybersecurity Framework` and `ISO/IEC 27001 Security Management System` — which Google will be participating in to ensure planned updates are applicable to emerging technology like AI — and are consistent with SAIF elements.

2. Working directly with organizations, including customers and governments to help them understand how to assess AI security risks and mitigate them.
   1. This includes conducting workshops with practitioners and continuing to `publish best practices for deploying AI systems securely`.

3. Sharing insights from Google’s leading threat intelligence teams like Mandiant and TAG on cyber activity involving AI systems.

4. Expanding our bug hunters programs (including Vulnerability Rewards Program) to reward and incentivize research around AI safety and security.

5. Continuing to deliver secure AI offerings with partners like GitLab and Cohesity, and further develop new capabilities to help customers build secure systems.
   1. That includes our commitment to the open source community and we will soon publish several open source tools to help put SAIF elements into practice for AI security.

---
