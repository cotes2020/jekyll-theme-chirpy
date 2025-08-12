---
title: Security Tools
date: 2020-02-10 11:11:11 -0400
categories: [02Security]
tags: [SecurityTools]
math: true
image:
---

# Security Tools

- [Security Tools](#security-tools)
  - [OSS Security](#oss-security)
  - [Scanner by AI](#scanner-by-ai)

---

## OSS Security

Gecko:
- [gecko](https://www.gecko.security):
- Become Secure by Default. Find and fix broken authentication, logic bugs, and complex vulnerabilities that rules-based scanners and humans miss.
- Gecko uses an AI native engine to build a semantic understanding of your application. It links together context from your code, infrastructure and documentation to trace how data flows and trust boundaries occur. By threat modelling targeted attack scenarios, Gecko surfaces multi step and business logic vulnerabilities that pattern matching tools overlook.

## Scanner by AI

GitHub Copilot CodeQL + AI Enhancements
- Traditional static analysis via CodeQL, enhanced by GitHub's Copilot AI for security autofixes, vulnerability explanations, and code suggestions.
- AI role: Copilot can detect insecure patterns and `suggest safer alternatives` directly `in the IDE`.
- Good for: Developers using GitHub workflows, early-stage secure coding practices.

Microsoft Security Copilot + Defender for DevOps
- AI assistant (GPT-powered) for security analysts, integrated with Azure DevOps or GitHub to scan IaC, containers, and application code.
- AI role: Generates insights, explanations, and `recommended remediations` for discovered vulnerabilities.
- Good for: Enterprises in the Microsoft ecosystem.

Jit Security + AI Remediation
- Continuous security tool that scans repos using SAST tools (e.g., Semgrep, Bandit), then `uses AI to prioritize and explain results, and propose code fixes`.
- AI role: `Generates developer-contextual explanations and patch suggestions` using LLMs.
- Good for: Teams wanting lightweight DevSecOps automation with clear actionability.

Snyk AI
- AI-enhanced version of the Snyk platform with scanning for SAST, dependencies, IaC, containers.
- AI role: Offers `AI-generated remediation advice, code explanations, and prioritization of security issues`.
- Good for: Developers needing security insights in `real time` within GitHub/IDE/CI pipelines.

Bearer AI
- SAST tool that uses code context and AI heuristics to find data security and privacy risks (e.g., secrets, PII usage, unsafe flows).
- AI role: Trains models to `understand code patterns and sensitive data flows` beyond regex or rule-based systems.
- Good for: Privacy/security-aware companies `handling sensitive data`.

Oxeye AI (for cloud-native apps)
- Cloud-native security scanner that combines runtime context with static and dynamic analysis.
- AI role: Helps `correlate findings` across layers (e.g., vulnerabilities and actual exploit paths).
- Good for: K8s/containerized environments.

CodeSentry by GrammaTech (AI-enhanced SCA + SAST)
- Combines SCA (Software Composition Analysis) with AI to `identify known and unknown vulnerabilities` in binaries and source code.
- AI role: Uses machine learning for `vulnerability prediction and classification`.
- Good for: Embedded software, complex C/C++ systems.

Astra Security AI Scanner
- Web app security scanner with AI-supported threat detection and remediation guidance.
- AI role: Natural language `summaries of issues and guided fixing` steps using LLMs.
- Good for: Web apps and small-to-medium businesses.


.
