---
title: Security Conference
date: 2021-08-11 11:11:11 -0400
description:
categories: [51AI]
img: /assets/img/sample/rabbit.png
tags: [AI, ML]
---

# Security Conference

**Table of contents:**
- [Security Conference](#security-conference)
  - [Black Hat](#black-hat)
    - [Agentic AI Attack Surface](#agentic-ai-attack-surface)
    - [Security Controls](#security-controls)
    - [AI-Defensive Use Cases](#ai-defensive-use-cases)
    - [Agentic AI Threat Surface](#agentic-ai-threat-surface)
    - [Context as the Primary Attack Vector](#context-as-the-primary-attack-vector)
    - [Evolution of AI Agents](#evolution-of-ai-agents)
    - [Security Controls \& Recommendations](#security-controls--recommendations)
    - [Defensive AI Use Cases](#defensive-ai-use-cases)
    - [Lockheed Martin’s AI Security Approach](#lockheed-martins-ai-security-approach)

---

## Black Hat

---

###  Agentic AI Attack Surface

| Attack Type                 | Description                                                   | Example                                                       | Risk Level |
| --------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------- | ---------- |
| **Prompt Injection**        | Malicious instructions hidden in prompts or external content. | AI told to give away secrets or execute destructive commands. | 🔴 High     |
| **Context Poisoning**       | Corrupting training or inference-time data fed to model.      | Adding misleading info to a database LLM queries.             | 🔴 High     |
| **Access Control Issues**   | “Confused deputy” – agent has more privileges than user.      | Low-priv user triggers high-priv agent action.                | 🔴 High     |
| **Tool Misuse**             | Exploiting agents’ tool access to run harmful commands.       | AI executes file delete instead of file read.                 | 🟠 Medium   |
| **Memory Poisoning**        | Altering short/long-term memory to influence future behavior. | Fake “facts” stored in agent memory.                          | 🟠 Medium   |
| **Cascading Hallucination** | Error from one agent spreads to others.                       | Wrong data from Agent A causes bad output in Agent B.         | 🟠 Medium   |


### Security Controls

| Control                  | Purpose                                      | Tools/Methods                               |
| ------------------------ | -------------------------------------------- | ------------------------------------------- |
| **Model Scanning**       | Detect malware in model files.               | Antivirus + AI-specific scanners.           |
| **Runtime Security**     | Catch abnormal AI behavior during execution. | Anomaly detection, runtime firewalls.       |
| **Human-in-the-loop**    | Approve/deny high-risk AI actions.           | Review checkpoints in workflows.            |
| **AuthN/AuthZ**          | Control agent-to-agent communications.       | OAuth 2.0, OIDC, mTLS, X.509 certs.         |
| **Context Sanitization** | Clean and validate external inputs.          | Filtering, regex checks, schema validation. |
| **Logging & Auditing**   | Trace agent actions & context changes.       | Centralized logging, immutable audit logs.  |



### AI-Defensive Use Cases

| Use Case                | How AI Helps                                                | Example                                             |
| ----------------------- | ----------------------------------------------------------- | --------------------------------------------------- |
| **Threat Intel**        | Correlate CVEs, exploits, chatter in real time.             | AI parses GitHub, social media, CVE feeds.          |
| **Vuln Prioritization** | Rank vulnerabilities by exploit activity & business impact. | Prioritize based on attacker chatter.               |
| **AI Code Review**      | Multi-agent PR analysis by security domain.                 | Developer agent + Architect agent + Security agent. |



---

### Agentic AI Threat Surface

- AI agents can execute tools, access data, and interact with other agents — all of which expand the attack surface.

- Risks include:
  - **Prompt injection** (malicious instructions in prompts)
  - **Context poisoning** (compromising the data fed into the model)
  - **Access control issues** (e.g., confused deputy problem)
  - **Tool misuse** and cascading hallucinations
  - **Memory poisoning** (short-term & long-term)

- Orchestrator → specialized agent architectures multiply risks.

---

### Context as the Primary Attack Vector

- Context comes from instructions, memory, tools, and other models.

- Protect both inference-time and training-time context.

- Context is King
  - Sources of context:
  - Direct instructions (prompts)
  - Short-term memory (chat history)
  - Long-term memory (vector DB, docs, code)
  - Tool outputs
  - Other LLMs

- LLMs do NOT distinguish between system prompt and user prompt — corruption anywhere is dangerous.
  - LLMs treat all context (system prompts, user prompts) as one — making poisoned context highly dangerous.


- Context Protection
  - Techniques to validate, sanitize, and isolate context sources.
  - Monitoring for unexpected context changes.

---

### Evolution of AI Agents

- LLM apps → Tool-using agents → MCP-based agents → upcoming auto-discovery & self-modifying agents → general-purpose agents.

---

### Security Controls & Recommendations

- Scan models for malicious code.
- Add runtime security and logging for AI apps.
- Use human-in-the-loop for high-risk actions.
- Apply traditional security principles (auth, authz, least privilege) to AI systems.
- Consider protocols for agent-to-agent authentication (OAuth2.0, mTLS, certificates).

- Authentication & Authorization for AI Agents
  - How OAuth2.0, OpenID Connect, mTLS, and certificate-based auth can be applied to agent-to-agent communications.
  - Limitations of existing protocols in AI agent ecosystems.

---

### Defensive AI Use Cases

- Threat intelligence enrichment (correlating CVEs, exploits, and chatter).

- AI-augmented vulnerability prioritization.

- AI-driven code review (multi-agent review pipeline).

---

### Lockheed Martin’s AI Security Approach

- Prioritize open-weight models from trusted vendors.

- Key pillars:

  - Traceable & transparent stack

  - Secure stack (T&E, monitoring, adversarial testing, AI BOM)

- Guardrails (e.g., Llama Guard) help but don’t solve model misalignment.

- Model Alignment & Governance
  - Guardrails vs. alignment issues.
  - AI Bill of Materials (AI BOM) concept.
  - Approaches to AI red teaming.






.