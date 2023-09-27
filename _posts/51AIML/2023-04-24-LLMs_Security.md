---
title: AIML - OWASP Top 10 for LLM
date: 2023-04-24 11:11:11 -0400
description:
categories: [51AIML]
# img: /assets/img/sample/rabbit.png
tags: [AIML]
---

# OWASP Top 10 for LLM

- [OWASP Top 10 for LLM](#owasp-top-10-for-llm)
  - [overall](#overall)
  - [OWASP Top 10 for LLM](#owasp-top-10-for-llm-1)
    - [Top 10](#top-10)
  - [Benefits of LLMs in Cybersecurity](#benefits-of-llms-in-cybersecurity)
    - [Debugging and Coding](#debugging-and-coding)
    - [Analysis of Threat Patterns](#analysis-of-threat-patterns)
    - [Response Automation](#response-automation)

---

## overall

> - [OWASP Top 10 for LLM VERSION 1.0 Published: August 1, 2023 ](https://owasp.org/www-project-top-10-for-large-language-model-applications/assets/PDF/OWASP-Top-10-for-LLMs-2023-v1_0.pdf)

- This undertaking is not a one-time effort but a continuous process, mirroring the ever-evolving nature of cyber threats. With the rapid advancements in LLMs, their potential for both utility and abuse will continue to grow, making the task of security a continually moving target that demands our attention and expertise.

- In closing, the quest for robust security measures for LLMs is ongoing.

- ensure that the tools are not just powerful and effective, but also `secure and ethically used`.



---

## OWASP Top 10 for LLM

- The frenzy of interest in Large Language Models (LLMs) following the mass-market pre-trained chatbots in late 2022 has been remarkable.

- Businesses, eager to harness the potential of LLMs, are rapidly integrating them into their operations and client-facing offerings. Yet, the breakneck speed at which LLMs are being adopted has outpaced the establishment of comprehensive security protocols, leaving many applications vulnerable to high-risk issues.

- The absence of a `unified resource addressing these security concerns in LLMs` was evident. Developers, unfamiliar with the specific risks associated with LLMs, were left with scattered resources and **OWASP’s mission seemed a perfect fit to help drive safer adoption of this technology**.


Who is it for?
- Our primary audience is developers, data scientists and security experts tasked with designing and building applications and plug-ins leveraging LLM technologies. We aim to provide practical, actionable, and concise security guidance to help these professionals navigate the complex and evolving terrain of LLM security.

The Making of the List
- The creation of the OWASP Top 10 for LLMs list was a major undertaking, built on the collective expertise of an international team of nearly 500 experts, with over 125 active contributors. Our contributors come from diverse backgrounds, including AI companies, security companies, ISVs, cloud hyperscalers, hardware providers and academia.

- Over the course of a month, we brainstormed and proposed `potential vulnerabilities`, with team members writing up 43 distinct threats. Through multiple rounds of voting, we refined these proposals down to a concise list of the ten most critical vulnerabilities. Each vulnerability was then further scrutinized and refined by dedicated sub-teams and subjected to public review, ensuring the most comprehensive and actionable final list.

- Each of these vulnerabilities, along with common examples, prevention tips, attack scenarios, and references, was further scrutinized and refined by dedicated sub-teams and subjected to public review, ensuring the most comprehensive and actionable final list.


Relating to other OWASP Top 10 Lists
- While our list shares DNA with vulnerability types found in other OWASP Top 10 lists, we do not simply reiterate these vulnerabilities. Instead, we delve into the unique implications these vulnerabilities have when encountered in applications utilizing LLMs.

- Our goal is to bridge the divide between general application security principles and the specific challenges posed by LLMs. This includes exploring how conventional vulnerabilities may pose different risks or might be exploited in novel ways within LLMs, as well as how traditional remediation strategies need to be adapted for applications utilizing LLMs.

The Future
- This first version of the list will not be our last. We expect to update it on a periodic basis to keep pace with the state of the industry. We will be working with the broader community to push the state of the art, and creating more educational materials for a range of uses. We also seek to collaborate with standards bodies and governments on AI security topics. We welcome you to join our group and contribute.


### Top 10

![LLM Blog Cover -- OG](/assets/img/post/LLM%20Blog%20Cover%20--%20OG.webp)

LLM01: **Prompt Injection**
  - This manipulates a large language model (LLM) through `crafty inputs, causing unintended actions by the LLM`. Direct injections overwrite system prompts, while indirect ones manipulate inputs from external sources.

LLM02: **Insecure Output Handling**
  - This vulnerability occurs when an `LLM output is accepted without scrutiny, exposing backend systems`. Misuse may lead to severe consequences like XSS, CSRF, SSRF, privilege escalation, or remote code execution.

LLM03: **Training Data Poisoning**
  - This occurs when `LLM training data is tampered, introducing vulnerabilities or biases that compromise security, effectiveness, or ethical behavior`. Sources include Common Crawl, WebText, OpenWebText, & books.

LLM04: **Model Denial of Service**
  - Attackers cause `resource-heavy operations on LLMs, leading to service degradation or high costs`. The vulnerability is magnified due to the resource-intensive nature of LLMs and unpredictability of user inputs.

LLM05: **Supply Chain Vulnerabilities**
  - LLM application lifecycle can be `compromised by vulnerable components or services`, leading to security attacks. `Using third-party datasets, pre-trained models, and plugins` can add vulnerabilities.

LLM06: **Sensitive Information Disclosure**
  - LLM’s may `inadvertently reveal confidential data in its responses, leading to unauthorized data access, privacy violations, and security breaches`. It’s crucial to implement data sanitization and strict user policies to mitigate this.

LLM07: **Insecure Plugin Design**
  - LLM plugins can have insecure inputs and insufficient access control. This lack of application control makes them `easier to exploit and can result in consequences like remote code execution`.

LLM08: **Excessive Agency** 过度代理
  - LLM-based systems may undertake actions leading to unintended consequences. The issue arises from excessive functionality, permissions, or autonomy granted to the LLM-based systems.

LLM09: **Overreliance** 过度依赖
  - Systems or people overly depending on LLMs without oversight may face `misinformation, miscommunication, legal issues, and security vulnerabilities due to incorrect or inappropriate content generated by LLMs`.

- LLM10: **Model Theft**
  - This involves `unauthorized access, copying, or exfiltration of proprietary LLM models`. The impact includes economic losses, compromised competitive advantage, and potential access to sensitive information.


---

## Benefits of LLMs in Cybersecurity


### Debugging and Coding

- There are already debuggers that do a pretty good job. But with LLMs you can literally write code and debug at a much faster rate.
- ensure that the LLM is provided by a company that doesn’t have the potential to use your data – like Samsung found out when their proprietary code was leaked by accident.

![image-235](https://www.freecodecamp.org/news/content/images/2023/04/image-235.png)


### Analysis of Threat Patterns

- LLMs have the feature of `pattern finding` and this could be utilised to analyse behaviours and tactics of Advanced Persistent Threats in order to better attribute incidents and mitigate them if such patterns are recognised in real-time.


### Response Automation

- LLMs have a lot of potential in the Security Operations Center and response automation.
- Scripts, tools, and even reports can be written using these models, reducing the total amount of time professionals require to do their work.

---
