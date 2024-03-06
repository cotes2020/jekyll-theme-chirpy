---
title: AIML - LLM with Vulnerability Detection
date: 2023-04-24 11:11:11 -0400
description:
categories: [51AIML]
# img: /assets/img/sample/rabbit.png
tags: [AIML]
---


# LLM with Vulnerability Detection

- [LLM with Vulnerability Detection](#llm-with-vulnerability-detection)
  - [overall](#overall)
  - [Vulnerability Detection using Large Language Models](#vulnerability-detection-using-large-language-models)


---


## overall


Motivation
- Vulnerability detection is a very critical task for systems security.

- Current analysis techniques suffer from the trade-off between coverage and accuracy.

- ML-based analysis tools are non-robust, black-box and unreliable to use in real-world [^Dos].

[^Dos]:DanielArp,ErwinQuiring,FeargusPendlebury,AlexanderWarnecke,andFabioPierazzi. Dos and don’ts of machine learning in computer security, 2021.


- LLMs demonstrate revolutionizing capabilities for programming language-related tasks but they are also studied in a black-box fashion for both vulnerability detection and its repair.


> Security experts follow a step-by-step approach for vulnerability detection. Can using the same approach help LLMs performing better at the vulnerability detection task?


## Vulnerability Detection using Large Language Models

Step-by-Step Vulnerability Detection using Large Language Models [^USENIX_23_Poster]

[^USENIX_23_Poster]: https://www.bu.edu/peaclab/files/2023/08/USENIX_23_Poster.pdf

Objective
- Design a framework to emulate step-by-step reasoning process of a human security expert using LLMs, to **efficiently detect vulnerabilities in source code.**



Methodology
- uses few-shot in-context learning to guide LLMs to follow a step-by-step human-like reasoning model for vulnerability detection.
- make sure that the model first generates chain-of-thought reasoning [^Chain-of-thought] and then makes a decision based on that reasoning (Figure 1 and 3b).

[^Chain-of-thought]: JasonWei,XuezhiWang,DaleSchuurmans,MaartenBosma,BrianIchter,FeiXia,EdChi,QuocLe,andDennyZhou. Chain-of-thought prompting elicits reasoning in large language models, 2023.

![Screenshot 2023-11-08 at 10.45.34](/assets/img/Screenshot%202023-11-08%20at%2010.45.34.png)



Visualizing the Process of Vulnerability Detection
- the behavior of an LLM when it is asked to detect a vulnerability in two different scenarios.
  - First, when it is asked to give a direct answer (Figure 3a);
  - second, when it is first asked to perform human-expert like reasoning and then make a decision (Figure 3b).
- We choose GPT-3.5 as an LLM and a code snip- pet containing an out-of- bound write vulnerability as a running example.

![Screenshot 2023-11-08 at 10.48.39](/assets/img/Screenshot%202023-11-08%20at%2010.48.39.png)

![Screenshot 2023-11-08 at 10.48.52](/assets/img/Screenshot%202023-11-08%20at%2010.48.52.png)



Evaluation
- it shows that step-by-step reasoning guides the LLM to detect the (CWE-787) vulnerability.
- To systematically evaluate this approach, we create our own diverse synthetic dataset based on a subset of the MITRE 2022 top 25 most dangerous vulnerabilities.
- **For each vulnerability** we create vulnerable examples and their patches with varying levels of complexity.
- We use the ‘gpt-3.5-turbo-16k’ chat API to compare our approach with SoTA tools (Table 1).

![Screenshot 2023-11-08 at 10.50.29](/assets/img/Screenshot%202023-11-08%20at%2010.50.29.png)

Takeaway
- Following a human-like step-by-step reasoning approach helps LLMs to efficiently analyze code and detect vulnerabilities.
- Our approach provides an explanation for the detected vulnerabilities, which helps user to better contextualize them and to find their root cause.
- Systematic evaluation of this approach on real-world datasets is still required to determine its reliability in real-world use cases.
