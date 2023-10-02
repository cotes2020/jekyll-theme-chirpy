---
title: AIML - 4
date: 2023-04-24 11:11:11 -0400
description:
categories: [51AIML]
# img: /assets/img/sample/rabbit.png
tags: [AIML]
---

# OWASP Top 10 for LLM

- [OWASP Top 10 for LLM](#owasp-top-10-for-llm)
    - [LLM04: Model Denial of Service](#llm04-model-denial-of-service)
      - [Vulnerability Examples](#vulnerability-examples)
      - [Attack Scenario Examples](#attack-scenario-examples)
      - [Prevention Solution](#prevention-solution)


---

### LLM04: Model Denial of Service

- An attacker interacts with a LLM in a method that `consumes an exceptionally high amount of resources, which results in a decline in the quality of service` for them and other users, as well as potentially incurring high resource costs.

- Furthermore, an emerging major security concern is the possibility of `an attacker interfering with or manipulating the context window of an LLM`.
  - This issue is becoming more critical due to the increasing use of LLMs in various applications, their intensive resource utilization, the unpredictability of user input, and a general unawareness among developers regarding this vulnerability.
  - In LLMs, the context window represents the maximum length of text the model can manage, covering both input and output. It's a crucial characteristic of LLMs as it dictates the complexity of language patterns the model can understand and the size of the text it can process at any given time.
  - The size of the context window is defined by the model's architecture and can differ between models.


#### Vulnerability Examples

- `resource-limitation`
  - Posing queries that lead to  `recurring resource usage through high-volume generation of tasks in a queue`
    - e.g. with LangChain or AutoGPTz

  - Sending queries that are unusually `resource-consuming`, perhaps because they use unusual orthography or sequencesz

- `Continuous input overflow`:
  - An attacker sends a stream of input to the LLM that exceeds its context window, causing the model to consume excessive computational resourcesz

- `Repetitive long inputs`:
  - The attacker repeatedly sends long inputs to the LLM, each exceeding the context windows

- `Recursive context expansion`:
  - The attacker constructs input that triggers recursive context expansion, forcing the LLM to repeatedly expand and process the context windows

- `Variable-length input flood`:
  - The attacker floods the LLM with a large volume of variable-length inputs, where each input is carefully crafted to just reach the limit of the context window.
  - This technique aims to exploit any inefficiencies in processing variable-length inputs, straining the LLM and potentially causing it to become unresponsive.


#### Attack Scenario Examples

- An attacker `repeatedly sends multiple requests` to a hosted model that are difficult and costly for it to process, leading to worse service for other users and increased resource bills for the host

- `A piece of text on a webpage is encountered` while an LLM-driven tool is collecting information to respond to a benign query.
  - This leads to the tool making many more web page requests, resulting in large amounts of resource consumption

- An attacker continuously `bombards the LLM with input that exceeds its context window`.
  - The attacker may use automated scripts or tools to send a high volume of input, overwhelming the LLM's processing capabilities. As a result, the LLM consumes excessive computational resources, leading to a significant slowdown or complete unresponsiveness of the system

- An attacker `sends a series of sequential inputs to the LLM, with each input designed to be just below the context window's limit`.
  - By repeatedly submitting these inputs, the attacker aims to `exhaust the available context window capacity`.
  - As the LLM struggles to process each input within its context window, system resources become strained, potentially resulting in degraded performance or a complete denial of service

- An attacker leverages the LLM's recursive mechanisms to `trigger context expansion repeatedly`.
  - By crafting input that exploits the recursive behavior of the LLM, the attacker forces the model to repeatedly expand and process the context window, consuming significant computational resources.
  - This attack strains the system and may lead to a DoS condition, making the LLM unresponsive or causing it to crash

- An attacker `floods the LLM with a large volume of variable-length inputs, carefully crafted to approach or reach the context window's limit`.
  - By overwhelming the LLM with inputs of varying lengths, the attacker aims to exploit any inefficiencies in processing variable-length inputs.
  - This flood of inputs puts excessive load on the LLM's resources, potentially causing performance degradation and hindering the system's ability to respond to legitimate requests.

---

#### Prevention Solution

- **Implement input validation and sanitization** to ensure user input adheres to defined limits and filters out any malicious content

- **Cap resource use per request or step**, so that requests involving complex parts execute more slowly

- **Enforce API rate limits** to restrict the number of requests an individual user or IP address can make within a specific timeframe

- **Limit the number of queued actions and total actions** in a system reacting to LLM responses

- **Continuously monitor the resource utilization** of the LLM to identify abnormal spikes or patterns that may indicate a DoS attack.

- Set **strict input limits based on the LLM's context window** to prevent overload and resource exhaustionk

- Promote awareness among developers about potential DoS vulnerabilities in LLMs and provide guidelines for secure LLM implementation.



Reference Links
- [LangChain max_iterations](https://twitter.com/hwchase17/status/160846749387757977D)
- [Sponge Examples Energy-Latency Attacks on Neural Networks](https://arxiv.org/abs/2006.03469)
- [OWASP DOS Attack](https://owasp.org/www-community/attacks/Denial_of_Servic)
- [Learning From Machines: Know Thy Context](https://lukebechtel.com/blog/lfm-know-thy-context)
