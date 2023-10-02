---
title: AIML - 5
date: 2023-04-24 11:11:11 -0400
description:
categories: [51AIML]
# img: /assets/img/sample/rabbit.png
tags: [AIML]
---

# OWASP Top 10 for LLM

- [OWASP Top 10 for LLM](#owasp-top-10-for-llm)
    - [Supply Chain Vulnerabilities (LLM05)](#supply-chain-vulnerabilities-llm05)
      - [Vulnerability Examples](#vulnerability-examples)
      - [Attack Scenario Examples](#attack-scenario-examples)
      - [Prevention Solution](#prevention-solution)

---

### Supply Chain Vulnerabilities (LLM05)

- The supply chain in LLMs can be vulnerable, impacting the `integrity of training data, ML models, and deployment platforms`. These vulnerabilities can lead to `biased outcomes, security breaches, or even complete system failures`.

- Traditionally, vulnerabilities are focused on `software components`

- But Machine Learning extends this with the pre-trained models and training data supplied by third parties susceptible to tampering and poisoning attacks. Finally, `LLM Plugin extensions` can bring their own vulnerabilities. These are described in LLM - Insecure Plugin Design, which covers writing LLM Plugins and provides information useful to evaluate third-party plugins.


#### Vulnerability Examples

- `Traditional third-party package vulnerabilities`, including outdated or deprecated components

- Using a `vulnerable pre-trained model for fine-tuning`

- Use of `poisoned crowd-sourced data for training`

- Using `outdated or deprecated models` that are no longer maintained leads to security issues

- Unclear `T&Cs and data privacy policies of the model operators` lead to the application’s sensitive data being used for model training and subsequent sensitive information exposure.
  - This may also apply to risks from using copyrighted material by the model supplier.


#### Attack Scenario Examples

- An attacker `exploits a vulnerable Python library to compromise a system`. This happened in the first OpenAI data breach

- An attacker provides `an LLM plugin to search for flights which generates fake links` that lead to scamming plugin users

- An attacker `exploits the PyPi package registry to trick model developers into downloading a compromised package` and exfiltrating data or escalating privilege in a model development environment. This was an actual attack

- An attacker `poisons a publicly available pre-trained model` specialising in economic analysis and social research to create a backdoor which generates misinformation and fake news. They deploy it on a model marketplace (e.g. HuggingFace) for victims to use

- An attacker `poisons publicly available data set` to help create a backdoor when fine-tuning models. The backdoor subtly favours certain companies in different markets

- A `compromised employee of a supplier (outsourcing developer, hosting company, etc) exfiltrates data, model, or code stealing IP`.

- An `LLM operator changes its T&Cs and Privacy Policy` so that it requires an explicit opt-out from using application data for model training, leading to memorization of sensitive data.


Reference Links

- [ChatGPT Data Breach Confirmed as Security Firm Warns of Vulnerable Component Exploitation](https://www.securityweek.com/chatgpt-data-breach-confirmed-as-security-firm-warns-of-vulnerable-component-exploitation)
- [Open AI’s Plugin review process](https://platform.openai.com/docs/plugins/review)
- [Compromised PyTorch-nightly dependency chain](https://pytorch.org/blog/compromised-nightly-dependency)
- [PoisonGPT: How we hid a lobotomized LLM on Hugging Face to spread fake news](https://blog.mithrilsecurity.io/poisongpt-how-we-hid-a-lobotomized-llm-on-hugging-face-to-spread-fake-news)
- [Army looking at the possibility of AI BOMs](https://defensescoop.com/2023/05/25/army-looking-at-the-possibility-of-ai-boms-bill-of-materials)
- [Failure Modes in Machine Learning](https://learn.microsoft.com/en-us/security/engineering/failure-modes-in-machine-learnin)
- [ML Supply Chain Compromise](https://atlas.mitre.org/techniques/AML.T0010)
- [Transferability in Machine Learning: from Phenomena to Black-Box Attacks using Adversarial Samples](https://arxiv.org/pdf/1605.07277.pdf)
- [BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain](https://arxiv.org/abs/1708.0673/)
- [VirusTotal Poisoning](https://atlas.mitre.org/studies/AML.CS0002)


#### Prevention Solution

- **Carefully vet data sources and suppliers, including T&Cs and their privacy policies, only using trusted suppliers**.
  - Ensure adequate and independently-audited security is in place and that model operator policies align with the data protection policies, i.e., the data is not used for training their models;
  - similarly, seek assurances and legal mitigations against using copyrighted material from model maintainers

- **Only use reputable plug-ins and ensure they have been tested for the application requirements**. LLM-Insecure Plugin Design provides information on the LLM-aspects of Insecure Plugin design you should test against to mitigate risks from using third-party plugins.

- **Understand and apply the mitigations found in the OWASP Top Ten's A06:2021 – Vulnerable and Outdated Components**.
  - This includes vulnerability scanning, management, and patching components.
  - For development environments with access to sensitive data, apply these controls in those environments, too。

- **Maintain an up-to-date inventory of components**
  - using a `Software Bill of Materials (SBOM)` to ensure you have an up-to-date, accurate, and signed inventory preventing tampering with deployed packages.
  - SBOMs can be used to detect and alert for new, zero-date vulnerabilities quickly
  - At the time of writing, SBOMs do not cover models, their artefacts, and datasets; If the LLM application uses its own model, you should use `MLOPs best practices` and platforms offering secure model repositories with data, model, and experiment tracking

- You should also **use model and code signing** when using external models and suppliers

- **Anomaly detection and adversarial robustness tests** on supplied models and data can help detect tampering and poisoning as discussed in Training Data Poisoning;
  - ideally, this should be part of MLOps pipelines; however, these are emerging techniques and may be easier implemented as part of red teaming exercises

- **Implement sufficient monitoring** to cover
  - component and environment vulnerabilities scanning,
  - use of unauthorised plugins,
  - out-of-date components, including the model and its artefacts

- **Implement a patching policy to mitigate vulnerable or outdated components**. Ensure that the application relies on a maintained version of APIs and the underlying model

- **Regularly review and audit** supplier Security and Access, ensuring no changes in their security posture or T&Cs.


---
