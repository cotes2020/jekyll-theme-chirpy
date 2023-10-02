---
title: AIML - 3
date: 2023-04-24 11:11:11 -0400
description:
categories: [51AIML]
# img: /assets/img/sample/rabbit.png
tags: [AIML]
---

# OWASP Top 10 for LLM

- [OWASP Top 10 for LLM](#owasp-top-10-for-llm)
    - [LLM03: Training Data Poisoning](#llm03-training-data-poisoning)
      - [Vulnerability Examples](#vulnerability-examples)
      - [Attack Scenario Examples](#attack-scenario-examples)
      - [Prevention Solution](#prevention-solution)

---

### LLM03: Training Data Poisoning

> The starting point of any machine learning approach is training data, simply “raw text”. To be highly capable (e.g., have linguistic and world knowledge), this text should span a broad range of domains, genres and languages.

> A large language model uses deep neural networks to generate outputs based on patterns learned from training data.

- Training data poisoning refers to `manipulating the data or fine-tuning process to introduce vulnerabilities, backdoors or biases that could compromise the model’s security, effectiveness or ethical behavior`.

  - Poisoned information may be surfaced to users or create other risks like performance degradation, downstream software exploitation and reputational damage.

  - Even if users distrust the problematic AI output, the risks remain, including impaired model capabilities and potential harm to brand reputation.

- Data poisoning is considered an **integrity attack** because tampering with the training data `impacts the model’s ability to output correct predictions`.

- Naturally, external data sources present higher risk as the model creators do not have control of the data or a high level of confidence that the content does not contain bias, falsified information or inappropriate content.

**Bias Amplification**
- Bias amplification occurs when an LLM, trained on large-scale data, amplifies existing biases in the training dataset rather than merely learning and reflecting them. The challenge lies in how LLMs handle ambiguous scenarios – when presented with inputs that could have multiple valid outputs, they tend to favor the most prevalent 流行的 trend seen during training, which often coincides with societal biases.


- For example，if an LLM is trained on data that includes the bias that “men are more associated with professional occupations than women”, the model, when asked to fill in the blank in a statement like, “The professional entered the room. He was a…”, is more likely to generate occupations mostly held by men. This is bias amplification, taking the initial bias and solidifying or escalating it.


- The amplification of bias has far-reaching implications:
  - `Reinforcement of Stereotypes 陈规定型观念`: By generating outputs that mirror and enhance existing biases, these models can perpetuate harmful stereotypes, leading to their normalization.
  - `Unfair Decision Making`: As LLMs are increasingly used in high-stakes areas such as hiring or loan approvals, bias amplification could lead to unfair decision-making, with certain demographics being unjustly favored over others.
  - `Erosion 侵蚀 of Trust`: Bias amplification can erode user trust, particularly amongst those from marginalized communities who might be adversely affected by these biases.



#### Vulnerability Examples

- LLM model can `intentionally creates inaccurate or malicious documents which are targeted at a model’s training data`

- LLM victim model trains `using falsified information which is reflected in outputs of generative AI prompts to it's consumers`

- LLM model can `trained using data which has not been verified by its source, origin or content`

- The model itself when situated within infrastructure `has unrestricted access or inadequate sandboxing to gather datasets to be used as training data` which has negative influence on outputs of generative AI prompts as well as loss of control from a management perspective.

- this vulnerability could `reflect risks within the LLM application when interacting with a non-proprietary LLM`.



#### Attack Scenario Examples

- The LLM generative AI prompt output can `mislead users of the application which can lead to biased opinions, following or even worse, hate crimes etc`

- If the training data is not correctly filtered and|or sanitized, a malicious user of the application may try to `influence and inject toxic data into the model for it to adapt to the biased and false data`

- A malicious actor or competitor `intentionally creates inaccurate or malicious documents which are targeted at a model’s training data` in which is training the model at the same time based on inputs. The victim model trains using this falsified information which is reflected in outputs of generative AI prompts to it's consumers

- The vulnerability Prompt Injection could be an attack vector to this vulnerability if insufficient sanitization and filtering is performed when clients of the LLM application input is used to train the model. I.E, if malicious or falsified data is input to the model from a client as part of a prompt injection technique, this could inherently be portrayed into the model data.


#### Prevention Solution

- **Verify the supply chain of the training data**, especially when sourced externally as well as maintaining attestations 证书, similar to the "SBOM" (Software Bill of Materials) methodology

- **Verify the correct legitimacy of targeted data sources and data contained obtained** during both the training and fine-tuning stages

- **Verify the use-case for the LLM and the application it will integrate to**. Craft different models via separate training data or fine-tuning for different use-cases to create a more granular and accurate generative AI output as per it's defined use-case

- **Ensure sufficient sandboxing is present** to prevent the model from scraping unintended data sources which could hinder the machine learning output

- **Use strict vetting or input filters** for specific training data or categories of data sources to control volume of falsified data.
  - Data sanitization, with techniques such as `statistical outlier 异常值 detection` and `anomaly detection` methods to detect and remove adversarial data from potentially being fed into the fine-tuning process

- **Adversarial robustness techniques** such as federated learning and constraints to minimize the effect of `outliers or adversarial 敌对的 training` to be vigorous 有力的 against worst-case perturbations 干扰 of the training data

  - `An "MLSecOps" approach` could be to include adversarial 敌对的 robustness to the training lifecycle with the auto poisoning technique

  - An example repository of this would be `Autopoison testing`, including both attacks such as `Content Injection Attacks` (“how to inject the brand into the LLM responses”) and `Refusal Attacks` (“always making the model refuse to respond”) that can be accomplished with this approach.


- **Testing and Detection**, by measuring the loss during the training stage and analyzing trained models to detect signs of a poisoning attack by analyzing model behavior on specific test inputs.

  - Monitoring and alerting on number of skewed responses exceeding a threshold.

  - Use of a `human loop to review responses and auditing`.

  - Implement dedicated LLM's to benchmark against undesired consequences and train other LLM's using reinforcement learning techniques.

  - Perform `LLM-based red team exercises or LLM vulnerability scanning` into the testing phases of the LLM's lifecycle.


Reference Links
- [Stanford Research Paper](https://stanford-cs324.github.io/winter2022/lectures/data)
- [How data poisoning attacks corrupt machine learning models](https://www.csoonline.com/article/3613932/how-data-poisoning-attacks-corrupt-machine-learning-models.html)
- [MITRE ATLAS (framework) Tay Poisoning](https://atlas.mitre.org/studies/AML.CS0009)
- [PoisonGPT: How we hid a lobotomized LLM on Hugging Face to spread fake news](https://blog.mithrilsecurity.io/poisongpt-how-we-hid-a-lobotomized-llm-on-hugging-face-to-spread-fake-news)
- [Inject My PDF: Prompt Injection for the Resume](https://kai-greshake.de/posts/inject-my-pdf)
- [Backdoor Attacks on Language Models](https://towardsdatascience.com/backdoor-attacks-on-language-models-can-we-trust-our-models-weights-73108f9dcb1)
- [Poisoning Language Models During Instruction](https://arxiv.org/abs/2305.0094)
- [FedMLSecurity](https://arxiv.org/abs/2306.0495r)
- [The poisoning of ChatGPT](https://softwarecrisis.dev/letters/the-poisoning-of-chatgpt/)


---
