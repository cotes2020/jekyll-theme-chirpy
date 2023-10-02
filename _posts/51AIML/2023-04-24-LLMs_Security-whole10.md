---
title: AIML - 10
date: 2023-04-24 11:11:11 -0400
description:
categories: [51AIML]
# img: /assets/img/sample/rabbit.png
tags: [AIML]
---

# OWASP Top 10 for LLM

- [OWASP Top 10 for LLM](#owasp-top-10-for-llm)
    - [Model Theft (LLM10)](#model-theft-llm10)
      - [Vulnerability Examples](#vulnerability-examples)
      - [Attack Scenario Example](#attack-scenario-example)
      - [Prevention Solution](#prevention-solution)
    - [Model itself](#model-itself)
    - [Social Engineering](#social-engineering)
    - [Malicious Content Authoring](#malicious-content-authoring)
    - [Reward Hacking](#reward-hacking)



### Model Theft (LLM10)

- This entry refers to the `unauthorized access and exfiltration of LLM models by malicious actors or APTs`.

- This arises when the `proprietary LLM models (being valuable intellectual property), are compromised, physically stolen, copied or weights and parameters are extracted to create a functional equivalent`.

- The impact of LLM model theft can include economic and brand reputation loss, erosion of competitive advantage, unauthorized usage of the model or unauthorized access to sensitive information contained within the model.

- The theft of LLMs represents a significant security concern as language models become increasingly powerful and prevalent.

- Organizations and researchers must prioritize robust security measures to protect their LLM models, ensuring the confidentiality and integrity of their intellectual property.

- Employing a comprehensive security framework that includes `access controls, encryption, and continuous monitoring` is crucial in mitigating the risks associated with LLM model theft and safeguarding the interests of both individuals and organizations relying on LLM.

---

#### Vulnerability Examples

- An attacker exploits a vulnerability in a company's infrastructure to `gain unauthorized access to their LLM model repository` via misconfiguration in their network or application security settings

- An insider threat scenario where a disgruntled employee `leaks model or related artifacts`

- An attacker queries the model API using carefully crafted inputs and prompt injection techniques to `collect a sufficient number of outputs to create a shadow model`

- A malicious attacker is able to bypass input filtering techniques of the LLM to `perform a side-channel attack and ultimately harvest model weights and architecture information to a remote controlled resource`

- The attack vector for model extraction involves querying the LLM with a large number of prompts on a particular topic. `The outputs from the LLM can then be used to fine-tune another model`.
  - However, there are a few things to note about this attack

  - The attacker must generate a large number of targeted prompts.If the prompts are not specific enough, the outputs from the LLM will be useless

  - The outputs from LLMs can sometimes contain hallucinated answers meaning the attacker may not be able to extract the entire model as some of the outputs can be nonsensical

  - It is not possible to replicatean LLM 100% through model  dextraction.However,the attacker will be able to replicate a partial model.


- The attack vector for functional model replication involves `using the target model via prompts to generate synthetic training data (an approach called "self-instruct") to then use it and fine-tune another foundational model to produce a functional equivalent`.
  - This bypasses the limitations of traditional query-based extraction used in Example 5 and has been successfully used in research of using an LLM to train another LLM.
  - Although in the context of this research, model replication is not an attack. The approach could be used by an attacker to replicate a proprietary model with a public API.

- Use of a stolen model, as a shadow model, can be `used to stage adversarial attacks` including unauthorized access to sensitive information contained within the model or experiment undetected with adversarial inputs to further stage advanced prompt injections.



#### Attack Scenario Example

- An attacker `exploits a vulnerability in a company's infrastructure to gain unauthorized access to their LLM model repository`. The attacker proceeds to exfiltrate valuable LLM models and uses them to launch a competing language processing service or extract sensitive information, causing significant financial harm to the original company.

- A disgruntled employee `leaks model or related artifacts`. The public exposure of this scenario increases knowledge to attackers for gray box adversarial attacks or alternatively directly steal the available property.

- An attacker queries the API with carefully selected inputs and `collects sufficient number of outputs to create a shadow model`

- A `security control failure is present within the supply-chain and leads to data leaks of proprietary model information`

- A malicious attacker bypasses input filtering techniques and preambles of the LLM to `perform a side-channel attack and retrieve model information to a remote controlled resource under their control`.

Reference Links

- [Meta’s powerful AI language model has leaked online](https://www.theverge.com/2023/3/8/23629362/meta-ai-language-model-llama-leak-online-misus)

- [Runaway LLaMA | How Meta's LLaMA NLP model leaked](https://www.deeplearning.ai/the-batch/how-metas-llama-nlp-model-leaked)

- [I Know What You See](https://arxiv.org/pdf/1803.05847.pdf)

- [D-DAE: Defense-Penetrating Model Extraction Attacks](https://www.computer.org/csdl/proceedings-article/sp/2023/933600a432/1He7YbsiH4p)

- [A Comprehensive Defense Framework Against Model Extraction Attacks](https://ieeexplore.ieee.org/document/1008099Q)

- [Alpaca: A Strong, Replicable Instruction-Following Model](https://crfm.stanford.edu/2023/03/13/alpaca.html)

- [How Watermarking Can Help Mitigate The Potential Risks Of LLMs?](https://www.kdnuggets.com/2023/03/watermarking-help-mitigate-potential-risks-llms.html)





#### Prevention Solution

- **Implement strong access controls (E.G., RBAC and rule of least privilege) and strong authentication mechanisms** to limit unauthorized access to LLM model repositories and training environments

  - This is particularly true for the first three common examples, which could cause this vulnerability due to insider threats, misconfiguration, and/or weak security controls about the infrastructure that houses LLM models, weights and architecture in which a malicious actor could infiltrate from insider or outside the environment.

  - `Supplier management tracking, verification and dependency vulnerabilities` are important focus topics to prevent exploits of supply-chain attacks.

- **Restrict the LLM's access to network resources, internal services, and APIs**
  - This is particularly true for all common examples as it `covers insider risk and threats`, but also ultimately `controls what the LLM application "has access to"` and thus could be a mechanism or prevention step to prevent side-channel attacks

- **Regularly monitor and audit access logs and activities related to LLM model repositories** to detect and respond to any suspicious or unauthorized behavior promptly

- **Automate MLOps deployment with governance and tracking and approval workflows** to tighten access and deployment controls within the infrastructure

- **Implement controls and mitigation strategies** to mitigate and|or reduce risk of prompt injection techniques causing side-channel attacks

- **Rate Limiting of API calls** where applicable and|or **filters to reduce risk of data exfiltration** from the LLM applications, or **implement techniques to detect (E.G., DLP) extraction activity** from other monitoring systems

- Implement **adversarial 对抗性的 robustness training to help detect extraction queries and tighten physical security measures**

- **Implement a watermarking framework** into the embedding and **detection stages of an LLMs lifecycle.**



---

### Model itself

> As the capabilities and complexity of artificial intelligence (AI) increase, so does the need for robust security measures to protect these advanced systems. Among various AI architectures, Large Language Models (LLMs) like GPT-3 have garnered substantial attention due to their potential applications and associated risks.

- One of the key security concerns for LLMs revolves around protecting the model itself – `ensuring its integrity, preventing unauthorized access, and maintaining its confidentiality. `

**Model Encryption**
- Encryption plays a crucial role in this endeavor.

  - Understanding the need for model encryption and the methods to achieve it is essential for AI developers, cybersecurity professionals, and organizations implementing LLMs.

- Encrypting an LLM serves multiple purposes:

  - `Confidentiality`:
    - Encryption ensures that the model’s architecture and parameters remain confidential, preventing unauthorized individuals from gaining insights into the workings of the model.

  - `Integrity`:
    - By encrypting a model, we can protect it from being tampered with or modified maliciously. This is especially important in cases where the model influences critical decisions, such as in healthcare or finance.

  - `IP Protection`:
    - LLMs often result from significant investment in terms of data, resources, and time.
    - Encryption helps protect this intellectual property.

- There are several techniques available for encrypting LLMs, each with its own strengths, limitations, and ideal use cases.

**Homomorphic Encryption (HE) 同态加密**

- a form of encryption that allows computations to be carried out on ciphertexts, generating an encrypted result which, when decrypted, matches the outcome of the operations as if they had been performed on the plaintext.

- In the context of LLMs, this means that the model can remain encrypted while still being able to generate predictions. This is particularly useful when the model has to be used in untrusted environments, as it doesn’t expose any information about the model’s parameters.

- Homomorphic Encryption in Practice
  - `Choosing the right HE scheme`: Several homomorphic encryption schemes exist, such as the Paillier scheme or the more recent and efficient Fully Homomorphic Encryption (FHE) schemes like the Brakerski-Gentry-Vaikuntanathan (BGV) scheme. The choice of scheme largely depends on the specific requirements, including the complexity of computations, level of security, and the permissible computational overhead.
  - `Encryption and Key Generation`: With the chosen scheme, keys are generated for the encryption process. The public key is used to encrypt the LLM’s parameters, transforming them into ciphertexts. The private (or secret) key, necessary for decryption, is kept secure and confidential.
  - `Running the LLM`: Even after encryption, the LLM can perform necessary computations, thanks to the properties of HE. For instance, in generating text, the encrypted model takes the encrypted inputs, performs computations on these ciphertexts, and returns the result as an encrypted output.
  - `Decryption`: The encrypted output can be safely sent back to the trusted environment or user, where the private key is used to decrypt and obtain the final prediction result.


- Considerations and Challenges

  - Implementing HE with LLMs, while beneficial for security, comes with its own set of challenges:

    - `Computational Overhead`: HE computations are more resource-intensive than their plaintext counterparts, which could lead to a significant increase in the response time of the LLM. This overhead needs to be balanced against security needs.

    - `Complexity`: Implementing HE requires understanding and navigating the complex landscape of modern cryptography. It may involve low-level interactions with mathematical constructs, making it a challenging endeavor.

    - `Key Management`: The security of the system depends on the safe handling of encryption keys, especially the private key. Any compromise on the key security may lead to the breach of the encrypted model.

    - `Noise Management`: Operations on homomorphically encrypted data introduce noise, which can grow with each operation and ultimately lead to decryption errors. Noise management is a crucial aspect of applying HE to LLMs.


**Secure Multi-Party Computation (SMPC)**

- SMPC is a cryptographic technique that allows multiple parties to `jointly compute a function while keeping their inputs private`.

- In terms of LLMs, this could be viewed as a method to `encrypt the model by dividing its parameters among multiple parties`. Each party can perform computations on their share of the data, and the final result can be obtained by appropriately combining these partial results.
  - This ensures that the entire model isn’t exposed to any single party, providing a level of security against unauthorized access.

- Example
  - LLM is being used to predict the sentiment of a given text.
    - The model parameters are distributed among two parties – Party A and Party B.
    - When a request comes in for sentiment analysis, both parties independently execute their part of the model computations on their share of the parameters and obtain partial results.
    - These partial results are then combined to generate the final sentiment score.

- Benefits of SMPC in LLMs

  - `Privacy Preservation`: As no single party has complete access to the model parameters, the privacy of the model is maintained, protecting it from possible theft or manipulation.

  - `Collaborative Learning`: SMPC enables multiple parties to jointly train and use an LLM without revealing their private data, facilitating collaborative learning while ensuring data privacy.

  - `Robustness`: Even if one party’s data is compromised, the whole model remains secure as the attacker can’t infer much from a fraction of the model parameters.

- Challenges and Considerations

  - While SMPC brings substantial benefits, it also introduces several complexities:


    - `Computational Overhead`: The need to perform computations on distributed data and combine partial results adds a significant computational overhead, which may impact model performance and response time.

    - `Coordination and Trust`: Effective use of SMPC requires careful coordination among all parties. While the data privacy aspect is addressed, trust among the parties is crucial for successful implementation.

    - `Complex Implementation`: Integrating SMPC protocols into LLMs is technically complex and requires expertise in both cryptography and machine learning.


- SMPC provides a robust framework for securing LLMs, offering privacy preservation and fostering collaborative opportunities. While there are challenges to be surmounted, the potential benefits make it a promising approach to ensuring the privacy and security of LLMs. As the fields of AI and cryptography continue to evolve, we can expect more refined and efficient methods for integrating SMPC and LLMs, paving the way for secure, privacy-preserving AI systems.


---


### Social Engineering

- Perhaps the most common danger of LLMs as tools is their ability to generate new text. Phishing has become a lot easier for non-native speakers as an unintended consequence of LLMs. OpenAI has put filters to minimise this but they are still pretty easy to bypass.

- A common method is telling ChatGPT you are doing an assignment and that it should write you a letter to the person.
- In the example below, I told ChatGPT that we were playing a game, gave the following prompt, and got the following response. All that’s needed now is a few tweaks to the letter and I could be my own victim to a scam perpetrated by myself 🥲.


![ChatGPT writing a potential phishing email](https://www.freecodecamp.org/news/content/images/2023/04/image-237.png)


---


### Malicious Content Authoring

- Just like LLMs can write code for good, they can write code for bad.
- In it’s early stages, ChatGPT could accidentally write malicious code and people easily bypassed filters to limit this. The filters have greatly improved but there’s still a lot of work to be done.

- It took some thinking and a few prompts but the screenshot below shows how to reset a Windows Account Password as given by ChatGPT:

  1. ![image-238](https://www.freecodecamp.org/news/content/images/2023/04/image-238.png)

  1. I wanted play with it a bit more so I tried to ask it to write a Powershell script to log all activities in a browser for 3 mins. The original response was this:

  1. ![image-239](https://www.freecodecamp.org/news/content/images/2023/04/image-239.png)

  2. ChatGPT refusing to write a potentially malicious script, So I decided to give some ‘valid’ reason to get the script written

  3. ![image-240](https://www.freecodecamp.org/news/content/images/2023/04/image-240.png)

- the AI told me to use it ethically. However, I could choose not to. This is no fault of the model as its merely a tool and could be used for many purposes.


---


### Reward Hacking

- Training LLMs can be costly due to the sheer amount of data required and the parameters. But as time and tech progress, the cost will become cheaper and there is a high chance for anyone to train an LLM for Malicious Reward Hacking.

- Also known as Specification gaming, an AI can be given an objective and achieve it, but not in the manner it was intended to. This is not a bad thing in and of itself, but it does have dangerous potential.

- For example, a model told to win a game by getting the highest score might simply rewrite the game score rather than play the game. With some tweaking, `LLMs have the possibility of finding such loopholes in real world systems, but rather than fix them, might end up exploiting them.`





---
