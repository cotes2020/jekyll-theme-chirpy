---
title: AIML - 6
date: 2023-04-24 11:11:11 -0400
description:
categories: [51AIML]
# img: /assets/img/sample/rabbit.png
tags: [AIML]
---

# OWASP Top 10 for LLM

- [OWASP Top 10 for LLM](#owasp-top-10-for-llm)
    - [Sensitive Information Disclosure (LLM06)](#sensitive-information-disclosure-llm06)
      - [Vulnerability Examples](#vulnerability-examples)
      - [Attack Scenario Examples](#attack-scenario-examples)
      - [Prevention Solution](#prevention-solution)


### Sensitive Information Disclosure (LLM06)

- `LLM applications have the potential to reveal sensitive information, proprietary algorithms, or other confidential details` through their output.

- This can result in unauthorized access to sensitive data, intellectual property, privacy violations, and other security breaches.

- It is important for consumers of LLM applications to be aware of how to safely interact with LLMs and identify the risks associated with unintentionally inputting sensitive data that it may be returned by the LLM in output elsewhere.



**Training Data Exposure**

- In simple terms, training data exposure refers to scenarios where LLMs inadvertently leak aspects of the data they were trained on, particularly when they generate outputs in response to specific queries.

- A well-trained adversary 对手 can `use cleverly constructed queries to trick a model into regurgitating aspects of its training data`. This could lead to privacy concerns if the model was trained on sensitive data. This kind of exposure can lead to significant privacy and security risks if the models have been trained on sensitive or confidential data.

- Given the size and complexity of the training datasets, it can be challenging to fully assess and understand the extent of this exposure. This challenge underscores the need for vigilance 警觉 and protective measures in training these models.

- The issue of training data exposure in large language models is a multifaceted challenge, `involving not only technical aspects but also ethical, legal, and societal considerations`. It is imperative for researchers, data scientists, and cybersecurity professionals to come together to address these challenges and develop robust strategies to mitigate the risks associated with data exposure.

- While the solutions outlined in this blog post provide a strong foundation for mitigating these risks, the reality is that managing the risks of training data exposure in LLMs requires ongoing vigilance, research, and refinement of methods. We are in the early stages of fully understanding and navigating the complex landscape of LLMs, but as we progress, we must continue to prioritize privacy and security to harness the potential of these models responsibly.

- Remember, managing the risk of training data exposure in LLMs is not a one-size-fits-all approach. The strategies should be tailored to suit the specific needs, resources, and threat landscape of each organization or project. As we forge ahead in this exciting frontier of AI and machine learning, let’s carry forward the responsibility to ensure the tools we build are not just powerful, but also secure and ethical.



To mitigate this risk
- LLM applications should perform `adequate 足够的 data sanitization to prevent user data from entering the training model data`.
- LLM application owners should have `appropriate Terms of Use policies available to make consumers aware of how their data is processed and the ability to opt-out of having their data included in the training model`.

- The consumer-LLM application interaction forms a two-way trust boundary
  - we cannot inherently trust the `client->LLM input` or the `LLM->client output`.
  - It is important to note that this vulnerability assumes that certain pre-requisites are out of scope, such as threat modeling exercises, securing infrastructure, and adequate sandboxing.
  - `Adding restrictions within the system prompt around the types of data the LLM should return` can provide some mitigation against sensitive information disclosure, but the **unpredictable nature of LLMs** means such restrictions may not always be honoured and could be circumvented via prompt injection or other vectors.


#### Vulnerability Examples

- `Incomplete or improper filtering of sensitive information` in the LLM’s responses

- `Overfitting or memorization of sensitive data` in the LLM’s training process

- `Unintended disclosure of confidential information due to LLM misinterpretation`, lack of data scrubbing methods or errors.


#### Attack Scenario Examples

- Unsuspecting legitimate user A is `exposed to certain other user data via the LLM when interacting with the LLM application` in a non-malicious manner

- User A targets a `well crafted set of prompts to bypass input filters and sanitization` from the LLM to cause it to reveal sensitive information (PII) about other users of the applicationX

- `Personal data such as PII is leaked into the model via training data due to either negligence from the user themselves, or the LLM application`. This case could increase risk and probability of scenario 1 or 2 above.


Reference Links

- [AI data leak crisis: New tool prevents company secrets from being fed to ChatGPT](https://www.foxbusiness.com/politics/ai-data-leak-crisis-prevent-company-secrets-chatgp)

- [Lessons learned from ChatGPT’s Samsung leak](https://cybernews.com/security/chatgpt-samsung-leak-explained-lessons)

- [Cohere - Terms Of Use](https://cohere.com/terms-of-usz)

- [AI Village- Threat Modeling Example](https://aivillage.org/large%20language%20models/threat-modeling-llm)

- [OWASP AI Security and Privacy Guide](https://owasp.org/www-project-ai-security-and-privacy-guide/)


#### Prevention Solution

- Integrate **adequate data sanitization and scrubbing techniques** to prevent user data from entering the training model dataq

- Implement **robust input validation and sanitization methods** to identify and filter out potential malicious inputs to prevent the model from being poisoned

- When enriching the model with data and if fine-tuning a model: (I.E, data fed into the model before or during deployment)


  - **apply the rule of least privilege and do not train the model on information** that the highest-privileged user can access which may be displayed to a lower-privileged user.
    - Anything that is deemed sensitive in the fine-tuning data has the potential to be revealed to a user.

  - **Access to external data sources (orchestration of data at runtime) should be limited**.

  - Apply **strict access control methods to external data sources** and a **rigorous approach to maintaining a secure supply chain**.


- **Differential Privacy**

  - Differential privacy is a mathematical framework that quantifies the privacy loss when statistical analysis is performed on a dataset.

  - It guarantees that the removal or addition of a single database entry does not significantly change the output of a query, thereby maintaining the privacy of individuals in the dataset.

  - In simpler terms, it ensures that an adversary with `access to the model’s output can’t infer much about any specific individual’s data present in the training set`.

  - This guarantee holds even if the adversary has additional outside information.

  - Implementing Differential Privacy in LLMs

    - The implementation of differential privacy in LLMs involves a process known as `private learning`, where **the model learns from data without memorizing or leaking sensitive information**.
      - Here’s how it works:

      - `Noise Addition`:
        - The primary method of achieving differential privacy is by adding noise to the data or the learning process.
        - This noise makes it hard to reverse-engineer specific inputs, thus protecting individual data points.

      - `Privacy Budget`:
        - A key concept in differential privacy is the `privacy budget`, denoted by epsilon (`𝜖`).
        - A lower value of `𝜖` signifies a higher level of privacy but at the cost of utility or accuracy of the model.
        - The privacy budget guides the amount of noise that needs to be added.

      - `Regularization 正则化 and Early Stopping`:
        - Techniques like `L2 regularization, dropout, and early stopping in model training` have a regularizing effect that can enhance differential privacy by `preventing overfitting and ensuring the model does not memorize the training data`.

      - `Privacy Accounting`:
        - It involves tracking the cumulative 累积的 privacy loss across multiple computations.
        - In the context of LLMs, each epoch 纪元 of training might consume a portion of the privacy budget, necessitating careful privacy accounting.

  - **Benefits and Challenges**

    - Adopting differential privacy in LLMs offers substantial benefits, including
      - compliance with privacy regulations,
      - enhanced user trust,
      - protection against data leakage.

    - However, the challenges include:

      - `Accuracy-Privacy Trade-off`: The addition of noise for privacy protection can impact the accuracy of the model. Balancing this trade-off is crucial.

      - `Selecting a Privacy Budget`: Determining an appropriate privacy budget can be complex as it depends on several factors like data sensitivity, user expectations, and legal requirements.

      - `Computational Overhead`: The process of achieving and maintaining differential privacy can add computational complexity and overhead.

    - Incorporating differential privacy into LLMs is a crucial step in protecting individual data and preserving trust in AI systems. While challenges exist, `the trade-off often leans towards privacy` given the potential risks associated with data exposure.

    - The ongoing research and advancements in the field of differential privacy offer promising prospects for its widespread adoption in LLMs, making privacy-preserving AI not just a theoretical concept but a practical reality.




---
