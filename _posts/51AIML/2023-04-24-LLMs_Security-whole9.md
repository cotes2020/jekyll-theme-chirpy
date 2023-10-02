---
title: AIML - 9
date: 2023-04-24 11:11:11 -0400
description:
categories: [51AIML]
# img: /assets/img/sample/rabbit.png
tags: [AIML]
---

# OWASP Top 10 for LLM
- [OWASP Top 10 for LLM](#owasp-top-10-for-llm)
    - [LLM09: Overreliance](#llm09-overreliance)
      - [Vulnerability Examples](#vulnerability-examples)
      - [Attack Scenario Example](#attack-scenario-example)
      - [Prevention Solution](#prevention-solution)


### LLM09: Overreliance

- Overreliance occurs `when systems or people depend on LLMs for decision-making or content generation without sufficient oversight`.

- LLMs can produce creative and informative content, LLMs can also

  - generate content that is factually incorrect, inappropriate or unsafe. This is referred to as `hallucination` or `confabulation` and can result in misinformation, miscommunication, legal issues, and reputational damage.

  - LLM-generated source code can introduce `unnoticed security vulnerabilities`. This poses a significant risk to the operational safety and security of applications.

- These risks show the importance of a rigorous review processes, with:
  - Oversight
  - Continuous validation mechanisms
  - Disclaimers on risk


**Misuse of Generated Content**

- LLMs learn from a massive amount of text data and generate responses or content based on that.

  - In the right hands, this can lead to innovative applications like drafting emails, writing code, creating articles, etc.

  - However, this very capability can be manipulated for harmful purposes, leading to misuse of the generated content.

    - `Sophisticated LLMs` can be used to `create realistic but false news articles, blog posts, or social media content`. This capability can be exploited to spread disinformation, manipulate public opinion, or conduct propaganda campaigns on a large scale.

    - LLMs can also be manipulated to `mimic a specific writing style or voice`. This can potentially be used for impersonation or identity theft, sending messages that seem like they are from a trusted person or entity, leading to scams or phishing attacks.

    - LLMs can `generate harmful, violent, or inappropriate content`. Even with content filtering mechanisms in place, there might be cases where harmful content slips through.


#### Vulnerability Examples

- LLM `provides inaccurate information as a response, causing misinformation`

- LLM `produces logically incoherent or nonsensical text that, while grammatically correct, doesn't make sense`

- LLM `melds information from varied sources, creating misleading content`

- LLM `suggests insecure or faulty code, leading to vulnerabilities when incorporated into a software system`

- `Failure of provider to appropriately communicate the inherent risks to end users`, leading to potential harmful consequences.


#### Attack Scenario Example

- A news organization `heavily uses an AI model to generate news articles`
  - A malicious actor exploits this over-reliance, feeding the AI misleading information, causing the spread of disinformation. The AI unintentionally plagiarizes content, leading to copyright issues and decreased trust in the organizationE

- A software development team `utilizes an AI system like Codex to expedite the coding process`
  - Over-reliance on the AI's suggestions introduces security vulnerabilities into the application due to insecure default settings or recommendations inconsistent with secure coding practices

- A software development firm uses an LLM to assist developers.
  - The LLM suggests a non-existent code library or package, and a developer, trusting the AI, unknowingly integrates a malicious package into the firm's software.
  - This highlights the importance of `cross-checking AI suggestions`, especially when involving third-party code or libraries.


Reference Links
- [Understanding LLM Hallucinations](https://towardsdatascience.com/llm-hallucinations-ec831dcd7780)
- [How Should Companies Communicate the Risks of Large Language Models to Users](https://techpolicy.press/how-should-companies-communicate-the-risks-of-large- language-models-to-users)
- [A news site used AI to write articles. It was a journalistic disaster](https://www.washingtonpost.com/media/2023/01/17/cnet-ai-articles-journalism-corrections)
- [AI Hallucinations: Package Risk](https://vulcan.io/blog/ai-hallucinations-package-risk)
- [How to Reduce the Hallucinations from Large Language Models](https://thenewstack.io/how-to-reduce-the-hallucinations-from-large-language-models)
- [Practical Steps to Reduce Hallucination](https://newsletter.victordibia.com/p/practical-steps-to-reduce-hallucination)


#### Prevention Solution

- Addressing misuse of generated content necessitates comprehensive strategies:

- **Robust Content Filters**: Developing and implementing robust content filtering mechanisms is crucial.
  - These filters can help detect and prevent the generation of harmful or inappropriate content.
  - This could involve techniques such as `Reinforcement Learning from Human Feedback (RLHF)` where the model is trained to avoid certain types of outputs.
  - **Build APIs and user interfaces that encourage responsible and safe use of LLMs**, such as `content filters, user warnings about potential inaccuracies, and clear labeling of AI-generated content`.

- **Regularly monitor and review the LLM outputs**.
  - Use `self-consistency or voting techniques` to filter out inconsistent text.
  - Comparing multiple model responses for a single prompt can better judge the quality and consistency of outputx

- **Adversarial Testing**:
  - Regular adversarial testing can help identify potential misuse scenarios and help in developing effective countermeasures.

- **Collaboration with Policymakers**: Collaborating with regulators and policymakers to establish guidelines and laws can deter misuse and ensure proper repercussions.

- **Cross-check the LLM output with trusted external sources**.
  - This additional layer of validation can help ensure the information provided by the model is accurate and reliablex

- **Enhance the model with fine-tuning or embeddings to improve output quality**.
  - Generic pre-trained models are more likely to produce inaccurate information compared to tuned models in a particular domain.
  - Techniques such as `prompt engineering, parameter efficient tuning (PET), full model tuning, and chain of thought prompting` can be employed for this purpose.

- Implement **automatic validation mechanisms that can cross-verify the generated output against known facts or data**.
  - This can provide an additional layer of security and mitigate the risks associated with hallucinationsE

- **Break down complex tasks into manageable subtasks and assign them to different agents**.
  - This not only helps in managing complexity, but it also reduces the chances of hallucinations as each agent can be held accountable for a smaller task.

- **Communicate the risks and limitations associated with using LLMs**.
  - This includes potential for information inaccuracies, and other risks. Effective risk communication can prepare users for potential issues and help them make informed decisions.

- **User Verification and Rate Limiting**:
  - To prevent mass generation of misleading information, platforms could use stricter user verification methods and impose rate limits on content generation.

- **Awareness and Education**:
  - Informing users about the potential misuse of LLM-generated content can encourage responsible use and enable them to identify and report instances of misuse.

- When using LLMs in development environments, **establish secure coding practices and guidelines** to prevent the integration of possible vulnerabilities.
