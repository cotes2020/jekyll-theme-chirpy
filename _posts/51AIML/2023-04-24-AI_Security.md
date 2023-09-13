---
title: AIML - Security
date: 2023-04-24 11:11:11 -0400
description:
categories: [51AIML]
# img: /assets/img/sample/rabbit.png
tags: [AIML]
---

# AIML - Security

- [AIML - Security](#aiml---security)
  - [Overall](#overall)
    - [OWAPS Top10 for LLM](#owaps-top10-for-llm)
    - [Hallucinations](#hallucinations)
      - [Hallucinations in Large Language Models](#hallucinations-in-large-language-models)
      - [Using Hallucinations](#using-hallucinations)
      - [Mitigating Hallucinations](#mitigating-hallucinations)

ref:
- [OWAPS Top10 for LLM v1](https://owasp.org/www-project-top-10-for-large-language-model-applications/assets/PDF/OWASP-Top-10-for-LLMs-2023-v1_0.pdf)


---

## Overall

- OWAPS Top10 for LLM

---

### OWAPS Top10 for LLM

link:

---

### Hallucinations

#### Hallucinations in Large Language Models

> Large Language Models (LLMs) are known to have `hallucinations`

hallucinations
- behavior in that the model speaks false knowledge as if it is accurate.

- when a model generates text, it can’t tell if the generation is accurate.
  - A large language model is a trained machine learning model that generates text based on the prompt you provided. The model’s training equipped it with some knowledge derived from the `training data` provided. It is difficult to tell what knowledge a model remembers or what it does not.

In the context of LLMs,
- “hallucination”: a phenomenon where the model generates text that is incorrect, nonsensical, or not real.
- Since LLMs are not databases or search engines, `they would not cite where their response is based on`.
- These models generate text as an extrapolation from the prompt you provided.
- The result of extrapolation is not necessarily supported by any training data, but is the most correlated from the prompt.

- For example
  - build a two-letter bigrams Markov model from some text: Extract a long piece of text, build a table of every pair of neighboring letters and tally the count.
  - “hallucinations in large language models” would produce “HA”, “AL”, “LL”, “LU”, etc. and there is one count of “LU” and two counts of “LA.”
  - when started with a prompt of “L”, you are twice as likely to produce “LA” than “LL” or “LS”.
  - with a prompt of “LA”, you have an equal probability of producing “AL”, “AT”, “AR”, or “AN”.
  - with a prompt of “LAT” and continue this process.
  - Eventually, this model invented a new word that didn’t exist.
  - This is a result of the statistical patterns. You may say your Markov model hallucinated a spelling.

- Hallucination in LLMs is not much more complex than this, even if the model is much more sophisticated. From a high level, hallucination is caused by limited contextual understanding since the model is obligated to transform the prompt and the training data into an abstraction, in which some information may be lost. Moreover, noise in the training data may also provide a skewed statistical pattern that leads the model to respond in a way you do not expect.


#### Using Hallucinations

- You may consider hallucinations a feature in large language models.

- You want to see the models hallucinate if you want them to be creative.
  - For example, if you ask ChatGPT or other Large Language Models to give you a plot of a fantasy story, you want it not to copy from any existing one but to generate a new character, scene, and storyline. This is possible only if the models are not looking up data that they were trained on.

- you want hallucinations when looking for diversity
  - for example, asking for ideas. It is like asking the models to brainstorm for you. You want to have derivations from the existing ideas that you may find in the training data, but not exactly the same. Hallucinations can help you explore different possibilities.

Many language models have a “temperature” parameter.
- control the temperature in ChatGPT using the API instead of the web interface.
- This is a parameter of randomness. The higher temperature can introduce more hallucinations.


#### Mitigating Hallucinations

- Language models are not search engines or databases.
- Hallucinations are unavoidable. What is annoying is that the `models generate text with mistakes that is hard to spot`.

- If the contaminated training data caused the hallucination, you can **clean up the data and retrain the model**.
  - However, most models are too large to train on your own devices. Even fine-tuning an existing model may be impossible on commodity hardware.

- The best mitigation may be **human intervention in the result**
  - asking the model to regenerate if it went gravely wrong.

- The other solution to avoid hallucinations is **controlled generation**.
  - It means providing enough details and constraints in the prompt to the model.
  - Hence the model has limited freedom to hallucinate.
  - The reason for prompt engineering is to specify the role and scenario to the model to guide the generation, so that it does not hallucinate unbounded.

.
