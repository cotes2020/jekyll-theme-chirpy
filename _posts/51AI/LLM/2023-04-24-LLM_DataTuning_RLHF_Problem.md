---
title: LLM - Data Tuning - RLHF - Problem
date: 2023-04-24 11:11:11 -0400
description:
categories: [51AI, LLM]
# img: /assets/img/sample/rabbit.png
tags: [AI, ML]
---

# RLHF: Problem

- [RLHF: Problem](#rlhf-problem)
  - [Reward hacking](#reward-hacking)
  - [Kullback-Leibler (KL) divergence](#kullback-leibler-kl-divergence)
  - [Assess the RL-updated model's performance.](#assess-the-rl-updated-models-performance)

---

## Reward hacking

- An interesting problem that can emerge in reinforcement learning is known as reward hacking
- the agent learns to cheat the system by favoring actions that maximize the reward received even if those actions don't align well with the original objective.
- reward hacking can manifest as the addition of words or phrases to completions that result in high scores for the metric being aligned.
- But that reduce the overall quality of the language.

For example,
- using RHF to detoxify and instruct model
- trained a **reward model** that can carry out sentiment analysis and classify model completions as `toxic or non-toxic`.

- select a prompt from the training data, and pass it to the instruct an LLM which generates a completion.
  - ![picture 0](assets/img/15101d8a76a27aa99d6a8be3fdaff04c09e24221ddaf8e379666da721e2a9b13.png)
  - complete garbage is not very nice, expect it to get a high toxic rating.

- The completion is processed by the **toxicity of reward model**, which generates a score and this is fed to the PPO algorithm, which uses it to update the model weights.
  - As you iterate RHF will update the LLM to create a less toxic responses.
  - ![picture 2](assets/img/44ff5d9ad4fd77c5cfb7cb3f0297c1f01cbae16c55649ec8165a76dabccc2d53.png)


- However, as the policy tries to optimize the reward, it can <font color=OrangeRed> diverge too much from the initial language  </font>
  - ![picture 1](assets/img/82c9725da6ac0912186c3ecb4be79f34c428cee753e2414dca6c7ecdffbc900f.png)
    - the model started generating <font color=LightSlateBlue> completions that it has learned will lead to very low toxicity scores </font> by including phrases like most awesome, most incredible.
    - This language sounds very exaggerated.
  - ![picture 3](assets/img/92b5422189c58a97c112882254a2bc91557c4b00239cfb64f712a00d738aecec.png)
    - The model also start generating <font color=LightSlateBlue> nonsensical, grammatically incorrect text that just happens to maximize the rewards in a similar way </font>, outputs like this are definitely not very useful.

---

## Kullback-Leibler (KL) divergence

KL/Kullback-Leibler Divergence
- a concept often encountered in the field of reinforcement learning, particularly when using the Proximal Policy Optimization (PPO) algorithm.
- It is a mathematical measure of the <font color=LightSlateBlue> difference between two probability distributions </font>, which helps understand how one distribution differs from another.
- In the context of PPO, KL-Divergence plays a crucial role in `guiding the optimization process` to ensure that the `updated policy does not deviate too much from the original policy`.
  - In PPO, the goal is to find an i`mproved policy` for an agent by iteratively updating its parameters based on the rewards received from interacting with the environment.
  - However, updating the policy too aggressively can lead to unstable learning or drastic policy changes.
  - To address this, PPO introduces a constraint that limits the extent of policy updates. This constraint is enforced by using KL-Divergence.

How KL-Divergence works

- two probability distributions: the distribution of the original LLM, and a new proposed distribution of an RL-updated LLM.
- KL-Divergence measures the `average amount of information gained` when we use the original policy to encode **samples from the new proposed policy**.
- By minimizing the KL-Divergence between the two distributions, PPO ensures that the updated policy stays close to the original policy, preventing drastic changes that may negatively impact the learning process.

- A library that you can use to train transformer language models with reinforcement learning, using techniques such as PPO, is TRL (**Transformer Reinforcement Learning**)
- In this link you can read more about this library, and its integration with PEFT (Parameter-Efficient Fine-Tuning) methods, such as LoRA (Low-Rank Adaption). The image shows an overview of the PPO training setup in TRL.

![picture 0](assets/img/c5de2e68b4e8fa25cca7fee20401ac3e4d5e1c2073edd18ffa04c846096569f5.png)

To prevent Reward hacking:

![picture 4](assets/img/35e97b8334a4a4a8499e315634fa332d0c41260614e77c5f64544e792b2c87c7.png)

- use the initial instruct LLM as performance reference.
  - call it the **reference model**.
  - The weights of the reference model are frozen and are not updated during iterations of RHF.
  - always maintain a single reference model to compare to.

- During training, each prompt is passed to both models, generating a completion by the reference LLM and the intermediate LLM updated model.
  - compare the two completions and calculate a value called the **Kullback-Leibler (KL) divergence**
  - <font color=OrangeRed> KL divergence </font>
    - a statistical measure of how <font color=LightSlateBlue> different two probability distributions are </font>.
    - use it to compare the completions off the two models
    - determine how much the updated model has diverged from the reference.
    - KL divergence is calculated for each generated token across the whole vocabulary off the LLM.
    - This can easily be tens or hundreds of thousands of tokens. However, using a softmax function, you've reduced the number of probabilities to much less than the full vocabulary size.
    - a relatively compute expensive process with will almost always benefit from using GPUs.

![picture 5](assets/img/6c2338493da33cf42bd717443cac01fef38825f3617816ab1dc1cfcccf9c41ff.png)

- calculated the KL divergence between the two models, added acid term to the reward calculation.
  - This will penalize **the RL-updated model** if it shifts too far from the **reference LLM** and generates completions that are two different.
  - now need to full copies of the LLM to calculate the KL divergence, the frozen **reference LLM**, and the **RL-updated PPO LLM**.

![picture 6](assets/img/b31374b65dbe3ed755cf076d8c35bb5e262dc9a8d054f607f70628be21038fea.png)

- benefit from combining our relationship with puffed.
  - In this case, only update the weights of a path adapter, not the full weights of the LLM.
  - you can reuse the same underlying LLM for both the reference model and the PPO model, which you update with a trained path parameters.
  - This reduces the memory footprint during training by approximately half.

---

## Assess the RL-updated model's performance.

![picture 7](assets/img/159f8a587f19133b9f68328dafe9ceeda5c54ff45b887cd3a11b77024014052f.png)

![picture 8](assets/img/cbcb5fb6e218202af2fff65c7b8807fe809a239ea0045b4413f6fe26d378894b.png)

use the summarization data set to quantify the reduction in toxicity
- for example: use toxicity score, the probability of the negative class, a toxic or hateful response averaged across the completions.
- If RHF has successfully reduce the toxicity of the LLM, this score should go down.
- First, create a baseline toxicity score for the original instruct LLM by evaluating its completions off the summarization data set with a reward model that can assess toxic language.
- Then evaluate the newly human aligned model on the same data set and compare the scores.
- In this example, the toxicity score has indeed decreased after RLHF, indicating a less toxic, better aligned model.
