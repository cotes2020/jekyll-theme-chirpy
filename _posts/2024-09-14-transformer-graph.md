---
title: Message Passing Transformers
author: jake
date: 2024-09-14 12:00:00 +0800
categories: [Software Engineering]
tags: [statistics, math]
math: true
mermaid: false
image:
  path: /assets/img/custom/attention.png
  alt: Diagram of soft dictionary lookup from "Dive into deep learning"
---
This post shows a graph ("message passing" or "soft dictionary lookup") interpretation of [Attention](https://arxiv.org/pdf/1706.03762). It combines two really great references on the topic that show that forward passes through transformers are similar to message passing algorithms for probabilistic graphical models. But instead of the NP-hardness of the [Belief Propagation algorithm](https://en.wikipedia.org/wiki/Belief_propagation), transformers scale as $O(n^2 * d)$ with the data.

## Attention as soft dictionary lookup
This was inspired from `Section 15.4.1` of [Kevin Murphy's Probabilistic Machine Learning (PML) textbook](https://probml.github.io/pml-book/). Attention operation accepts a **query** (from a node) and a set of **(key, value)** tuples (from all other nodes):

$$
\text{Attn}(\pmb q, (\pmb k_1, \pmb v_1), \dots, (\pmb k_m, \pmb v_m)) = \text{Attn}(\pmb q, (\pmb k_{1:m}, \pmb v_{1:m})) = \sum_{i=1}^m \alpha_i (\pmb q, \pmb k_{1:m})\pmb v_i \in \mathbb R^v
$$

The **attention weights** $\alpha_i$:

$$
\alpha_i(\pmb q, \pmb k_{1:m}) = \text{softmax}_i([a(\pmb q, \pmb k_1), \dots, a(\pmb q, \pmb k_m)]) = \frac{\exp{a(\pmb q, \pmb k_i)}}{\sum_{j=1}^m \exp{a(\pmb q, \pmb k_j)}}
$$

have the properties:

$$
0 \leq \alpha_i(\pmb q, \pmb k_{1:m}) \leq 1
$$

$$
\sum_i \alpha_i(\pmb q, \pmb k_{1:m}) = 1
$$

For **attention score** $a$ which computes the similarity between $\pmb q$ and $\pmb k_i$:

$$
a: \pmb q \times \pmb k_i \rightarrow \mathbb R
$$

Finally, the computation graph can be visualized as (credit to [Dive into Deep Learning](https://d2l.ai/chapter_attention-mechanisms-and-transformers/attention-scoring-functions.html)):
![alt text](assets/img/custom/dict_lookup.png)

## Message Passing (Python)
This python code directly taken from [Andrej Karpathy's lecture on Transformers](https://youtu.be/XfpMkf4rD6E?si=QuQUK7XDCM0Wen5B). The idea is to represent each data point (embeddings of tokens in a sequence) as a node in a graph and then connect them (fully or casually). Instead of dealing with this data directly, we project the data into the **query**, **key**, and **value** data. This improves the representation of the underlying data. The algorithm finally updates itself by taking a convex combination of the values with attention scores (i.e. the attention weights).
{% include markdown/attention_graph.md %}

## Summary
From the Attention as a soft dictionary lookup formalism, we can see that Attention's forward pass is similar to a message passing algorithm. During each step of training, attention tunes the $Q$, $K$, & $V$ projection matrices to achieve more coherent predictions similar to [clique calibration](https://ermongroup.github.io/cs228-notes/inference/jt/) in graphical models. In contrast, attention weights do not define joint probability distributions among random variables (like a graphical model), but only distributions over which values should be weighted higher during the forward pass. So while [marginal inference](https://ermongroup.github.io/cs228-notes/inference/ve/) is not possible on a trained network with Attention mechanisms, correlations should be preserved from the original nodes.