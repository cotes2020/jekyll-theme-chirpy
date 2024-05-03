---
title: LLM - Evaluation
date: 2023-04-24 11:11:11 -0400
description:
categories: [51AI, LLM]
# img: /assets/img/sample/rabbit.png
tags: [AI, ML]
---


# Evaluation


## overall


### Sample

### LlamaIndex

```py
from deepeval.integrations.llama_index import (
    DeepEvalAnswerRelevancyEvaluator,
    DeepEvalFaithfulnessEvaluator,
    DeepEvalContextualRelevancyEvaluator,
    DeepEvalSummarizationEvaluator,
    DeepEvalBiasEvaluator,
    DeepEvalToxicityEvaluator,
)
```

![Screenshot 2024-04-29 at 12.15.14](/assets/img/Screenshot%202024-04-29%20at%2012.15.14_xjzmg6dsi.png)

![Screenshot 2024-04-29 at 12.16.25](/assets/img/Screenshot%202024-04-29%20at%2012.16.25_1t3jx1uox.png)

Evaluating Response Faithfulness (i.e. Hallucination)
- The `FaithfulnessEvaluator` evaluates if the answer is <font color=LightSlateBlue> faithful </font> to the retrieved contexts (in other words, whether if there's hallucination).

![Screenshot 2024-04-29 at 12.33.12](/assets/img/Screenshot%202024-04-29%20at%2012.33.12_frl8djwa0.png)

Evaluating Query + Response Relevancy
- The `RelevancyEvaluator` evaluates if the retrieved context and the answer is <font color=LightSlateBlue> relevant and consistent </font> for the given query.

![Screenshot 2024-04-29 at 12.39.39](/assets/img/Screenshot%202024-04-29%20at%2012.39.39.png)

![Screenshot 2024-04-29 at 12.39.24](/assets/img/Screenshot%202024-04-29%20at%2012.39.24.png)
