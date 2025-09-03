---
title: Week 3 brief report
author: wxz
date: 2025-09-03 08:40:00 +0800
categories: [Weekly Report]
tags: [weekly report]
---

*This blog carries a dual purpose of testing and reporting*

## Week 3

1. Refining LLaRA. 
    + Did not finish the cross attention due to high complexity.
    + Tested the adaptive model on the three datasets.

    |                   | Movielens | Steam  | Lastfm |
    |-------------------|:---------:|:------:|:------:|
    | linear            |  0.4421   | 0.4949 | 0.4508 |
    | adaptive          |  0.4526   | 0.4924 | 0.4508 |

    + The curriculum learning in the paper was only incremental. Changing the learning policies has limited effect on the performance.

2. Contriever

    Learnt about the [paper](https://arxiv.org/pdf/2112.09118) and after training reproduced its results on the BEIR benchmark,close to the reported value.

    | nDCG@10 | MSMACRO | Fever |
    |---------|:-------:|:-----:|
    |         |   20.3  | 67.6  | 

## Prospects for next week

Haven't considered other reproductions yet……

(Several **irrelevant but important commitments** coming this September, temporarily limiting my ability to engage wholly into lab work.)
+ GRE on 5th Sept.
+ ICPC rounds on 7th, 14th and 20th Sept.
+ Opening of the fall semester

That being said, I sincerely appreciate the opportunity to be in the lab, and I am eager to participate in hands-on research. **Please let me know if I may be of assistance in lab work.**