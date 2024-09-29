---
title: Bradley Terry Preference Model on NFL Games
author: jake
date: 2024-09-28 12:00:00 +0800
categories: [Math]
tags: [statistics, math]
math: true
mermaid: false
image:
  path: /assets/img/custom/bradley_terry.png
  alt: Team connectivity of the NFL 2023 Season Graph
---
A post on the Bradley Terry model with NFL game data. The Bradley Terry model:

$$
\log{\frac{p_{ij}}{1 - p_{ij}}} = \beta_i - \beta_j
$$

Is a classical model for ranking or preference data and is sometimes referred to as a *preference model*.

{% include markdown/bradley_terry.md %}