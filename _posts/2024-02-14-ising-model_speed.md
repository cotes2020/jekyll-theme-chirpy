---
title: Ising Model (Python, Cython, Numba, Mojo, Julia, pt. 3)
author: jake
date: 2024-03-04 12:00:00 +0800
categories: [Math]
tags: [physics]
math: true
mermaid: true
image:
  path: /assets/img/ising_speed.png
  alt: Ising Model with Checkerboard Coloring
---
A post about the *speed* of the computation of the [Ising Model](https://en.wikipedia.org/wiki/Ising_model). I compare Python, Cython, Numba, Mojo, & Julia on the Ising Model.
> I am sure there are optimizations I missed, e.g. vectorize, parallel, etc. This serves as a comparison of the naive implementation I have blogged about before, plus some tricks that I am aware of.
{: .prompt-info }

## TL;DR

| Language      | Naive Speed | Development Effort Rank |
| ----------- | ----------- | ----------- |
| Python      | 140ms       | 1 |
| Cython   | 0.580ms        | 5 |
| Numba   | 1.1ms        | 2 |
| Mojo   | 3ms        | 4 |
| Julia   | 3.2ms        | 3 |

<!-- jupyter nbconvert --NbConvertApp.output_files_dir="../assets/img/ising_model_out" --to markdown _includes/ising_model.ipynb -->
{% include ising_model_speed.md %}
{% include ising_model_speed_2.md %}
{% include ising_model_speed_3.md %}
## Summary
The naive translation of each language beats Python by ~100x, which is pretty good for basic applications. If you are a fan of Swift, then Mojo will come very naturally since both are [syntactic sugar for LLVM](https://news.ycombinator.com/item?id=21832060). If you come from R, then Julia has similar syntax and emphasis on scientific computing. Numba can be applied with a simple decorator, making it a great first pass attempt on existing code. Cython provides the most speed, but at the highest development cost (IMO).