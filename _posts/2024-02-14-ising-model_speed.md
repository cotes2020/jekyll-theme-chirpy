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
> I do not claim to be an expert in any of these languages. I am sure there are optimizations I missed, e.g. vectorize, parallel. This serves as a comparison of the naive implementation I have blogged about before, plus some tricks that I am aware of.
{: .prompt-info }

<!-- jupyter nbconvert --NbConvertApp.output_files_dir="../assets/img/ising_model_out" --to markdown _includes/ising_model.ipynb -->
{% include ising_model_speed.md %}
{% include ising_model_speed_2.md %}
{% include ising_model_speed_3.md %}
