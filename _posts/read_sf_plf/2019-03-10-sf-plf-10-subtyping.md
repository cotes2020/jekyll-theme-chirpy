---
title: "「SF-PLF」10 Sub"
subtitle: "Programming Language Foundations - Subtyping (子类型化)"
layout: post
author: "Hux"
header-style: text
hidden: true
tags:
  - SF (软件基础)
  - PLF (编程语言基础)
  - Coq
  - 笔记
---



Concepts
--------



### The Subsumption Rule


### The Subtype Relation














### Slide QA1

Record Subtyping... 

row type


index? record impl as list


width/depth/permulation
- multiple step rules



---


Java

1. class - no index (thinking about offset)

having both width/permulation subtyping make impl slow
- OOP - hmm
- ML has no permulation - for perf reason (static structure) as C

ML has depth?
- a little bit by equality


OCaml objection has all three


### Slide QA2

Looking at Contravariant!

1. (2) `{i1:S,i2:T}→U <: {i1:S,i2:T,i3:V}→U`

2. (4) `{i1:T,i2:V,i3:V} <: {i1:S,i2:U} * {i3:V}` is interesting:

the interesting thing is, why don't we make some subtyping rules for that as well?

- there are definitely _code_ can do that
- their _runtime_ semantics are different tho they carry same information
- __coercion__ can used for that

3 and 4. (5) ...


A <: Top   =>   Top -> A <: A -> A  -- contravariant

if we only care `(A*T)`, can use `T:Top`

but to type the whole thing `: A`

`Top -> A`?
but noticed that we said `\z:A.z`

can we pass `A -> A` into `Top -> A`? 
      more specific        more general
      
smallest -> most specific -> `A -> A`
largest  -> most specific -> `Top -> A`


5. 
"The type Bool has no proper subtypes." (I.e., the only type smaller than Bool is Bool itself.)
Ture unless we have Bottom

hmm seems like `Bottom` in subtyping is different with Empty/Void, which is closer to logical `Bottom ⊥` since Bottom here is subtyping of everything..
OH they are the same: (nice)
> <https://en.wikipedia.org/wiki/Bottom_type>

6. True



### Inversion Lemmas for Subtyping

`inversion` doesn't lose information, `induction` does.

auto rememeber?? --- dependent induction
hetergeous equaltiy



In soundness proof

- subtyping only affects Canonical Forms + T_Sub case in induction


> Lemma: If Gamma ⊢ \x:S1.t2 ∈ T, then there is a type S2 such that x⊢>S1; Gamma ⊢ t2 ∈ S2 and S1 → S2 <: T.

why `T` not arrow? Top...


if including Bottom...many proof becomes hard, canonical form need to say...might be Bottom?

> no, no value has type Bottom (Void)...








