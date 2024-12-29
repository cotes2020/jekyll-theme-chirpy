---
title: "「SF-PLF」14 RecordSub"
subtitle: "Programming Language Foundations - Subtyping with Records"
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


```coq
Inductive ty : Type :=
  (* record types *)
  | RNil : ty
  | RCons : string → ty → ty → ty.
```

we need typecon to identify record...


```coq
Inductive tm : Type :=
  | rproj ...?  isn't it as well?
  (* record terms *)
  | rnil : tm
  | rcons : string → tm → tm → tm.
``

as a list...


for Record, can compiler reorder the fields? (SML and OCaml)





