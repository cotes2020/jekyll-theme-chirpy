---
title: "「SF-PLF」12 Records"
subtitle: "Programming Language Foundations - Adding Records To STLC"
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


## Adding Records


```coq
t ::=                          Terms:
    | {i1=t1, ..., in=tn}         record
    | t.i                         projection
    | ...

v ::=                          Values:
    | {i1=v1, ..., in=vn}         record value
    | ...

T ::=                          Types:
    | {i1:T1, ..., in:Tn}         record type
    | ...
```


## Formalizing Records





