---
title: "「SF-LC」14 ImpCEvalFun"
subtitle: "Logical Foundations - An Evaluation Function For Imp"
layout: post
author: "Hux"
header-style: text
hidden: true
tags:
  - LF (逻辑基础)
  - SF (软件基础)
  - Coq
  - 笔记
---


Step-Indexed Evaluator
----------------------

...Copied from `12-imp.md`:

> Chapter `ImpCEvalFun` provide some workarounds to make functional evalution works:
> 1. _step-indexed evaluator_, i.e. limit the recursion depth. (think about _Depth-Limited Search_). 
> 2. return `option` to tell if it's a normal or abnormal termination.
> 3. use `LETOPT...IN...` to reduce the "optional unwrapping" (basicaly Monadic binding `>>=`!)
>    this approach of `let-binding` became so popular in ML family.


```coq
Notation "'LETOPT' x <== e1 'IN' e2"
   := (match e1 with
         | Some x ⇒ e2
         | None ⇒ None
       end)
   (right associativity, at level 60).

Open Scope imp_scope.
Fixpoint ceval_step (st : state) (c : com) (i : nat)
                    : option state :=
  match i with
  | O ⇒ None       (* depth-limit hit! *)
  | S i' ⇒
    match c with
      | SKIP ⇒
          Some st
      | l ::= a1 ⇒
          Some (l !-> aeval st a1 ; st)
      | c1 ;; c2 ⇒
          LETOPT st' <== ceval_step st c1 i' IN    (* option bind *)
          ceval_step st' c2 i'
      | TEST b THEN c1 ELSE c2 FI ⇒
          if (beval st b)
            then ceval_step st c1 i'
            else ceval_step st c2 i'
      | WHILE b1 DO c1 END ⇒
          if (beval st b1)
          then LETOPT st' <== ceval_step st c1 i' IN
               ceval_step st' c i'
          else Some st
    end
  end.
Close Scope imp_scope.
```



Relational vs. Step-Indexed Evaluation
--------------------------------------

Prove `ceval_step` is equiv to `ceval`


### ->

```coq
Theorem ceval_step__ceval: forall c st st',
      (exists i, ceval_step st c i = Some st') ->
      st =[ c ]=> st'.
```

The critical part of proof:

- `destruct` for the `i`.
- `induction i`, generalize on all `st st' c`. 
  1. `i = 0` case contradiction
  2. `i = S i'` case;
     `destruct c`. 
      - `destruct (ceval_step ...)` for the `option`
        1. `None` case contradiction
        2. `Some` case, use induction hypothesis...
    

### <-

```coq
Theorem ceval__ceval_step: forall c st st',
      st =[ c ]=> st' ->
      exists i, ceval_step st c i = Some st'.
Proof.
  intros c st st' Hce.
  induction Hce.
```



