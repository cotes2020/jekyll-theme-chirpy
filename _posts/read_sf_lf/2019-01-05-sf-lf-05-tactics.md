---
title: "「SF-LC」5 Tactics"
subtitle: "Logical Foundations - More Basic Tactics"
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

## `apply`

- _exactly_ the same as some hypothesis
- can be used to __finish__ a proof (shorter than `rewrite` then `reflexivity`)

It also works with _conditional_ hypotheses:

```coq
n, m, o, p : nat
eq1 : n = m
eq2 : forall q r : nat, q = r -> [q; o] = [r; p]
============================
[n; o] = [m; p]

apply eq2.
n = m
```

It works by working backwards. 
It will try to _pattern match_ the universally quantified `q r`. (i.e. universal var)
We match the _conclusion_ and generates the _hypothesis_ as a _subgoal_.

```coq
Theorem trans_eq : forall (X:Type) (n m o : X), n = m -> m = o -> n = o.

Example trans_eq_example' : forall (a b c d e f : nat),
     [a;b] = [c;d] -> [c;d] = [e;f] -> [a;b] = [e;f].
Proof. 
  intros a b c d e f eq1 eq2.
  apply trans_eq. (* Error: Unable to find an instance for the variable m. *)
```

The _unification algo_ won't happy since:
- it can find instance for `n = o` from `[a;b] = [e;f]` (matching both conclusion)
- but what should be `m`? It could be anything as long as `n = m` and `m = o` holds.

So we need to tell Coq explicitly which value should be picked for `m`:

```coq
apply trans_eq with (m:=[c;d]).   (* <- supplying extra info, [m:=] can be ommited *)
```

> Prof Mtf: As a PL person, you should feel this is a little bit awkward since now function argument name must be remembered. (but it's just local and should be able to do any alpha-conversion).
> named argument is more like a record.

In Coq Intensive 2 (2018), someone proposed the below which works:

```coq
Example trans_eq_example'' : forall (a b c d e f : nat),
  [a;b] = [c;d] -> [c;d] = [e;f] -> [a;b] = [e;f]. 
Proof.
  intros a b c d e f.
  apply trans_eq.          (* Coq was able to match three at all at this time...hmm *)
Qed.

```


## `injection` and `discrinimate`

### Side Note on Terminologys of Function

                         relation

> function is defined as _a special kind of binary relation_. 
> it requires `xRy1 ∧ xRy2 → y1 = y2`  called "functional" or "univalent", "right-unique", or "deterministic"
> and also `∀x ∈ X, ∃y ∈ Y s.t. xRy`   called "left-total"

                    x       ↦      f(x)
                  input     ↦     output
                argument    ↦     value

                    X       ↦       Y
                 domain 域  ↦  co-domain 陪域      
           what can go into ↦  what possibly come out

                  A ⊆ X     ↦  f(A) = {f(x) | x ∈ A}
                            ↦     image
                            ↦  what actually come out

    f⁻¹(B)={x ∈ X|f(x) ∈ B} ↦     B ⊆ Y
                 preimage   ↦

                when A = X  ↦       Y
                            ↦     range  
                               image of domain

Besides subset, the notation of `image` and `pre-image` can be applied to _element_ as well.
However, by definition:
- the image    of an element `x` of domain    ↦  always single element of codomain (singleton set)
- the preimage of an element `y` of codomain  ↦  may be empty, or one, or many!
  - `<= 1 ↦ 1` : injective   (left-unique)
  - `>= 1 ↦ 1` : surjective  (right-total)
  - `   1 ↦ 1` : bijective

Noted that the definition of "function" doesn't require "right-total"ity) until we have `surjective`.

graph = `[(x, f(x))]`, these points form a "curve", 真的是图像

### Total vs Partial

For math, we seldon use partial function since we can simply "define a perfect domain for that".
But in Type Theory, Category Theory, we usually consider the _domain_ `X` and the _domain of definition_ `X'`.

Besides, `f(x)` can be `undefined`. (not "left-total", might not have "right")

### Conclusion - the road from Relation to Function


                bi-relation 
                     | + right-unique 
              partial function
                     | + left-total   
              (total) function
     + left-unique /   \ + right-total
          injection     surjection
                   \   /
                 bijection



### Original notes on [Injective, surjective, Bijective](https://en.wikipedia.org/wiki/Function)

All talk about the propeties of _preimage_!

- Injective:  `<= 1 ↦ 1` or `0, 1 ↦ 1` (distinctness) 
- Surjective: `>= 1 ↦ 1` (at least 1 in the domain)
- Bijective:  `   1 ↦ 1` (intersection of Inj and Surj, so only `1` preimage, _one-to-one correspondence_)


### _injectivitiy_ and _disjointness_, or `inversion`.

Recall the definition of `nat`:

```coq
Inductive nat : Type :=
| O : nat
| S : nat → nat.
```

Besides there are two forms of `nat` (for `destruct` and `induction`), there are more facts:

1. The constructor `S` is _injective_ (distinct), i.e `S n = S m -> n = m`.
2. The constructors `O` and `S` are _disjoint_, i.e. `forall n, O != S n `.


### `injection`

- can be used to prove the _preimages_ are the same.
- `injection` leave things in conclusion rather than hypo. with `as` would be in hypo.


### `disjoint`

- _principle of explosion_ (a logical principle)
  - asserts a contraditory hypothesis entails anything. (even false things)
  - _vacously true_
- `false = true` is contraditory because they are distinct constructors.

### `inversion`

- the big hammer: inversion of the definition.
- combining `injection` and `disjoint` and even some more `rewrite`.
  - IMH, which one to use depends on _semantics_

from Coq Intensive (not sure why it's not the case in book version).

```coq
Theorem S_injective_inv : forall (n m : nat),
  S n = S m -> n = m.
Proof.
  intros n m H. inversion H. reflexivity. Qed. 


Theorem inversion_ex1 : forall (n m : nat),
  [n] = [m] -> n = m.
Proof.
  intros n m H. inversion H. reflexivity. Qed.
```

> Side question: could Coq derive equality function for inductive type?
> A: nope. Equality for some inductive types are _undecidable_.

### Converse of injectivity

```coq
Theorem f_equal : ∀(A B : Type) (f: A → B) (x y: A),
  x = y → f x = f y.
Proof. 
  intros A B f x y eq. 
  rewrite eq. reflexivity. Qed.
```


### Slide Q&A 1

1. The tactic fails because tho `negb` is injective but `injection` only workks on constructors.

## Using Tactics in Hypotheses

### Reasoning Backwards and Reasoning Forward (from Coq Intensive 2)

Style of reasoning

- Backwards: start with _goal_, applying tactics `simpl/destruct/induction`, generate _subgoals_, until proved.
  - iteratively reasons about what would imply the goal, until premises or previously proven theorems are reached.
- Forwards:  start with _hypo_, applying tactics, iteratively draws conclusions, until the goal is reached. 

Backwards reasoning is dominated stratgy of theroem prover (and execution of prolog). But not natural in informal proof.

> True forward reasoning derives fact, but in Coq it's like hypo deriving hypo, very imperative.

### `in`

> most tactics also have a variant that performs a similar operation on a statement in the context.

```coq
simpl in H.
simpl in *. (* in all hypo and goal *)

symmetry in H.
apply L in H.
```

### `apply`ing in hypothesis and in conclusion

`apply`ing in hypo is very different with `apply`ing in conclusion.

> it's not we unify the ultimate conclusion and generate premises as new goal, but trying to find a hypothesis to match and left the residual conclusion as new hypothesis.

```coq
Theorem silly3'' : forall (n : nat),
  (true = (n =? 5) -> true = ((S (S n)) =? 7)) ->
  true = (n =? 5)  ->
  true = ((S (S n)) =? 7).
Proof.
  intros n eq H.
  apply eq in H.  (* or *)  apply eq. (* would be different *)
  apply H.  Qed.
```

Also if we add one more premises `true = true ->`, 
the subgoal generated by `apply` would be in reversed order: 

```coq
Theorem silly3'' : forall (n : nat),
  (true = true -> true = (n =? 5) -> true = ((S (S n)) =? 7)) ->
  true = (n =? 5)  ->
  true = ((S (S n)) =? 7).
Proof.
```
> Again: "proof engineering": proof can be done in so many different ways and in different orders.


## Varying the Induction Hypothesis

Sometimes it's important to control the exact form of the induction hypothesis!!

Considering:

```coq
Theorem double_injective: ∀n m,
        double n = double m → n = m.
```

if we begin with `intros n m. induction n.`
then we get stuck in the inductive case of `n`, where the induction hypothesis `IHn'` generated is:

```coq
IHn' : double n' = double m -> n' = m
IHn' : double n' = double (S m') -> n' = S m'  (* m = S m' *)
```

This is not what we want!! 

To prove `double_injective`, we hope `IHn'` can give us `double n' = double m' -> n' = m'` (i.e. the `P(n-1)` case).

The problem is `intros` implies _for these particular `n` and `m`_. (not more `forall` but _const_).  And when we `intros n m. induction n`, we are trying to prove a statement involving _every_ n but just a _single_ m...


### _How to keep `m` generic (universal)?_

By either `induction n` before `intros m` or using `generalize dependent m`, we can have:

```coq
IHn' : forall m : nat, double n' = double m -> n' = m
```
where the `m` here is still universally quantified, so we can instaniate `m` with `m'` by `apply`ing it with `double n' = double m'` to yield `n' = m'` or vice versa. (recall conditional statements can be `apply`ed in 2 ways.)


### Notes on `generalize dependent`

Usually used when the argument order is conflict with instantiate (`intros`) order.

> ? _reflection_: turing a computational result into a propositional result 



## Unfolding Definitions.

> tactics like `simpl`, `reflexivity`, and `apply` will often unfold the definitions of functions automatically.
> However, this automatic unfolding is somewhat _conservative_. 

`simpl.` only do unfolding when it can furthur simplify after unfolding. But sometimes you might want to explicitly `unfold` then do furthur works on that.


## Using `destruct` on Compound Expressions

destruct the whole arbitrary expression.

`destruct` by default throw away the whole expression after it, which might leave you into a stuck state.
So explicitly saying `eqn:Name` would help with that!


## Micro Sermon - Mindless proof-hacking

From Coq Intensive...

- a lot of fun 
- ...w/o thinking at all
- terrible temptation
- you shouldn't always resist...

But after 5 mins...you should step back and try to think

A typical coq user
- sitting and does not have their brain engaged all the time...
- at some point...(get stuck)
  - oh I have to reengage brain..

what is this really saying...

One way: good old paper and pencil

5 mins is good time!


