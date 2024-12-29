---
title: "「SF-LC」2 Induction"
subtitle: "Logical Foundations - Proof by Induction"
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

## Review (only in slide)

```coq
Theorem review2: ∀b, (orb true b) = true.
Theorem review3: ∀b, (orb b true) = true.
```

Whether or not it can be just `simpl.` depending on the definition of `orb`.

In _Proof Engineering_, we probably won't need to include `review2` but need to include `review3` in library.

> Why we have `simpl.` but not `refl.` ?


Proving `0` is a "neutral element" for `+` (additive identity)
--------------------------------------------------------------

### Proving `0 + n = n`

```coq
Theorem plus_O_n : forall n : nat, 0 + n = n.
Proof.
  intros n. simpl. reflexivity. Qed.
```

This can be simply proved by _simplication_ bcuz the definition of `+` is defined by pattern matching against 1st operand:

```coq
Fixpoint plus (n : nat) (m : nat) : nat :=
  match n with
    | O ⇒ m
    | S n' ⇒ S (plus n' m)
  end.
```

We can observe that if `n` is `0`(`O`), no matter `m` is, it returns `m` as is.


### Proving `n + 0 = n`

#### 1st try: Simplication

```coq
Theorem plus_O_n_1 : forall n : nat,  n + 0 = n.
Proof.
  intros n.
  simpl. (* Does nothing! *)
Abort.
```

This cannot be proved by _simplication_ bcuz `n` is unknown so _unfold_ the definition `+` won't be able to simplify anything.

#### 2nd try: Case Analysis

```coq
Theorem plus_n_O_2 : ∀n:nat,
  n = n + 0.
Proof.
  intros n. destruct n as [| n'] eqn:E.
  - (* n = 0 *)
    reflexivity. (* so far so good... *)
  - (* n = S n' *)
    simpl. (* ...but here we are stuck again *)
Abort.
```

Our 2nd try is to use _case analysis_ (`destruct`), but the proof stucks in _inductive case_ since `n` can be infinitely large (destructed)


#### Induction to the resucue

> To prove interesting facts about numbers, lists, and other inductively defined sets, we usually need a more powerful reasoning principle: induction.

Princeple of induction over natural numbers (i.e. _mathematical induction_)

```coq
P(0); ∀n' P(n') → P(S n')  ====>  P(n)
```

In Coq, like `destruct`, `induction` break `P(n)` into 2 subgoals:

```coq
Theorem plus_n_O : ∀n:nat, n = n + 0.
Proof.
  intros n. induction n as [| n' IHn'].
  - (* n = 0 *) reflexivity.
  - (* n = S n' *) simpl. rewrite <- IHn'. reflexivity. Qed.
```


Proving `n - n = 0`
-------------------

```coq
Theorem minus_diag : ∀n,
  minus n n = 0.
Proof.
  (* WORKED IN CLASS *)
  intros n. induction n as [| n' IHn'].
  - (* n = 0 *)
    simpl. reflexivity.
  - (* n = S n' *)
    simpl. rewrite → IHn'. reflexivity. Qed
```

Noticed that the definition of `minus`:

```coq
    Fixpoint minus (n m:nat) : nat :=
      match n, m with
      | O   , _    => O
      | S _ , O    => n
      | S n', S m' => minus n' m'
      end.
```

`rewrite`
---------

`rewrite` would do a (DFS) preorder traversal in the syntax tree.








