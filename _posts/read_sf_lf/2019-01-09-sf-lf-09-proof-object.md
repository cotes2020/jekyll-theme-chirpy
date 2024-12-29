---
title: "「SF-LC」9 ProofObjects"
subtitle: "Logical Foundations - The Curry-Howard Correspondence "
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

> "Algorithms are the computational content of proofs." —Robert Harper

So the book material is designed to be gradually reveal the facts that 
> Programming and proving in Coq are two sides of the same coin. 


e.g.
- `Inductive` is useds for both data types and propositions.
- `->` is used for both type of functions and logical implication.

The fundamental idea of Coq is that:

> _provability_ in Coq is represented by _concrete evidence_. When we construct the proof of a basic proposition, we are actually _building a tree of evidence_, which can be thought of as a data structure.

e.g.
- implication like `A → B`, its proof will be an _evidence transformer_: a recipe for converting evidence for A into evidence for B.

> Proving manipulates evidence, much as programs manipuate data.


Curry-Howard Correspondence
---------------------------

> deep connection between the world of logic and the world of computation:

    propositions             ~  types
    proofs / evidence        ~  terms / data values 


`ev_0 : even 0`
- `ev_0` __has type__                             `even 0`
- `ev_0` __is a proof of__ / __is evidence for__  `even 0`

`ev_SS : ∀n, even n -> even (S (S n))`
- takes a nat `n` and evidence for `even n` and yields evidence for `even (S (S n))`.

This is _Props as Types_.


Proof Objects
-------------

Proofs are data! We can see the _proof object_ that results from this _proof script_.

```coq
Print ev_4.
(* ===> ev_4 = ev_SS 2 (ev_SS 0 ev_0) 
             : even 4  *)

Check (ev_SS 2 (ev_SS 0 ev_0)).     (* concrete derivation tree, we r explicitly say the number tho *)
(* ===> even 4 *)
```

These two ways are the same in principle!


Proof Scripts
-------------

`Show Proof.`  will show the _partially constructed_ proof terms / objects.
`?Goal` is the _unification variable_. (the hold we need to fill in to complete the proof)

more complicated in branching cases
one hole more subgoal

```coq
Theorem ev_4'' : even 4.   (*  match? (even 4) *)
Proof.
  Show Proof.              (*  ?Goal  *)
  apply ev_SS.
  Show Proof.              (*  (ev_SS 2 ?Goal)  *)
  apply ev_SS.
  Show Proof.              (*  (ev_SS 2 (ev_SS 0 ?Goal))  *)
  apply ev_0. 
  Show Proof.              (*  ?Goal (ev_SS 2 (ev_SS 0 ev_0))  *)
Qed.
```

> Tactic proofs are useful and convenient, but they are not essential: 
> in principle, we can always construct the required evidence by hand

Agda doesn't have tactics built-in. (but also Interactive)


Quantifiers, Implications, Functions
------------------------------------

In Coq's _computational universe_ (where data structures and programs live), to give `->`:
- constructors (introduced by `Indutive`)
- functions

in Coq's _logical universe_ (where we carry out proofs), to give implication:
- constructors
- functions!
 
 
So instead of writing proof scripts e.g._

```coq
Theorem ev_plus4 : ∀n, even n → even (4 + n).
Proof.
  intros n H. simpl.
  apply ev_SS.
  apply ev_SS.
  apply H.
Qed.
```

we can give proof object, which is a _function_ here, directly!

```coq
Definition ev_plus4' : ∀n, even n → even (4 + n) :=    (* ∀ is syntax for Pi? *)
  fun (n : nat)    ⇒ 
  fun (H : even n) ⇒
    ev_SS (S (S n)) (ev_SS n H).


Definition ev_plus4'' (n : nat) (H : even n)           (* tricky: implicitly `Pi` when `n` get mentioned?  *)
                    : even (4 + n) :=
  ev_SS (S (S n)) (ev_SS n H).
```

two interesting facts:
1. `intros x` corresponds to `λx.` (or `Pi x.`??)
2. `apply` corresponds to...not quite function application... but more like _filling the hole_.
3. `even n` mentions the _value_ of 1st argument `n`. i.e. _dependent type_!


Recall Ari's question in "applying theorem as function" e.g. `plus_comm` 
why we can apply value in type-level fun.
becuz of dependent type.

Now we call them `dependent type function`



### `→` is degenerated `∀` (`Pi`)

> Notice that both implication (`→`) and quantification (`∀`) correspond to functions on evidence. 
> In fact, they are really the same thing: `→` is just a shorthand for a degenerate use of `∀` where there is no dependency, i.e., no need to give a name to the type on the left-hand side of the arrow:

```coq
  ∀(x:nat), nat 
= ∀(_:nat), nat 
= nat → nat

  ∀n, ∀(E : even n), even (n + 2).
= ∀n, ∀(_ : even n), even (n + 2).
= ∀n, even n → even (n + 2).
```

> In general, `P → Q` is just syntactic sugar for `∀ (_:P), Q`.

TaPL also mention this fact for `Pi`.


Q&A - Slide 15
--------------

1. `∀ n, even n → even (4 + n)`. (`2 + n = S (S n)`)




Programming with Tactics.
-------------------------

If we can build proofs by giving explicit terms rather than executing tactic scripts, 
you may be wondering whether we can _build programs using tactics_? Yes!

```coq
Definition add1 : nat → nat.
  intro n.
  Show Proof.      
(** 
the goal (proof state):
    
    n : nat
    =======
    nat
    
the response:

    (fun n : nat => ?Goal) 
    
What is really interesting here, is that the premies [n:nat] is actually the arguments!
again, the process of applying tactics is _partial application_
**)

  apply S.
  Show Proof.      
(** 
    (fun n : nat => S ?Goal) 
**)
  apply n. 
Defined.

Print add1.
(* ==> add1 = fun n : nat => S n
            : nat -> nat *)
```

> Notice that we terminate the Definition with a `.` rather than with `:=` followed by a term.
> This tells Coq to enter _proof scripting mode_ (w/o `Proof.`, which did nothing)

> Also, we terminate the proof with `Defined` rather than `Qed`; this makes the definition _transparent_ so that it can be used in computation like a normally-defined function
> (`Qed`-defined objects are _opaque_ during computation.).

`Qed` make things `unfold`able, 
thus `add 1` ends with `Qed` is not computable...
(becuz of not even `unfold`able thus computation engine won't deal with it)

> Prof.Mtf: meaning "we don't care about the details of Proof"

see as well [Smart Constructor](https://wiki.haskell.org/Smart_constructors)


> This feature is mainly useful for writing functions with dependent types

In Coq      - you do as much as ML/Haskell when you can...?
Unlike Agda - you program intensively in dependent type...?

When Extracting to OCaml...Coq did a lot of `Object.magic` for coercion to bypass OCaml type system. (Coq has maken sure the type safety.)


Logical Connectives as Inductive Types
--------------------------------------

> Inductive definitions are powerful enough to express most of the connectives we have seen so far. 
> Indeed, only universal quantification (with implication as a special case) is built into Coq; 
> all the others are defined inductively. 
Wow...

> CoqI: What's Coq logic? Forall + Inductive type (+ coinduction), that's it.

### Conjunctions

```coq
Inductive and (P Q : Prop) : Prop :=
| conj : P → Q → and P Q.

Print prod.
(* ===>
   Inductive prod (X Y : Type) : Type :=
   | pair : X -> Y -> X * Y. *)
```

similar to `prod` (product) type... more connections happening here.

> This similarity should clarify why `destruct` and `intros` patterns can be used on a conjunctive hypothesis. 

> Similarly, the `split` tactic actually works for any inductively defined proposition with exactly one constructor
(so here, `apply conj`, which will match the conclusion and generate two subgoal from assumptions )

A _very direct_ proof:

```coq
Definition and_comm'_aux P Q (H : P ∧ Q) : Q ∧ P :=
  match H with
  | conj HP HQ ⇒ conj HQ HP
  end.
```



### Disjunction

```coq
Inductive or (P Q : Prop) : Prop :=
| or_introl : P → or P Q
| or_intror : Q → or P Q.
```

this explains why `destruct` works but `split` not..


Q&A - Slide 22 + 24
-------------------

Both Question asked about what's the type of some expression

```coq
fun P Q R (H1: and P Q) (H2: and Q R) ⇒
    match (H1,H2) with
    | (conj _ _ HP _, conj _ _ _ HR) ⇒ conj P R HP HR
    end.

fun P Q H ⇒
    match H with
    | or_introl HP ⇒ or_intror Q P HP
    | or_intror HQ ⇒ or_introl Q P HQ
    end.
```
But if you simply `Check` on them, you will get errors saying:
`Error: The constructor conj (in type and) expects 2 arguments.` or 
`Error: The constructor or_introl (in type or) expects 2 arguments.`.


### Coq Magics, "Implicit" Implicit and Overloading??

So what's the problem?
Well, Coq did some magics...

```coq
Print and.
(* ===> *)
Inductive and (A B : Prop) : Prop :=  conj : A -> B -> A /\ B
For conj: Arguments A, B are implicit
```

constructor `conj` has implicit type arg w/o using `{}` in `and` ...

```coq
Inductive or (A B : Prop) : Prop :=
    or_introl : A -> A \/ B | or_intror : B -> A \/ B

For or_introl, when applied to no more than 1 argument:
  Arguments A, B are implicit
For or_introl, when applied to 2 arguments:
  Argument A is implicit
For or_intror, when applied to no more than 1 argument:
  Arguments A, B are implicit
For or_intror, when applied to 2 arguments:
  Argument B is implicit
```

this is even more bizarre...
constructor `or_introl` (and `or_intror`) are _overloaded_!! (WTF)


And the questions're still given as if they're inside the modules we defined our plain version of `and` & `or` (w/o any magics), thus we need `_` in the positions we instantiate `and` & `or` so Coq will infer.



### Existential Quantification

> To give evidence for an existential quantifier, we package a witness `x` together with a proof that `x` satisfies the property `P`:

```coq
Inductive ex {A : Type} (P : A → Prop) : Prop :=
| ex_intro : ∀x : A, P x → ex P.

Check ex.                    (* ===> *) : (?A -> Prop) -> Prop 
Check even.                  (* ===> *) : nat -> Prop  (* ?A := nat  *)
Check ex even.               (* ===> *) : Prop 
Check ex (fun n => even n)   (* ===> *) : Prop     (* same *)
```

one interesting fact is, _outside_ of our module, the built-in Coq behaves differently (_magically_):

```coq
Check ev.                    (* ===> *) : ∀ (A : Type), (A -> Prop) -> Prop
Check even.                  (* ===> *) : nat -> Prop  (* A := nat  *)
Check ex (fun n => even n)   (* ===> *) : ∃ (n : nat) , even n : Prop  (* WAT !? *)
```

A example of explicit proof object (that inhabit this type):

```coq
Definition some_nat_is_even : ∃n, even n :=
  ex_intro even 4 (ev_SS 2 (ev_SS 0 ev_0)).
```

the `ex_intro` take `even` first then `4`...not sure why the order becomes this... 

```coq
Check (ex_intro).            (* ===> *) : forall (P : ?A -> Prop) (x : ?A), P x -> ex P
```

To prove `ex P`, given a witness `x` and a proof of `P x`. This desugar to `∃ x, P x`

- the `P` here, is getting applied when we define prop `∃ x, P x`.
- but the `x` is not mentioned in type constructor...so it's a _existential type_.
  - I don't know why languages (including Haskell) use `forall` for _existential_ tho.

`exists` tactic = applying `ex_intro`



### True and False

```coq
Inductive True : Prop :=
  | I : True.

(* with 0 constructors, no way of presenting evidence for False *)
Inductive False : Prop := .
```


Equality
--------

```coq
Inductive eq {X:Type} : X → X → Prop :=
| eq_refl : ∀x, eq x x.

Notation "x == y" := (eq x y)
                    (at level 70, no associativity)
                    : type_scope.
```


> given a set `X`, it defines a _family_ of propositions "x is equal to y,", _indexed by_ pairs of values (x and y) from `X`.

> Can we also use it to construct evidence that `1 + 1 = 2`? 
> Yes, we can. Indeed, it is the very same piece of evidence!

> The reason is that Coq treats as "the same" any two terms that are convertible according to a simple set of computation rules.

nothing in the unification engine but we relies on the _reduction engine_.

> Q: how much is it willing to do?  
> Mtf: just run them! (since Coq is total!)

```coq
Lemma four: 2 + 2 == 1 + 3.
Proof.
  apply eq_refl.
Qed.
```

The `reflexivity` tactic is essentially just shorthand for `apply eq_refl`.


Slide Q & A
-----------

- (4) has to be applicable thing, i.e. lambda, or "property" in the notion! 

In terms of provability of `reflexivity`

```coq
(fun n => S (S n)) = (fun n => 2 + n)          (* reflexivity *)
(fun n => S (S n)) = (fun n => n + 2)          (* rewrite add_com *)
```

### Inversion, Again

> We've seen inversion used with both equality hypotheses and hypotheses about inductively defined propositions. Now that we've seen that these are actually the same thing

In general, the `inversion` tactic...

1. take hypo `H` whose type `P` is inductively defined
2. for each constructor `C` in `P`
   1. generate new subgoal (assume `H` was built with `C`)
   2. add the arguments (i.e. evidences of premises) of `C` as extra hypo (to the context of subgoal)
   3. (apply `constructor` theorem), match the conclusion of `C`, calculates a set of equalities (some extra restrictions)
   4. adds these equalities
   5. if there is contradiction, `discriminate`, solve subgoal.


### Q

> Q: Can we write `+` in a communitive way?  
> A: I don't believe so.


[Ground truth](https://en.wikipedia.org/wiki/Ground_truth)
 - provided by direct observation (instead of inference)

[Ground term](https://en.wikipedia.org/wiki/Ground_expression#Ground_terms) 
 - that does not contain any free variables.

Groundness
 - 根基性?

> Weird `Axiomness` might break the soundness of generated code in OCaml...





