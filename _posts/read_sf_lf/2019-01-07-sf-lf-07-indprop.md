---
title: "「SF-LC」7 Ind Prop"
subtitle: "Logical Foundations - Inductively Defined Propositions (归纳定义命题)"
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

Inductively Defined Propositions 
--------------------------------

### The 3rd way to state Evenness...

Besides: 

```coq
Theorem even_bool_prop : ∀n,
  evenb n = true ↔ ∃k, n = double k.
 (*bool*)                 (*prop*)
```

we can write an _Inductive definition_ of the `even` property!


### Inference rules

In CS, we often uses _inference rules_ 

                        ev n
    ---- ev_0       ------------ ev_SS
    ev 0            ev (S (S n))

and _proof tree_ (i.e. evidence), there could be multiple premieses to make it more tree-ish.

    ---- ev_0
    ev 0
    ---- ev_SS
    ev 2
    ---- ev_SS
    ev 4

So we can literally translate them into a GADT:


### Inductive Definition of Evenness

```coq
Inductive even : nat → Prop :=
  | ev_0  : even 0
  | ev_SS : ∀n, even n → even (S (S n)). 

Check even_SS.
(* ==> : forall n : nat, even n -> even (S (S n)) *)
```

There are two ways to understand the `even` here:


### 1. A Property of `nat` and two theorems (Intuitively) 

> the thing we are defining is not a `Type`, but rather a function `nat -> Prop` — i.e., a property of numbers. 

we have two ways to provide an evidence to show the `nat` is `even`, either or:
1. it's `0`, we can immediately conclude it's `even`.
2. for any `n`, if we can provide a evidence that `n` is `even`, then `S (S n)` is `even` as well.

> We can think of the definition of `even` as defining a Coq property `even : nat → Prop`, together with primitive theorems `ev_0 : even 0` and `ev_SS : ∀ n, even n → even (S (S n))`.


### 2. An "Indexed" GADT and two constructors (Technically)

> In an Inductive definition, an argument to the type constructor on the left of the colon is called a "parameter", whereas an argument on the right is called an "index". -- "Software Foundaton"

Considered a "parametrized" ADT such as the polymorphic list, 

```coq
Inductive list (X:Type) : Type :=
  | nil
  | cons (x : X) (l : list X).

Check list. (* ===> list : Type -> Type *)
```

where we defined type con `list : Type -> Type`, by having a type var `X` in the left of the `:`.
the `X` is called a _parameter_ and would be _parametrized i.e. substituted, globally_, in constructors.



Here, we write `nat` in the right of the `:` w/o giving it a name (to refer and to substitute),
which allows the `nat` taking different values in different constructors (as constraints).
it's called an _index_ and will form a family of type indexed by `nat` (to type check?)


From this perspective, there is an alternative way to write this GADT:

```coq
Inductive even : nat → Prop :=
| ev_0                         : even 0
| ev_SS (n : nat) (H : even n) : even (S (S n)).
```

we have two ways to construct the `even` type (`Prop <: Type`), either or:
1. `ev_0` takes no argument, so simply instantiate `even` with `nat` 0
2. `ev_SS` takes a `nat` `n` and a `H` typed `even n`, 
  - the _dependency_ between two arguments thus established! 
  - as long as the _constraint on same `n`_ is fullfilled, we can build type `even` with `S (S n)`
  
The take way is that _dependent type (Pi-type)_ allow us to constriant constructors with different values.

> _indexed_ way is more general. it formed a larger type, and is only used when extra power needed. 
> every parametrized one can be represented as indexed one (it's just that index happended to be the same)


### "Constructor Theorems"

> Such "constructor theorems" have the same status as proven theorems. In particular, we can use Coq's `apply` tactic with the rule names to prove `even` for particular numbers...

```coq
Theorem ev_4 : even 4.
Proof. apply ev_SS. apply ev_SS. apply ev_0. Qed.
```

Proof States Transition:

    even 4
    ------ apply ev_SS.
    even 2
    ------ apply ev_SS.
    even 0
    ------ apply ev_0.
           Qed.


I believed what `apply` do is trying to _backward reasoning_, i.e. matching the goal and leave the "evidence" need to be proved (to conclude the goal).

we can write it as normal function application syntax w/o using tactics like other Dependent-typed PL as well

```coq
Theorem ev_4' : even 4.
Proof. apply (ev_SS 2 (ev_SS 0 ev_0)). Qed.
```


Using Evidence in Proofs
------------------------

> Besides _constructing evidence_ that numbers are even, we can also _reason_ about such evidence.

> Introducing `even` with an `Inductive` declaration tells Coq that these two constructors are the __only__ ways to build evidence that numbers are `even`. 

> In other words, if someone gives us evidence `E` for the assertion `even n`, then we know that `E` must have one of two shapes

> This suggests that it should be possible to analyze a hypothesis of the form `even n` much _as we do inductively defined data structures_; in particular, it should be possible to argue by __induction__ and __case analysis__ on such evidence.

This starts to get familiar as what we did for many calculi, ranging from Logics to PLT.
This is called the __Inversion property__.


### Inversion on Evidence

We can prove the inersion property by ourselves:

```coq
Theorem ev_inversion :
  ∀(n : nat), even n →
    (n = 0) ∨ (∃n', n = S (S n') ∧ even n').
Proof.
  intros n E.
  destruct E as [ | n' E'].
  - (* E = ev_0 : even 0 *)                  left. reflexivity.
  - (* E = ev_SS n', E' : even (S (S n')) *) right. ∃n'. split. reflexivity. apply E'.
Qed.
```

But Coq provide the `inversion` tactics that does more! (not always good tho, too automagical)

> The inversion tactic does quite a bit of work. When applied to equalities, as a special case, it does the work of both `discriminate` and `injection`. In addition, it carries out the `intros` and `rewrite`s

> Here's how inversion works in general. Suppose the name `H` refers to an assumption `P` in the current context, _where `P` has been defined by an `Inductive` declaration_. Then, for each of the constructors of `P`, `inversion H` generates a subgoal in which `H` has been replaced by the _exact, specific conditions under which this constructor could have been used to prove `P`_. 
> Some of these subgoals will be self-contradictory; inversion throws these away. The ones that are left represent the cases that must be proved to establish the original goal. For those, inversion adds all equations into the proof context that must hold of the arguments given to `P` (e.g., `S (S n') = n` in the proof of `evSS_ev`).
(`9-proof-object.md` has a better explaination on `inversion`)

`inversion` is a specific use upon `destruct` (both do case analysis on constructors), but many property need `induction`!. 
By `induction (even n)`, we have cases and subgoals splitted, and induction hypothesis as well.


### Induction on Evidence

Similar to induction on inductively defined data such as `list`: 
> To prove a property of (for any `X`)                       `list X` holds, we can use `induction` on `list X`.
> To prove a property of `n` holds for all numbers for which `even n` holds, we can use `induction` on `even n`.


#### Notes on induction

_The principle of induction_ is to prove `P(n-1) -> P(n)` (多米诺) for some (well-founded partial order) set of `n`. 

Here, we are induction over "the set of numbers fullfilling the property `even`". 
Noticed that we r proving things over this set, meaning we already have it (i.e. a proof, or a evidence) in premises, instead of proving the `even`ness of the set.


#### Proof by Mathematical Induction is Deductive Reasoning

> "Proof by induction," despite the name, is deductive. The reason is that proof by induction does not simply involve "going from many specific cases to the general case." Instead, in order for proof by induction to work, we need a deductive proof that each specific case implies the next specific case. Mathematical induction is not philosophical induction. 
<https://math.stackexchange.com/a/1960895/528269>

> Mathematical induction is an inference rule used in formal proofs. Proofs by mathematical induction are, in fact, examples of deductive reasoning.
> Equivalence with the well-ordering principle: The principle of mathematical induction is usually stated as an axiom of the natural numbers; see Peano axioms. However, it can be proved from the well-ordering principle. Indeed, suppose the following:
<https://en.wikipedia.org/wiki/Mathematical_induction>


#### Also, Structual Induction is one kind of Math. Induction

> 和标准的数学归纳法等价于良序原理一样，结构归纳法也等价于良序原理。

> ...A _well-founded_ _partial order_ is defined on the structures...
> ...Formally speaking, this then satisfies the premises of an _axiom of well-founded induction_...
<https://en.wikipedia.org/wiki/Structural_induction>

In terms of Well-ordering and Well-founded:

> If the set of all structures of a certain kind admits a well-founded partial order, 
> then every nonempty subset must have a minimal element. (This is the definition of "well-founded".)
> 如果某种整个结构的集容纳一个良基偏序， 那么每个非空子集一定都含有最小元素。（其实这也是良基的定义





Inductive Relations
-------------------

Just as a single-argument proposition defines a _property_, 性质
a two-argument proposition defines a _relation_. 关系

```coq
Inductive le : nat → nat → Prop :=
  | le_n n                : le n n
  | le_S n m (H : le n m) : le n (S m).

Notation "n ≤ m" := (le n m).
```

> It says that there are two ways to _give evidence_ that one number is less than or equal to another:

1. either same number
2. or give evidence that `n ≤ m` then we can have `n ≤ m + 1`.

and we can use the same tactics as we did for properties.




## Slide Q&A - 1

1. First `destruct` `even n` into 2 cases, then `discriminate` on each.

Another way... 
rewriting `n=1` on `even n`. It won't compute `Prop`, but `destruct` can do some `discriminate` behind the scene.



## Slide Q&A - 2

`inversion` and `rewrite plus_comm` (for `n+2`)




`destruct` vs. `inversion` vs. `induction`.
-------------------------------------------

> `destruct`, `inversion`, `induction` (on general thing)... similar/specialized version of each...

Trying to internalize this concept better: _When to use which?_

For any inductively defined proposition (`<: Type`) in hypothesis:
meaning from type perspective, it's already a "proper type" (`::*`)

```coq
Inductive P = C1 : P1 | C2 : A2 -> P2 | ...
```

1. `destruct`     case analysis on inductive type 

* simply give you each cases, i.e. each constructors.
* we can destruct on `a =? b` since `=?` is inductively defined.


2. `induction`    use induction principle

* proving `P` holds for all base cases
* proving `P(n)` holds w/ `P(n-1)` for all inductive cases
(`destruct` stucks in this case because of no induction hypothesis gained from induction principle)


3. `inversion`    invert the conclusion and give you all cases with premises of that case.

For GADT, i.e. "indexed" `Prop` (property/relation), `P` could have many shape
`inversion` give you `Ax` for shape `P` assuming built with `Cx`

`inversion` discards cases when shape `P != Px`.
(`destruct` stucks in this case because of no equation gained from inversion lemma)






Case Study: Regular Expressions
-------------------------------


### Definition

_Definition of RegExp in formal language can be found in FCT/CC materials_

```coq
Inductive reg_exp {T : Type} : Type :=
  | EmptySet                 (* ∅ *)
  | EmptyStr                 (* ε *)
  | Char (t : T)
  | App (r1 r2 : reg_exp)    (* r1r2 *)
  | Union (r1 r2 : reg_exp)  (* r1 | r2 *)
  | Star (r : reg_exp).      (* r*  *)
```


> Note that this definition is _polymorphic_. 
> We depart slightly in that _we do not require the type `T` to be finite_. (difference not significant here)

> `reg_exp T` describe _strings_ with characters drawn from `T` — that is, __lists of elements of `T`__. 


### Matching

The matching is somewhat similar to _Parser Combinator_ in Haskell... 

e.g.
`EmptyStr` matches `[]`
`Char x`   matches `[x]`

> we definied it into an `Inductive` relation (can be displayed as _inference-rule_). 
somewhat type-level computing !

```coq
Inductive exp_match {T} : list T → reg_exp → Prop :=
| MEmpty : exp_match [] EmptyStr
| MChar x : exp_match [x] (Char x)
| MApp s1 re1 s2 re2
            (H1 : exp_match s1 re1)
            (H2 : exp_match s2 re2) :
            exp_match (s1 ++ s2) (App re1 re2)
(** etc. **)

Notation "s =~ re" := (exp_match s re) (at level 80).  (* the Perl notation! *)
```

## Slide Q&A - 3

The lack of rule for `EmptySet` ("negative rule") give us what we want as PLT


### `Union` and `Star`.

> the informal rules for `Union` and `Star` correspond to _two constructors_ each.

```coq
| MUnionL s1 re1 re2
              (H1 : exp_match s1 re1) :
              exp_match s1 (Union re1 re2)
| MUnionR re1 s2 re2
              (H2 : exp_match s2 re2) :
              exp_match s2 (Union re1 re2)
| MStar0 re : exp_match [] (Star re)
| MStarApp s1 s2 re
              (H1 : exp_match s1 re)
              (H2 : exp_match s2 (Star re)) :
              exp_match (s1 ++ s2) (Star re).
```

Thinking about their _NFA_: they both have non-deterministic branches!
The recursive occurrences of `exp_match` gives as _direct argument_ (evidence) about which branches we goes.

> we need some _sanity check_ since Coq simply trust what we declared...
> that's why there is even Quick Check for Coq.

### Direct Proof 

In fact, `MApp` is also non-deterministic about how does `re1` and `re2` collaborate...
So we have to be explicit:

```coq
Example reg_exp_ex2 : [1; 2] =~ App (Char 1) (Char 2).
Proof.
  apply (MApp [1] _ [2]).
  ...
```

### Inversion on Evidence

This, if we want to prove via `destruct`, 
we have to write our own _inversion lemma_ (like `ev_inversion` for `even`).
Otherwise we have no equation (which we should have) to say `contradiction`.

```coq
Example reg_exp_ex3 : ~ ([1; 2] =~ Char 1).
Proof.
  intros H. inversion H.
Qed.
```

### Manual Manipulation

```coq
Lemma MStar1 :
  forall T s (re : @reg_exp T) ,
    s =~ re ->
    s =~ Star re.
Proof.
  intros T s re H.
  rewrite <- (app_nil_r _ s).  (* extra "massaging" to convert [s] => [s ++ []] *)
  apply (MStarApp s [] re).    (* to the shape [MStarApp] expected thus can pattern match on *)

      (* proving [MStarApp] requires [s1 s2 re H1 H2]. By giving [s [] re], we left two evidence *)
      | MStarApp s1 s2 re
          (H1 : exp_match s1 re)
          (H2 : exp_match s2 (Star re)) :
          exp_match (s1 ++ s2) (Star re).

  - apply H.                   (* evidence H1 *)
  - apply MStar0.              (* evidence H2 *)
Qed.                           (* the fun fact is that we can really think the _proof_
                                  as providing evidence by _partial application_. *)
```

### Induction on Evidence

> By the recursive nature of `exp_match`, proofs will often require induction.

```coq
(** Recursively collecting all characters that occur in a regex **)
Fixpoint re_chars {T} (re : reg_exp) : list T :=
  match re with
  | EmptySet ⇒ []
  | EmptyStr ⇒ []
  | Char x ⇒ [x]
  | App re1 re2 ⇒ re_chars re1 ++ re_chars re2
  | Union re1 re2 ⇒ re_chars re1 ++ re_chars re2
  | Star re ⇒ re_chars re
  end.
```

The proof of `in_re_match` went through by `inversion` on relation `s =~ re`. (which gives us all 7 cases.)
The interesting case is `MStarApp`, where the proof tree has two _branches_ (of premises):

                      s1 =~ re    s2 =~ Star re
                     ---------------------------            (MStarApp)
                        s1 ++ s2 =~ Star re

So by induction on the relation (rule), we got _two induction hypotheses_!
That's what we need for the proof.



The `remember` tactic (Induction on Evidence of A Specific Case)
----------------------------------------------------------------

One interesting/confusing features is that `induction` over a term that's _insuffciently general_. e.g. 

```coq
Lemma star_app: ∀T (s1 s2 : list T) (re : @reg_exp T),
  s1 =~ Star re →
  s2 =~ Star re →
  s1 ++ s2 =~ Star re.
Proof.
  intros T s1 s2 re H1.
```

Here, we know the fact that both `s1` and `s2` are matching with the form `Star re`. 
But by `induction`. it will give us _all 7 cases_ to prove, but _5 of them are contradictory_!

That's where we need `remember (Star re) as re'` to get this bit of information back to `discriminate`.


### Sidenotes: `inversion` vs. `induction` on evidence

We might attemp to use `inversion`,
which is best suitted for have a specific conclusion of some rule and inverts back to get its premises.

But for _recursive cases_ (e.g. `Star`), we always need `induction`. 

`induction` on a specific conclusion then `remember + contradiction` is similar with how `inversion` solves contradictionary cases. (They both `destruct` the inductively defined things for sure)




Exercise: 5 stars, advanced (pumping)
-------------------------------------

FCT/Wikipedia "proves" [pumping lemma for regex](https://en.wikipedia.org/wiki/Pumping_lemma_for_regular_languages) in a non-constructive way.

Here we attempts to give a constructive proof.




Case Study: Improving Reflection (互映)
-------------------------------------

> we often need to relate boolean computations to statements in `Prop`

```coq
Inductive reflect (P : Prop) : bool → Prop :=
| ReflectT (H : P) : reflect P true
| ReflectF (H : ¬P) : reflect P false.
```

The _only_ way to construct `ReflectT/F` is by showing (a proof) of `P/¬P`,
meaning invertion on `reflect P bool` can give us back the evidence. 


`iff_reflect` give us `eqbP`.

```coq
Lemma eqbP : ∀n m, reflect (n = m) (n =? m).
Proof.
  intros n m. apply iff_reflect. rewrite eqb_eq. reflexivity.
Qed.
```

This gives us a small gain in convenience: we immediately give the `Prop` from `bool`, no need to `rewrite`.
> Proof Engineering Hacks...


### SSReflect - small-scale reflection

> a Coq library
> used to prove 4-color theorem...!
> simplify small proof steps with boolean computations. (somewhat automation with decision procedures)






Extended Exercise: A Verified Regular-Expression Matcher
--------------------------------------------------------

> we have defined a _match relation_ that can _prove_ a regex matches a string.
> but it does not give us a _program_ that can _run_ to determine a match automatically...

> we hope to translate _inductive rules (for constructing evidence)_ to _recursive fn_.
> however, since `reg_exp` is recursive, Coq won't accept it always terminates 

theoritically, the regex = DFA so it is decidable and halt.
technically, it only halts on finite strings but not infinite strings. 
(and infinite strings are probably beyond the scope of halting problem?)

> Heavily-optimized regex matcher = translating into _state machine_ e.g. NFA/DFA.
> Here we took a _derivative_ approach which operates purely on string.

```coq
Require Export Coq.Strings.Ascii.
Definition string := list ascii.
```

Coq 标准库中的 ASCII 字符串也是归纳定义的，不过我们这里为了之前定义的 match relation 用 `list ascii`.

> to define regex matcher over `list X` i.e. polymorphic lists.
> we need to be able to _test equality_ for each `X` etc.


### Rules & Derivatives.

Check paper [Regular-expression derivatives reexamined - JFP 09]() as well.

`app` and `star` are the hardest ones. 


#### Let's take `app` as an example 

##### 1. 等价 helper 

```coq
Lemma app_exists : ∀(s : string) re0 re1,
    s =~ App re0 re1 ↔ ∃s0 s1, s = s0 ++ s1 ∧ s0 =~ re0 ∧ s1 =~ re1.
```

this _helper rules_ is written for the sake of convenience:
- the `<-` is the definition of `MApp`.
- the `->` is the `inversion s =~ App re0 re1`.

##### 2. `App` 对于 `a :: s` 的匹配性质

```coq
Lemma app_ne : ∀(a : ascii) s re0 re1,
    a :: s =~ (App re0 re1) ↔
    ([ ] =~ re0 ∧ a :: s =~ re1) ∨
    ∃s0 s1, s = s0 ++ s1 ∧ a :: s0 =~ re0 ∧ s1 =~ re1.
```
the second rule is more interesting. It states the _property_ of `app`:
> App re0 re1 匹配 a::s 当且仅当  (re0 匹配空字符串 且 a::s 匹配 re1)  或  (s=s0++s1，其中 a::s0 匹配 re0 且 s1 匹配 re1)。


这两条对后来的证明很有帮助，`app_exists` 反演出来的 existential 刚好用在 `app_ne` 中.
> https://github.com/jiangsy/SoftwareFoundation/blob/47543ce8b004cd25d0e1769f7444d57f0e26594d/IndProp.v


##### 3. 定义 derivative 关系

the relation _`re'` is a derivative of `re` on `a`_ is defind as follows:

```coq
Definition is_der re (a : ascii) re' :=
  ∀s, a :: s =~ re ↔ s =~ re'.
```

##### 4. 实现 derive

Now we can impl `derive` by follwing `2`, the property.
In paper we have:

    ∂ₐ(r · s) = ∂ₐr · s + ν(r) · ∂ₐs       -- subscriprt "a" meaning "respective to a" 

    where 
      ν(r) = nullable(r) ? ε : ∅ 

In our Coq implementation, `nullable(r) == match_eps(r)`, 

Since we know that 
`∀r, ∅ · r = ∅`, 
`∀r, ε · r = r`, 
we can be more straightforward by expanding out `v(r)`:

```coq
Fixpoint derive (a : ascii) (re : @reg_exp ascii) : @reg_exp ascii :=
...
 | App r1 r2 => if match_eps r1                            (** nullable(r) ? **)
      then Union (App (derive a r1) r2) (derive a r2)      (**  ∂ₐr · s + ∂ₐs **)
      else App (derive a r1) r2                            (**  ∂ₐr · s       **)
```
