---
title: "「SF-QC」2 TypeClasses"
subtitle: "Quickcheck - A Tutorial on Typeclasses in Coq"
layout: post
author: "Hux"
header-style: text
hidden: true
tags:
  - SF (软件基础)
  - QC (Quickcheck)
  - Coq
  - 笔记
---

Considerring printing different types with this common idiom:

```coq
showBool : bool → string
showNat : nat → string
showList : {A : Type} (A → string) → (list A) → string
showPair : {A B : Type} (A → string) → (B → string) → A * B → string

Definition showListOfPairsOfNats := showList (showPair showNat showNat)   (* LOL *)
```

> The designers of Haskell addressed this clunkiness through _typeclasses_, a mechanism by which the typechecker is instructed to automatically construct "type-driven" functions [Wadler and Blott 1989].

Coq followed Haskell's lead as well, but

> because Coq's type system is so much richer than that of Haskell, and because typeclasses in Coq are used to automatically construct not only programs but also proofs, Coq's presentation of typeclasses is quite a bit less "transparent"


Basics
------

### Classes and Instances

```coq
Class Show A : Type := {
  show : A → string
}.

Instance showBool : Show bool := {
  show := fun b:bool ⇒ if b then "true" else "false"
}.
```

Comparing with Haskell:

```haskell
class Show a where
  show :: a -> string

--  you cannot override a `instance` so in reality you need a `newtype` wrapper to do this
instance Show Bool where
  show b = if b then "True" else "Fasle"
```

> The show function is sometimes said to be overloaded, since it can be applied to arguments of many types, with potentially radically different behavior depending on the type of its argument.


Next, we can define functions that use the overloaded function show like this:

```coq
Definition showOne {A : Type} `{Show A} (a : A) : string :=
  "The value is " ++ show a.
  
Compute (showOne true).
Compute (showOne 42).

Definition showTwo {A B : Type}
           `{Show A} `{Show B} (a : A) (b : B) : string :=
  "First is " ++ show a ++ " and second is " ++ show b.

Compute (showTwo true 42).
Compute (showTwo Red Green).
```

> The parameter `` `{Show A}`` is a _class constraint_, which states that the function showOne is expected to be applied only to types A that belong to the Show class.

> Concretely, this constraint should be thought of as an _extra parameter_ to showOne supplying _evidence_ that A is an instance of Show — i.e., it is essentially just a show function for A, which is implicitly invoked by the expression show a.

读时猜测（后来发现接下来有更正确的解释）：`show` 在 name resolution 到 `class Show` 时就可以根据其参数的 type（比如 `T`）infer 出「我们需要一个 `Show T` 的实现（`instance`，其实就是个 table）」，在 Haskell/Rust 中这个 table 会在 lower 到 IR 时才 made explicit，而 Coq 这里的语法就已经强调了这里需要 implicitly-and-inferred `{}` 一个 table，这个 table 的名字其实不重要，只要其 type 是被 `A` parametrized 的 `Show` 就好了，类似 ML 的 `functor` 或者 Java 的 generic `interface`。

This is _Ad-hoc polymorphism_.


#### Missing Constraint

What if we forget the class constrints:

```coq
Error:
Unable to satisfy the following constraints:
In environment:
A : Type
a : A

?Show : "Show A"
```


#### Class `Eq`

```coq
Class Eq A :=
  {
    eqb: A → A → bool;
  }.

Notation "x =? y" := (eqb x y) (at level 70).

Instance eqBool : Eq bool :=
  {
    eqb := fun (b c : bool) ⇒ 
       match b, c with
         | true, true ⇒ true
         | true, false ⇒ false
         | false, true ⇒ false
         | false, false ⇒ true
       end
  }.

Instance eqNat : Eq nat :=
  {
    eqb := Nat.eqb
  }.
```

> Why should we need to define a typeclass for boolean equality when _Coq's propositional equality_ (`x = y`) is completely generic? 
> while it makes sense to _claim_ that two values `x` and `y` are equal no matter what their type is, it is not possible to write a _decidable equality checker_ for arbitrary types. In particular, equality at types like `nat → nat` is undecidable.

`x = y` 返回一个需要去证的 `Prop` (relational) 而非 executable `Fixpoint` (functional)  
因为 function 的 equality 有时候会 undeciable，所以才需要加 Functional Extensionality `Axiom`（见 LF-06）

```coq
Instance eqBoolArrowBool: Eq (bool -> bool) :=
  {
    eqb := fun (f1 f2 : bool -> bool) =>
      (f1 true) =? (f2 true) && (f1 false) =? (f2 false) 
  }.

Compute (id =? id).      (* ==> true *)
Compute (negb =? negb).  (* ==> true *)
Compute (id =? negb).    (* ==> false *)
```

这里这个 `eqb` 的定义也是基于 extensionality 的定义，如果考虑到 effects（divergence、IO）是很容易 break 的（类似 parametricity）



### Parameterized Instances: New Typeclasses from Old

Structural recursion 

```coq
Instance showPair {A B : Type} `{Show A} `{Show B} : Show (A * B) :=
  {
    show p :=
      let (a,b) := p in 
        "(" ++ show a ++ "," ++ show b ++ ")"
  }.
Compute (show (true,42)).
```

Structural equality

```coq
Instance eqPair {A B : Type} `{Eq A} `{Eq B} : Eq (A * B) :=
  {
    eqb p1 p2 :=
      let (p1a,p1b) := p1 in
      let (p2a,p2b) := p2 in
      andb (p1a =? p2a) (p1b =? p2b)
  }.
```

Slightly more complicated example: typical list:

```coq
(* the book didn't use any from ListNotation *)
Fixpoint showListAux {A : Type} (s : A → string) (l : list A) : string :=
  match l with
    | nil ⇒ ""
    | cons h nil ⇒ s h
    | cons h t ⇒ append (append (s h) ", ") (showListAux s t)
  end.
Instance showList {A : Type} `{Show A} : Show (list A) :=
  {
    show l := append "[" (append (showListAux show l) "]")
  }.
  
(* I used them though *)
Fixpoint eqListAux {A : Type} `{Eq A} (l1 l2 : list A) : bool :=
  match l1, l2 with
    | nil, nil => true
    | (h1::t1), (h2::t2) => (h1 =? h2) && (eqListAux t1 t2)
    | _, _ => false
  end.

Instance eqList {A : Type} `{Eq A} : Eq (list A) :=
  {
    eqb l1 l2 := eqListAux l1 l2
  }.
```



### Class Hierarchies

> we might want a typeclass `Ord` for "ordered types" that support both equality and a less-or-equal comparison operator.

A bad way would be declare a new class with two func `eq` and `le`.

It's better to establish dependencies between typeclasses, similar with OOP `class` inheritence and subtyping (but better!), this gave good code reuses.

> We often want to organize typeclasses into hierarchies.

```coq
Class Ord A `{Eq A} : Type :=
  {
    le : A → A → bool
  }.
Check Ord. (* ==>
Ord
     : forall A : Type, Eq A -> Type
*)
```

class `Eq` is a "super(type)class" of `Ord` (not to be confused with OOP superclass)

This is _Sub-typeclassing_.

```coq
Fixpoint listOrdAux {A : Type} `{Ord A} (l1 l2 : list A) : bool := 
  match l1, l2 with
  | [], _ => true
  | _, [] => false
  | h1::t1, h2::t2 => if (h1 =? h2)
                     then (listOrdAux t1 t2)
                     else (le h1 h2)
  end.

Instance listOrd {A : Type} `{Ord A} : Ord (list A) :=
  {
    le l1 l2 := listOrdAux l1 l2
  }.

(* truthy *)
Compute (le [1] [2]).
Compute (le [1;2] [2;2]).
Compute (le [1;2;3] [2]).

(* falsy *)
Compute (le [1;2;3] [1]).
Compute (le [2] [1;2;3]).
```



How It works
------------

### Implicit Generalization

所以 `` `{...}`` 这个 "backtick" notation is called _implicit generalization_，比 implicit `{}` 多做了一件自动 generalize 泛化 free varabile 的事情。

> that was added to Coq to support typeclasses but that can also be used to good effect elsewhere.

```coq
Definition showOne1 `{Show A} (a : A) : string :=
  "The value is " ++ show a.

Print showOne1.
(* ==>
    showOne1 = 
      fun (A : Type) (H : Show A) (a : A) => "The value is " ++ show a
           : forall A : Type, Show A -> A -> string

    Arguments A, H are implicit and maximally inserted
*)
```

> notice that the occurrence of `A` inside the `` `{...}`` is unbound and automatically insert the binding that we wrote explicitly before.

> The "implicit and maximally generalized" annotation on the last line means that the automatically inserted bindings are treated (注：printed) as if they had been written with `{...}`, rather than `(...)`.

> The "implicit" part means that the type argument `A` and the `Show` witness `H` are usually expected to be left implicit  
> whenever we write `showOne1`, Coq will automatically insert two _unification variables_ as the first two arguments.

> This automatic insertion can be disabled by writing `@`, so a bare occurrence of `showOne1` means the same as `@showOne1 _ _`

这里的 witness `H` 即 `A` implements `Show` 的 evidence，本质就是个 table or record，可以 written more explicitly:

```coq
Definition showOne2 `{_ : Show A} (a : A) : string :=
  "The value is " ++ show a.

Definition showOne3 `{H : Show A} (a : A) : string :=
  "The value is " ++ show a.
```

甚至 

```coq
Definition showOne4 `{Show} a : string :=
  "The value is " ++ show a.
```

```coq
showOne = 
fun (A : Type) (H : Show A) (a : A) => "The value is " ++ show a
     : forall A : Type, Show A -> A -> string

Set Printing Implicit.

showOne = 
fun (A : Type) (H : Show A) (a : A) => "The value is " ++ @show A H a     (* <-- 注意这里 *)
     : forall A : Type, Show A -> A -> string
```

#### vs. Haskell

顺便，Haskell 的话，`Show` 是可以直接 inferred from the use of `show` 得

```haskell
Prelude> showOne a = show a
Prelude> :t showOne
showOne :: Show a => a -> String
```

但是 Coq 不行，会退化上「上一个定义的 instance Show」，还挺奇怪的（

```coq
Definition showOne5 a : string :=  (* not generalized *)
  "The value is " ++ show a.
```

#### Free Superclass Instance

``{Ord A}` led Coq to fill in both `A` and `H : Eq A` because it's the superclass of `Ord` (appears as the second argument). 

```coq
Definition max1 `{Ord A} (x y : A) :=
  if le x y then y else x.

Set Printing Implicit.
Print max1.
(* ==>
     max1 = 
       fun (A : Type) (H : Eq A) (H0 : @Ord A H) (x y : A) =>
         if @le A H H0 x y then y else x

   : forall (A : Type) (H : Eq A), 
       @Ord A H -> A -> A -> A    
*)
Check Ord.
(* ==> Ord : forall A : Type, Eq A -> Type *)
```

`Ord` type 写详细的话可以是：

```coq
Ord : forall (A : Type), (H: Eq A) -> Type
```


#### Other usages of `` `{} ``

Implicit generalized `Prop` mentioning free vars.

```coq
Generalizable Variables x y.

Lemma commutativity_property : `{x + y = y + x}.
Proof. intros. omega. Qed.

Check commutativity_property.
```

Implicit generalized `fun`/`λ`, however...

```coq
Definition implicit_fun := `{x + y}.
Compute (implicit_fun 2 3)  (* ==> Error *)
Compute (@implicit_fun 2 3)
```

Implicitly-generalized but inserted as explicit via `` `(...)``

```coq
Definition implicit_fun := `(x + y).
Compute (implicit_fun 2 3)
```

这里可以看到 Coq 的所有语法都是正交的（非常牛逼……）
- `()`/`{}` 控制是否是 implicit argument
- `` ` ``-prefix 控制是否做 implicit generalization
  - N.B. 可能你忘记了但是 `→` is degenerated `∀` (`Π`)，所以 generalization 自然会生成 `fun`


### Records are Products

> Record types must be declared before they are used. For example:

```coq
Record Point :=
  Build_Point
    {
      px : nat;
      py : nat
    }.

(* built with constructor *)
Check (Build_Point 2 4).

(* built with record syntax *)
Check {| px := 2; py := 4 |}.
Check {| py := 2; px := 4 |}.

(* field access, with a clunky "dot notation" *)
Definition r : Point := {| px := 2; py := 4 |}.
Compute (r.(px) + r.(py)).
```

和 OCaml 一样是 nominal typing 而非 structural typing。
类似于 OCaml 中的 record 其实到 backend 了就会和 tuple 等价：都会 lower 到 Heap Block），
Coq 中的 Record 其实和 Pair/Product 也是等价：都是 arity 为 2 的 Inductive type：

```coq
Inductive Point : Set := 
  | Build_Point : nat → nat → Point.
```

我仿造 `Print px.` 输出的定义模拟了一下：

```coq
Inductive Point2 : Set := 
  | Build_Point2 (px2:nat) (py2:nat).
Definition px2 := fun p : Point2 => let (px, _) := p in px.
Definition py2 := fun p : Point2 => let (_, py) := p in py.

Definition r2 : Point2 := Build_Point2 2 4.
Compute (r2.(px2) + r2.(py2)).                        (* => 6 *)

Definition r2 : Point2 := {| px2 := 2; py2 := 4 |}.   (* Error: px2 is not a projection *)
```

可以发现 dot notation 是可以工作的，`.` 应该只是一个 pipe
但是 `{|...|}` 不知道为什么这里会认为 `px2` 不是一个 record projection.


> Note that the field names have to be different. Any given field name can belong to only one record type. 
> This greatly simplifies type inference!


### Typeclasses are Records

> Typeclasses and instances, in turn, are basically just syntactic sugar for record types and values (together with a bit of magic for using proof search to fill in appropriate instances during typechecking...

> Internally, a typeclass declaration is elaborated into a _parameterized_ `Record` declaration:

```coq
Class Show A : Type := { show : A → string }.

Print Show.
Record Show (A : Type) : Type := 
    Build_Show { show : A -> string }

Set Printing All.
Print Show.
Variant Show (A : Type) : Type :=
    Build_Show : forall _ : forall _ : A, string, Show A

(* to make it more clear... *)
Inductive Show (A : Type) : Type :=
  | Build_Show : ∀(show : ∀(a : A), string), Show A
  
(* or more GADT looking, i.e., implicit generalized *)
Inductive Show (A : Type) : Type :=
  | Build_Show : (A -> string) -> Show A
```

Coq actually call a single-field record `Variant`. 
Well actually, I found it's for any single-constructor `Inductive`ly constructed type. 
You can even use `Variant` nonchangbly with `Inductive` as a keyword...

```coq
Set Printing All.
Print Point.
Variant Point : Set :=
    Build_Point : forall (_ : nat) (_ : nat), Point
```

> Analogously, Instance declarations become record values:

```coq
Print showNat.
showNat = {| show := string_of_nat |}
    : Show nat
```

> Similarly, overloaded functions like show are really just _record projections_, which in turn are just functions that select a particular argument of a one-constructor Inductive type.

```coq
Print show.
show = 
  fun (A : Type) (Show0 : Show A) => 
    let (show) := Show0 in show
      : forall A : Type, Show A -> A -> string

Set Printing All.
Print show.
show = 
  fun (A : Type) (Show0 : Show A) =>
    match Show0 return (forall _ : A, string) with
    | Build_Show _ show => show
    end
      : forall (A : Type) (_ : Show A) (_ : A), string
```


### Inferring Instances

> appropriate instances are automatically inferred (and/or constructed!) during typechecking.

```coq
Definition eg42 := show 42.

Set Printing Implicit.
Print eg42.
eg42 = @show nat showNat 42 : string
```

different with `Compute`, `Print` 居然还可以这么把所有 implicit argument (after inferred) 都给 print 出来……

type inferrence: 

- `show` is expanded to `@show _ _ 42`
- obviously it's `@show nat __42`
- obviously it's `@show nat (?H : Show Nat) 42`

Okay now where to find this witness/evidence/instance/record/table/you-name-it `?H` 

> It attempts to find or construct such a value using a _variant of the `eauto` proof search_ procedure that refers to a "hint database" called `typeclass_instances`.

```coq
Print HintDb typeclass_instances.  (* too much to be useful *)
```

"hint database" to me is better understood as a reverse of environment or typing context `Γ`. Though specialized with only `Instance` there.
（这么一看实现一个 Scala 的 `Implicit` 也不难啊）

Coq can even print what's happening during this proof search!

```coq
Set Typeclasses Debug.
Check (show 42).
(* ==>
     Debug: 1: looking for (Show nat) without backtracking
     Debug: 1.1: exact showNat on (Show nat), 0 subgoal(s)
*)

Check (show (true,42)).
(* ==>
     Debug: 1: looking for (Show (bool * nat)) without backtracking
     Debug: 1.1: simple apply @showPair on (Show (bool * nat)), 2 subgoal(s)
     Debug: 1.1.3 : (Show bool)
     Debug: 1.1.3: looking for (Show bool) without backtracking
     Debug: 1.1.3.1: exact showBool on (Show bool), 0 subgoal(s)
     Debug: 1.1.3 : (Show nat)
     Debug: 1.1.3: looking for (Show nat) without backtracking
     Debug: 1.1.3.1: exact showNat on (Show nat), 0 subgoal(s)      *)
Unset Typeclasses Debug.
```

> In summary, here are the steps again:

```coq
show 42
    ===>   { Implicit arguments }
@show _ _ 42
    ===>   { Typing }
@show (?A : Type) (?Show0 : Show ?A) 42
    ===>   { Unification }
@show nat (?Show0 : Show nat) 42
    ===>   { Proof search for Show Nat returns showNat }
@show nat showNat 42
```


Typeclasses and Proofs
----------------------

### Propositional Typeclass Members

```coq
Class EqDec (A : Type) {H : Eq A} := 
  { 
    eqb_eq : ∀ x y, x =? y = true ↔ x = y 
  }.
```

```coq
Instance eqdecNat : EqDec nat := 
  {
    eqb_eq := Nat.eqb_eq
  }.
```

这里可以用于抽象 LF-07 的 reflection


### Substructures

> Naturally, it is also possible to have typeclass instances as members of other typeclasses: these are called _substructures_. 

这里的 `relation` 来自 Prelude 不过和 LF-11 用法一样：

```coq
Require Import Coq.Relations.Relation_Definitions.
Class Reflexive (A : Type) (R : relation A) :=
  { 
    reflexivity : ∀ x, R x x
  }.
Class Transitive (A : Type) (R : relation A) :=
  {
    transitivity : ∀ x y z, R x y → R y z → R x z
  }.
```

```coq
Class PreOrder (A : Type) (R : relation A) :=
  { PreOrder_Reflexive :> Reflexive A R ;
    PreOrder_Transitive :> Transitive A R }.
```

> The syntax `:>` indicates that each `PreOrder` can be seen as a `Reflexive` and `Transitive` relation, so that, any time a reflexive relation is needed, a preorder can be used instead.

这里的 `:>` 方向和 subtyping 的 _subsumption_ 是反着的……跟 SML 的 ascription `:>` 一样……

- subtyping  `T :> S` : value of `S` can safely be used as value of `T`
- ascription `P :> R` : value of `P` can safely be used as value of `R`

Why?



Some Useful Typeclasses
-----------------------

### `Dec`

> The `ssreflect` library defines what it means for a proposition `P` to be _decidable_ like this...

```coq
Require Import ssreflect ssrbool.
Print decidable.
(* ==>
     decidable = fun P : Prop => {P} + {~ P}
*)
```

> .. where `{P} + {¬ P}` is an "informative disjunction" of `P` and `¬P`.

即两个 evidence（参考 LF-07)

```coq
Class Dec (P : Prop) : Type :=
  {
    dec : decidable P
  }.
```

### Monad

> In Haskell, one place typeclasses are used very heavily is with the Monad typeclass, especially in conjunction with Haskell's "do notation" for monadic actions.

> Monads are an extremely powerful tool for organizing and streamlining code in a wide range of situations where computations can be thought of as yielding a result along with some kind of "effect."

说话很严谨「in a wide range of situations where ... "effect"」

> most older projects simply define their own monads and monadic notations — sometimes typeclass-based, often not — while newer projects use one of several generic libraries for monads. Our current favorite (as of Summer 2017) is the monad typeclasses in Gregory Malecha's `ext-lib` package:

<https://github.com/coq-ext-lib/coq-ext-lib/blob/v8.5/theories/Structures/Monad.v>

```coq
Require Export ExtLib.Structures.Monads.
Export MonadNotation.
Open Scope monad_scope.
```

```coq
Class Monad (M : Type → Type) : Type := { 
  ret : ∀ {T : Type}, T → M T ;
  bind : ∀ {T U : Type}, M T → (T → M U) → M U
}.

Instance optionMonad : Monad option := {
  ret T x := Some x ;
  bind T U m f :=
    match m with
      None ⇒ None
    | Some x ⇒ f x
    end
}.
```

Compare with Haskell:

```haskell
class Applicative m => Monad (m :: * -> *) where
  return :: a -> m a
  (>>=) :: m a -> (a -> m b) -> m b
  
instance Monad Maybe where
  return = Just
  (>>=)  = (>>=)
  where
    (>>=) :: Maybe a -> (a -> Maybe b) -> Maybe b
    Nothing  >>= _ = Nothing
    (Just x) >>= f = f x
```

After mimic `do` notation: (as PLF-11)

```coq
Definition sum3 (l : list nat) : option nat :=
  x0 <- nth_opt 0 l ;;
  x1 <- nth_opt 1 l ;;
  x2 <- nth_opt 2 l ;;
  ret (x0 + x1 + x2).
```


Controlling Instantiation
-------------------------

### "Defaulting"

Would better explicitly typed. searching can be stupid

### Manipulating the Hint Database

> One of the ways in which Coq's typeclasses differ most from Haskell's is the lack, in Coq, of an automatic check for "overlapping instances."

在 Haskell 中一大 use case 是可以做类似 C++ 的 partial specification（偏特化）

- Check out [this](https://kseo.github.io/posts/2017-02-05-avoid-overlapping-instances-with-closed-type-families.html) on the pros and cons of overlapping instances in Haskell
- Check out [this] (https://www.ibm.com/developerworks/community/blogs/12bb75c9-dfec-42f5-8b55-b669cc56ad76/entry/c__e6_a8_a1_e6_9d_bf__e7_a9_b6_e7_ab_9f_e4_bb_80_e4_b9_88_e6_98_af_e7_89_b9_e5_8c_96?lang=en) on template partial specification in C++

> That is, it is completely legal to define a given type to be an instance of a given class in two different ways.
> When this happens, it is unpredictable which instance will be found first by the instance search process;

Workarounds in Coq when this happen:
1. removing instances from hint database
2. priorities



Debugging
---------

TBD.

- Instantiation Failures
- Nontermination


Alternative Structuring Mechanisms
----------------------------------

_large-scale structuring mechanisms_

> Typeclasses are just one of several mechanisms that can be used in Coq for structuring large developments. Others include:
>
> - canonical structures
> - bare dependent records
> - modules and functors

Module and functors is very familiar!


Further Reading
----------------------------------

On the origins of typeclasses in Haskell:

- How to make ad-hoc polymorphism less ad hoc Philip Wadler and Stephen Blott. 16'th Symposium on Principles of Programming Languages, ACM Press, Austin, Texas, January 1989.
  <http://homepages.inf.ed.ac.uk/wadler/topics/type-classes.html>  

The original paper on typeclasses In Coq:

- Matthieu Sozeau and Nicolas Oury. First-Class Type Classes. TPHOLs 2008.
  <https://link.springer.com/chapter/10.1007%2F978-3-540-71067-7_23>
  
