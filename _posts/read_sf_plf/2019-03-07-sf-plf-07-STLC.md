---
title: "「SF-PLF」7 Stlc"
subtitle: "Programming Language Foundations - The Simply Typed Lambda-Calculus"
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

this chapter:
- (change to new syntax...)
- function abstraction
- variable binding  -- 变量绑定
- substitution      -- 替换


Overview
--------

"Base Types", only `Bool` for now.   -- 基类型
...again, exactly following TAPL.


```coq
t ::= 
    | x                         variable
    | \x:T1.t2                  abstraction       -- haskell-ish lambda
    | t1 t2                     application
    | tru                       constant true
    | fls                       constant false
    | test t1 then t2 else t3   conditional

T ::= 
    | Bool
    | T → T                     arrow type

-- example
\x:Bool. \y:Bool. x
(\x:Bool. \y:Bool. x) fls tru
\f:Bool→Bool. f (f tru)
```

Some known λ-idioms:
> two-arg functions are higher-order one-arg fun, i.e. curried
> no named functions yet, all "anonymous"  -- 匿名函数


## Slide QA 1

1. 2
2. `Bool`, `fls`







Syntax
------

Formalize syntax.
things are, as usual, in the `Type` level, so we can "check" them w/ dependent type.

```coq
Inductive ty : Type :=
  | Bool : ty
  | Arrow : ty → ty → ty.

Inductive tm : Type :=
  | var : string → tm
  | app : tm → tm → tm
  | abs : string → ty → tm → tm
  | tru : tm
  | fls : tm
  | test : tm → tm → tm → tm.
```

> Noted that, `\x:T.t` (formally, `abs x T t`), the argument type is explicitly annotated (not inferred.)


另外，这里介绍了一个小 trick: 用 `Notation` (更接近 宏 ) 而非 `Defintion` 使得我们可以使用 `auto`...

```coq
(** idB = \x:Bool. x **)
Notation idB := (abs x Bool (var x)).
```






Operational Semantics
---------------------


### Values 值

- `tru` and `fls` are values
- what about function?
  1. `\x:T. t` is value iff `t` value. -- Coq 
  2. `\x:T. t` is always value         -- most FP lang, either CBV or CBN

Coq 这么做挺奇怪的，不过对 Coq 来说: 
> terms can be considered equiv up to the computation VM (在其项化简可以做到的范围内都算相等)
> this rich the notion of Coq's value (所以 Coq 的值的概念是比一般要大的)

Three ways to construct `value` (unary relation = predicate)

```coq
Inductive value : tm → Prop :=
  | v_abs : ∀x T t, value (abs x T t)
  | v_tru : value tru
  | v_fls : value fls.
```


### STLC Programs 「程序」的概念也是要定义的

- _closed_    = term not refer any undefined var = __complete program__
- _open term_ = term with _free variable_

> Having made the choice not to reduce under abstractions, we don't need to worry about whether variables are values, since we'll always be reducing programs "from the outside in," and that means the step relation will always be working with closed terms.

if we could reduce under abstraction and variables are values... What's the implication here? 始终不懂...


### Substitution (IMPORTANT!) 替换

> `[x:=s]t` and pronounced "substitute s for x in t."

    (\x:Bool. test x then tru else x) fls   ==>    test fls then tru else fls


Important _capture_ example:

    [x:=tru] (\x:Bool. x)  ==>  \x:Bool. x     -- x is bound, we need α-conversion here
                           !=>  \x:Bool. tru


Informal definition...

    [x:=s]x               = s
    [x:=s]y               = y                     if x ≠ y
    [x:=s](\x:T11. t12)   = \x:T11. t12
    [x:=s](\y:T11. t12)   = \y:T11. [x:=s]t12     if x ≠ y
    [x:=s](t1 t2)         = ([x:=s]t1) ([x:=s]t2)
    ...

and formally:

```coq
Reserved Notation "'[' x ':=' s ']' t" (at level 20).
Fixpoint subst (x : string) (s : tm) (t : tm) : tm :=
  match t with
  | var x' ⇒ if eqb_string x x' then s else t    (* <-- computational eqb_string *)
  | abs x' T t1 ⇒ abs x' T (if eqb_string x x' then t1 else ([x:=s] t1))
  | app t1 t2 ⇒ app ([x:=s] t1) ([x:=s] t2)
  ...
```

> Computable `Fixpoint` means _meta-function_! (in metalanguage, Coq here)


### 如果我们考虑用于替换掉某个变量的项 s 其本身也含有自由变量， 那么定义替换将会变得困难一点。

Is `if x ≠ y` for function abstraction one sufficient?  -- 在 PLT 中我们采取了更严格的定义
> Only safe if we only consider `s` is closed term. 

Prof.Mtf:
> here...it's not really "_defining_ on closed terms". Technically, you can still write open terms.
> if we want, we could define the real `closed_term`...more works to prove things tho.

Prof.Mtf:
> In some more rigorous setting...we might define `well_typed_term`
> and the definition itself is the proof of `Preservation`! 


### Slide QA 2

1. (3)


### Reduction （beta-reduction） beta-归约

Should be familar

                    value v2
          ----------------------------                   (ST_AppAbs)   until value, i.e. function  (β-reduction)
          (\x:T.t12) v2 --> [x:=v2]t12

                    t1 --> t1'
                ----------------                           (ST_App1)   reduce lhs, Function side
                t1 t2 --> t1' t2

                    value v1
                    t2 --> t2'
                ----------------                           (ST_App2)   reduce rhs, Arg side 
                v1 t2 --> v1 t2'


Formally,
(I was expecting they invents some new syntax for this one...so we only have AST)

```coq
Reserved Notation "t1 '-->' t2" (at level 40).
Inductive step : tm → tm → Prop :=
  | ST_AppAbs : ∀x T t12 v2,
         value v2 →
         (app (abs x T t12) v2) --> [x:=v2]t12
  | ST_App1 : ∀t1 t1' t2,
         t1 --> t1' →
         app t1 t2 --> app t1' t2
  | ST_App2 : ∀v1 t2 t2',
         value v1 →
         t2 --> t2' →
         app v1 t2 --> app v1 t2'
...
```


### Slide QA 3

1. (1)  `idBB idB -> idB`
2. (1)  `idBB (idBB idB) -> idB`
3. if () ill-typed `idBB (notB tru) -> idBB fls ....`
   - we don't type check in step
4. (3)  `idB fls`
5. NOT...ill-typed one & open term








Typing
------


### Typing Contexts 类型上下文

we need something like environment but for Types.

> three-place typing judgment, informally written   -- 三元类型断言

    Gamma ⊢ t ∈ T
    
> "under the assumptions in Gamma, the term t has the type T."

```coq
Definition context := partial_map ty.
(X ⊢> T11, Gamma) 
```

Why `partial_map` here? 
IMP can use `total_map` because it gave default value for undefined var.


### Typing Relations


                              Gamma x = T
                            ----------------                            (T_Var)   look up
                            Gamma |- x \in T

                   (x |-> T11 ; Gamma) |- t12 \in T12
                   ----------------------------------                   (T_Abs)   type check against context w/ arg
                    Gamma |- \x:T11.t12 \in T11->T12

                        Gamma |- t1 \in T11->T12
                          Gamma |- t2 \in T11
                         ----------------------                         (T_App)
                         Gamma |- t1 t2 \in T12


```coq
Example typing_example_1 :
  empty ⊢ abs x Bool (var x) ∈ Arrow Bool Bool.
Proof.
  apply T_Abs. apply T_Var. reflexivity. Qed.
```


`example_2`
- `eapply`
- `A` ?? looks like need need another environment to look up `A`...



### Typable / Deciable


> decidable type system = decide term if typable or not.
> done by type checker...

> can we prove...?
> `∀ Γ e, ∃ τ, (Γ ⊢ e : τ) ∨ ¬(Γ ⊢ e : τ)` -- a type inference algorithm!

> Provability in Coq witness decidabile operations.


### show term is "not typeable"

Keep inversion till the contradiction.



