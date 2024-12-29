---
title: "「SF-PLF」11. TypeChecking"
subtitle: "Programming Language Foundations - A Typechecker for STLC"
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

> The `has_type` relation is good but doesn't give us a _executable algorithm_ -- 不是一个算法
> but it's _syntax directed_, just one typing rule for one term (unique typing) -- translate into function!


Comparing Types
---------------

首先我们需要 check equality for types.
这里非常简单，如果是 SystemF 会麻烦很多，对 `∀` 要做 local nameless 或者 alpha renaming:

```coq
Fixpoint eqb_ty (T1 T2:ty) : bool :=
  match T1,T2 with
  | Bool, Bool ⇒
      true
  | Arrow T11 T12, Arrow T21 T22 ⇒
      andb (eqb_ty T11 T21) (eqb_ty T12 T22)
  | _,_ ⇒
      false
  end.
```

然后我们需要一个 refl 和一个 reflection，准确得说：「define equality by computation」，反方向用 refl 即可易证

```coq
Lemma eqb_ty_refl : ∀T1,
  eqb_ty T1 T1 = true.

Lemma eqb_ty__eq : ∀T1 T2,
  eqb_ty T1 T2 = true → T1 = T2.
```



The Typechecker
---------------

直接 syntax directed，不过麻烦的是需要 pattern matching `option`...

```coq
Fixpoint type_check (Gamma : context) (t : tm) : option ty :=
  match t with
  | var x =>
      Gamma x
  | abs x T11 t12 =>
      match type_check (update Gamma x T11) t12 with     (** <-- 对应 t12 的 rule **)
      | Some T12 => Some (Arrow T11 T12)                 
      | _ => None
      end
  | app t1 t2 =>
      match type_check Gamma t1, type_check Gamma t2 with
      | Some (Arrow T11 T12),Some T2 =>
          if eqb_ty T11 T2 then Some T12 else None       (** eqb_ty 见下文 **)
      | _,_ => None
      end
  ...
```

在课堂时提到关于 `eqb_ty` 的一个细节（我以前也经常犯，在 ML/Haskell 中……）：
我们能不能在 pattern matching 里支持「用同一个 binding 来 imply 说他们两需要 be equal」？

```coq
(** instead of this **)
| Some (Arrow T11 T12),Some T2 => if eqb_ty T11 T2 then ...

(** can we do this? **)
| Some (Arrow T   T' ),Some T  => ...
```

> the answer is __NO__ because this demands a _decidable equality_. 
> 我好奇的是，用 typeclass 是不是就可以 bake in 这个功能了？尤其是在 Coq function 还是 total 的情况下






Digression: Improving the Notation
----------------------------------

这里我们可以自己定义一个 Haskell `do` notation 风格的 _monadic_ notation:

```coq
Notation " x <- e1 ;; e2" := (match e1 with
                              | Some x ⇒ e2
                              | None ⇒ None
                              end)
         (right associativity, at level 60).

Notation " 'return' e "
  := (Some e) (at level 60).

Notation " 'fail' "
  := None.
```

好看一些吧反正：

```coq
Fixpoint type_check (Gamma : context) (t : tm) : option ty :=
  match t with
  | var x ⇒
      Gamma x 
  | abs x T11 t12 ⇒
      T12 <- type_check (update Gamma x T11) t12 ;;
      return (Arrow T11 T12)
  | app t1 t2 ⇒
      T1 <- type_check Gamma t1 ;;
      T2 <- type_check Gamma t2 ;;
      match T1 with 
      | Arrow T11 T12 ⇒ if eqb_ty T11 T2 then return T12 else fail
      | _ ⇒ fail
      end
```


Properties
----------

最后我们需要验证一下算法的正确性：
这里的 soundness 和 completess 都是围绕 "typechecking function ~ typing relation inference rule" 这组关系来说的：

```coq
Theorem type_checking_sound : ∀Gamma t T,
  type_check Gamma t = Some T → has_type Gamma t T.

Theorem type_checking_complete : ∀Gamma t T,
  has_type Gamma t T → type_check Gamma t = Some T.

```



Exercise
--------

给 `MoreStlc.v` 里的 StlcE 写 typechecker, 然后 prove soundness / completeness （过程中用了非常 mega 的 tactics）

```coq
(** 还不能这么写 **)
| fst p =>
    (Prod T1 T2) <- type_check Gamma p ;;


(** 要这样……感觉是 notation 的缘故？并且要提供 fallback case 才能通过 exhaustive check 是真的 **)
| fst p =>
    Tp <- type_check Gamma p ;;
    match Tp with
    | (Prod T1 T2) => T1
    | _ => fail
    end.
```


Extra Exercise (Prof.Mtf)
-------------------------

> I believe this part of exercise was added by Prof. Fluet (not found in SF website version)

给 `MoreStlc.v` 的 operational semantics 写 Interpreter (`stepf`), 然后 prove soundness / completeness... 


### `step` vs. `stepf` 

首先我们定义了 `value` 关系的函数版本 `valuef`，
然后我们定义 `step` 关系的函数版本 `stepf`:

以 pure STLC 为例：

```coq
Inductive step : tm -> tm -> Prop :=
  | ST_AppAbs : forall x T11 t12 v2,
         value v2 ->
         (app (abs x T11 t12) v2) --> [x:=v2]t12
  | ST_App1 : forall t1 t1' t2,
         t1 --> t1' ->
         (app t1 t2) --> (app t1' t2)
  | ST_App2 : forall v1 t2 t2',
         value v1 ->
         t2 --> t2' ->
         (app v1 t2) --> (app v1 t2')
```
```coq
Fixpoint stepf (t : tm) : option tm :=
  match t with
  | var x        => None (* We only define step for closed terms *)
  | abs x1 T1 t2 => None (* Abstraction is a value *)
  | app t1 t2    =>
    match stepf t1, stepf t2, t1 with
    | Some t1', _       , _           =>                     Some (app t1' t2)
    | None    , Some t2', _           => assert (valuef t1) (Some (app t1 t2')) (* otherwise [t1]      is a normal form *)
    | None    , None    , abs x T t11 => assert (valuef t2) (Some ([x:=t2]t11)) (* otherwise [t1], [t2] are normal forms *)
    | _       , _       , _           =>                     None
    end

Definition assert (b : bool) (a : option tm) : option tm := if b then a else None.
```

1. 对于关系，一直就是 implicitly applied 的，在可用时即使用。
   对于函数，我们需要手动指定 match 的顺序

2. `stepf t1 => None` 只代表这是一个 `normal form`，但不一定就是 `value`，还有可能是 stuck 了，所以我们需要额外的 `assert`ion. (失败时返回异常)
   __dynamics__ 本身与 __statics__ 是正交的，在 `typecheck` 之后我们可以有 `progress`，但是现在还没有



### Soundness

```coq
Theorem sound_stepf : forall t t',
    stepf t = Some t'  ->  t --> t'.
```

证明用了一个 given 的非常夸张的 automation...

不过帮助我找到了 `stepf` 和 `step` 的多处 inconsistency: 
- 3 次做 `subst` 时依赖的 `valuef` 不能省
- `valuef pair` 该怎么写才合适？
  最后把 `step` 中的 `value p ->` 改成了 `value v1 -> value v2 ->`，
  因为 `valuef (pair v1 v2)` 出来的 `valuef v1 && valuef v2` 比较麻烦。
  但底线是：__两者必须 consistent！__ 这时就能感受到 Formal Methods 的严谨了。


### Completeness

发现了 pair 实现漏了 2 个 case……然后才发现了 `Soundness` 自动化中的 `valuef pair` 问题



Extra (Mentioned)
-----------------
-----

[Church Style vs. Curry Style](https://lispcast.com/church-vs-curry-types/)
[Rice's Theorem](https://en.wikipedia.org/wiki/Rice%27s_theorem)

CakeML 
- prove correctness of ML lang compiler
- latest paper on verifying GC
