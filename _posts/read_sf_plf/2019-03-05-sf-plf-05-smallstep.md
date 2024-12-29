---
title: "「SF-PLF」5 Smallstep"
subtitle: "Programming Language Foundations - Small-Step Operational Semantics"
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


Recall Big-step Pros & Cons
---------------------------

## Big-step

> 一步到位 :  _eval to its final value (plus final store)_ 

### Pros - natural (so called _natural semantics_), "all in one big step"

### Cons - not catch the _essence of how program behave_ 

> 大步语义只是一个 `程序 ↦ 结果` 这样的 pair 集合，而「如何一步步处理」才是程序「执行」的本质

not just input state get mapped to output state.
but also _intermediate state_ (which could be observed by _concurrent_ code!)


### Cons - not technically expressive enough to express _exception / crash / non-termination_ 

> 比如说，大步语义无法区分「不停机」与「卡住」
> two quite different reasons of "fail to map a given state to any ending state"

1. 不停机 nontermination - we want to allow this (infinite loop is the price paid for usability)
2. 卡住 getting stuck / undefiend behaviour 未定义行为  - we want to prevent (wrong)

- `WHILE_true_nonterm` 仅仅表达了「程序不能再 take step」，无法与「卡住」区分
- `WHILE_true` 更是直接让任何「无限循环」的程序都「等价」了...而忽略了中间状态和 effect (作用)

> we need _a way of presenting semantics that distinguish_ nontermination from erroneous "stuck states"


## Small-step 

> 更精细化 :  a _finer-grained_ way of defining and reasoning about program behaviors. 
> 原子步骤 :  _"atomic steps"_ of computation are performed. 







A Toy Language
--------------

Only Constant and Plus

```coq
Inductive tm : Type :=
  | C : nat → tm (* Constant *)
  | P : tm → tm → tm. (* Plus *)
```
  
### Big-Step

`==>` is really `⇓`
  
                ---------        (E_Const)
                C n ==> n

                t1 ==> n1
                t2 ==> n2
            -------------------  (E_Plus)
            P t1 t2 ==> n1 + n2
  

### Small-Step

> single reduction step
> find leftmost redex

      -------------------------------   (ST_PlusConstConst)
      P (C n1) (C n2) --> C (n1 + n2)

              t1 --> t1'
          --------------------          (ST_Plus1)
          P t1 t2 --> P t1' t2

              t2 --> t2'
      ----------------------------      (ST_Plus2)
      P (C n1) t2 --> P (C n1) t2'






Relations
---------

> Check notes of `rel` and `tactics` for more details about bi-relation.


### Deterministic 确定性

> a.k.a Partial Function. 
> in terms of its _right uniqueness_ under mathematical context, not its emphasise on _partial_ under programming context)

```coq
Definition deterministic {X : Type} (R : relation X) :=
  ∀x y1 y2 : X, R x y1 → R x y2 → y1 = y2.
```

`deterministic step` can be proved by induction on derivation `x --> y1` 
- use `generalize dependent y2`!
- in informal proof, we usually just take `∀ y2` by default.


### `Ltac solve_by_inverts n`

```coq
Ltac solve_by_inverts n :=
  match goal with | H : ?T ⊢ _ ⇒
  match type of T with Prop ⇒
    solve [
      inversion H;
      match n with S (S (?n')) ⇒ subst; solve_by_inverts (S n') end ]
  end end.
```


### Values 值

#### Abstract Machine 抽象机!

> think of the `-->` relation as defining an _abstract machine_:

- term = _state_ of machine 项 = 机器状态
- step = atomic unit of computation (think as assembly opcode / CPU instructrion)
- _halting state_ = no more computation. 停机状态

> execute a term `t`:

- starting state = `t`
- repeatedly use `-->` 
- when halt, _read out_ the _final state_ as result of execution

> Intutively, we call such (final state) terms _values_.
Okay so the point is...this language is simple enough (no stuck state).
and in this lang, value can only be `C`onst:

> 在这个语言中，我们「规定」只有 `C`onst 是「值」:

```coq
Inductive value : tm → Prop :=
  | v_const : ∀n, value (C n).
```

> and we can write `ST_Plus2` more elegant:
well...in this lang, not really, since only one form of value to write out.
in cases we have multiple form of value, by doing this we don't have to write out any cases.

             value v1
            t2 --> t2'
        --------------------  (ST_Plus2)
        P v1 t2 --> P v1 t2'



### Strong Progress and Normal Forms 强可进性和正规式


> _strong progress_: every term either is a value or can "make progress"

```coq
Theorem strong_progress : ∀t,
  value t ∨ (∃t', t --> t').
```


> terms that cannot make progress.
> for an arbitrary relation `R` over an arbitrary set `X`


> _normal form_: term that cannot make progress (take a step)
> 其实我个人比较喜欢理解为「常态」或「无能量稳定态」

```coq
Definition normal_form {X : Type} (R : relation X) (t : X) : Prop :=
  ¬∃t', R t t'.
```


> theorem: _in this language_, normal forms and values are actually the same thing. 

```coq
Lemma value_is_nf : v, value v → normal_form step v.
Lemma nf_is_value : ∀t, normal_form step t → value t.
Corollary nf_same_as_value : ∀t, normal_form step t ↔ value t.
```


#### Value != Normal Form (not always)

> value       is a _syntactic_ concept : it is defined by looking at the form of a term 
> normal form is a _semantic_  one     : it is defined by looking at how the term steps.


> E.g. we can defined term that can take a step as "value":
> 添加一个不是 normal form 的 value

```coq
Inductive value : tm → Prop :=
  | v_const : ∀n, value (C n)
  | v_funny : ∀t1 n2, value (P t1 (C n2)). (* <--- it can actually progress! *)
```

> 或者更改 `step` 让 value 不是 normal form...

```coq
Inductive step : tm -> tm -> Prop :=
  | ST_Funny : forall n,   
      C n --> P (C n) (C 0)                (* <--- or a weird  *)
```








Multi-Step Reduction `-->*` 多步规约
----------------------------------

> relation `multi R`: _multi-step closure of R_ 
> same as `clos_refl_trans_1n` in `Rel` chapter.

```coq
Inductive multi {X : Type} (R : relation X) : relation X :=
  | multi_refl : ∀(x : X), multi R x x
  | multi_step : ∀(x y z : X),
                    R x y →
                    multi R y z →
                    multi R x z.
```

以上是一种方便的定义，而以下则给了我们两个 helper 定理：

```coq
Theorem multi_R : ∀(X : Type) (R : relation X) (x y : X),
    R x y → 
    multi R x y.

Theorem multi_trans : ∀(X : Type) (R : relation X) (x y z : X),
    multi R x y →
    multi R y z →
    multi R x z.
```


### Normal Forms Again


```coq
Definition step_normal_form := normal_form step.  (** 这个是一个「性质」 Property : _ -> Prop , 从 polymorphic 的 [normal_form] 以 [step] 实例化而来 **) 
Definition normal_form_of (t t' : tm) :=          (** 是两个项之间的（i.e. 定义在 [tm] 集合上的) 二元关系, 即 t' 是 t 的正规式 **)
  (t -->* t' ∧ step_normal_form t').
  
Theorem normal_forms_unique:                      (** single-step reduction is deterministic 可以推出 normal form is unique for a given term **)
  deterministic normal_form_of.
```


### Normalizing 总是可正规化得  -- "Evaluating to completion"

> something stronger is true for this language (though not for all languages)
> reduction of _any_ term `t` will eventually reach a normal form (我们知道 STLC 也有这个特性)

```coq
Definition normalizing {X : Type} (R : relation X) :=
  ∀t, ∃t',
    (multi R) t t' ∧ normal_form R t'.
```

To prove this, we need lemma showing some _congruence_ of `-->*`: 
同余关系，不过这次是定义在 `-->*` 这个关系上，again，同余指的是「关系对于结构上的操作保持」

```coq
Lemma multistep_congr_1 : ∀t1 t1' t2,
     t1 -->* t1' → 
     P t1 t2 -->* P t1' t2.

Lemma multistep_congr_2 : ∀t1 t2 t2',
     value t1 →
     t2 -->* t2' →
     P t1 t2 -->* P t1 t2'.
```

Then we can prove...

```coq
Theorem step_normalizing :
  normalizing step.
```



### Equivalence of Big-Step and Small-Step

```coq
Theorem eval__multistep : ∀t n,
  t ==> n → t -->* C n.

Theorem multistep__eval : ∀t t',
  normal_form_of t t' → ∃n, t' = C n ∧ t ==> n.    (* might be better to say value here? *)
```




Additional: Combined Language
-----------------------------

What if we combined the lang `Arith` and lang `Boolean`?
Would `step_deterministic` and `strong_progress` still holds?

Intuition:
- `step_deterministic` should still hold
- but `strong_progress` would definitely not!!
  - now we mixed two _types_ so we will have stuck terms e.g. `test 5` or `tru + 4`.
  - we will need type check and then we would be able to prove `progress` (which require well-typeness)

```coq
Theorem strong_progress :
  (forall t, value t \/ (exists t', t --> t')) \/
  ~ (forall t, value t \/ (exists t', t --> t')).
Proof.
  right. intros Hcontra.
  remember (P tru fls) as stuck.   (** 类似 disprove equiv = 举一个反例就好 **)
  specialize (Hcontra stuck).
  destruct Hcontra as [Hcvalue | Hcprogress]; subst.
  - inversion Hcvalue; inversion H.
  - destruct Hcprogress. inversion H. inversion H3. inversion H4.
Qed.
```





Small-Step IMP 
--------------

又到了老朋友 IMP……还好没练习……简单看一下

首先对于定义小步语义，我们需要定义 `value` 和 `-->` (step)

### `aexp`, `bexp` 

```coq
Inductive aval : aexp → Prop :=
  | av_num : ∀n, aval (ANum n).
```

`bexp` 不需要 `value` 因为在这个语言里 `BTrue` 和 `BFalse` 的 step 总是 disjointed 得，所以并没有任何复用 `value` predicate 的时候


### `-->a`, `-->b` 

这里，我们先为 `aexp`, `bexp` 定义了它们各自的小步语义，

> 但是，其实 from PLT we know, 我们其实也可以直接复用 `aexp`, `bexp` 的大步语义！
> 1. 大步语义要短得多
> 2. `aexp`, `bexp` 其实并不会出
>   - 「不停机」: 没有 jump 等控制流结构
>   - 「异常」/「卡住」: 我们在 meta-language 的 AST 里就区分了 `aexp` 和 `bexp`，相当于主动约束了类型，所以不会出现 `5 || 3` 这样 type error 的 AST


### `cmd`, `-->`

> 我们把 `SKIP` 当作一个「命令值（command value）」 i.e. 一个已经到达 normal form 的命令。
> - 赋值命令归约到 `SKIP` （和一个新的 state）。
> - 顺序命令等待其左侧子命令归约到 `SKIP`，然后丢弃它，并继续对右侧子命令归约。

> 对 `WHILE` 命令的归约是把 `WHILE` 命令变换为条件语句，其后紧跟同一个 `WHILE` 命令。

> 这些都与 PLT 是一致的






Concurrent IMP
--------------

为了展示 小步语义 的能力，let's enrich IMP with concurrency.
- unpredictable scheduling (subcommands may be _interleaved_)
- _share same memory_

It's slightly confusing here to use `Par` (meaning _in parallel_) 
I mean, concurrency _could_ be in parallel but it doesn't have to...

```coq
Inductive com : Type :=
  | CPar : com → com → com. (* <--- NEW *)

Inductive cstep : (com * state) → (com * state) → Prop :=
  (* New part: *)
  | CS_Par1 : ∀st c1 c1' c2 st',
      c1 / st --> c1' / st' →
      (PAR c1 WITH c2 END) / st --> (PAR c1' WITH c2 END) / st'
  | CS_Par2 : ∀st c1 c2 c2' st',
      c2 / st --> c2' / st' →
      (PAR c1 WITH c2 END) / st --> (PAR c1 WITH c2' END) / st'
  | CS_ParDone : ∀st,
      (PAR SKIP WITH SKIP END) / st --> SKIP / st
```








A Small-Step Stack Machine  小步栈机
-----------------------------------

啊哈！IMP 章节 Stack Machine，我们之前仅仅定义了 `Fixpoint s_execute` 和 `Fixpoint s_compile`，这里给出其小步语义
> 对于本身就与「小步语义」在精神上更统一的「抽象机」，我怀疑其语义都应该是足够「小」的（即大小步将是一致的？)

```coq
Definition stack := list nat.
Definition prog  := list sinstr.

Inductive stack_step : state -> prog * stack -> prog * stack -> Prop :=
  | SS_Push : forall st stk n p',
    stack_step st (SPush n :: p', stk)      (p', n :: stk)
  | SS_Load : forall st stk i p',
    stack_step st (SLoad i :: p', stk)      (p', st i :: stk)
  | SS_Plus : forall st stk n m p',
    stack_step st (SPlus :: p', n::m::stk)  (p', (m+n)::stk)
  | SS_Minus : forall st stk n m p',
    stack_step st (SMinus :: p', n::m::stk) (p', (m-n)::stk)
  | SS_Mult : forall st stk n m p',
    stack_step st (SMult :: p', n::m::stk)  (p', (m*n)::stk).
    
(** closure of stack_step **)
Definition stack_multistep st := multi (stack_step st).
```

### Compiler Correctness

> 「编译器的正确性」= the notion of _semantics preservation_ (in terms of observable behaviours)
>   S  = `e`
>   C  = `s_compile e`
> B(S) = `aeval st e` 
> B(C) = functional `s_execute` 
>      | relational `stack_multistep` 

之前我们证明过 _functional/computational_ `Fixpoint` 的性质

```coq
Theorem s_compile_correct : forall (st : state) (e : aexp),
  s_execute st [] (s_compile e) = [ aeval st e ].

(** 重要的是这个更一般的「描述了 prog 如何与 stack 交互」的定理 **)
Theorem s_execute_theorem : forall (st : state) (e : aexp) (stack : list nat) (prog : list sinstr),
    s_execute st                  stack  (s_compile e ++ prog) 
  = s_execute st ((aeval st e) :: stack)                 prog.

```

现在则是证明 _relational_ `Inductive` 的性质，同样我们需要一个更一般的定理（然后原命题作为推论）

```coq
Theorem stack_step_theorem : forall (st : state) (e : aexp) (stack : list nat) (prog : list sinstr),
  stack_multistep st
                  ((s_compile e ++ prog),                 stack) 
                  (                prog , (aeval st e) :: stack).      (** 这里 prog 和 stack 的交互本质上和上面是一样的 **)
Proof.
  unfold stack_multistep.
  induction e; intros; simpl in *;        (** 证明 induction on aexp，然后利用 transivitiy、constructor 与 IH 即可，非常快 **)
    try (apply multi_R; constructor);
    try (
        repeat (rewrite <- app_assoc);
        eapply multi_trans; try apply IHe1;
        eapply multi_trans; try apply IHe2;
        eapply multi_R; constructor
      ).
      
Definition compiler_is_correct_statement : Prop := forall (st : state) (e : aexp),
  stack_multistep st (s_compile e, []) ([], [aeval st e]).
```






Aside: A `normalize` Tactic
---------------------------

Even with `eapply` and `auto`...manual normalization is tedious:

```coq
Example step_example1' :
  (P (C 3) (P (C 3) (C 4)))
  -->* (C 10).
Proof.
  eapply multi_step. auto. simpl.
  eapply multi_step. auto. simpl.
  apply multi_refl.
Qed.
```

We could write custom `Tactic Notation`...(i.e. tactic macros)

```coq
Tactic Notation "print_goal" :=
  match goal with ⊢ ?x ⇒ idtac x end.
  
Tactic Notation "normalize" :=
  repeat (print_goal; eapply multi_step ;
            [ (eauto 10; fail) | (instantiate; simpl)]);
  apply multi_refl.
```

`instantiate` seems here for intros `∃`?

```coq
Example step_example1''' : exists e',
  (P (C 3) (P (C 3) (C 4)))
  -->* e'.
Proof.
  eapply ex_intro. normalize.
Qed.
```

But what surprise me is that we can `eapply ex_intro`, which leave the `∃` as a hole `?ex` (unification variable).
