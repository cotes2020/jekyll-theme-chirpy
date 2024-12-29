---
title: "「SF-PLF」13 References"
subtitle: "Programming Language Foundations - Typing Mutable References"
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

> Hux: this chapter is very similar to TAPL - ch13 References
> But under a "formal verification" concept, it's more interesting and practical and push you to think about it!


_computational effects_ - "side effects" of computation - _impure_ features
- assign to mutable variables (reference cells, arrays, mutable record fields, etc.)
- perform input and output to files, displays, or network connections; 
- make non-local transfers of control via exceptions, jumps, or continuations; 
- engage in inter-process synchronization and communication


> The main extension will be dealing explicitly with a 
> - _store_ (or _heap_) and 
> - _pointers_ (or _reference_) that name _store locations_, or _address_...

interesting refinement: type preservation



Definition
----------

forms of assignments:
- rare : Gallina   - No 
- some : ML family - Explicit _reference_ and _dereference_
- most : C  family - Implicit ...

For formal study, use ML's model.



Syntax
------

### Types & Terms

```coq
T ::= 
    | Nat
    | Unit
    | T → T
    | Ref T

t ::= 
    | ...                Terms
    | ref t              allocation
    | !t                 dereference
    | t := t             assignment
    | l                  location
```
```coq
Inductive ty : Type :=
  | Nat : ty
  | Unit : ty
  | Arrow : ty → ty → ty
  | Ref : ty → ty.

Inductive tm : Type :=
  (* STLC with numbers: *)
  ...
  (* New terms: *)
  | unit : tm
  | ref : tm → tm
  | deref : tm → tm
  | assign : tm → tm → tm
  | loc : nat → tm.         (** 这里表示 l 的方式是 wrap 一个 nat as loc **)
```


### Typing


                           Gamma |- t1 : T1
                       ------------------------                         (T_Ref)
                       Gamma |- ref t1 : Ref T1

                        Gamma |- t1 : Ref T11
                        ---------------------                         (T_Deref)
                          Gamma |- !t1 : T11

                        Gamma |- t1 : Ref T11
                          Gamma |- t2 : T11
                       ------------------------                      (T_Assign)
                       Gamma |- t1 := t2 : Unit


### Values and Substitution

```coq
Inductive value : tm → Prop :=
  ...
  | v_unit :     value unit
  | v_loc  : ∀l, value (loc l).  (* <-- 注意这里是一个 Π (l:nat) . value (loc l) *)
```

```coq
Fixpoint subst (x:string) (s:tm) (t:tm) : tm :=
  match t with
  ...
  | unit         ⇒ t
  | ref t1       ⇒ ref (subst x s t1)
  | deref t1     ⇒ deref (subst x s t1)
  | assign t1 t2 ⇒ assign (subst x s t1) (subst x s t2)
  | loc _        ⇒ t
  end.
```




Pragmatics
----------


### Side Effects and Sequencing

    r:=succ(!r); !r

can be desugar to

    (\x:Unit. !r) (r:=succ(!r)).
    
then we can write some "imperative programming"

    r:=succ(!r); 
    r:=succ(!r); 
    r:=succ(!r); 
    !r


### References and Aliasing

_shared reference_ brings __shared state_

    let r = ref 5 in
    let s = r in
    s := 82;
    (!r)+1


### Shared State

_thunks_ as _methods_ 

```haskell

    let c = ref 0 in
    let incc = \_:Unit. (c := succ (!c); !c) in
    let decc = \_:Unit. (c := pred (!c); !c) in (
      incc unit; 
      incc unit;          -- in real PL: the concrete syntax is `incc()`
      decc unit
    )

```


### Objects 

_constructor_ and _encapsulation_!

```haskell

    newcounter =
      \_:Unit.            -- add `(self, init_val)` would make it more "real"
        let c = ref 0 in  -- private and only accessible via closure (特权方法)
        let incc = \_:Unit. (c := succ (!c); !c) in
        let decc = \_:Unit. (c := pred (!c); !c) in
        { i=incc, 
          d=decc  }       -- return a "record", or "struct", or "object"!
          
```


### References to Compound Types (e.g. Function Type)

Previously, we use _closure_ to represent _map_, with _functional update_
这里的"数组" （这个到底算不算数组估计都有争议，虽然的确提供了 index 但是这个显然是 O(n) 都不知道算不算 random access...
并不是 in-place update 里面的数据的，仅仅是一个 `ref` 包住的 map 而已 （仅仅是多了可以 shared

其实或许 `list (ref nat)` 也可以表达数组？ 反正都是 O(n) 每次都 linear search 也一样……

```haskell

    newarray = \_:Unit. ref (\n:Nat.0)
    lookup = \a:NatArray. \n:Nat. (!a) n   
    update = \a:NatArray. \m:Nat. \v:Nat.
               let oldf = !a in
               a := (\n:Nat. if equal m n then v else oldf n);

```


### Null References

_nullptr_!

Deref a nullptr:
- exception in Java/C#
- insecure in C/C++     <-- violate memory safety!!

```haskell

    type Option T   = Unit + T
    type Nullable T = Option (Ref T)

```


Why is `Option` outside?
think about C, `nullptr` is A special _const_ location, like `Unit` (`None` in terms of datacon) here.


### Garbage Collection

last issue: store _de-allocation_ 

> w/o GC, extremely difficult to achieve type safety...if a primitive for "explicit deallocation" provided
> one can easily create _dangling reference_ i.e. references -> deleted

One type-unsafe example: (pseudo code)

```haskell

   a : Ref Nat = ref 1;       -- alloc loc 0
   free(a);                   -- free  loc 0
   b : Ref Bool = ref True;   -- alloc loc 0
   
   a := !a + 1                -- BOOM!

```





Operational Semantics
---------------------


### Locations

> what should be the _values_ of type `Ref T`? 

`ref` allocate some memory/storage!

> run-time store is essentially big array of bytes.
> different datatype need to allocate different size of space (region)

> we think store as _array of values_, _abstracting away different size of different values_
> we use the word _location_ here to prevent from modeling _pointer arithmetic_, which is un-trackable by most type system

location `n` is `float` doesn't tell you anything about location `n+4`...
 


### Stores

we defined `replace` as `Fixpoint` since it's computational and easier. The consequence is it has to be total.



### Reduction


   


Typing
------

typing context:

```coq
Definition context := partial_map ty.
```

### Store typings

why not just make a _context_ a map of pair? 
we don't want to complicate the dynamics of language,
and this store typing is only for type check.



### The Typing Relation






Properties
----------

### Well-Typed Stores

### Extending Store Typings

### Preservation, Finally

### Substitution Lemma

### Assignment Preserves Store Typing

### Weakening for Stores

### Preservation!

### Progress





References and Nontermination
-----------------------------
