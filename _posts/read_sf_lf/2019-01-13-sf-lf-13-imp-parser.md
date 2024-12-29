---
title: "「SF-LC」13 ImpParser"
subtitle: "Logical Foundations - Lexing And Parsing In Coq"
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

> the parser relies on some "monadic" programming idioms

basically, _parser combinator_ (But 非常麻烦 in Coq)


Lex
---

```coq
Inductive chartype := white | alpha | digit | other.

Definition classifyChar (c : ascii) : chartype :=
  if      isWhite c then white
  else if isAlpha c then alpha
  else if isDigit c then digit
  else                   other.
  

Definition token := string.
```




Syntax
------

带 error msg 的 `option`:

```coq
Inductive optionE (X:Type) : Type :=
  | SomeE (x : X)
  | NoneE (s : string).       (** w/ error msg **)

Arguments SomeE {X}.
Arguments NoneE {X}.
```


Monadic: 

```coq
Notation "' p <- e1 ;; e2"
   := (match e1 with
       | SomeE p ⇒ e2
       | NoneE err ⇒ NoneE err
       end)
   (right associativity, p pattern, at level 60, e1 at next level).

Notation "'TRY' ' p <- e1 ;; e2 'OR' e3"
   := (match e1 with
       | SomeE p ⇒ e2
       | NoneE _ ⇒ e3
       end)
   (right associativity, p pattern,
    at level 60, e1 at next level, e2 at next level).
```


```coq
Definition parser (T : Type) :=
  list token → optionE (T * list token).
```

```haskell
newtype Parser a = Parser (String -> [(a,String)])

instance Monad Parser where
   -- (>>=) :: Parser a -> (a -> Parser b) -> Parser b
   p >>= f = P (\inp -> case parse p inp of
                           []        -> []
                           [(v,out)] -> parse (f v) out)
```


### combinator `many` 

Coq vs. Haskell 
1. explicit recursion depth, .e. _step-indexed_
2. explicit exception `optionE`  (in Haskell, it's hidden behind the `Parser` Monad as `[]`)
3. explicit string state `xs`    (in Haskell, it's hidden behind the `Parser` Monad as `String -> String`)
4. explicit `acc`epted token     (in Haskell, it's hidden behind the `Parser` Monad as `a`, argument)

```coq
Fixpoint many_helper {T} (p : parser T) acc steps xs :=
  match steps, p xs with
  | 0, _ ⇒
      NoneE "Too many recursive calls"
  | _, NoneE _ ⇒
      SomeE ((rev acc), xs)
  | S steps', SomeE (t, xs') ⇒
      many_helper p (t :: acc) steps' xs'
  end.

Fixpoint many {T} (p : parser T) (steps : nat) : parser (list T) :=
  many_helper p [] steps.
```

```haskell
manyL :: Parser a -> Parser [a]
manyL p = many1L p <++ return []   -- left biased OR

many1L :: Parser a -> Parser [a]
many1L p = (:) <$> p <*> manyL p
-- or
many1L p = do x <- p
              xs <- manyL p
              return (x : xs)
```


### `ident`


```coq
Definition parseIdentifier (xs : list token) : optionE (string * list token) :=
  match xs with
  | [] ⇒ NoneE "Expected identifier"
  | x::xs' ⇒ if forallb isLowerAlpha (list_of_string x)
             then SomeE (x, xs')
             else NoneE ("Illegal identifier:'" ++ x ++ "'")
  end.
```

```haskell
ident :: Parser String
ident = do x  <- lower
           xs <- many alphanum
           return (x:xs)
```
