---
layout: post
title: "Avoiding success at all cost"
subtitle: 'Watching "Escape from the Ivory Tower: The Haskell Journey"'
author: "Hux"
header-style: text
lang: en
tags:
  - Haskell
  - 笔记
  - En
---

"Avoiding success at all cost" is the informal motto behinds [Haskell](https://www.haskell.org/). It could be parenthesized in two ways, either "Avoiding (success at all cost)" or "(Avoiding sucess) (at all cost)". 

I'm not going to interpret them directly but rather to share some thoughts on "the success vs. costs" basing on my very own understanding and experience.

### The success vs. cost of language design

There're always trade offs (or compromises) in any software design, and programming language design has no exceptions.

In other words, all language design decision that made them "successful" i.e. being popular and widely-used in industry or education for some reasons, all comes with their own "costs": being unsafe, limited expressiveness, or having bad performance, etc.

Whether or not the "cost" is a problem really depends on scenarios, or their goals. For instances, Python/JavaScript are both very expressive and beginner-friendly by being dynamically-typed, sacrifing the type safety and performance. Java, in constrast, uses a much safer and optimization-friendly type system but being much less expressive. Another typicial comparison would be memory management in programming languages, where languages that are "managed" (by either ARC or Gabage Collector) could be much easier and safer (in terms of memory) for most programmers but also considerred slower than languages that are "closer to the metal". 

None of these "costs", or "differences", really prevent them from being immortally popular.

For Haskell, the story becomes quite different: being research-oriented means the goal of this language is to pursue some "ultimate" things: the "ultimate" simplicity of intermediate representation, the "ultimate" type system where safety and expressiveness can coexist, the "ultimate" compilation speed and runtime performance, the "ultimate" concise and elegant concrete syntax, the "ultimate"...I don't know. But it has to be some "ultimate" things that is very difficult, probably endless and impossible, to achieve. 

This, as a result, made all language decisions in Haskell became very hard and slow, because **almost nothing can be scarified**. That's why Haskell insisted to be lazy to "guard" the purity regardless of some problems of being "call-by-need"; a decent IO mechanisms is missing in the first 4 yrs after the project's start until P Walder found _Monad_; and the _Type Class_, which is first proposed in P Walder's 1989 paper, spent yrs long to implement and popularize.

As a side note though, it doesn't mean there is no compromise in Haskell at all. It's just as minimized as it could be during its progress. When one audience asking why we have Haskell and OCaml, which're quite similar in very high level, both survived, SPJ replies:

> There's just a different set of compromises.

### The success vs. cost of language design process

Another common but extremely controversial (if not the most) topics of programming language design is about its design process: Would you prefer dictatorship or a committee (in other words, a dictatorship of many?)? Would you prefer being proprietary or standardized? In which form would you write the standards, in human nature language, pseudo code, or formal semantics? How many and how frequently breaking changes dare you make? Would you let open source community involve in?  

Again, I think there is no THE answer for all those questions. Majority of popular programming languages came and are still on going with very different paths.

Python, whose creater, Guido van Rossum, known as the "Benevolent Dictator For Life" (BDFL), i.e. good kind of dictator, still play the central role (until July 2018) of the Python's development after Python getting popular and adapt a open source and community-based development model. This factor direcly contribute to the fact that Python 3, as a breaking (not completely backward-compatible and not easy to port) but good (in terms of language design and consistency) revision of the language can still be landed, despite of many communities' pressures. There're many language (Ruby, Perl, Elm) also choose to follow this route.

JavaScript, widely known as being created by Brendan Eich in 10 days, in comparision, quickly involved into a committee (TC39) and standardized (ECMAScript) language due to both the open nature of the Web and fast adoption of itself. But Brendan, as the creater, wasn't even powerful enough to push the committee landing ES4, which is also a breaking but much better revision, but ended up with the ES5 (Harmony), a backward-compatible, yet much less ambitious version due to many political "fights" between different parties (e.g. Mozilla, Microsoft, Yahoo etc.) thus the history wasn't changed. Even the latest rising and yearly releasing of the "modern" JavaScript (ES6 or ES2015, 2016, 2017...) are mainly driven by the new generation of committee parties (+ Google, Facebook, Airbnb etc.) and still in a very open and standardized way.

As you can see here, even the history and progress of two rather similar languages can be so different, not to mention more proprietary languages such as Java from Sun/Oracle, C# from Microsoft, OC/Swift from Apple (though the latter was open sourced) or more academia and standardized language like SML and Scheme which both has a standard written in formal semantics.

So it's not not obvious that Haskell, also chose its own unique process to suit its unique goal. Although it backs on academia, it chose a rather practical/less-formal approach to define the language, i.e. the compiler implementation over standardization (plus many "formal" fragments among papers though), which is more like C++/OCaml from this point of view. It has a committee, but instead of being very open and conservative, it's more dictatorial (in terms of average users) and super aggressive in terms of making breaking changes. As a result however, it trained a group of very change-tolerant people in its community...All of these quirks and odds combined works very well and avoid the Haskell "becoming too success too quickly".


### End thoughts

To be fair, Haskell has alreay been very "successful" nowdays, in particular academia (for education, sexy type laboratory etc.) but also industry, either being used in real business or being very reputable among programmers (as being both hard and fun).

I am not confident and qualified to say Haskell is success in the right degree at the right time. But it's great to see it, after more than 20 and now almost 30 yrs, slowly figure out its very own way, to "Escape from the Ivory Tower", and keep going beyond.
