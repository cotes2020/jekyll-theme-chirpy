---
title: Lazy Evaluation in Scala
description: A quick tour of lazy evaluation to cover its benefits and drawbacks in Scala. This quick lesson in lazy evaluation covers its benefits and drawbacks your options to implement it in your Scala code.
tags: ["scala", "jvm"]
category: ["programming", "tutorial"]
date: 2018-02-16
permalink: '/scala/lazy-evaluation/'
counterlink: 'scala-lazy-evaluation/'
---

As per Wikipedia, “Lazy Evaluation is an evaluation strategy which delays the evaluation of an expression until its value is needed.” And today most of the modern programming languages support Lazy Evaluation. In contrast with strict or eager evaluation, which computes values as soon as possible, lazy evaluation can certainly deliver a few benefits, such as:

* Lazy evaluation can help to resolve circular dependencies
* It can provide performance enhancement by not doing calculations until needed — and they may not be done at all if the calculation is not used.
* It can increase the response time of applications by postponing the heavy operations until required.

However, lazy evaluation has the drawback that performance may not be predictable — because you cannot say exactly when the value is going to be evaluated.

Scala has the following features to support lazy evaluation.

1. Lazy vals
A `val` can be declared as lazy by using the lazy keyword. The value of val not be initialized until it is called.
   
```scala 
lazy val lval = 10
```

2. Call by name parameters 
A method in Scala can accept calls by name parameters. The value of the parameter will only be evaluated whenever it will be used within the method.

```scala
def method(n :=> Int)
```

3. Lazy sequences
Sometimes, we need to allow our client to use as much as they want, and lazy sequences address the problem. Lazy `sequence` or `streams` let you define a sequence without any upper bound.
   
```scala
import Stream.cons;
def stream(n: Int): Stream[Int] = Stream.cons(n, addStream(n+1))
```

The tutorial provides a detailed hands-on example of Scala's lazy evaluation.
<iframe src="https://www.youtube.com/embed/iromVyC0mDs"></iframe>

