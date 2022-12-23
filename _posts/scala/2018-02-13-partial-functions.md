---
title: Partial Functions in Scala
description: Interested in learning some finer points of the Scala language? Read on to get a quick overview of Scala's partial functions, and how they aid in development. This quick hands-on lesson on partial functions in Scala and how they help in development. The article also provides partial function code snippet
tags: ["scala", "jvm"]
category: ["programming", "tutorial"]
date: 2018-02-13
permalink: '/scala/partial-functions/'
counterlink: 'scala-partial-functions/'
---


As the name suggests, partial functions are only partial implementations. They do not cover every possible scenario of incoming parameters. A partial function caters to only a subset of possible data for which it has been defined. In order to assist developers, if the partial function is defined for a given input, Scala's `PartialFunction` trait provides the `isDefinedAt` method. The `isDefinedAt` method can be queried if it can handle a given value.

Partial functions in Scala can be defined by using the `case` statement. Let us define a simple partial function, `squareRoot`. The function would take in a `double` input parameter and would `return` the square root.

```scala
val squareRoot: PartialFunction[Double, Double] = { 
    case d: Double if d > 0 => Math.sqrt(d) 
} 
```

As is evident from the above example, we are not aware what would happen if `d` is less than 0.

### Advantages
Consider this `list` of numbers having some values.

```scala
val list: List[Double] = List(4, 16, 25, -9)
```

If I use a simple `map` function with `Math.sqrt()`, then I'll get an annoying `NaN` at the end of my `result` list.

```scala
val result = list.map(Math.sqrt)
result: List[Double] = List(2.0, 4.0, 5.0, NaN)
```

We never intended to have a `NaN` value in our result. What could be worse? We could have got an exception.

Let us try to use our previously defined `squareRoot` partial function along with `collect`.

```scala
val result = list.collect(squareRoot)
result: List[Double] = List(2.0, 4.0, 5.0)
```

And this time, we can observe that we do not have any unwanted elements in our result list. Thus, partial functions can help us to get rid of any side effects.

There are other helpful functions such as `orElse` and `andThen` that can be used with partial functions.

Below is a nice tutorial that provides few more hands-on examples of partial functions in Scala.

<iframe src="https://www.youtube.com/embed/DJ3RbPYbNHg"></iframe>
