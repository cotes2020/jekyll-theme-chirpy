---
title: Repeated Method Parameters in Scala
description: Let's see how Scala supports variable arguments and repeated method parameters, and the conditions to consider when using them.
tags: ["scala", "jvm"]
category: ["programming", "tutorial"]
date: 2018-02-28
permalink: '/scala/scala-repeated-method-parameters/'
counterlink: 'scala-repeated-method-parameters/'
---

Similar to Java, Scala also supports variable arguments or repeated method parameters. The concept is really useful in situations when you don't know how many parameters you need to pass to a method, or you have to pass an unlimited number of arguments to a method.

However, there are few conditions to using repeated method parameters in Scala

* All the repeated parameters must be of the same type.

* We can only have one argument as a repeated parameter in the method definition. We cannot declare 2 repeated parameters for a method.

* Scala only allows the last parameter of the method call to be repeated.

To denote a repeated parameter, place an asterisk after the type of the parameter. For example, below is a sum method that would calculate the sum of all the numbers passed to the method.

```scala
def sum(args: Int*): Int = args.fold(0)(_+_)
```

You can call sum as `sum()` or `sum(3,4)` or `sum(1,3,4,5,7,8,9)`. Scala treats incoming parameters as arrays. However, if you try to pass an array to `sum()`, Scala will throw a type mismatch error.

```shell
scala> sum(Array(1,2))
<console>:13: error: type mismatch
 found   : Array[Int]
 required: Int
       sum(Array(1,2))
```

In order to pass an array, we need to append the argument with a colon and an` _*` symbol.

This notation will ask the compiler to pass each element of the array as a single argument. So array elements are passed one by one to `sum()`, rather than all of it as a single argument.

The video tutorial talks about the same concepts in more detail and provides few more examples.

<iframe src="https://www.youtube.com/embed/tyHswiV2gvk?list=PLiRMk2ipn1vqwRMy6NhroeQOmY1x1-rTH"></iframe>
