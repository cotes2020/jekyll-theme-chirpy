---
title: Partially Applied Functions in Scala
description: A great primer for Scala devs or functional programming buffs in general, check out how (and why) to make use of partially applied functions.
tags: ["scala", "jvm"]
category: ["programming", "tutorial"]
date: 2017-12-29
permalink: '/scala/partially-applied-functions/'
---

<hr>
Scala, like many other functional languages, allows developers to apply functions partially. What this means is that, when applying a function, a developer does not pass in all the arguments defined by the function. But, provides only for some of them, leaving remaining parameters to be passed later.

Once you have provided the required initial parameters, what you get back is a new function whose parameter list only contains those parameters from the original function that were left blank.

I will provide an example to explain the concept. Consider the example below, where we define a method to calculate product price after discount. The method takes in two parameters — the first is the `discount` to be applied, and the second is `product price`.

```scala
def calculateProductPrice(discount: Double, productPrice: Double): Double =
(1 - discount/100) * productPrice
```

We cannot ask shopkeepers to provide discounts every time. We will set the `discount` once for all the products.
```scala
val dicountApplied = calculateProductPrice(30, _: Double)
```

Notice how I have used placeholder syntax to inform Scala that I am going to provide the value of `productPrice` later on. Now we can use `discountApplied` again and again without being bothered about the value of the discount.

### Advantage
This technique has many advantages:

* We have reduced a method that used to accept multiple arguments to a function that accepts only a single argument, thus making it easier for consumers to use the method.

* We can safeguard our code by exposing only a partially applied function so that no one else can pass in incorrect arguments by mistake.

Below is a small tutorial video for more explanation and examples:

<iframe src="https://www.youtube.com/embed/9RYCGOKpk6E"></iframe>


