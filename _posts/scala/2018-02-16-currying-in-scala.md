---
title: Currying in Scala
description: This simple example of function currying shows a simple use case as well as code snippets to show how function currying saves time. This quick hands-on example of currying, article shows a simple use case as well as code snippets to show how currying can save time.
tags: ["scala", "jvm"]
category: ["programming", "tutorial"]
date: 2018-01-02
permalink: '/scala/currying-in-scala/'
---

Currying is named after Haskell Curry, an American mathematician. He is known for his work in combinatory logic.

Currying is a means of transforming a function that takes more than one argument into a chain of calls to functions, each of which takes a single argument.

Let us consider the function below to calculate the final price of the product. The function takes in 3 parameters:

* VAT (`vat`) for the region
* Service charge (`serviceCharge`) of the shop
* Product price (`productPrice`)

```scala
def finalPrice(vat: Double,
        serviceCharge: Double,
        productPrice: Double): Double =
        productPrice + productPrice*serviceCharge/100 + productPrice*vat/100
```

But, if you think about the function finalPrice again, a shopkeeper has to provide all the above values time and again whenever he wants to calculate the final price. Of course, that ignores the fact that the values of:

* VAT is already defined for a country
* Service charge for a shop is constant

So we will try to make life of our client a little easier. Let us define curried `finalPrice`:
```scala
def finalPriceCurried(vat: Double)
(serviceCharge: Double)
(productPrice: Double): Double = 
productPrice + productPrice*serviceCharge/100 + productPrice*vat/100
```

We are taking this approach because our vat and `serviceCharge` will not change very often. So, let's use currying to split our method. We will declare a new `val: vatApplied`. I will provide the value of vat to the `finalPriceCurried` method and assign it to `vatApplied`.

```scala
val vatApplied = finalPriceCurried(20) _
```

Next, we will provide a service charge to my `vatApplied` val, and we will leave the `price` to be provided by the shopkeeper whenever they need it.

```scala
val serviceChargeApplied = vatApplied(12.5)
```

Let us test our `serviceChargeApplied` function to calculate the final price of the product.

```scala
val finalProductPrice = serviceChargeApplied(120)

finalProductPrice: Double = 159.0
```
We have reduced our method from accepting 3 parameters to accept one parameter. So, we have split our method in such a way that we don't have to provide all the arguments at the same time. I can provide these arguments whenever they are available. This transformation is called currying.

We can also convert our existing methods to curry methods using function curried method. I have seen currying used a lot in my production code. It is always good to understand how these things work. You can refer to the video below to understand the concept in more detail and to check out few more examples.

<iframe src="https://www.youtube.com/embed/txNAZXPSbiE"></iframe>
