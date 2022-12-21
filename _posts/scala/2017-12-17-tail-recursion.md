---
title: Tail Recursion in Scala
description: Sure, recursion can be error-prone, but that's what tail recursion tries to solve. See how Scala helps developers out with recursive code.
tags: ["scala", "jvm"]
category: ["programming", "tutorial"]
date: 2017-12-17
permalink: '/scala/tail-recursion/'
---

<hr>

### Introduction
Recursion is quite common in the programming world. As you probably know, it's the process of solving a problem by breaking it down into smaller subproblems. You can easily spot recursion if you see a method calling itself with a smaller subset of inputs.

### Why Recursion?
Many programmers consider recursion tough and error-prone, but in Scala, we use recursion because:

* It avoids the mutable variables that you need to use while writing loops.
* It is a natural way of describing many algorithms.

### Problems With Recursion
Even though recursion can help you eliminate mutable variables from code, there are still some problems with it. As a developer you should take care of:

* Memory consumption in each recursive call.
* The fact that deep recursion can blow up your stack.

### Tail Recursion
In order to avoid the problems of recursion, we can use tail recursion. So, the first question is, <em>"What is tail recursion?"</em> If the recursive call is the last operation performed by the function, and no operations need to be saved when the function returns, that is called tail recursion.
If there is nothing on the stack frame, the compiler can reuse the stack frame and will convert the recursion into a loop.

### Scala Support of Tail Recursion
Scala can guide developers through tail recursion. You can use `@tailrec` to check if the recursion is tail recursive or not. The annotation is available as a part of the scala.annotation._ package. If the recursion is not tail recursive, then Scala will throw a compile-time error.

People who are really keen on hands-on examples can refer to YouTube link tutorial below.

<iframe src="https://www.youtube.com/embed/hjY_mC7dPxc"></iframe>
