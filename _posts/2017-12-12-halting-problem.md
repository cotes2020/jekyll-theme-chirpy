---
layout: post
title: "如何通俗地解释停机问题？"
subtitle: "How to explain the Halting Problem?"
author: "Hux"
header-img: "img/post-bg-halting.jpg"
header-mask: 0.3
tags:
  - 知乎
  - 计算理论
---

> 这篇文章转载自[我在知乎上的回答]( https://www.zhihu.com/question/20081359/answer/275107187)

我用 Python 伪代码来解释下，我觉得对这个问题有兴趣的应该都是有点编程基础的，所以直接上 code 应该是最容易的。

## 背景知识

「停机问题」研究的是：是否存在一个「程序」，能够判断另外一个「程序」在特定的「输入」下，是会给出结果（停机），还是会无限执行下去（不停机）。

在下文中，我们用「函数」来表示「程序」，「函数返回」即表示给出了结果。

## 正文

我们假设存在这么一个「停机程序」，不管它是怎么实现的，但是它能够回答「停机问题」：它接受一个「程序」和一个「输入」，然后判断这个「程序」在这个「输入」下是否能给出结果：

```py
def is_halt(program, input) -> bool:
  # 返回 True  如果 program(input) 会返回
  # 返回 False 如果 program(input) 不返回
```

（在这里，我们通过把一个函数作为另一个函数的输入来描述一个「程序」作为另一个「程序」的「输入」，如果你不熟悉「头等函数」的概念，你可以把所有文中的函数对应为一个具备该函数的对象。）

为了帮助大家理解这个「停机程序」的功能，我们举个使用它的例子：

```py
from halt import is_halt

def loop():
  while(True):
      pass

# 如果输入是 0，返回，否则无限循环
def foo(input):
  if input == 0:
    return
  else:
    loop()

is_halt(foo, 0)  # 返回 True
is_halt(foo, 1)  # 返回 False
```

是不是很棒？

不过，如果这个「停机程序」真的存在，那么我就可以写出这么一个「hack 程序」：

```py
from halt import is_halt

def loop():
  while(True):
      pass

def hack(program):
  if is_halt(program, program):
    loop()
  else:
    return
```

这个程序说，如果你 `is_halt` 对 `program(program)` 判停了，我就无限循环；如果你判它不停，我就立刻返回。

那么，如果我们把「hack 程序」同时当做「程序」和「输入」喂给「停机程序」，会怎么样呢？

```py
is_halt(hack, hack)
```

你会发现，如果「停机程序」认为 `hack(hack)` 会给出结果，即 `is_halt(hack, hack)`) 返回 `True`) ，那么实际上 `hack(hack)` 会进入无限循环。而如果「停机程序」认为 `hack(hack)` 不会给出结果，即 `is_halt(hack, hack)` 返回 ，那么实际上 `hack(hack)` 会立刻返回结果……

这里就出现了矛盾和悖论，所以我们只能认为，我们最开始的假设是错的：

**这个「停机程序」是不存在的。**

## 意义

简单的说，「停机问题」说明了现代计算机并不是无所不能的。

上面的例子看上去是刻意使用「自我指涉」来进行反证的，但这只是为了证明方便。实际上，现实中与「停机问题」一样是现代计算机「不可解」的问题还有很多，比如所有「判断一个程序是否会在某输入下怎么样？」的算法、Hilbert 第十问题等等，wikipedia 甚至有一个 [List of undecidable problems](https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/List_of_undecidable_problems)。

## 漫谈

如果你觉得只是看懂了这个反证法没什么意思：

1.  最初图灵提出「停机问题」只是针对「图灵机」本身的，但是其意义可以被推广到所有「算法」、「程序」、「现代计算机」甚至是「量子计算机」。
2.  实际上「图灵机」只能接受（纸带上的）字符串，所以在图灵机编程中，无论是「输入」还是另一个「图灵机」，都是通过编码来表示的。
3.  「图灵机的计算能力和现代计算机是等价的」，更严谨一些，由于图灵机作为一个假象的计算模型，其储存空间是无限大的，而真实计算机则有硬件限制，所以我们只能说「不存在比图灵机计算能力更强的真实计算机」。
4.  这里的「计算能力」（power）指的是「能够计算怎样的问题」（capablity）而非「计算效率」（efficiency），比如我们说「上下文无关文法」比「正则表达式」的「计算能力」强因为它能解决更多的计算问题。
5.  「图灵机」作为一种计算模型形式化了「什么是算法」这个问题（[邱奇－图灵论题](https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/Church%25E2%2580%2593Turing_thesis)）。但图灵机并不是唯一的计算模型，其他计算模型包括「[Lambda 算子](https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/Lambda_calculus)」、[μ-递归函数](https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/%25CE%259C-recursive_function)」等，它们在计算能力上都是与「图灵机」等价的。因此，我们可以用「图灵机」来证明「[可计算函数](https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/Computable_function)」的上界。也因此可以证明哪些计算问题是超出上界的（即不可解的）。
6.  需要知道的是，只有「可计算的」才叫做「算法」。
7.  「停机问题」响应了「哥德尔的不完备性定理」。

## 拓展阅读：

中文：

- [Matrix67: 不可解问题(Undecidable Decision Problem)](https://link.zhihu.com/?target=http%3A//www.matrix67.com/blog/archives/55)

- [Matrix67: 停机问题、Chaitin 常数与万能证明方法](https://link.zhihu.com/?target=http%3A//www.matrix67.com/blog/archives/901)

- [刘未鹏：康托尔、哥德尔、图灵--永恒的金色对角线(rev#2) - CSDN 博客](https://link.zhihu.com/?target=http%3A//blog.csdn.net/pongba/article/details/1336028)

- [卢昌海：Hilbert 第十问题漫谈 (上)](https://link.zhihu.com/?target=http%3A//www.changhai.org/articles/science/mathematics/hilbert10/1.php)

英文：

- [《Introduction to the Theory of Computation》](https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/Introduction_to_the_Theory_of_Computation)

- [Turing Machines Explained - Computerphile](https://link.zhihu.com/?target=https%3A//www.youtube.com/watch%3Fv%3DdNRDvLACg5Q)

- [Turing & The Halting Problem - Computerphile](https://link.zhihu.com/?target=https%3A//www.youtube.com/watch%3Fv%3DmacM_MtS_w4%26t%3D29s)

- [Why, really, is the Halting Problem so important?](https://link.zhihu.com/?target=https%3A//cs.stackexchange.com/questions/32845/why-really-is-the-halting-problem-so-important)
