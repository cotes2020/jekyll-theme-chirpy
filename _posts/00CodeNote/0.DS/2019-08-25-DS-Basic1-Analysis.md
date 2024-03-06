---
title: Data Structures - Basic 1 - Analysis
# author: Grace JyL
date: 2019-08-25 11:11:11 -0400
description:
excerpt_separator:
categories: [00CodeNote, DS]
tags:
math: true
# pin: true
toc: true
# image: /assets/img/sample/devices-mockup.png
---

- [Data Structures - Basic 1 - Analysis](#data-structures---basic-1---analysis)
  - [Objectives](#objectives)
  - [Experimental Studies](#experimental-studies)
  - [Counting Primitive Operations](#counting-primitive-operations)
    - [The Seven Functions](#the-seven-functions)
      - [The Constant Function `f(n) = c`](#the-constant-function-fn--c)
      - [The Logarithm Function `f(n) = logb n`](#the-logarithm-function-fn--logb-n)
      - [The Linear Function `f(n) = n`](#the-linear-function-fn--n)
      - [The N-Log-N Function `f(n) = nlogn`](#the-n-log-n-function-fn--nlogn)
      - [The Quadratic Function `f(n) = n^2`](#the-quadratic-function-fn--n2)
      - [The Cubic Function and Other Polynomials `f(n) = n3`](#the-cubic-function-and-other-polynomials-fn--n3)
      - [The Exponential Function `f(n) = b^n`](#the-exponential-function-fn--bn)
  - [Asymptotic Analysis 渐近分析](#asymptotic-analysis-渐近分析)
    - [Big O Notation `f(n)≤c·g(n), for n≥n0` 最坏情况的度量](#big-o-notation-fncgn-for-nn0-最坏情况的度量)
    - [Big Omega Ω Notation `f(n) ≥ cg(n), for n ≥ n0.` 算法的最好情况](#big-omega-ω-notation-fn--cgn-for-n--n0-算法的最好情况)
    - [Big Theta Θ Notation `c′g(n) ≤ f (n) ≤ c′′g(n), for n ≥ n0.`](#big-theta-θ-notation-cgn--f-n--cgn-for-n--n0)
    - [Little O Notation](#little-o-notation)
    - [Little ω Omega Notation](#little-ω-omega-notation)
  - [Amortized analysis](#amortized-analysis)
    - [running time of operations on dynamic arrays.](#running-time-of-operations-on-dynamic-arrays)
  - [Comparative Analysis 对比分析](#comparative-analysis-对比分析)
    - [T(n)](#tn)
    - [Examples of Algorithm Analysis](#examples-of-algorithm-analysis)
      - [题目](#题目)
      - [O(1): Constant-Time Operations](#o1-constant-time-operations)
      - [O(㏒N): 二分查找算法](#on-二分查找算法)
      - [O(n): 线性算法 linear time](#on-线性算法-linear-time)
      - [O(N㏒N: 二分查找算法](#onn-二分查找算法)
      - [O($n^2$)](#on2)
      - [O($n^3$) 做两个n阶矩阵的乘法运算](#on3-做两个n阶矩阵的乘法运算)
      - [O($2^n$): 求具有n个元素集合的所有子集的算法](#o2n-求具有n个元素集合的所有子集的算法)
      - [O(n!): 求具有N个元素的全排列的算法](#on-求具有n个元素的全排列的算法)
  - [Simple Justification Techniques](#simple-justification-techniques)
    - [counterexample: By Example](#counterexample-by-example)
    - [contrapositive and the contradiction: The “Contra” Attack](#contrapositive-and-the-contradiction-the-contra-attack)
    - [Induction and Loop Invariants](#induction-and-loop-invariants)
- [3.5. Performance of Python Data Structures](#35-performance-of-python-data-structures)
  - [3.6. Lists](#36-lists)
  - [3.7. Dictionaries](#37-dictionaries)

---

# Data Structures - Basic 1 - Analysis

source:
- DS - pythonds3 - 3. Analysis
- Problem Solving with Algorithms and Data Structures using Python 3
- Data Structures and Algorithms in Java, 6th Edition.pdf

---

## Objectives

- **data structure** is a systematic way of organizing and accessing data
- **algorithm** is a step-by-step procedure for performing some task in a finite amount of time.
- to classify some data structures and algorithms as “good,” we must have precise ways of analyzing them.

what we really mean by computing resources.two different ways to look at this.

1. **Running time**
   1. `the amount of time` require to execute.
   2. “execution time, running time of the algorithm.
   3. One way to measure is to do a benchmark analysis. track the actual time required for the program to compute its result.
      1. In Python, we can benchmark a function by noting the starting time and ending time with respect to the system we are using.
      2. In the time module, function called time, will return the current system clock time in seconds since some arbitrary starting point.
      3. By calling this function twice, at the beginning and at the end, and then computing the difference, we can get an exact number of seconds (fractions in most cases) for execution.

1. **Space usage**
   1. `the amount of space or memory` an algorithm requires to solve the problem.
   2. The amount of space required by a problem solution is typically dictated by the problem instance itself.
   3. Every so often, however, there are algorithms that have very specific space requirements, and in those cases we will be very careful to explain the variations.


---

## Experimental Studies

- implement it and experiment by running the program on various test inputs while recording the time spent during each execution
  - However, the measured times reported by both methods `currentTimeMillis` and `nanoTime` will vary greatly from machine to machine, and may likely vary from trial to trial, even on the same machine.
  - because many processes share use of a computer’s **central processing unit (or CPU)** and **memory system**;
  - therefore, the elapsed time will depend on what other processes are running on the computer when a test is performed.
  - While the precise running time may not be dependable, experiments are quite useful when comparing the efficiency of two or more algorithms, so long as they gathered under similar circumstances.

- three major limitations to algorithm analysis:
  - Experimental running times of two algorithms are difficult to directly compare unless the experiments are performed in the same hardware and software environments.
  - Experiments can be done only on a limited set of test inputs; hence, they leave out the running times of inputs not included in the experiment (and these inputs may be important).
  - An algorithm must be fully implemented in order to execute it to study its running time experimentally.
    - the most serious drawback
    - At early stages of design, when considering a choice of data structures or algorithms, it would be foolish to spend a significant amount of time implementing an approach that could easily be deemed inferior by a higher-level analysis.


```py
long startTime = System.currentTimeMillis();
long endTime = System.currentTimeMillis();
long elapsed = endTime − startTime;


# =============== 1 =================
def sum_of_n_2(n):
    start = time.time()
    the_sum = 0
    for i in range(1, n + 1):
        the_sum = the_sum + i
    end = time.time()
    return the_sum, end - start

print("Sum is %d required %10.7f seconds" % sum_of_n_2(10000))
print("Sum is %d required %10.7f seconds" % sum_of_n_2(100000))
print("Sum is %d required %10.7f seconds" % sum_of_n_2(1000000))
# Sum is 50005000     required  0.0018950 seconds
# Sum is 5000050000   required  0.0199420 seconds
# Sum is 500000500000 required  0.1948988 seconds


# =============== 2 =================
def sum_of_n_3(n):
    return (n * (n + 1)) / 2
print(sum_of_n_3(10))
# Sum is 50005000         required 0.00000095 seconds
# Sum is 5000050000       required 0.00000191 seconds
# Sum is 500000500000     required 0.00000095 seconds
# Sum is 50000005000000   required 0.00000095 seconds
# Sum is 5000000050000000 required 0.00000119 seconds
```

> First, the times recorded above are shorter than any of the previous examples.
> Second, they are very consistent no matter what the value of n.
> It appears that sum_of_n_3 is hardly impacted by the number of integers being added.

> the iterative solutions is doing more work since some program steps are being repeated.



```java
public static String repeat1(char c, int n) {
    String answer = "";
    for (int j=0; j < n; j++) answer += c; return answer;
    return answer;
}

public static String repeat2(char c, int n) {
    StringBuilder sb = new StringBuilder();
    for (int j=0; j < n; j++) sb.append(c);
    return sb.toString();
}
```

![Screen Shot 2022-03-04 at 12.36.59](https://i.imgur.com/7qvnjE0.png)

![Screen Shot 2022-03-04 at 12.37.25](https://i.imgur.com/jvLO3uz.png)

---

## Counting Primitive Operations
- To analyze the running time of an algorithm without performing experiments, perform an analysis directly on a `high-level description of the algorithm`
- define a set of primitive operations such as the following:
  - Assigning a value to a variable
  - Following an object reference
  - Performing an arithmetic operation (for example, adding two numbers)
  - Comparing two numbers
  - Accessing a single element of an array by index
  - Calling a method
  - Returning from a method

**Focusing on the Worst-Case Input**
- An algorithm may run faster on some inputs than it does on others of the same size.
- to express the running time of an algorithm as the function of the input size obtained by taking the average over all possible inputs of the same size.
- Unfortunately, such an average-case analysis is typically quite challenging. It requires us to define a probability distribution on the set of inputs, which is often a difficult task.

---

### The Seven Functions

![Screen Shot 2022-03-04 at 18.59.27](https://i.imgur.com/jGC0vAA.png)

- Ideally, we would like `data structure operations` to run in times proportional to the **constant or logarithm function**
- would like our `algorithms` to run in **linear or n-log-n time**.
- Algorithms with quadratic or cubic running times are less practical,
- algorithms with exponential running times are infeasible for all but the smallest sized inputs.

- analysis of an algorithm may sometimes involve the use of the `floor function` and `ceiling function`, which are defined respectively as follows:

![Screen Shot 2022-03-04 at 19.01.37](https://i.imgur.com/TcZ7wKZ.png)


f(n) | Name
---|---
`1`      | <kbd>Constant</kbd>    表示算法的运行时间为常量
`log𝑛`   | <kbd>Logarithmic</kbd> 二分查找算法
`𝑛`      | <kbd>Linear</kbd>
`𝑛log𝑛`  | <kbd>Log Linear</kbd>
$𝑛^2$    | <kbd>Quadratic</kbd>   对数组进行排序的各种简单算法，例如直接插入排序的算法。
$𝑛^3$    | <kbd>Cubic</kbd>       做两个n阶矩阵的乘法运算
$2^𝑛$    | <kbd>Exponential</kbd> 求具有n个元素集合的所有子集的算法
O(n!)    | 求具有N个元素的全排列的算法

![Screen Shot 2021-10-26 at 3.07.39 AM](https://i.imgur.com/mwyIctk.png)

![newplot](https://i.imgur.com/wDUItRW.png)

![newplot2](https://i.imgur.com/ZE0qJ9I.png)

> when n is small, the functions are not very well defined with respect to one another.
> as n grows, there is a definite relationship and it is easy to see how they compare with one another.

优<---------------------------<劣

`O(1)<O(㏒n)<O(n)<O(n ㏒n)<O(n^2)<O(2^n)<O(n!)`

常数阶O(1)、对数阶O(log2n)、线性阶O(n)、线性对数阶O(nlog2n)、平方阶O(n2)、立方阶O(n3)、……k次方阶O(nk)、指数阶O(2n)。


---




#### The Constant Function `f(n) = c`
- The simplest functio
- for some fixed constant c, such as c=5, c=27, or c=210.
- for any argument n, the constant function f(n) assigns the value c.
- it does not matter what the value of n is; f(n) will always be equal to the constant value c.

- it characterizes the number of steps needed to do a basic operation on a computer, like
  - adding two numbers,
  - assigning a value to a variable,
  - or comparing two numbers.

#### The Logarithm Function `f(n) = logb n`
- the ubiquitous presence of the logarithm function, f(n) = logb n, for some constant b > 1.
- This function is defined as the inverse of a power, as follows:

```
x=logbn ifandonlyif b^x =n.
⌈log3 27⌉ = 3, because ((27/3)/3)/3 = 1;
⌈log4 64⌉ = 3, because ((64/4)/4)/4 = 1;
⌈log2 12⌉ = 4, because (((12/2)/2)/2)/2 = 0.75 ≤ 1.
```

- The value b is known as `the base of the logarithm`.
- for any base b > 0, we have that logb 1 = 0.
- The most common base for the logarithm function in computer science is 2 as computers store integers in binary. In fact, this base is so common that we will typically omit it from the notation when it is 2.: `log n = log2 n.`


#### The Linear Function `f(n) = n`
- given an input value n, the linear function f assigns the value n itself.
- This function arises in algorithm analysis any time we have to do a single basic operation for each of n elements.
- For example
  - comparing a number x to each element of an array of size n: require n comparisons.
- The linear function also represents the `best running time we can hope to achieve for any algorithm that processes each of n objects` that are not already in the computer’s memory, because reading in the n objects already requires n operations.


#### The N-Log-N Function `f(n) = nlogn`
- the function that assigns to an input n, the value of n times the logarithm base-two of n.
- This function grows a little more rapidly than the linear function and a lot less rapidly than the quadratic function;
- For example,
  - the fastest possible algorithms for sorting n arbitrary values require time proportional to n log n.


#### The Quadratic Function `f(n) = n^2`
- given an input value n, the function f assigns the product of n with itself (“n squared”).
- algorithms that have **nested loops**, where the inner loop performs a linear number of operations and the outer loop is performed a linear number of times.
  - the operations in the inner loop increase by one each time,
  - then the total number of operations is quadratic in the number of times, n, we perform the outer loop.
  - the algorithm performs n · n = n^2 operations.
- To be fair, the number of operations is `n^2/2 + n/2`
- over half the number of operations than an algorithm that uses n operations each time the inner loop is performed.
- But the order of growth is still quadratic in n.


#### The Cubic Function and Other Polynomials `f(n) = n3`

**Cubic Function**
- The cubic function appears less frequently in the context of algorithm analysis than the constant, linear, and quadratic functions previously mentioned, but it does appear from time to time.

**Polynomials**
- The linear, quadratic and cubic functions can each be viewed as being part of a larger class of functions, the polynomials.

`f(n) = a0 +a1n+a2n^2 +a3n3 +···+adn^d`

- a0 , a1 , . . . , ad are constants, called the **coefficients** of the polynomial, and ad ̸= 0.
- Integer d, which indicates the highest power in the polynomial, is called the **degree** of the polynomial.
- For example, the following functions are all polynomials:
  - f(n) = 2+5n+n2
  - f(n)=1+n^3
  - f(n)=1
  - f(n)=n
  - f(n)=n^2

**Summations**
- A notation that appears again and again in the analysis of data structures and algo- rithms is the summation, which is defined as follows:

![Screen Shot 2022-03-04 at 18.55.08](https://i.imgur.com/l85IEU2.png)


#### The Exponential Function `f(n) = b^n`
- where b is a positive constant, called the **base**,
- and the argument n is the **exponent**.
- That is, function f (n) assigns to the input argument n the value obtained by multiplying the base b by itself n times.
- As was the case with the logarithm function, the most common base for the exponential function in algorithm analysis is b = 2.
  - For example,
  - an integer word containing n bits can represent all the nonnegative integers less than 2n.
  - If we have a loop that starts by performing one operation and then doubles the number of operations performed with each iteration, then the number of operations performed in the nth iteration is 2n.
- the following exponent rules are quite helpful.
  1. (b^a)^c = b^ac
  2. b^ab^c=b^a+c
  3. b^a/b^c = b^a−c


**Geometric Sums**
- Suppose we have a loop for which each iteration takes a multiplicative factor longer than the previous one.
- This loop can be analyzed using the following proposition.

![Screen Shot 2022-03-04 at 18.58.48](https://i.imgur.com/i5isO72.png)

---





## Asymptotic Analysis 渐近分析


### Big O Notation `f(n)≤c·g(n), for n≥n0` 最坏情况的度量

![bigO](https://i.imgur.com/Sm0gaI0.png)

- 如果存在正数c和N，对于所有的n>=N，有f(n)<=c*g(n)，则f(n)=O(g(n))
- 求一个算法的worst-case，即是一个最坏情况的度量，求的是上界。
* used to describe the upper bound of a particular algorithm.
* Big O is used to describe **worst case scenarios**
- The big-Oh notation allows us to say that
  - a function f(n) is “less than or equal to” another function g(n) up to a constant factor and in the asymptotic sense as n grows toward infinity.
  - f(n) is order of g(n)
- **describe the function in the big-Oh in simplest terms.**

- use the names of these functions to refer to the running times of the algorithms
  - for example
    - an algorithm that runs in worst-case time `4n^2 + n log n` is a **quadratic-time algorithm**, since it runs in `O(n2)` time.
    - an algorithm running in time at most `5n + 20 log n + 4` would be called a **linear-time algorithm**.


**example**


> 5$n^4$ + $3n^3$ + $2n^2$ + 4n + 1 is O($n^4$).
> Justification: $5n^4$ + $3n^3$ + $2n^2$ + 4n+1 ≤ (5+3+2+4+1)$n^4$ =c$n^4$, for c=15, whenn≥n0 =1.

> If f (n) is a polynomial of degree d: f(n) = a0 +a1n+···+adnd, and ad > 0,
> Justification: a0 +a1n+a2n2 +···+adnd ≤ (|a0|+|a1|+|a2|+···+|ad|)n^d
> f(n) is O(n^d).

> 5n^2 + 3nlogn + 2n + 5 is O(n^2).
> Justification: 5n2 + 3nlogn + 2n + 5 ≤ (5+3+2+5)n^2 = cn^2,for c=15, when n≥n0 =1.


---


### Big Omega Ω Notation `f(n) ≥ cg(n), for n ≥ n0.` 算法的最好情况

- 如果存在正数c和N，对于所有的n>=N，有f(n)>=c*g(n)，则f(n)=Omega(g(n))

- 和Big O相反，这个玩意儿是很乐观的，求得是一个算法的最好情况，即下界，即best-case。

* used to provide an asymptotic lower bound on a particular algorithm

![bigOmega](https://i.imgur.com/ZzgElTm.png)


> 3n log n − 2n is Ω(n log n).
> Justification: 3nlogn−2n = nlogn+2n(logn−1) ≥ nlogn for n ≥ 2;
> hence, c=1 and n0 =2


---

### Big Theta Θ Notation `c′g(n) ≤ f (n) ≤ c′′g(n), for n ≥ n0.`

- 如果存在正数c1，c2和N，对于所有的n>=N，有c1*g(n)<=f(n)<=c2*g(n)，则f(n)=Theta(g(n))

- 这个记法表示一个算法不会好于XX，也不会坏于XX，太中庸了，没有激情啊。
- 所以也就是求average-case。


* used to provide a bound on a particular algorithm such that it can be "sandwiched" between two constants (one for an upper limit and one for a lower limit) for sufficiently large values.

![theta](https://i.imgur.com/aGbSgxo.png)

> Example 4.15: 3nlogn+4n+5logn is Θ(nlogn).
> Justification: 3nlogn ≤ 3nlogn+4n+5logn ≤ (3+4+5)nlogn for n ≥ 2.

---

### Little O Notation
* used to describe an upper bound of a particular algorithm;
* however, Little O provides a bound that is not asymptotically tight

- 对于任意正数c，均存在正数N，对于所有的n>=N，有f(n)<c*g(n)，则f(n)=o(g(n))

---


### Little ω Omega Notation
* used to provide a lower bound on a particular algorithm that is not asymptotically tight

- 对于任意正数c，均存在正数N，对于所有的n>=N，有f(n)>c*g(n)，则f(n)=omega(g(n))



---


## Amortized analysis


**amortization** 分期偿还
- an algorithmic design pattern
- amortized analysis,
  - view the computer as a coin-operated appliance that requires the payment of one cyber-dollar for a constant amount of computing time.
  - When an operation is executed, should have enough cyber-dollars available in our current “bank account” to pay for that operation’s running time.
  - the total amount of cyber-dollars spent for any computation will be proportional to the total time spent on that computation.
  - we can overcharge some operations in order to save up cyber-dollars to pay for others.


### running time of operations on dynamic arrays.

- the insertion of an element to be the last element in an array list as a push operation.


- The strategy of replacing an array with a new, larger array
  - might at first seem slow, because a single push operation may require Ω(n) time to perform, where n is the current number of elements in the array.
  - However, by doubling the capacity during an array replacement, our new array allows us to add n further elements before the array must be replaced again.
  - In this way, there are many simple push operations for each expensive one
  - a series of push operations on an initially empty **dynamic array** is efficient in terms of its total running time.


- Using amortization, performing a sequence of push operations on a dynamic array is actually quite efficient.


**Proposition**
- Let L be an initially empty array list with capacity one, implemented by means of a dynamic array that doubles in size when full.
- The total time to perform a series of n push operations in L is O(n).
**Justification**:
- assume that `one` cyber-dollar for the execution of each **push** operation in L, excluding the time spent for growing the array.
- assume that **growing the array from size k to size 2k** requires `k` cyber-dollars for the time spent initializing the new array.
- charge each **push** operation `three` cyber-dollars. Thus, we overcharge each push operation that does not cause an overflow by two cyber-dollars.
- Think of the two cyber-dollars profited in an insertion that does not grow the array as being “stored” with the cell in which the element was inserted.
- An overflow occurs when the array L has 2^i elements, for some integer i ≥ 0, and the size of the array used by the array representing L is 2i.
- Thus, doubling the size of the array will require 2^i cyber-dollars.
- Fortunately, these cyber-dollars can be found stored in cells 2i−1 through 2i − 1.

- In other words, the amortized running time of each push operation is O(1); hence, the total running time of n push operations is O(n).










---

## Comparative Analysis 对比分析

- an algorithm A has a running time of O(n), algorithm B has a running time of O(n^2).
- algorithm A is **asymptotically better** than algorithm B, although for a small value of n, B may have a lower running time than A.

**Some Words of Caution**
- the use of the big-Oh and related notations can be somewhat misleading should the constant factors they “hide” be very large.
- For example
  - function `10^100n is O(n)`, if this is the running time of an algorithm being compared to one whose running time is `10n log n`, we should prefer the O(n log n)- time algorithm, even though the linear-time algorithm is asymptotically faster.
  - This preference is because the constant factor, 10^100 “one googol,” is believed by many astronomers to be an upper bound on the number of atoms in the observable universe. So we are unlikely to ever have a real-world problem that has this number as its input size.
- The observation above raises the issue of what constitutes a “fast” algorithm.
- Generally speaking, any algorithm running in **O(nlogn)** time (with a reasonable constant factor) should be considered efficient.
  - Even an O(n^2)-time function may be fast enough in some contexts, an algorithm whose running time is an **exponential function**, e.g., O(2n), `should almost never be considered efficient`


Sometimes the performance of an algorithm depends on the `exact values of the data` rather than simply `the size of the problem`.
- For these kinds of algorithms, characterize performance in terms of `best case`, `worst case`, or `average case` performance.
- The `worst case performance` refers to a particular data set where the algorithm performs especially poorly. Whereas a different data set for the exact same algorithm might have extraordinarily good performance. However, in most cases the algorithm performs somewhere in between these two extremes (average case).


**Common Data Structure Operations**
![Screen Shot 2021-10-26 at 3.08.29 AM](https://i.imgur.com/ZOHJix6.png)


**Array Sorting Algorithms**
![Screen Shot 2021-10-26 at 3.09.05 AM](https://i.imgur.com/QKG1tjP.png)

---

### T(n)

<kbd>T(n)</kbd> is **the time it takes to solve a problem of size n**

> the time required to solve the larger case would be greater than for the smaller case.
> Our goal then is to show how `the algorithm’s execution time changes` with respect to `the size of the problem`.

It turns out that the exact number of operations is not as important as determining the most dominant part of the `𝑇(𝑛)` function.
- as the problem gets larger, some portion of the `𝑇(𝑛)` function tends to overpower the rest.
- This dominant term is what, in the end, is used for comparison.


variable name define info.
- 存储空间
- 执行时间: `import time: time.time()=8888888`


The **order of magnitude** function
- describes the part of `𝑇(𝑛)` that increases the fastest as the value of `n` increases.
- **Order of magnitude重要性** is often called **Big-O notation** (for “order”) and written as <kbd>𝑂(𝑓(𝑛))</kbd>.

It provides a useful approximation to the actual number of steps in the computation.
The function `𝑓(𝑛)` provides a simple representation of the dominant part of the original `𝑇(𝑛)`.

> example 1

`𝑇(𝑛)=1+𝑛`
- The parameter n is often referred to as the “size of the problem,”
- read this as “T(n) is the time it takes to solve a problem of size n, namely 1 + n steps.”
- As `n` gets large, the constant 1 will become less and less significant to the final result.
- If looking for an approximation for `𝑇(𝑛)`, then can drop the 1 and simply say that the running time is <kbd>𝑂(𝑛)</kbd>.
- It is important to note that the **1 is certainly significant for `𝑇(𝑛)`**.
- However, as `n` gets large, our approximation will be just as accurate without it.


> example 2

`𝑇(𝑛)=5𝑛^2+27𝑛+1005`.
- When `n` is small, the constant 1005 seems to be the dominant part of the function.
- However, as n gets larger
  - the `𝑛^2` term becomes the most important.
  - the other two terms become insignificant in the role that they play in determining the final result.
  - can ignore the other terms and focus on `5𝑛^2`.
  - the `coefficient 5` also becomes insignificant as `n` gets large.
- the function `𝑇(𝑛)` has an **order of magnitude** 𝑓(𝑛)=𝑛^2, or simply that it is <kbd>𝑂(𝑛^2)</kbd>.

---

### Examples of Algorithm Analysis

#### 题目

- **The j th element can be found**
  - not by iterating through the array one element at a time,
  - but by validating the index, and using it as an `offset from the beginning of the array in determining the appropriate memory address`.
  - Therefore, `A[j]` is evaluated in O(1) time for an array.


    ```java
    sum=1;
    // T(n)=1
    // O(1): 表示算法的运行时间为常量
    a.length;
    a[j];
    ```

- **finding the largest element of an array.**
  - loop through elements of the array while maintaining as a variable the largest element seen thus far.

    ```java
    public static double arrayMax(double[ ] data) {
        int n = data.length;            // constant number of primitive operations.
        double currentMax = data[0];    // constant number of primitive operations.
        // Each iteration of the loop requires only a constant number of primitive operations,

        // and the loop executes n − 1 times.
        for (int j=1; j < n; j++){
            if (data[j] > currentMax) currentMax = data[j];
        }

        return currentMax;              // constant number of primitive operations.
    }
    // a(n−1)+b = an+(b−a) ≤ an
    // arrayMax is O(n)
    ```

- **Composing Long Strings**
  1. `String`
     - strings in Java are **immutable objects**. Once created, an instance cannot be modified.
     - answer += c
       - does not cause a new character to be added to the existing String instance;
       - instead it produces a new String with the desired sequence of characters,
       - and then it reassigns the variable, answer, to refer to that new string.
     - the creation of a new string as a result of a concatenation, requires time that is proportional to the length of the resulting string.
     - Therefore, the overall time taken by this algorithm is proportional to `1+2+···+n <= n^2`
  2. `StringBuilder`
     - uses Java’s StringBuilder class, demonstrate a trend of approximately doubling each time the problem size doubles.
     - The StringBuilder class relies on an advanced technique with a worst-case running time of `O(n)` for composing a string of length n;


        ```java
        public static String repeat1(char c, int n) {
            String answer = "";
            for (int j=0; j < n; j++) answer += c;
            return answer;
        }
        // total time complexity of the repeat1 algorithm is O(n2)
        ```


- **Three-Way Set Disjointness**
  - if A and B are each sets of distinct elements, there can be at most `O(n)` such pairs with a equal to b.
  - Therefore, the innermost loop, over C, executes at most n times.


    ```java
    public static boolean disjoint2(int[ ] groupA, int[ ] groupB, int[ ] groupC) {
        for (int a : groupA) {
            for (int b : groupB){
                if (a == b){  // n times
                    for (int c : groupC) if (a == c) return false;
                }
            }
        }
        return true;
    }
    // the worst-case running time for disjoint2 is O(n^2).
    ```

- **Element Uniqueness**
  1. looping through all distinct pairs of indices j < k: `O(n2)`
  2. Using Sorting as a Problem-Solving Tool:
     - The best sorting algorithms (including those used by Array.sort in Java) guarantee a worst-case running time of `O(nlogn)`.
     - the subsequent loop runs in O(n) time,
     - and so the entire unique2 algorithm runs in `O(n log n)` time.

        ```java
        public static boolean unique2(int[] data) {
            int n = data.length;
            int[] temp = Arrays.copyOf(data, n);
            Arrays.sort(temp);             // O(nlogn)
            for (int j=0; j < n−1; j++){   // n
                if (temp[j] == temp[j+1]) return false;
            }
            return true;
        }
        ```


- **Prefix Averages**
  - given a sequence x consisting of n numbers, compute a sequence a such that aj is the average of elements x0,...,xj
  1. Quadratic-Time Algorithm: `O(n^2)`


        ```java
        public static double[ ] prefixAverage1(double[ ] x) {
            int n = x.length;                // O(1) time.
            double[ ] a = new double[n];     // O(1) time.
            for (int j=0; j < n; j++) {      // n
                double total = 0;
                for (int i=0; i <= j; i++){  // n-1
                    total += x[i];
                    a[j] = total / (j+1);
                }
            }
            return a;
        }
        ```

  2. Linear-Time Algorithm: `O(n)`

      ```java
      public static double[ ] prefixAverage2(double[] x) {
          int n = x.length;                // O(1) time.
          double[] a = new double[n];      // O(1) time.
          a[0] = x[0];
          for (int j=1; j < n; j++) {      // n
              a[j] = (a[j-1]+x[j]) / (j+1)
          }
          return a;
      }

      public static double[ ] prefixAverage2(double[] x) {
          int n = x.length;                // O(1) time.
          double[] a = new double[n];      // O(1) time.
          double total = 0;
          for (int j=0; j < n; j++) {      // n
              total += x[j];
              a[j] = total/(j+1);
          }
          return a;
      }
      ```






---


#### O(1): Constant-Time Operations


- code that executes in the same amount of time no matter how big the array is

```java
sum=1;
// T(n)=1
// O(1): 表示算法的运行时间为常量
a.length;
a[j];
```


---

#### O(㏒N): 二分查找算法

$2^t$ < n

t < log2(n)

- when data being used is decreased roughly by 50% each time through the algorithm
- as ㏒N increases or N specifically increases
- the different between N and logN will be dramatically different

```java
int i=1;                   // 1
while (i<=n) {             //2,3,....n
    i=i*2;                 // 设语句2的频度是t
}
// 2^t <= n
// t <= log2(n)
// 取最大值t = log2(n),
// T(n) = O(log2n)


aFunc(int n) {
    for (int i = 2; i < n; i++) {  //2,3,....n:  n-1
        i *= 2;                    //假设循环次数为 t，则循环条件满足 2^t < n。
    }
}
// 2^t < n
// t < log2(n)
```


---


#### O(n): 线性算法 linear time


```java
a=0,b=1;               ① 1+1
for (i=1;i<=n;i++) ② n
{
    s=a+b;　　　　③ n-1
    b=a;　　　　　④ n-1
    a=s;　　　　　⑤ n-1
}
// 解: T(n)=2+n+3(n-1)=4n-1= O(n).


int aFunc(int n) {
    for(int i = 0; i<n; i++) {         // 需要执行 (n + 1) 次
        printf("Hello, World!\n");     // 需要执行 n 次
    }
    return 0;       // 需要执行 1 次
}
// 解: T(n)= (n + 1 + n + 1) = 2n + 2 = O(n).
```


---


#### O(N㏒N: 二分查找算法



```java
for(int i = 2; i < n; i++) {   // n
    int i=1;                   // 1
    while (i<=n) {
        i=i*2;                 // 设语句2的频度是t
    }
}
// T(n)=O(n * log2(N))
```


---


#### O($n^2$)

`Big o n square`
- 对数组进行排序的简单算法，例如直接插入排序的算法。
- The time to complete will be proportional to the square of the amount of data

```java
for (i=1;i<n;i++){            // n
    for (j=0;j<=n;j++) {      // n
        x++;
    }
}
// T(n) = O(n^2)
```


#### O($n^3$) 做两个n阶矩阵的乘法运算

`n cube`
- n^3 增长速度远超 n^2，n^2 增长速度远超 n

```java
for(i=0;i<n;i++){           // i=m
    for(j=0;j<i;j++){       // j=(m-1)*m
        for(k=0;k<j;k++)    // k=(m-1)m-1
            x=x+2;
    }
}
```

#### O($2^n$): 求具有n个元素集合的所有子集的算法




---


#### O(n!): 求具有N个元素的全排列的算法



---



## Simple Justification Techniques
- make claims about an algorithm, such as showing that it is correct or that it runs fast.
- we must justify or prove our statements.



### counterexample: By Example
- Some claims are of the generic form,
- “There is an element x in a set S that has property P.”
  - only need to produce a particular x in S that has property P.
- “Every element x in a set S has property P.”
  - To justify that such a claim is false,
  - we only need to produce a particular x from S that does not have property P.
- Such an instance is called a counterexample.

> Example 4.17: Professor Amongus claims that every number of the form 2i − 1 is a prime, when i is an integer greater than 1. Professor Amongus is wrong.
> Justification: To prove Professor Amongus is wrong, we find a counterexample. 24 −1 = 15 = 3·5.




### contrapositive and the contradiction: The “Contra” Attack


- Another set of justification techniques involves the use of the negative. The two primary such methods are the use of the contrapositive and the contradiction.

**contrapositive**
- “if p is true, then q is true,”
  - we establish that “if q is not true, then p is not true” instead.
  - Logically, these two statements are the same, but the latter, which is called the contrapositive of the first, may be easier to think about.


> Example 4.18: Let a and b be integers. If ab is even, then a is even or b is even.
> Justification: To justify this claim, consider the contrapositive,
> “If a is odd and b is odd, then ab is odd.”
> a = 2 j + 1 and b = 2k + 1, for some integers j and k. Then ab = 4jk+2j+2k+1 = 2(2jk+ j+k)+1; hence, ab is odd.

- de Morgan’s law
  - the negation of a statement of the form “p or q” is “not p and not q.”
  - the negation of a statement of the form “p and q”is“not p or not q.”


**Contradiction**
- we establish that a statement q is true by first supposing that q is false and then showing that this assumption leads to a contradiction (such as 2 ̸= 2 or 1 > 3).
- By reaching such a contradiction, we show that no consistent situation exists with q being false, so q must be true. Of course, in order to reach this conclusion, we must be sure our situation is consistent before we assume q is false.

> Example 4.19: Let a and b be integers. If ab is odd, then a is odd and b is odd.
> Justification: suppose a is even or b is even. In fact, without loss of generality, we can assume that a is even (since the case for b is symmetric). Then a = 2 j for some integer j.
> Hence, ab = (2 j)b = 2( jb), that is, ab is even.
> But this is a contradiction: ab cannot simultaneously be odd and even. Therefore, a is odd and b is odd.


---


### Induction and Loop Invariants

- Most of the claims we make about a running time or a space bound involve an integer parameter n (usually denoting an intuitive notion of the “size” of the problem).
- Moreover, most of these claims are equivalent to saying some statement q(n) is true “for all n ≥ 1.”
- Since this is making a claim about an infinite set of numbers, we cannot justify this exhaustively in a direct fashion.


**Induction**
- justify claims as true by using **induction**.
- This technique amounts to showing that, for any particular n ≥ 1, there is a finite sequence of implications that starts with something known to be true and ultimately leads to showing that q(n) is true.
- begin a justification by induction by showing that q(n) is true for n = 1 (and possibly some other values n = 2, 3, . . . , k, for some constant k).
- justify that the inductive “step” is true for n > k
- we show “if q(j) is true for all j < n, then q(n) is true.”
- The combination of these two pieces completes the justification by induction.

> Consider the Fibonacci function F(n), which is defined such that F(1) = 1, F(2) = 2, and `F(n) = F(n−2)+F(n−1)` for n > 2. F(n) < 2^n.
> Justification:
> Basecases:(n≤2).
> F(1)=1<2=2^1
> F(2)=2<4=2^2
> Induction step: (n > 2). Suppose our claim is true for all j < n.
> Since both n − 2 and n − 1 are less than n, we can apply the inductive assumption (sometimes called the “inductive hypothesis”) to imply that Since
> F(n) = F(n−2)+F(n−1) < 2^n−2 +2^n−1.
> 2^n−2 +2^n−1 < 2^n−1 +2^n−1 = 2·2^n−1 = 2^n
> we have that F(n) < 2n, thus showing the inductive hypothesis for n.



**Loop Invariants**
- To prove some statement L about a loop is correct,
- define L in terms of a series of smaller statements L0,L1,...,Lk, where:
  1. The initial claim, L0, is true before the loop begins.
  2. If Lj−1 is true before iteration j, then Lj will be true after iteration j.
  3. The final statement, Lk, implies the desired statement L to be true.



---



# 3.5. Performance of Python Data Structures

![Screen Shot 2020-05-26 at 00.32.28](https://i.imgur.com/3Fma2jK.png)

![Screen Shot 2020-05-26 at 00.43.30](https://i.imgur.com/1FedT9U.png)

---

## 3.6. Lists

Common programming task is to grow a list.

```py
def test1():
    l = []
    for i in range(1000):
        l = l + [i]


def test2():
    l = []
    for i in range(1000):
        l.append(i)


def test3():
    l = [i for i in range(1000)]


def test4():
    l = list(range(1000))
```

use Python’s `timeit` module: make cross-platform timing measurements by running functions in a consistent environment and using timing mechanisms that are as similar as possible across operating systems.
- create a `Timer` object
- parameters are two Python statements.
  - The first parameter is a `Python statement that want to time`;
  - the second parameter is a `statement that will run once to set up the test`.
- The timeit module will then time how long it takes to execute the statement some number of times.
- By default timeit will try to run the statement one million times.
- When its done it returns the time as a floating point value representing the total number of seconds.
- However, since it executes the statement a million times you can read the result as the number of microseconds to execute the test one time. You can also pass timeit a named parameter called number that allows you to specify how many times the test statement is executed.
- The following session shows how long it takes to run each of our test functions 1000 times.

```py
from timeit import Timer

t1 = Timer("test1()", "from __main__ import test1")
print(f"concatenation: {t1.timeit(number=1000):15.2f} milliseconds")

t2 = Timer("test2()", "from __main__ import test2")
print(f"appending: {t2.timeit(number=1000):19.2f} milliseconds")

t3 = Timer("test3()", "from __main__ import test3")
print(f"list comprehension: {t3.timeit(number=1000):10.2f} milliseconds")

t4 = Timer("test4()", "from __main__ import test4")
print(f"list range: {t4.timeit(number=1000):18.2f} milliseconds")

# concatenation:           6.54 milliseconds
# appending:               0.31 milliseconds
# list comprehension:      0.15 milliseconds
# list range:              0.07 milliseconds
```

> In this case the statement from __main__ import test1 imports the function test1 from the __main__ namespace into the namespace that timeit sets up for the timing experiment.
> The timeit module does this because it wants to run the timing tests in an environment that is uncluttered by any stray variables you may have created, that may interfere with your function’s performance in some unforeseen way.

all of the times include some overhead for actually calling the test function,
- but we can assume that the function call overhead is identical in all four cases so we still get a meaningful comparison of the operations.
- So it would not be accurate to say that the concatenation operation takes 6.54 milliseconds but rather `the concatenation test function takes 6.54 milliseconds`.


Big-O Efficiency of Python List Operators

Operation | Big-O Efficiency
---|--
index []  | <kbd>O(1)</kbd>
index assignment  | <kbd>O(1)</kbd>
append  | <kbd>O(1)</kbd>
pop()  | <kbd>O(1)</kbd>
pop(i)  | <kbd>O(n)</kbd>
insert(i,item)  | <kbd>O(n)</kbd>
del operator  | <kbd>O(n)</kbd>
iteration  | <kbd>O(n)</kbd>
contains (in)  | <kbd>O(n)</kbd>
`get slice [x:y]`  | <kbd>O(k)</kbd>
del slice  | <kbd>O(n)</kbd>
set slice  | <kbd>O(n+k)</kbd>
reverse  | <kbd>O(n)</kbd>
concatenate  | <kbd>O(k)</kbd>
sort  | <kbd>O(n log n)</kbd>
multiply  | <kbd>O(nk)</kbd>



different times for pop.
- When pop is called on the end of the list it takes <kbd>𝑂(1)</kbd>
- when pop is called on the first element in the list or anywhere in the middle it is <kbd>𝑂(𝑛)</kbd>
  - The reason for this lies in how Python chooses to implement lists.
  - When an item is taken from the front of the list, in Python’s implementation, all the other elements in the list are shifted one position closer to the beginning.


As a way of demonstrating this difference in performance let’s do another experiment using the timeit module. Our goal is to be able to verify the performance of the pop operation on a list of a known size when the program pops from the end of the list, and again when the program pops from the beginning of the list. We will also want to measure this time for lists of different sizes. What we would expect to see is that the time required to pop from the end of the list will stay constant even as the list grows in size, while the time to pop from the beginning of the list will continue to increase as the list grows.

Listing 4 shows one attempt to measure the difference between the two uses of pop. As you can see from this first example, popping from the end takes 0.0003 milliseconds, whereas popping from the beginning takes 4.82 milliseconds. For a list of two million elements this is a factor of 16,000.

> the statement from __main__ import x. Although we did not define a function we do want to be able to use the list object x in our test. This approach allows us to time just the single pop statement and get the most accurate measure of the time for that single operation.
> Because the timer repeats 1000 times it is also important to point out that the list is decreasing in size by 1 each time through the loop. But since the initial list is two million elements in size we only reduce the overall size by 0.05%


to show that pop(0) is indeed slower than pop():

```py
pop_zero = Timer("x.pop(0)", "from __main__ import x")
pop_end = Timer("x.pop()", "from __main__ import x")

x = list(range(2000000))
print(f"pop(0): {pop_zero.timeit(number=1000):10.5f} milliseconds")

x = list(range(2000000))
print(f"pop(): {pop_end.timeit(number=1000):11.5f} milliseconds")

# pop(0):    2.09779 milliseconds
# pop():     0.00014 milliseconds
```


to validate the claim that pop(0) is 𝑂(𝑛) while pop() is 𝑂(1)
- look at the performance of both calls over a range of list sizes.

```py
pop_zero = Timer("x.pop(0)", "from __main__ import x")
pop_end = Timer("x.pop()", "from __main__ import x")
print(f"{'n':10s}{'pop(0)':>15s}{'pop()':>15s}")

for i in range(1_000_000, 100_000_001, 1_000_000):
    x = list(range(i))
    pop_zero_t = pop_zero.timeit(number=1000)
    x = list(range(i))
    pop_end_t = pop_end.timeit(number=1000)
    print(f"{i:<10d}{pop_zero_t:>15.5f}{pop_end_t:>15.5f}")
```

![poptime](https://i.imgur.com/30bdfeg.png)

- the list gets longer and longer
- the time it takes to pop(0) also increases
- while the time for pop stays very flat.
- This is exactly what we would expect to see for a 𝑂(𝑛) and 𝑂(1) algorithm.

> Some sources of error in our little experiment include the fact that there are other processes running on the computer as we measure that may slow down our code,
> That is why the loop runs the test one thousand times in the first place to statistically gather enough information to make the measurement reliable.

---

## 3.7. Dictionaries

- the `get item` and `set item` operations on a dictionary are 𝑂(1).
- Checking to see `whether a key is in the dictionary` or not is also 𝑂(1).

> the efficiencies we provide in the table are for average performance.
> In some rare cases the contains, get item, and set item operations can degenerate into 𝑂(𝑛) performance

Big-O Efficiency of Python Dictionary Operations

operation | Big-O Efficiency
---|---
copy | <kbd>O(n)</kbd>
get item | <kbd>O(1)</kbd>
set item | <kbd>O(1)</kbd>
delete item | <kbd>O(1)</kbd>
contains (in) | <kbd>O(1)</kbd>
iteration | <kbd>O(n)</kbd>


compare the performance of the `contains operation` between **lists** and **dictionaries**.
- make a list with a range of numbers in it.
- pick numbers at random and check to see if the numbers are in the list.
- If our performance tables are correct, the bigger the list the longer it should take to determine if any one number is contained in the list.

We will repeat the same experiment for a dictionary that contains numbers as the keys.
- determining whether or not a number is in the dictionary is not only much faster,
- but the time it takes to check should remain constant even as the dictionary grows larger.

Listing 6 implements this comparison. Notice that we are performing exactly the same operation, number in container. The difference is that on line 8 x is a list, and on line 10 x is a dictionary.

```py
import timeit
import random

print(f"{'n':10s}{'list':>10s}{'dict':>10s}")

for i in range(10_000, 1_000_001, 20_000):
    t = timeit.Timer(f"random.randrange({i}) in x", "from __main__ import random, x")

    x = list(range(i))
    lst_time = t.timeit(number=1000)

    x = {j: None for j in range(i)}
    dict_time = t.timeit(number=1000)

    print(f"{i:<10,} {lst_time:>10.3f} {dict_time:>10.3f}")

# n               list      dict
# 10,000          0.085      0.001
# 30,000          0.225      0.001
# 50,000          0.381      0.001
# 70,000          0.542      0.001
# 90,000          0.770      0.001
# 110,000         1.104      0.001
# 130,000         0.993      0.001
# 150,000         1.121      0.001
# 170,000         1.243      0.001
# 190,000         1.375      0.001
# 210,000         1.546      0.001
```

![listvdict](https://i.imgur.com/ziaxv2F.png)


---






























.
