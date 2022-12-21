---
title: What is Microbenchmarking
description: Microbenchmarking with jvm is hard and should be avoided. An introduction to Microbenchmarking, when to use it and pitfalls to avoid.
tags: ["java", "jvm", "microbenchark"]
category: ["programming"]
date: 2021-03-30
permalink: '/java/java-benchmarking/'
---

<hr>
### Introduction
Optimisation of code is an endless struggle. It is often even hard to produce meaningful metrics using `jvm` as it is an adaptive virtual machine. The article is
- a brief introduction to microbenchmarking,
- why microbenchmark
- when to consider it, and finally,
- pitfalls to avoid

### What is a Microbenchmark
A microbenchmark is an attempt to measure the performance of a small unit of code. The tests are usually in the sub-millisecond range. The tests can help determine how the code is going to behave when released into production. These tests are guide to a better implementation.

### Why Microbenchmark
Profiling a whole app in a production or production-like environment is difficult. Profiling cannot pinpoint a specific piece of code. Moreover, it also counts external factors such as logging. Yet, profiling does produce realistic results.

Micro-benchmarking focuses on a specific piece of code, removing everything else. But, you need to be careful of the results produced by benchmarking, as these are somewhat artificial.

If we are asking JVM to manage a small piece of code, it means we are asking JVM to handle a different problem. JVM may optimize the code differently from that in production. GC may be much more effective for this small program, but it can take long pauses in actual production application. Besides, the machine architecture of the production server could be completely different from the local machine used for benchmarking.

__Why should I benchmark then?__<br>
Usually, it should be the job of `jvm` to optimize the code. As a good practice, a developer should only focus on principles of clean coding. But it is always a good idea to review the code. You should consider asking can I break this loop early, or can I reduce the complexity of an algorithm?

### When to Consider Microbenchmark
A developer should test an application in a way it is supposed to be used, with a similar kind of inputs. But, it might not be possible every time. For example, if you are trying to write an underlying support infrastructure for a variety of applications, or if you are producing a library. Thus, it is not possible to predict the range of inputs or to monitor and optimize the code for specific scenarios. In these scenarios, you can consider benchmarking.
Micro benchmarking may still not provide definitive answers to problems you may face, but, it can point towards a better design. 


### Code to Microbenchmark
Let us try to microbenchmark a piece of code. I am going to write a program to calculate nth Narcissistic number or a plus perfect number. If you have never heard about these numbers then you can refer to [Wikipedia](https://en.wikipedia.org/wiki/Narcissistic_number). These are the numbers where the number is equal to the sum of its own digits each raised to the power of the number of digits.

The first step is to define `isNarNumber()` that would check if a provided number is a Narcissistic number.

```java
public class NarcissisticNumber {
    public static boolean isNarNumber(int number) {
        int length = String.valueOf(number).length();
        int sum = 0;

        for (int i = number; i > 0; i = i / 10) {
            sum += pow(i % 10, length);
        }
        return sum == number;
    }
}
```

Next step is to define the method `findNarcissisticNumber()` to get the results

```java
public class NarcissisticNumber {
    ...
    public static int findNarcissisticNumber(int n) {
        int pointer = 0;
        int i;

        for (i=1; i<Integer.MAX_VALUE && pointer < n; i++) {
            if (isNarNumber(i)) {
                pointer = pointer+1;
            }
        }

        return i;
    }
}
```

### A Naive Approach
The concept of benchmarking seems straight. I can quickly measure the execution time of `findNarcissisticNumber` by using `System.currentTimeMillis()`.

```java
public class NarcissisticNumber {
    ...
    public static void main(String[] args) {
        int result;
        long before = System.currentTimeMillis();
        for (int i = 0; i < cycles; i++) {
            result = findNarcissisticNumber(20);
        }
        long after = System.currentTimeMillis();
        System.out.println("Time elapsed: " + (after-before)/cycles + " seconds" );
    }
}
```
But, the approach might not be best suited with jvm. There are few common pitfalls with the approach above. We will discuss the same in next section.

### Avoid Common Pitfalls

`jvm`  is quite advanced, and it can optimize the code itself. The features of `jvm` such as JIT compilation and garbage collection makes it hard to perform bencharmking. I am listing a few of the pitfalls, if are not considered, then may lead to inconsistent results.

#### 1. Skipping Warm-up Cycles
`jvm` optimizes the code with each run. The code execution gets faster, the longer it is executed. Skipping `jvm` warm-up will lead to inconsistent reading. Thus, benchmarking should include warm-up cycles to allow `jvm` to optimize the code. 

```java
for (int i = 0; i < warmupCycles; i++) {
    result = findNarcissisticNumber(20);
}

for (int i = 0; i < cycles; i++) {
    result = findNarcissisticNumber(20);
}
```

#### 2. Dead Code
Even after considering the warm-up cycles, it is possible that execution time is reported as 0. Since `result` is not used anywhere `jvm` can skip the loop altogether. It is also possible that the compiler performs partial iterations, thus, leading to even more inconsistent result.  To fix the issue it is better to read it is advisable to read the result.

#### 3. Not Testing a Range of Input
Even if the code starts using the `result` variable, we may still get irregular readings. The code above always calculates the 20th Narcissistic number. A smart compiler can figure that out and can replace the method call with the constant result. The iterations are redundant, and the compiler can skip a few of those.

Moreover, the code in production would calculate the Narcissistic number on a range of inputs. Hence, we should also benchmark the code with a similar range of inputs.

#### 4. Noisy Neighbours
While running benchmarking on the machine, you might not be aware of all the processes running your system. It is possible to have background apps contending for CPU and RAM on the local machine. Thus, creating a server like environment is difficult.  Mitigating the noisy neighbour problem is a bit tricky.

### Conclusion
It is difficult to achieve good benchmarking stats. We need to consider all the factors if we want to achieve reliable results. The article highlights the pitfalls of benchmarking. In the next post, we will explore Java Microbenchmark Harness (jmh). JMH is a powerful tool and can help us to benchmark the code.

__Reference__<br>
<sup><a href="https://github.com/openjdk/jmh" target="_blank">JMH Github</a></sup><br>
<sup><a href="https://shipilev.net/talks/devoxx-Nov2013-benchmarking.pdf" target="_blank">Devoxx-Nov2013 Slides</a></sup><br>
<sup><a href="https://github.com/google/caliper/wiki/JavaMicrobenchmarks" target="_blank">Google Caliper Github</a></sup><br>
<sup><a href="https://www.oracle.com/technical-resources/articles/java/architect-benchmarking.html" target="_blank">Oracle Avoiding Benchmarking Pitfalls on the JVM</a></sup>


