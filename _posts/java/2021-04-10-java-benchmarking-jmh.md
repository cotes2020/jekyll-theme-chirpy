---
title: Java Microbenchmark Harness (JMH)
description: Creating the first JMH project. A quick hands-on lesson to learn about Java Microbenchmark Harness (JMH). The article helps you get started and configure JMH project.
tags: ["java", "jvm", "microbenchark"]
category: ["programming", "tutorial"]
date: 2021-04-10
permalink: 'java/java-benchmarking-jmh/'
counterlink: 'java-benchmarking-jmh/'
image:
  path: https://raw.githubusercontent.com/Gaur4vGaur/traveller/master/images/java/2021-04-10-java-benchmarking-jmh.jpg
  width: 800
  height: 200
---

### Introduction
In my <a href="https://www.gaurgaurav.com/java/java-benchmarking" target="_blank">previous article</a> I established that microbenchmarking is hard with `jvm`. It is not enough to surround the code in a loop with `System.out.println()` and gather the time measurements. While benchmarking, a developer should consider warm-up cycles, JIT compilations, jvm optimizations, avoiding usual pitfalls and even more.

Thankfully, OpenJDK has a great tool Java Microbenchmark Harness (JMH) that can help to generated benchmarking stats. In this article, I will examine how JMH can help to avoid the pitfalls that we have <a href="https://www.gaurgaurav.com/java/java-benchmarking" target="_blank">discussed earlier</a>. 

### Getting Started with JMH
A quick way to start with JMH is to use the Maven archetype. The command below will generate a new Java project. The project will have `com/gaurav/MyBenchmark.java` class and `pom.xml`. The Maven `pom.xml` includes all the required dependencies to support JMH.

```shell
mvn archetype:generate -DarchetypeGroupId=org.openjdk.jmh -DarchetypeArtifactId=jmh-java-benchmark-archetype -DinteractiveMode=false -DgroupId=com.gaurav -DartifactId=benchmark -Dversion=1.0
```

### Good Benchmarks with JMH
Below are few features of JMH that help write better microbenchmarks.

* JMH, by default, makes several warm up cycles before collecting the stats. Thus, it makes sure that the results are not completely random and `jvm` has performed initial optimizations.
* `@benchmark` runs iteration over the code, before collecting the average. The more runs it makes through the code, the better stats it will collect.
* Use <a href="https://javadox.com/org.openjdk.jmh/jmh-core/1.6.3/org/openjdk/jmh/infra/Blackhole.html" target="_blank">`Blackhole`</a> class of JMH to avoid dead code elimination by `jvm`. If I pass the calculated results to `blackhole.consume()`, it would trick the `jvm`. `jvm` will never drop the code thinking that `consume()` method uses the result.

### Writing First Benchmark
Maven has already provided me with a template in `MyBenchmark` class to fill in. I am going to utilise the same class.

```java
package com.gaurav;
import org.openjdk.jmh.annotations.Benchmark;

public class MyBenchmark {
    @Benchmark
    public void testMethod() {
        // This is a demo/sample template for building your JMH benchmarks. Edit as needed.
        // Put your benchmark code here.
    }
}
```

I would like to keep my first benchmark pretty simple. Let me start by iterating over all the elements of a list and sum them up using a conventional `for` loop. As discussed, I will use `Blackhole` to fool the compiler and return the result. Here, I am asking JMH to calculate the average time, using `@BenchmarkMode`, which it takes to run the `testMethod()`.

```java
@Benchmark
@BenchmarkMode(Mode.AverageTime)
public static double testMethod(Blackhole blackhole) {
    double sum = 0;
    for(int i=0; i<list.size(); i++) {
        sum += list.get(i);
    }

    blackhole.consume(sum);
    return sum;
}
```

### Compiling the JMH Project
Compile and build the project like any other Maven project:

```shell
mvn clean install
```

The command will create a fully executable `jar` file under `benchmark/target` directory. Please note that Maven will always generate a `jar` file named `benchmarks.jar`, regardless of the project name.

The next step is to execute the `jar`.
```shell
java -jar target/benchmarks.jar
```

Executing above command produced below result for me. It means that test operation is taking approx. _0.053_ seconds on the current hardware.  

```shell
# Run progress: 80.00% complete, ETA 00:01:41
# Fork: 5 of 5
# Warmup Iteration   1: 0.052 s/op
# Warmup Iteration   2: 0.051 s/op
# Warmup Iteration   3: 0.053 s/op
# Warmup Iteration   4: 0.056 s/op
# Warmup Iteration   5: 0.055 s/op
Iteration   1: 0.054 s/op
Iteration   2: 0.053 s/op
Iteration   3: 0.053 s/op
Iteration   4: 0.054 s/op
Iteration   5: 0.059 s/op

Result "com.example.MyBenchmark.testMethod":
  0.053 ±(99.9%) 0.002 s/op [Average]
  (min, avg, max) = (0.052, 0.053, 0.061), stdev = 0.002
  CI (99.9%): [0.051, 0.055] (assumes normal distribution)

# Run complete. Total time: 00:08:27
```

### Benchmark Modes
In the previous example, I used `@BenchmarkMode(Mode.AverageTime)`. If you try to decompile JMH jar, you will find `enum Mode` has below options:

|Modes||
| : ---- | :-----------: |
|`Throughput("thrpt", "Throughput, ops/time")`| It will calculate the number of times your method can be executed with in a second |
|`AverageTime("avgt", "Average time, time/op")`| It will calculate the average time in seconds to execute the test method |
|`SampleTime("sample", "Sampling time")`| It randomly samples the time spent in test method calls |
|`SingleShotTime("ss", "Single shot invocation time")`| It works on single invocation of the method and is useful in calculating _cold_ performance |
|`All("all", "All benchmark modes")`| Calculates all of the above |

The default Mode is `Throughput`.

### Time measurement
It is evident from the console output above that calculations are in seconds. But, JMH allows to configure the time units using `@OutputTimeUnit` annotation. The `@OutputTimeUnit` accepts `java.util.concurrent.TimeUnit`, as shown below:

```java
@OutputTimeUnit(TimeUnit.SECONDS)
```

The `TimeUnit` enum has following values:

NANOSECONDS<br>
MICROSECONDS<br>
MILLISECONDS<br>
SECONDS<br>
MINUTES<br>
HOURS<br>
DAYS<br>

The default `TimeUnit` is `SECONDS`

### Configure Fork, Warmup and Iterations

The benchmark is currently executing 5 times, with 5 warmup iterations and 5 measurement iterations. JMH even allows to configure these values using `@Fork`, `@Warmup` and `@Measurement` annotations. The code snippet below would execute the test method twice, with a couple of warmup iterations and 3 measurement iterations.

```java
@Fork(value = 2)
@Warmup(iterations = 2)
@Measurement(iterations = 3)
```
`@Warmup` and `@Measurement` annotations also accepts parameters:
- `batchSize` - configures the number of test method calls to be performed per operation
- `time` - time spent for each iteration


### Practice
You can play around to compare execution times of different `for` loops i.e. a conventional `for` loop, a `forEach` loop and a `stream` iterator. Something like:

```java
private static final List<Integer> list = IntStream.rangeClosed(1, Integer.MAX_VALUE/100)
            .boxed().collect(Collectors.toList());

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    public static double conventionalLoop(Blackhole blackhole) {
        double sum = 0;
        for(int i=0; i<list.size(); i++) {
            sum += list.get(i);
        }
        
        blackhole.consume(sum);
        return sum;
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    public static double enhancedForLoop(Blackhole blackhole) throws InterruptedException {
        double sum = 0;
        for (int integer : list) {
            sum += integer;
        }

        blackhole.consume(sum);
        return sum;
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    public static double streamMap(Blackhole blackhole) {
        double sum = list.stream().mapToDouble(Integer::doubleValue).sum();
        blackhole.consume(sum);
        return sum;
    }
```

### Conclusion
In this post, we have gone through a hands-on example of creating a JMH project. We have seen how can we configure our JMH project to suit our needs. You can refer to <a href="https://github.com/openjdk/jmh/tree/master/jmh-samples/src/main/java/org/openjdk/jmh/samples" target="_blank">JMH Github Samples</a> for more in depth examples.

We have seen that JMH is a `jvm` tool. In the <a href="https://www.gaurgaurav.com/java/scala-benchmarking-jmh/" target="_blank">next article</a> we will try to explore if it can help us with other `jvm` based languages.

__Reference__<br>
<sup><a href="https://github.com/openjdk/jmh" target="_blank">JMH Github</a></sup><br>
<sup><a href="https://github.com/openjdk/jmh/tree/master/jmh-samples/src/main/java/org/openjdk/jmh/samples" target="_blank">JMH Github Samples</a></sup><br>
<sup><a href="https://javadox.com/org.openjdk.jmh/jmh-core/0.8/org/openjdk/jmh/annotations/Mode.html" target="_blank">JMH Javadox - Mode</a></sup><br>
<sup><a href="https://javadox.com/org.openjdk.jmh/jmh-core/1.7/org/openjdk/jmh/annotations/OutputTimeUnit.html" target="_blank">JMH Javadox - OutputTimeUnit</a></sup><br>
<sup><a href="https://javadox.com/org.openjdk.jmh/jmh-core/0.9/org/openjdk/jmh/annotations/Fork.html" target="_blank">JMH Javadox - Fork</a></sup><br>