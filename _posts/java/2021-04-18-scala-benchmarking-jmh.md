---
title: Benchmarking Scala Code with JMH
description: Creating the first Java Microbenchmark Harness (JMH) project using SBT. A quick hands-on lesson to integrate Java Microbenchmark Harness (JMH) with SBT.
tags: ["scala", "jvm", "microbenchark"]
category: ["programming", "tutorial"]
date: 2021-04-18
permalink: '/java/scala-benchmarking-jmh/'
counterlink: 'scala-benchmarking-jmh/'
---

### Introduction
We identified in the <a href="https://www.gaurgaurav.com/java/java-benchmarking-jmh/" target="_blank">previous post</a> that JMH is a `jvm` tool that can help benchmark the source code. Till now, we have used it to benchmark the Java code. But, since it is a `jvm`, it must be capable of benchmarking other `jvm` based languages. In this post, I pick up Scala, `jvm` language, to benchmark the code.

### Integrate JMH with SBT
Let us start by creating a new Scala project with SBT. I will use the below `giter8` template to produce the project structure.

```shell
sbt new scala/scala-seed.g8 --name=benchmarks
```

The next step is to configure JMH with the new project. We can achieve it by adding JMH plugin. Create a new file `plugins.sbt` under the `project` directory and add the below content to `plugins.sbt`.

```shell
addSbtPlugin("pl.project13.scala" % "sbt-jmh" % "0.4.0")
```

We must enable the above plugin through `build.sbt`.

```scala
enablePlugins(JmhPlugin)
```

Now, we can start experimenting with benchmarks. We are going to use the same JMH annotations as we have seen in the <a href="https://www.gaurgaurav.com/java/java-benchmarking-jmh/" target="_blank">previous post</a>. The `giter8` template has created `example` package inside directory `src/main/scala`. I am going to rename `example` package to `gaurav`. I will create a new class `MyBenchmark.scala` in the new package. Similar to the java example, I am going to benchmark the code to sum all elements inside a `list`.

```scala
import org.openjdk.jmh.annotations.Benchmark
import org.openjdk.jmh.infra.Blackhole

class MyBenchmark {

  @Benchmark
  def testMethod(blackHole: Blackhole): Double = {
    val list: List[Int] = List.range(1, Integer.MAX_VALUE/100)
    val sum: Double = list.sum
    blackHole.consume(sum)
    sum
  }
}
```

As we have observed in <a href="https://www.gaurgaurav.com/java/java-benchmarking/" target="_blank">earlier posts</a>, I am returning a value from `testMethod`. Additionally, I am using `BlackHole` to avoid `jvm` optimization.

### Executing JMH project in SBT

Unlike, maven which creates a JMH jar to execute the project, SBT can perform the operations from its console. Use the below command to both compile and execute the project. You can also use `sbt jmh:compile`, to just compile the project.

```scala
sbt jmh:run
```

I find it quite handy and quick, as compared to maven. Once executed, you can observe `jmh` log lines

```shell
[info] running (fork) org.openjdk.jmh.Main
```

Similar to what we have seen earlier, `jmh`, by default, will:
* execute 5 warm up iterations,
* execute 5 fork iterations,
* the mode will be `Throughput`, and 
* The default `TimeUnit` will be `SECONDS`

It produced the below result for me. It means that it is executing approx _69342 operations per sec_.

```shell
[info] # Run progress: 80.00% complete, ETA 00:01:40
[info] # Fork: 5 of 5
[info] # Warmup Iteration   1: 58060.100 ops/s
[info] # Warmup Iteration   2: 64730.638 ops/s
[info] # Warmup Iteration   3: 69149.250 ops/s
[info] # Warmup Iteration   4: 63715.739 ops/s
[info] # Warmup Iteration   5: 66027.235 ops/s
[info] Iteration   1: 70228.232 ops/s
[info] Iteration   2: 60943.758 ops/s
[info] Iteration   3: 63144.950 ops/s
[info] Iteration   4: 63729.494 ops/s
[info] Iteration   5: 63557.685 ops/s
[info] Result "com.gaurav.MyBenchmark.testMethod":
[info]   69342.082 ±(99.9%) 4236.598 ops/s [Average]
[info]   (min, avg, max) = (54795.094, 69342.082, 75332.645), stdev = 5655.737
[info]   CI (99.9%): [65105.484, 73578.681] (assumes normal distribution)
```

### Benchmark Configuration
Now, the project is up and running, the next step is to configure the `jmh`. We can configure the project for:

* Modes - `Throughput`, `AverageTime`, `SampleTime`, `SingleShotTime`, and `All`
* Time Units - `NANOSECONDS`, `MICROSECONDS`, `MILLISECONDS`, `SECONDS`, `MINUTES`, `HOURS`, and `DAYS`
* Iterations - `Fork` iterations, `Warmup` iterations and `Measurements` iterations

Below is my sample code. I have configured the code to use `AverageTime`, to run a couple of `fork`, `warmup` and `measurement` iterations.

```scala
import org.openjdk.jmh.annotations.{Benchmark, BenchmarkMode, Fork, Measurement, Mode, Warmup}
import org.openjdk.jmh.infra.Blackhole

class MyBenchmark {

    @Benchmark
    @BenchmarkMode(Array(Mode.AverageTime))
    @Fork(value = 2)
    @Warmup(iterations = 2)
    @Measurement(iterations = 2)
    def testMethod(blackHole: Blackhole): Double = {
        val list: List[Int] = List.range(1, 1000)
        val sum: Double = list.sum
        blackHole.consume(sum)
        sum
    }
}
```

Below is the output. As you can observe in the highlighted lines below, `jmh` first prints the summary of the configuration. The feature is quite handy. You can go through to make sure `jmh` is doing the right thing. 

```shell
[info] # Warmup: 2 iterations, 10 s each
[info] # Measurement: 2 iterations, 10 s each
[info] # Timeout: 10 min per iteration
[info] # Threads: 1 thread, will synchronize iterations
[info] # Benchmark mode: Average time, time/op
[info] # Benchmark: com.gaurav.MyBenchmark.testMethod
[info] # Run progress: 0.00% complete, ETA 00:01:20
[info] # Fork: 1 of 2
[info] # Warmup Iteration   1: ≈ 10⁻⁵ s/op
[info] # Warmup Iteration   2: ≈ 10⁻⁵ s/op
[info] Iteration   1: ≈ 10⁻⁵ s/op
[info] Iteration   2: ≈ 10⁻⁵ s/op
[info] # Run progress: 50.00% complete, ETA 00:00:40
[info] # Fork: 2 of 2
[info] # Warmup Iteration   1: ≈ 10⁻⁵ s/op
[info] # Warmup Iteration   2: ≈ 10⁻⁵ s/op
[info] Iteration   1: ≈ 10⁻⁵ s/op
[info] Iteration   2: ≈ 10⁻⁵ s/op
[info] Result "com.gaurav.MyBenchmark.testMethod":
[info]   ≈ 10⁻⁵ s/op
[info] # Run complete. Total time: 00:01:20
```

### Conclusion
In this article we have seen that `jmh` not only works with Java, but also with Scala, a `jvm` language. We have gone through a hand-on example of configuring a new Scala project with `jmh` plugin. You can refer to <a href="https://github.com/ktoso/sbt-jmh/tree/master/plugin/src/sbt-test/sbt-jmh/run/src/main/scala/org/openjdk/jmh/samples" target="_blank">JMH Github Scala Samples</a> for more in depth examples.

__Reference__<br>
<sup><a href="https://github.com/openjdk/jmh" target="_blank">JMH Github</a></sup><br>
<sup><a href="https://github.com/ktoso/sbt-jmh/tree/master/plugin/src/sbt-test/sbt-jmh/run/src/main/scala/org/openjdk/jmh/samples" target="_blank">JMH Github Scala Samples</a></sup><br>
<sup><a href="https://javadox.com/org.openjdk.jmh/jmh-core/0.8/overview-summary.html" target="_blank">JMH Javadox</a></sup><br>