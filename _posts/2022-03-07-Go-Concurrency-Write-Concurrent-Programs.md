---
title: Go Concurrency - Write Concurrent Programs
author: Hulua
date: 2022-03-17 20:55:00 +0800
categories: [Go, Concurrency]
tags: [go, concurrency, goroutine]
---

## Introduction

I have been learning Go for some time, I would like to say the concurrency feature is Go is fantastic. It makes life with concurrency program much easier. Think about most other pogramming languages, when we need concurrency, we need to create processes/threads explicitly. With Go, everything is simplied with the magic key word "go". Another innovation in Go is the way to synchronzie go routines, channel, which sounds like an revolution from conventional methods with mutex, semaphore, etc. For every Go programmers, the proficiency of concurrency skills is an esenstial part.

In this series of articles, we will explore and learn a few design paterns to design concurrent programs. Please note that this serie of articles are not for Go beginners. It is expected that readers already familar with basics of Go. OK, let's get started.


## Encapsulate a concurrent task

Let's say we want a task to run concurrently, the task will return the computation result (for simplicity, say an integer). For example, say we have a Go file test.go:

 
```go
package main

import "fmt"
import "time"

func concurrentTask() chan int {
    resCh := make(chan int)
    go func(){
        fmt.Println("I am doing the task, please wait")
        time.Sleep(time.Second * 5)
        value := 99
        resCh <- value
    }()
    return  resCh
}

func main() {
    resCh := concurrentTask()
    value := <- resCh
    fmt.Printf("Received value from concurrent task: %d\n", value)
}

```

Let's run the above Go program:
```console
$ go run test.go 
I am doing the task, please wait
Received value from concurrent task: 99
```

## Split a task with subtasks

The above example encapsulate a single task and created a channel to return the result. In most cases, we take advantage of concurrency to split a large computation task and use a few subtasks to accomplish the large task. In this case, we may use a single channel to aggregate results from subtasks. Let's look another example:


```go
package main

import "fmt"
import "time"

func subTask(id int, resChan chan int) {
    fmt.Printf("This is sub task %d, now doing the job..\n",id)
    time.Sleep(time.Second * 3)
    res := id * 10 //Just assume any computation result
    resChan <- res
}

func main() {
    resChan := make(chan int)
    for i:=0; i<3; i++ { //assume we have 3 subtask to do the job
        go subTask(i, resChan)
    }
    totalSum := 0
    for i:=0; i<3; i++ {//Collect result from the 3 subtask
        totalSum += <-resChan
    }
    fmt.Printf("Done at main: total %d\n", totalSum)
}
```

```console
$ go run test.go 
This is sub task 2, now doing the job..
This is sub task 0, now doing the job..
This is sub task 1, now doing the job..
Done at main: total 30
```

In this example, we launched three subtasks for a specific task, and the results of the subtasks are sent to a channel. Lastly, the results are agggregated (added) as the final results.

## Summary
See, it is very simple to write concurrent program with Go.  For the first example, the main idea is to use Go routine to run the task, and use a channel to transmit the result returned from the task. For the second example, we use subtasks and their results are sent to a shared channel for final aggregation. Unlike other programming languages, basically we did not explicicty create any threads. That is the most attractive feature with Go.
