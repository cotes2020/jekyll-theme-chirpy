---
title: Go Concurrency: Encapsulate A Concurrent Task
author: Hulua
date: 2021-03-17 20:55:00 +0800
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

Let's run the a
```console
$ go run test.go 
I am doing the task, please wait
Received value from concurrent task: 99
```

## Summary
See, it is very simple to encapsulte a computation task. The main idea is to use Go routine to run the task, and use a channel to transmit the result returned from the task.