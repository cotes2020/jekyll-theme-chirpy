---
title: Go Concurrency - Stop Signal and Timeout
author: Hulua
date: 2022-03-24 20:55:00 +0800
categories: [Go, Concurrency]
tags: [go, concurrency, goroutine]
---

In this post, we will see some other design patterns with go's channel. Specifically, we can use a channel to deliever a stop signal to goroutines. Meanwhile, when we are waiting on a channel, we can set a time limit, if after the dedicated time we still cannot get something we want, we can take actions.

## Stop Goroutines with Stop Signal

One commonly used pattern in Go programming is to use a channel to deliever a stop signal to goroutines, such that the goroutines can be stopped correctly. This can be done simply by closing the channel.  Let's say we have the below Go program called close.go:

```go
package main

import "fmt"
import "sync"
import "time"

func main(){
    var wg sync.WaitGroup
    quit := make(chan bool)

    for i:=0; i<3; i++ {
        wg.Add(1)
        go func(i int){
            defer wg.Done()
            doTask := func(){
                fmt.Printf("In routine %d, doing the task\n",i)
                time.Sleep(time.Second * 1)
            }
            for {
                select {
                case _, ok := <-quit:
                    if !ok {
                        fmt.Printf("routine %d received quit signal\n", i)
                        return
                    }
                default:
                    doTask()
                }
            }
        }(i)

    }

    time.Sleep(time.Second * 5) //Let routines run 5 se
    close(quit) //Signal quit thus routines will stop
    wg.Wait() //Wait routines to stop
}

```

In this case, we created a channel called ```quit```, to send stop signal we can simply close the channel. Then we started three goroutines, and each routine will monitor if the channel is closed using a select mechanism. If the channel is closed, we return from the goroutine. If the channel if not closed, it will do the task.  We also utilized a wait group so the main routine will wait until all the serving goroutines are stopped. Let's run the program:

```console

go run close.go
In routine 2, doing the task
In routine 0, doing the task
In routine 1, doing the task
In routine 1, doing the task
In routine 2, doing the task
In routine 0, doing the task
In routine 2, doing the task
In routine 0, doing the task
In routine 1, doing the task
In routine 2, doing the task
In routine 0, doing the task
In routine 1, doing the task
In routine 1, doing the task
In routine 2, doing the task
In routine 0, doing the task
routine 0 received quit signal
routine 2 received quit signal
routine 1 received quit signal

```

## Wait on Channel until Timeout

Another commonly used pattern in Go channel is to wait on the channel for some dedicated time. If after the dedicated time we still cannot receive data from the channel, we can take some actions. For example with timeout.go:

```go
package main

import "fmt"
import "sync"
import "time"

func main(){
    var wg sync.WaitGroup
    waitCh := make(chan int)
    wg.Add(1)
    go func(){
        fmt.Println("I am the go routine waiting on the channel...")
        select {
            case  _ = <- waitCh:
                fmt.Println("I received data from the channel.") //Not reachable
            case <- time.After(time.Second * 4):
                fmt.Println("Time out, I have waited enough time")
                wg.Done()
        }
    }()

    wg.Wait() //Wait routines to stop
}
```

Run the above program,we can see the below output.

```console

go run timeout.go
I am the go routine waiting on the channel...
Time out, I have waited enough time

```
