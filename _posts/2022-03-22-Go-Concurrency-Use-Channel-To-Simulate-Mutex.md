---
title: Go Concurrency - Use Channel to Simulate Mutex
author: Hulua
date: 2022-03-22 20:55:00 +0800
categories: [Go, Concurrency]
tags: [go, concurrency, goroutine]
---

Go's has two types of channels, namely unbuffered channel and buffered channel. With unbuffered channel, the sending or receiving party will be blocked until the other party is ready. While with buffered channel, senders can keep sending data as long as the buffer is not full.

This property can be utilized to simulate the concept of mutex to prevent critical data. 

Let's take a look at an example.  First we write the following program (mutex.go) using mutex:

```go
package main

import "fmt"
import "sync"

func main(){
    var mu sync.Mutex
    var wg sync.WaitGroup
    wg.Add(3)
    for i:=0; i<3; i++ {
        go func(i int){
            //Critical Section
            defer wg.Done()
            mu.Lock()
            for  j:=0; j<3; j++ {
                fmt.Printf("go routine %d , printing %d\n",i, j)
            }
            mu.Unlock()
            //End of Critical Section
        }(i)
    }
    wg.Wait()
}
```

Run the program we can see:

```console
go run mutex.go
go routine 2 , printing 0
go routine 2 , printing 1
go routine 2 , printing 2
go routine 0 , printing 0
go routine 0 , printing 1
go routine 0 , printing 2
go routine 1 , printing 0
go routine 1 , printing 1
go routine 1 , printing 2
```
We can achive the same effect using a buffered channel like below:

```go
package main

import "fmt"
import "sync"

func main(){
    var wg sync.WaitGroup
    wg.Add(3)
    ch := make(chan int, 1) //buffer with 1
    for i:=0; i<3; i++ {
        go func(i int){
            //Critical Section
            defer wg.Done()
            ch <- 1 //Other routine with block
            for  j:=0; j<3; j++ {
                fmt.Printf("go routine %d , printing %d\n",i, j)
            }
            <-ch //Other routine can now proceed
            //End of Critical Section
        }(i)
    }
    wg.Wait()
}
```

In this case, the buffered channel has a size of 1. Whenever a go routine enters the critical section, it sends a unit of data, and its job is done, the unit of data is pulled out. Thus, other go routines can proceed.
