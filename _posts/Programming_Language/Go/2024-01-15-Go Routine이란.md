---
title: Go Routine 이란?
date: 2024-01-15 15:54:32 +0900
author: kkankkandev
categories: [Programming Language, Go]
tags: [go, go-routine, go routine, goroutine, concurrency, parallelism]     # TAG names should always be lowercase
comments: true
image:
  path: https://github.com/War-Oxi/war-oxi.github.io/assets/72260110/89734e1d-6afb-45dc-a6f9-8da0266bf529
---
# Go Routine - [ Concurrency vs Parallelism ]

> Go Routine은 동시성과 병렬성을 매우 간결하고 효과적으로 다룰 수 있는 기능을 제공합니다.
> 

### 동시성(Concurrency)
- **싱글 코어에서 멀티 쓰레드 동작**
- 여러 작업을 시간을 나누어 사용함으로써 동시에 실행되는 것처럼 보이는 기술입니다.
- 실제로는 한 순간에 하나의 작업만 처리하지만, 작업들 사이를 빠르게 전환하면서 동시에 진행되는 것처럼 보이게 합니다.
- 단일 코어 환경에서 효율적인 자원사용과 빠른 응답 시간을 목표로 사용됩니다.

### 병렬성(Parallelism)
- **멀티 코어에서 멀티 쓰레드 동작**
- 여러 작업을 실제로 동시에 실행하는 기술입니다.
- 멀티코어 프로세서를 사용하며, 각 코어에서 별도의 작업을 동시에 수행합니다.
- 멀티코어 환경에서 성능을 극대화하기 위해 사용됩니다.

### 다중 CPU 처리

> `runtime.GOMAXPROCS()` 함수는 프로그램이 동시에 실행할 수 있는 최대 CPU 코어 수를 설정합니다.
Go 1.5 버전부터, `runtime.GOMAXPROCS()`의 디폴트 값은 시스템에서 사용 가능한 물리적 CPU 코어의 수로 설정되어 있습니다.
> 

### 예시 (Basic)

```go
package main

import (
	"fmt"
	"time"
)

func printHelloWorld(strIn string) {
	for i := 0; i < 10; i++ {
		fmt.Println(strIn, "hello world", i)
	}
}

func main() {
	// 기존 -> 동기적
	printHelloWorld("Sync")

	// Go Routine -> 비동기적
	go printHelloWorld("Async1")
	go printHelloWorld("Async2")
	go printHelloWorld("Async3")

	time.Sleep(time.Second * 3)
}
```

### 예시 (anonymous function)
```go
package main

import (
	"fmt"
	"sync"
)

func main() {
	// WaitGroup 생성. 2개의 Go Routine이 끝날때까지 기다리기
	var wait sync.WaitGroup
	wait.Add(2)

	go func() {
		defer wait.Done()
		fmt.Println("Hello")
	}()

	go func(msg string) {
		defer wait.Done()
		fmt.Prinln(msg)
	}("Hi")

	wait.Wait() //Go루틴이 모두 끝날 때까지 대기
}
```


<strong>궁금하신점이나 추가해야할 부분은 댓글이나 아래의 링크를 통해 문의해주세요.</strong>   
Written with [KKam.\_\.Ji](https://www.instagram.com/kkam._.ji/)
