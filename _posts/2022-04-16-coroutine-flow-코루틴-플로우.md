---
title: coroutine flow (코루틴 플로우)
authors: jongin_kim
date: 2022-04-16 00:00:00 +0900
categories: [kotlin]
tags: [kotlin, coroutine]
---
- 일시 중단 함수는 단일 값을 비동기적으로 반환한다.
- 그럼 어떻게 비동기적으로 계산된 여러 값을 반환할 수 있을까?
    - 이게 코루틴 플로우를 관통하는 질문이다.
- 여기서 kotlin coroutine flow 가 등장한다!

## 여러 값 표현

`collections`을 사용해 코틀린에서 여러 값을 나타낼 수 있다.

```kotlin
fun simple(): List<Int> = listOf(1, 2, 3)
 
fun main() {
    simple().forEach { value -> println(value) } 
}

1
2
3
```

## 시퀀스 (**Sequences)**

- CPU를 소모하는 일부 블로킹 코드(각 계산에 100ms 소요)를 통해 계산하는 경우 Sequences를 사용해 결과를 나타낼 수 있다.

```kotlin
fun simple(): Sequence<Int> = sequence { // sequence builder
    for (i in 1..3) {
        Thread.sleep(100) // pretend we are computing it
        yield(i) // yield next value
    }
}

fun main() {
    simple().forEach { value -> println(value) } 
}

1
2
3
```

이 코드는 위 예제와 동일한 결과를 출력하지만 각 결과를 출력하기 전에 100ms를 기다린다.

## 일시중단 함수

그러나 위 계산은 코드를 실행하는 주 스레드를 차단한다!

이러한 값이 비동기 코드로 계산되면 simple 함수를 suspend 함수로 바꿔 차단 없이 작업을 수행하고 결과를 목록으로 반환할 수 있다.

```kotlin
suspend fun simple(): List<Int> {
    delay(1000) // pretend we are doing something asynchronous here
    return listOf(1, 2, 3)
}

fun main() = runBlocking<Unit> {
    simple().forEach { value -> println(value) } 
}

1
2
3
```

이 코드는 1초 동안 기다린 후 결과를 출력한다.

## Flows

이제 flow가 나온다.

`List<Int>`타입을 사용해 결과를 나타내면 한번에 모든 값만 반환할 수 있다.

비동기식으로 계산되는 값의 스트림을 나타내기 위해 우리는 `Flow<Int>`타입을 사용할 수 있다. 

마치 동기식으로 계산된 값에 대해 `Sequence<Int>`타입을 사용하는 것 처럼

```kotlin
fun simple(): Flow<Int> = flow { // flow builder
    for (i in 1..3) {
        delay(100) // pretend we are doing something useful here
        emit(i) // emit next value
    }
}

fun main() = runBlocking<Unit> {
    // Launch a concurrent coroutine to check if the main thread is blocked
    launch {
        for (k in 1..3) {
            println("I'm not blocked $k")
            delay(100)
        }
    }
    // Collect the flow
    simple().collect { value -> println(value) } 
}

I'm not blocked 1
1
I'm not blocked 2
2
I'm not blocked 3
3
```

이 코드는 메인 스레드를 차단하지 않고 각 결과를 출력하기 전에 100ms를 기다립니다. 

메인 스레드에서 실행되는 별도의 코루틴에서 100ms마다 "I'm not blocking"을 출력함으로써 확인할 수 있다.

- Flow 타입의 빌더 함수를 `flow` 라고 한다
- `flow { ... }` 빌더 블록 내부의 코드는 **일시 중단될 수 있다.**
- simple 함수는 더이상 `suspend`로 표현하지 않는다!
- 값은 `emit`을 사용해 flow에서 방출한다.
- `collect`를 사용해 흐름에서 값을 수집한다 (그 외 여러가지 연산자 있음 마치 rxJava같이)

## Flows는 차갑다!

Flow는 Sequence와 유사한 `cold 스트림`이다.

flow빌더 내부의 코드는 flow가 수집될 때까지 실행되지 않는다 (= 차갑다)

```kotlin
fun simple(): Flow<Int> = flow { 
    println("Flow started")
    for (i in 1..3) {
        delay(100)
        emit(i)
    }
}

fun main() = runBlocking<Unit> {
    println("Calling simple function...")
    
		**val flow = simple()**

    println("Calling collect...")
    flow.collect { value -> println(value) } 
    println("Calling collect again...")
    flow.collect { value -> println(value) } 
}

Calling simple function...
Calling collect...
Flow started
1
2
3
Calling collect again...
Flow started
1
2
3
```

이 결과를 보면 알 수 있다.

flow를 반환하는 simple 함수가 `suspend`가 붙지 않는 이유를 

그 자체로 simple() 호출은 빠르게 바로 반환되고 아무것도 기다리지 않는다.

flow는 collect될때마다 시작되므로 다시 collect를 호출할때 `Flow started`가 출력된다.