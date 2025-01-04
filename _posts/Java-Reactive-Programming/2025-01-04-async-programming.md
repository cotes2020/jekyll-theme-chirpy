---
title: "비동기 프로그래밍(동기/비동기, block/non-block)"
categories: [Java, Reactive Programming]
tags: [async, 비동기 프로그래밍, 비동기api, blocking/non-blocking]
---

<!-- prettier-ignore -->
>  **비동기 프로그래밍?**
> 
> - **작업 실행의 완료를 기다리지 않고, 다른 작업을 동시에 진행할 수 있는게 하는 프로그래밍 방식**
> - 멀티스레딩과 유사하지만, 비동기 프로그래밍은 단일 스레드에서 이벤트 처리 및 콜백을 통해 구현되기도 하기 때문에 명시적 스레드 관리 없이도 동시성을 달성할 수 있다.
{: .prompt-info }

## 동기 API 와 비동기 API의 차이

![Image]({{"/assets/img/posts/2025-01-04-22-37-48.png" | relative_url }})

### `동기 vs 비동기`

- **작업 완료 처리의 주체**
- 동기: 호출자가 작업 완료 처리 (호출자가 직접 처리 결과를 반환하여 획득)
- 비동기: 메서드를 수행하는 피호출자가 작업 완료 처리 (처리 결과를 호출자에게 전달)

### `Blocking vs Non-Blocking`

- **함수의 처리값이 언제 반환되는지**
- blocking: 작업 완료 후 반환
- non-blocking: 즉시 반환
  - 💡 반환된 것은 최종 결과가 아닌 프로그램 제어권을 return
    - 프로그램 제어권? 비동기작업을 추적할 수 있는 권한

<!-- prettier-ignore -->
>  **용어 정리 ✏️**
>
> - 호출자: 메서드를 호출한 주체
> - 피호출자: 메서드를 실제로 수행하는 함수
{: .prompt-tip }

## 동기 + Blocking

- 호출자가 메서드 호출한 뒤, 메서드가 계산을 완료할 때 까지 대기한다.

- 메서드가 반환된 후, 반환된 값으로 다른 동작을 수행한다.

- 메서드를 호출한 호출자와 메서드를 수행하는 피호출자가 각각 다른 스레드에서 실행되는 상황에서도 호출자는 피호출자의 동작 완료를 대기한다.

- 메서드를 호출하고 반환될 때까지 대기하는 동기 API를 사용하는 상황을 **블록 호출**이라고 한다.

- ```java
  public class SyncBlockingExample {

      public static void main(String[] args) {
          System.out.println("Before method call");
          String result = performBlockingOperation();
          System.out.println("After method call: " + result);
      }

      private static String performBlockingOperation() {
          // 시간이 걸리는 작업을 가정
          try {
              Thread.sleep(2000); // 2초 대기
          } catch (InterruptedException e) {
              Thread.currentThread().interrupt();
          }
          return "Operation Completed";
      }
  }

  // 실행결과
  // Before method call
  // After method call: Operation Completed
  ```

## 동기 + Non-Blocking

- 메서드를 호출한 뒤, 메서드는 비동기 작업 추적 객체를 즉시 반환된다.

- 그 동안 호출자는 다른 작업을 할 수 있다.

- 호출자는 반환된 비동기 작업 추적 객체를 통해 피호출자에 작업완료 여부를 확인한다.

- 완료 응답을 받을 때까지 반복한다.

- 호출자는 완료 응답을 받으면 작업완료 여부 확인을 중지한다.

- ```java
  // Runnable 인터페이스를 구현하는 클래스 정의
  class MyTask implements Runnable {
      @Override
      public void run() {
          // 비동기로 실행할 작업
          System.out.println("Hello from a thread!");
      }
  }

  public class Main {
      public static void main(String[] args) {
          // Thread 객체 생성
          Thread thread = new Thread(new MyTask());

          // 스레드 실행
          thread.start();

          // Non-Blocking이므로 다른 작업 계속 가능
          System.out.println("Main thread is running...");

          // Sync를 위해 스레드의 작업 완료 여부 확인
          while (thread.isAlive()) {
              System.out.println("Waiting for the thread to finish...");
          }
          System.out.println("Thread finished!");

          System.out.println("Run the next tasks");
      }
  }

  // 실행결과
  // Main thread is running...
  // Waiting for the thread to finish...
  // Waiting for the thread to finish...
  // Waiting for the thread to finish...
  // Hello from a thread!
  // Thread finished!
  // Run the next tasks
  ```

## 비동기 + Blocking

- 메서드를 호출한 뒤, 호출자가 '작업이 완료될 때까지 대기' 메서드를 추가로 호출하여(`block()`, `get()` 등) 피호출자에게 전달한다.

- 호출자의 thread는 비동기 작업이 완료될 때 까지 실행을 멈추고 기다린다. => blocking

- 피호출자의 작업이 완료되면 callback 함수를 실행하여, 호출자에게 작업 결과를 반환한다.

- => 일반적으로 비동기 프로그래밍을 할 때, sleep() 이나 기타 블로킹 동작은 스레드 내에서 다른 태스크의 실행을 막기 때문에 최대한 배재하는 것이 좋다.``

- ```java
  @RestController
  public class BlockingController {

      @GetMapping("/block")
      public String block() {
          // Mono에서 값 가져오기(블로킹)
          String result = generateData()
                  .block(); // 비동기 처리 결과를 대기(Blocking)

          return "Blocking result: " + result;
      }

      private Mono<String> generateData() {
          return Mono.just("Some data")
                     .delayElement(Duration.ofSeconds(2)); // 비동기 처리를 위해 2초 지연
      }
  }

  // 실행결과
  // /block 호출 -> 바로 Mono 객체 반환 받음
  // block() 메서드가 호출되어, Mono에 의해 비동기 작업 결과 수행 2초간 대기 -> 블록 상태
  // 비동기 작업 완료 후, "Some data" 문자열 반환하여 block() 함수에 전달
  // block() 함수에서 최종 응답 반환 Blocking result: Some data
  // "/block" API의 HTTP 응답을 클라이언트에 전송
  ```

## 비동기 + Non-Blocking

- 메서드 호출한 뒤, 메서드는 계산 완료되기를 기다리지 않고 즉시 반환된다. (반환된 것은 최종 결과가 아닌 프로그램 제어권을 return)

  - 프로그램 제어권? 비동기작업을 추적할 수 있는 권한

- 완료되지 못한 계산을 호출자 스레드와 동기적으로 실행될 수 있도록 다른 스레드에 할당한다. 그동안 호출자의 thread에서는 다른 작업의 진행이 가능하다.

- 다른 스레드에 할당된 나머지 계산의 결과는 콜백 메서드를 통해 전달된다.

  - 호출자에 의해 선언

  - 실행은 비동기 작업이 완료될 때, 비동기 작업 수행중인 스레드에서 실행

- ```java
  import org.springframework.web.bind.annotation.GetMapping;
  import org.springframework.web.bind.annotation.RestController;
  import reactor.core.publisher.Mono;

  @RestController
  public class NonBlockingController {

      @GetMapping("/non-block")
      public Mono<String> nonBlock() {
          // 비동기적으로 데이터 생성 및 반환
          return generateData()
                  .map(data -> "Non-Blocking result: " + data);
      }

      private Mono<String> generateData() {
          return Mono.just("Some data")
                     .delayElement(Duration.ofSeconds(2)); // 비동기 처리를 위해 2초 지연
      }
  }

  // 실행결과
  // /non-block 호출 -> 바로 Mono 객체 반환받음
  // map() 연산을 통해 데이터가 준비되면, "Non-Blocking result: " 문자열과 결합하는 처리가 정의
  // - 실제 데이터 생성과 처리 완료를 기다리지 않고, 초기 HTTP 응답(헤더 등) 생성
  // - 데이터 처리가 완료되면 그 결과를 응답 본문에 포함하여 클라이언트에 전송
  // 2초 뒤 (2초 동안 nonBlock() 에서는 다른 작업 수행 가능-논블록 상태), "Some data" 문자열이 생성되고,
  // "Non-Blocking result: " 문자열과 결합하여 최종 문자열이 HTTP 응답 본문으로 클라이언트에 전송
  ```

## 비동기 프로그래밍 구현 방법

- **`Future` 인터페이스를 통한 `CompletableFutrue` 구현**

  - Java 8에서 도입된 `CompletableFuture`는 `Future`의 개선된 버전으로, 비동기 계산의 결과를 표현

  - `CompletableFuture`는 함수형 프로그래밍 방식과 결합하여 사용될 수 있으며, 비동기 연산을 파이프라인화하고 결과를 조합하고 변환하는 등의 고급 비동기 프로그래밍 기능을 제공

  - 일회성 값을 처리하는데 적합

- **Reactive Programming**

  - 발행-구독 프로토콜 기반
  - 비동기 데이터 흐름(스트림으로 처리)과 이벤트 기반 프로그래밍에 적합
