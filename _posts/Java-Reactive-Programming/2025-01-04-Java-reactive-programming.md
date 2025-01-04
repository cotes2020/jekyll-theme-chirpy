---
title: "Reactive Programming (with Spring WebFlux)"
categories: [Java, Reactive Programming]
tags: [async, 비동기 프로그래밍, 비동기api, blocking/non-blocking]
---

## 컨셉

![Image]({{"/assets/img/posts/2025-01-04-22-28-47.png" | relative_url }}){: .shadow }

<!-- prettier-ignore -->
> **주요 키워드📝**
> - **반응성**
> - **회복성**
> - **탄력성**
> - **메시지 주도**
{: .prompt-info }

## 이벤트 루프 기반 프로그래밍

- 웹에서 사용자의 클릭, 입력과 같은 이벤트가 발생할 때 마다, 해당 이벤트는 데이터 스트림으로 처리
- 새로운 데이터 스트림이 도착할 때 마다 반응(reacting)으로 프로그램이 동작
- 즉, 새로운 데이터가 스트림에 도착할 때 마다 자동으로 이를 감지하고 필요한 작업을 실행(데이터 처리, 조회, UI 업데이트 등..)

|             | 데이터 폴링                                                                                       | 이벤트 루프 기반                                                                                                         |
| ----------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| 데이터 감지 | 주기적 확인<br/>- 시스템이 정기적으로 데이터의 변경 여부를 확인                                   | 새로운 이벤트 발생 시 자동 반응<br/>- <u>시간에 따라 발생</u>하는 이벤트를 데이터 스트림에 할당                          |
| blocking    | blocking<br/>- 폴링 작업이 수행되는 동안 시스템은 새로운 데이터 대기 혹은 폴링작업 수행 완료 대기 | non-blocking<br/>- 이벤트 발생 시 즉시 처리<br/>- 시스템이 다른 작업 수행 시에도 새로운 이벤트 감지하고 있음             |
| 데이터 모델 | 데이터 풀 모델<br/>- 데이터의 업데이트를 가져오기 위해 데이터 소스에 지속적 요청                  | 데이터 푸시 모델<br/>- 데이터나 이벤트는 자동으로 관련 리스너에 푸시됨<br/>- 데이터 변경 시 시스템에 의해 자동 감지&처리 |
| 리소스 사용 | 새로운 데이터가 없어도 폴링이 계속 수행 => 자원의 낭비                                            | 데이터 변경 시 즉시 처리 => 불필요한 자원 사용 감소, 실시간 반응                                                         |

## 발행-구독 패턴

### 주요 개념

- **발행자 (Publisher)**

  - 데이터나 이벤트를 생성하고 배포하는 역할

  - 새로운 데이터가 생길때마다 "발행"함

  - 피호출자의 역할

- **구독자 (Subscriber)**

  - 발행자를 **구독(Subscription)** 하고 있다.

  - 발행자가 새로운 데이터를 발행하면, 새로운 데이터가 이벤트를 받아 처리한다.

  - 호출자의 역할

- **작업 완료 처리**

  - **동기**: 구독자가 작업의 완료를 직접 관리. 작업이 완료될 때까지 기다린 후 결과 반환

  - **비동기**: 발행자가 작업 완료를 관리. 구독자가 신경쓰지 않아도, 새로운 이벤트나 데이터가 준비되면 발행자로부터 알림을 받고 처리.

### 처리 흐름

![Image]({{"/assets/img/posts/2025-01-04-22-25-20.png" | relative_url }})
![Image]({{"/assets/img/posts/2025-01-04-22-26-17.png" | relative_url }})

1. 구독자는 발행자를 구독한다.

2. 구독 시 구독자는 발행자로부터 구독 객체(SubScription)를 받는다. Subscription 객체의 `onSubscribe` 함수에 새로운 데이터/이벤트 발생 시 수행할 동작이 정의된다.

3. 구독자는 새로운 데이터/이벤트 발생 시, 발행자에게 몇개의 데이터를 받을 지 요청 사항을 SubScription을 통해 발행자에게 전달한다.`=> request(n)` (혹은 구독 취소한다. `=> cancel()`)

4. 발행자가 새로운 데이터/이벤트를 생성하면, 발행자는 구독자의 요청에 따라 데이터를 전송한다. 이를 통해 구독자는 요청한 만큼의 데이터를 획득한다. (데이터 양 제어)

5. 모든 값을 다 받거나, 에러 발생으로 인해 더 이상 처리할 데이터가 없는 경우 종료한다.

> 구독 메서드
>
> - `onNext()`: 값이 있을 때 처리
>
> - `onError()`: 수행 도중 에러가 발생했을 때 처리
>
> - `onComplete()`: 값을 다 소진 or 에러발생으로 인해 더이상 처리할 데이터가 없을 때 처리

### 코드 예시

- 발행자/구독자를 직접 구현한 예시(플로 API) - 신문 구독 예시

```java
    public static void main(String[] args) {

        // 발행자: 신문사
        Publisher publisher = new Publisher() {
            // 4. 신문사는 새로운 신문을 발간한다.
            Iterable<Integer> iter = Arrays.asList(1, 2, 3, 4, 5);

            @Override
            public void subscribe(Subscriber subscriber) {
                Iterator<Integer> iterator = iter.iterator();
                // 2. 신문사는 신문 발간 시, 구독자에게 알려준다. (콜백으로 호출)
                subscriber.onSubscribe(new Subscription() {
                    @Override
                    public void request(long n) {
                        while (n-->0) {
                            if (iterator.hasNext()) {
                                // 5. 중개자는 구독자에게 신문을 보내준다.
                                subscriber.onNext(iterator.next());
                            } else {
                                // 더이상 발행할 신문이 없다면 구독자에게 모든 신문이 발행되었음을 알림
                                subscriber.onComplete();
                                break;
                            }
                        }
                    }

                    @Override
                    public void cancel() {

                    }
                });
            }
        };


        //구독자
        Subscriber<Integer> subscriber = new Subscriber<>() {
            Subscription subscription;

            @Override
            public void onSubscribe(Subscription subscription) {
                // 구독자는 신문 구독권을 가지고 있다.
                this.subscription = subscription;
                System.out.println("구독");
                // 3. 구독자는 신문을 몇 장 받을 수 있는지 구독권에 할당한다. <= 역압력
                subscription.request(2);
            }

            @Override
            public void onNext(Integer item) {
                // 5. 구독자는 신문을 받아 읽는다.
                System.out.println(item+ " 신문 읽기");
            }

            @Override
            public void onError(Throwable throwable) {
                System.out.println("onError");
            }

            @Override
            public void onComplete() {
                System.out.println("onComplete");
            }
        };

        Subscriber<Integer> s = subscriber;

        // 1. 구독자가 신문을 구독한다
        publisher.subscribe(subscriber);

    }


    // 실행결과
    // 구독
    // 1 신문 읽기
    // 2 신문 읽기
```

- Webflux

```java
import reactor.core.publisher.Flux;

public class SimpleReactiveExample {

    public static void main(String[] args) {
        // 문자열 목록을 발행하는 발행자(Publisher) 생성
        Flux<String> fruitFlux = Flux.just("Apple", "Banana", "Cherry", "Date");

        // 구독자(Subscriber)가 발행자를 구독하고, 데이터 처리 방법을 정의
        fruitFlux.subscribe(
            fruit -> System.out.println("Here's a fruit: " + fruit), // onNext: 스트림으로부터 새로 데이터 항목을 받았을 때의 처리
            error -> System.err.println("Something went wrong: " + error), // onError: 에러 발생 시의 처리
            () -> System.out.println("All fruits have been delivered!") // onComplete: 모든 데이터 처리가 완료되었을 때의 처리
        );
    }
}
```

### 장점

- 이벤트가 생길때마다 실시간으로 이를 처리할 수 있는 유연하고 효율적 방법 제공

- 리액티브 시스템에서 발행-구독 모델을 통해 데이터 스트림의 변화에 신속하게 반응

- 시스템 전반의 비동기성과 논블로킹 동작을 가능하게 함

<!-- prettier-ignore -->
> 주요 개념: **`역압력 (백프레셔)`**
>
> - 데이터의 발행자 <-> 구독자 사이에 데이터가 비동기적으로 전송될 때,
> - 발행자가 발행하는 데이터의 속도가 구독자가 처리할 수 있는 속도보다 빠르면, 구독자는 과부하에 빠질 수 있다.
> - 역압력은 이를 방지하기 위해 **구독자가 자신의 처리 능력에 맞게 데이터의 수신 속도를 조절**할 수 있게 하는 매커니즘이다.
{: .prompt-tip }

---

## Spring WebFlux

> 리액티브 프로그래밍을 지원하기 위해 설계된 모듈로,
>
> 비동기적으로 논블로킹 방식의 웹 어플리케이션 개발을 위해 사용된다.

### 장점

- 높은 동시성을 처리

- 고성능을 요구하는 웹 환경에서 유리

### 특징

- 비동기적 & 논블로킹 처리에 최적화

- 리액티브 스트림 지원

- 함수형 프로그래밍 모델 지원

  - 어노테이션 기반 프로그래밍 뿐 아니라 라우터 함수/핸들러 함수를 사용한 함수형 프로그래밍 모델 지원

- 다양한 클라이언트 지원

  - `WebClient`, `WebSocketClient` 등 클라이언트 제공하여, HTTP, SSE, SebSocket 을 통한 비동기 및 논블로킹 통신 지원

### 사용법

- 의존성 라이브러리 추가 (gradle version)

  ```groovy
  dependencies {
      implementation 'org.springframework.boot:spring-boot-starter-webflux'
  }
  ```

#### 1. 어노테이션 기반 컨트롤러 - 기본 케이스

- 컨트롤러 생성

  - `@RestController` 어노테이션을 사용하여 리액티브 컨트롤러를 생성하고, `Mono` 또는 `Flux`를 반환하도록 메서드를 정의

  - `Mono<T>`: 0 또는 1개의 결과를 제공하는 리액티브 타입 (주로 단건 데이터에 사용)

  - `Flux<T>`: 0개 이상의 결과를 제공하는 리액티브 타입 (주도 다건 데이터에 사용)

    ```java
    import org.springframework.web.bind.annotation.GetMapping;
    import org.springframework.web.bind.annotation.RestController;
    import reactor.core.publisher.Mono;

    @RestController
    public class HelloController {

     @GetMapping("/hello")
     public Mono<String> hello() {
         return Mono.just("Hello, WebFlux!");
     }
    }
    ```

#### 2. 어노테이션 기반 컨트롤러 - HTTP 요청 케이스

- Spring Webflux의 `WebClient`를 사용하요 외부 API를 호출 후 결과 처리

- `WebClient`는 비동기/논블로킹 방식으로 외부 HTTP요청을 수행할 수 있는 Spring Webflux의 클라이언트

- ```java
  import org.springframework.web.bind.annotation.GetMapping;
  import org.springframework.web.bind.annotation.RestController;
  import org.springframework.web.reactive.function.client.WebClient;
  import reactor.core.publisher.Mono;

  @RestController
  public class WebClientExampleController {

      private final WebClient webClient;

      public WebClientExampleController(WebClient.Builder webClientBuilder) {
          this.webClient = webClientBuilder.baseUrl("http://example.com").build(); // 외부 API의 기본 URL 설정
      }

      @GetMapping("/external-data")
      public Mono<String> getExternalData() {
          return this.webClient.get() // HTTP GET 요청
                               .uri("/data") // 요청할 URI
                               .retrieve() // 응답 받기
                               .bodyToMono(String.class); // 응답 본문을 String으로 변환
      }
  }
  ```

### 연산자 및 함수

#### 데이터 출력

- **`bodyToMono(Class<? extends T> elementClass)`** / **`bodyToFlux(Class<? extends T> elementClass)`**

  - 웹클라이언트의 응답 본문을 Mono 혹은 Flux로 변환할 때 사용
  - HTTP 요청의 결과로 받은 데이터를 Mono 혹은 Flux 타입으로 매핑
  - ```java
    WebClient.create().get()
        .uri("http://example.com")
        .retrieve()
        .bodyToMono(String.class)
        .subscribe(System.out::println);
    ```

#### 데이터 연산/처리

- **`flatMap(Function<? super T, ? extends Publisher<? extends R>> mapper)`**

  - `flatMap` 연산자는 스트림의 각 항목에 대해 비동기 연산을 수행하고, 그 결과로 새로운 `Publisher`를 생성
  - 이 연산자는 중첩된 비동기 작업의 결과를 평탄화(flatten)하여 하나의 스트림으로 생성
  - 주로 연쇄적인 비동기 호출이 필요할 때 사용
  - ```java
    Flux.just(1, 2, 3)
        .flatMap(i -> Mono.just(i * 2))
        .subscribe(System.out::println); // 2, 4, 6 출력
    ```

- **`map(Function<? super T, ? extends R> mapper)`**

  - 스트림의 각 항목에 동기적인 변환 작업을 수행

  - `flatMap`과 유사하지만, 변환 작업이 비동기적이지 않다는 차이

#### 데이터 소비

- **`subscribe()`**

  - 리액티브 스트림을 구독하고, 스트림의 데이터의 처리를 시작

  - 리액티브 스트림을 소비하는 가장 기본적 방법

  - 이 메서드에는 데이터 항목 처리(`onNext`), 에러 처리(`onError`), 스트림 완료 처리(`onComplete`) 등에 대한 콜백을 제공

  - 최대 3개의 람다 표현식을 인자로 받는다.

    - 첫번째: **`onNext` 처리 함수** => 스트림에서 전달된 각 아이템을 처리, 스트림의 각 요소가 발행될 때마다 호출

    - 두번째: **`onError` 처리 함수** => 에러 발생 시 호출되며, 발생한 예외를 파라미터로 받음. 발생 이후 종료.

    - 세번째: **`onComplete` 처리 함수** => 스트림이 성공적으로 완료되었을 때 호출, 더이상 스트림에 처리할 데이터 없음, 정상 종료

  - ```java
    import reactor.core.publisher.Flux;

    public class SubscribeExample {
        public static void main(String[] args) {
            Flux<String> fruitFlux = Flux.just("Apple", "Banana", "Cherry", "Date");

            fruitFlux.subscribe(
                fruit -> System.out.println("Here's a fruit: " + fruit), // onNext에 대응
                error -> System.err.println("Error: " + error), // onError에 대응
                () -> System.out.println("All fruits have been delivered!") // onComplete에 대응
            );
        }
    }
    ```

- **`then(Mono<? extends V> other)`**

  - `Mono`/`Flux`의 처리가 완료된 후에, 다른 `Mono`를 실행

  - 한 작업의 완료 후 다른 작업을 연쇄적으로 실행하려고 할 때 사용

  - **순차적인** 비동기 작업을 처리할 때 사용

  - ```java
    import reactor.core.publisher.Mono;

    public class ThenExample {
        public static void main(String[] args) {
            Mono.just("Hello")
                .doOnNext(System.out::println) // 첫 번째 작업: 문자열 출력
                .then(Mono.just("World"))
                .doOnNext(System.out::println) // 두 번째 작업: 다른 문자열 출력
                .subscribe(); // 구독하여 처리 시작
        }
    }
    ```

<!-- prettier-ignore -->
> **`subscribe` vs `then` ?**
>
> - `subscribe`: 스트림의 소비를 시작
>
> - `then`: 이전 작업이 완료된 후 순차적으로 다음 작업을 사용
{: .prompt-tip }

---

## CS Portal 사용 예시

CS Portal에서는 **Gitlab의 API를 호출하여 결과값을 반환받는 controller**와 **서버에 파일을 업로드하는 controller**에서 WebFlux를 사용하였습니다.

### Gitlab API 호출 케이스

```java
/**
 * 이슈 ID로 이슈 상세 조회
 *
 * @param projectId
 * @param issueId
 * @return
 */
public Mono<Issue> getIssueByIssueId(int projectId, int issueId) {

  return WebClient.create(getGitlabHost()).get()
      .uri(uriBuilder -> uriBuilder.path(
              getGitlabApiUri() + "/projects/{projectId}/issues/{issueId}")
          .build(projectId, issueId))
      .header("Authorization", "Bearer " + getGroupToken())
      .accept(MediaType.APPLICATION_JSON)
      .retrieve()
      .bodyToMono(Issue.class);
}
```

- WebClient에 host주소를 할당하여 create 하여 생성

- get() method로 선언

- uri를 통해 api path 선언

- build 하여 HTTP 요청 객체 생성

- accept를 통해 응답받을 데이터 타입 지정

- retrieve를 통해 반환 결과 획득

- bodyToMono 지정한 데이터 타입으로 매핑

### 파일 업로드 케이스

- **Controller**

  - 단건파일일 경우 `Mono<FilePart>` 타입으로, 다건파일인 경우 `Flux<FilePart>` 타입으로 데이터 타입을 지정

  - FilePart: Webflux에서 멀티파트 파일 업로드를 다룰 때 사용하는 인터페이스

  - `@RequestPart` 어노테이션으로 지정

```java
/**
 * 이슈 혹은 코멘트 내용 중간에 첨부 파일이 존재하는 경우 서버에 파일 업로드 (단건 단위)
 *
 * @param filePart
 * @return
 */
@PostMapping("/uploadToServer")
public Mono<String> uploadToServer(
    @RequestPart("file") Mono<FilePart> filePart,
    @RequestParam(name = "path") String path) {
  return issueService.uploadToServer(filePart, path);
}
```

- **Service**

  - I/O 작업을 비동기적으로 실행

```java
public Mono<String> uploadToServer(Mono<FilePart> filePartMono, String imagePath) {
  // 프로젝트 루트 경로
  String projectRootPath = System.getProperty("user.dir");

  // 저장할 파일의 경로
  String fullFilePath = projectRootPath + imagePath;

  // 파일을 저장할 경로에 대한 Path 객체 생성
  Path basePath = Paths.get(fullFilePath);

  // 입력받은 파일 `filePartMono`를 flatMap을 사용하여 스트림의 각 항목에 대한 처리를 정의
  return filePartMono
      .flatMap(fp -> Mono.fromCallable( // 에러 처리를 스트림의 일부로 통합하고, 에러가 발생했을 때 리액티브 스트림에 의해 처리
              () -> {
                // 파일이 존재하지 않는 경우 디렉토리 생성
                if (!Files.exists(basePath)) {
                  Files.createDirectories(basePath);
                }
                // 파일 경로에 파일명을 추가하여 저장할 최종 경로 생성
                return basePath.resolve(fp.filename());
              })
          // 비동기적으로 파일 저장
          .flatMap(filePath -> fp.transferTo((Path) filePath))
      )
      // 파일 저장 완료 후 imagePath 반환
      .then(Mono.just(imagePath))
      // 에러 발생 시 처리
      .onErrorResume(e -> {
        return Mono.error(new RuntimeException(e.getMessage(), e));
      });
}
```

---

[참고]

- 책: 모던 자바 인 액션(2019), 라울 가브리엘 우르마 저
