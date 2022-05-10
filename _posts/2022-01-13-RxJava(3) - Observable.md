---
title: RxJava(3) - Observable
author: Bean
date: 2022-01-13 16:13:00 +0800
categories: [Mobile frontend, Android]
tags: [RxJava]
layout: post
current: post
class: post-template
subclass: 'post'
navigation: True
cover:  assets/img/post_images/rxjava_cover.jpeg
---


## Observable 이란
***

Observable은 RxJava의 가장 핵심적인 요소이다. Observable을 잘 이해하는 것은 RxJava를 활용하는 데 매우 중요하다. 옵저버 패턴의 정의는 다음과 같다.

> 옵저버 패턴(observer pattern) 은 객체의 상태 변화를 관찰하는 관찰자들, 즉 옵저버들의 목록을 객체에 등록하여 상태 변화가 있을 때마다 메서드 등을 통해 객체가 직접 목록의 각 옵저버에게 통지하도록 하는 디자인 패턴이다. 주로 분산 이벤트 핸들링 시스템을 구현하는 데 사용된다. 발행/구독 모델로 알려져 있기도 하다.
> [옵저버 패턴 - 위키백과](https://ko.wikipedia.org/wiki/%EC%98%B5%EC%84%9C%EB%B2%84_%ED%8C%A8%ED%84%B4)

예를 들면, 안드로이드에서 Button이 클릭되었을 때 실행할 함수를 onclicklistener에 추가하는데 이와 같이 이벤트 핸들링 처리를 위해 사용되는 패턴이다. 이 패턴에는 Observable과 Observer가 등장한다.

* **Osbservable**: 등록된 Observer들을 관리하며, 새로운 데이터(이벤트)가 들어오면 등록된 Observer에게 데이터를 전달한다. 데이터를 생성해서 전달하기 때문에 Publisher(발행)라고 부른다.
* **Observer**: Observable로 부터 데이터(이벤트)를 받을 수 있다. 데이터를 전달 받기 때문에 Subscriber(구독)라고 부른다.

이처럼 Observable은 데이터 흐름에 맞게 알림을 보내 Observer가 데이터를 사용할 수 있도록 한다. 즉, Observable을 이용해 데이터를 회수하고 변환하는 메커니즘을 정의하고, Observer은 이를 구독해 데이터가 준비되면 이에 반응한다.

![Image](/assets/img/post_images/rxjava3-1.png)

그림에서 처럼 Observer pattern은 다음의 순서로 동작한다.

1. Observable이 데이터 스트림을 처리하고, 완료되면 데이터를 발행(**emit**)한다.
2. 데이터를 발행할 때마다 구독(**Subscribe**)하고 있는 모든 Observer가 알림을 받는다.
3. Observer는 수신한 데이터를 가지고 어떠한 일을 한다. (데이터를 소비(**Consume**)한다.)

데이터를 발행할 때 null은 발행할 수 없다.

\
&nbsp;
## Emit & Subscribe
***

### Observable의 데이터 발행 (Emit)
---

Observable이 데이터를 발행 한 후 보내는 알림에는 세 가지 종류가 있다.

```java
// Emitter를 통해 알림을 보낸다고 생각하면 된다
public interface Emitter<@NonNull T> {
    void onNext(@NonNull T value);
    void onError(@NonNull Throwable error);
    void onComplete();
}
```

* `onNext` : 데이터의 발행을 알림
* `onComplete` : 모든 데이터의 발행이 완료되었음을 알림, 딱 한 번만 발생하며 이후에 onNext가 발생하면 안됨
* `onError` : 오류가 발생했음을 알림, 이후에 onNext와 onComplete가 발생하지 않음

### Subscribe
---

구독(Subscribe)이란 단순하게 수신한 데이터를 가지고 할 행동을 정의하는 것이다. Observer는 `subsribe()` 메소드에서 수신한 각각의 알림에 대해 실행할 내용을 지정한다.

```java
public final Disposable subscribe()
public final Disposable subscribe(@NonNull Consumer<? super T> onNext)
public final Disposable subscribe(@NonNull Consumer<? super T> onNext, @NonNull Consumer<? super Throwable> onError)
public final Disposable subscribe(@NonNull Consumer<? super T> onNext, @NonNull Consumer<? super Throwable> onError, @NonNull Action onComplete)
public final void subscribe(@NonNull Observer<? super T> observer)
```

Disposable class는 구독의 정상적인 해지를 돕는다.
onComplete 이벤트가 발생하면 dispose()를 호출해 Observable이 더 이상 데이터를 발행하지 않도록 구독을 해지한다.
또한 isDisposed()를 통해 구독이 해지되었는지 확인할 수 있다.

\
&nbsp;
## Observable 생성하기
***

이제 본격적으로 Observable을 사용해보자. RxJava에서는 **연산자(Operator)** 를 통해 기존 데이터를 참조, 변형하여 Observable을 생성할 수 있다. Observable을 생성하는 함수를 **팩토리 함수** 라고 하는데, 이 팩토리 함수는 다음 표처럼 구분할 수 있다.

| 팩토리 함수 | 함수 |
| ------ | --- |
| RxJava 1.x 기본 팩토리 함수 | create(), just(), from() |
| RxJava 2.x 추가 팩토리 함수 (from() 함수 세분화) | fromArray(), fromIterable(), fromCallable(), fromFuture(), fromPublisher() |
| 기타 팩토리 함수 | interval(), range(), timer(), defer() 등 |

이 중 일부를 살펴보자.

* `just()` 함수
    ![just()](/assets/img/post_images/rxjava3_just.png)
    데이터를 발행하는 가장 쉬운 방법은 기존의 자료구조를 사용하는 것이다. just() 함수는 인자로 넣은 데이터를 차례로 발행하려고 Observable을 생성한다(실제 데이터의 발행은 subscribe() 함수를 호출해야 시작한다). 한 개의 값을 넣을 수도 있고 인자로 여러 개의 값(최대 10개)을 넣을 수도 있다. 단 타입은 모두 같아야 한다.
    ```java
    Observable<String> source = Observable.just("PIGBEAN", "Tech", "Blog");
    source.subscribe(System.out::println());
    ```

    ```
    PIGBEAN
    Tech
    Blog
    ```

* `create()` 함수
    ![create()](/assets/img/post_images/rxjava3_create.png)
    just() 함수는 데이터를 인자로 넣으면 자동으로 알림 이벤트가 발생하지만 create() 함수는 onNext, onComplete, onError 같은 알림을 개발자가 직접 호출해야 한다. 그래서 create()는 라이브러리가 무언가를 해준다기보다 개발자가 무언가를 직접 하는 느낌이 강한 함수이다.
    ```
    Observable<String> source = Observable.create(emitter -> {
        emitter.onNext("PIGBEAN");
        emitter.onNext("Tech");
        emitter.onNext("Blog");
        emitter.onComplete();
    });
    source.subscribe(System.out::println);
    ```

    ```
    PIGBEAN
    Tech
    Blog
    ```

    이처럼 실행 결과는 `just()`를 사용했을 때와 같지만 개발자가 직접 `onNext()`, `onComplete()` 를 호출해야 한다는 것에서 차이가 있다. 앞서 설명했듯이 onComplete 이후에는 아이템에 더 발행되더라도 구독자는 데이터를 받지 못한다. 또한 오류가 발생했을 때는 `onError()`를 호출해서 에러 상황을 처리해야 한다.
    ```
    Observable<String> source = Observable.create(emitter -> {
        emitter.onNext("PIGBEAN");
        emitter.onError(new Throwable());
        emitter.onNext("Tech");
        emitter.onNext("Blog");
    });
    source.subscribe(System.out::println,
        throwable -> System.out.println("Good bye")
    );
    ```

    ```
    PIGBEAN
    Good bye
    ```

    **Observable.create()를 사용할때는 주의해야 한다.**
    RxJava 문서에 따르면 create()는 RxJava에 익숙한 사용자만 활용하도록 권고한다. create()를 사용하지 않고 다른 팩토리 함수를 활용하면 같은 효과를 낼 수 있기 때문이다. 만약 그래도 사용해야 한다면 아래 사항을 확인해야 한다.

    1. Observable이 구독 해지(dispose)되었을때 등록된 콜백을 모두 해제해야 한다. 그렇지 않으면 잠재적으로 메모리 누수(memory leak)가 발생한다.
    2. 구독자가 구독하는 동안에만 onNext와 onComplete 이벤트를 호출해야 한다.
    3. 에러가 발생했을때는 오직 onError 이벤트로만 에러를 전달해야 한다.
    4. 배압(back pressure)을 직접 처리해야 한다.

\
&nbsp;
> just()나 create()는 단일 데이터를 다룬다. 단일 데이터가 아닐때는 fromXXX() 계열 함수를 사용한다. **배열, 리스트 등의 자료구조나 Future, Callable, Publisher 등은 from으로 시작하는 연산자를 통해 간단히 Observable로 변환할 수 있다.** 원래 RxJava 1.x에서는 from()과 fromCallable() 함수만 사용했었다. 그런데 from() 함수를 배열, 반복자, 비동기 계산 등에 모두 사용하다 보니 모호함이 있었다. 따라서 RxJava2에서는 from() 함수를 세분화했고 그중 하나가 아래 소개하는 fromArray() 함수이다.

![from](/assets/img/post_images/rxjava3_from.png)

* `fromArray()` 함수

    `fromArray()` 함수를 통해 배열의 아이템을 Observable로 바꿔 아이템을 순차적으로 발행할 수 있다.
    ```
    String[] itemArray = new String[]{"PIGBEAN", "Tech", "Blog"};
    Observable source = Observable.fromArray(itemArray);
    source.subscribe(System.out::println);
    ```

    ```
    PIGBEAN
    Tech
    Blog
    ```

* `fromCallable()` 함수

    RxJava는 비동기 프로그래밍을 하기 위한 라이브러리이다. 이전까지 기본적인 자료구조로 Observable을 생성하는 부분을 살펴봤다면 이번에는 기존 자바에서 제공하는 비동기 클래스나 인터페이스와의 연동을 살펴볼 차례이다. 먼저 살펴보는 것은 자바 5에서 추가된 동시성 API인 Callable 인터페이스이다. 비동기 실행 후 결과를 반환하는 call() 메서드를 정의한다.
    ```
    Callable<String> callable = () -> "PIGBEAN tech blog";
    Observable source = Observable.fromCallable(callable);
    source.subscribe(System.out::println);
    ```

    ```
    PIGBEAN Tech Blog
    ```

\
&nbsp;
## 다양한 Observable의 형태
***

Observable 스트림 이외에도 특별한 목적으로 사용되는 `Single`, `Maybe`, `Completable` 등의 특별한 스트림이 있다. 이들은 Observable로 변환될 수 있고, 반대로 Observable도 이들 스트림으로 변환될 수 있다.

### Single
---

![Single](/assets/img/post_images/rxjava3_single.png)
Single은 단일 아이템만 발행한다. 이 특징 때문에 http 요청/응답 같은 이벤트 처리에 많이 쓰인다. Single을 사용해 http 이벤트에 실행 결과에 따른 **응답 메시지** 를 전달받아 추후 프로그램에 활용할 수 있다. 데이터를 한 번만 발행하기 때문에 onNext(), onComplete() 대신 `onSuccess()`를 사용해 데이터 발행이 완료됨을 알려준다. 오류처리는 Observable과 마찬가지로 `onError()`을 사용한다.

```
Single.create(emitter -> emitter.onSuccess("Success"))
    .subscribe(System.out::println);
```

```
Success
```

&nbsp;
### Completable
---

Completable은 아이템을 발생하지 않고, 정상적으로 실행이 종료되었는 지에 대해 확인할 때 사용된다. http 이벤트를 처리할 때 응답 메시지를 받아보지 않고 그냥 http 이벤트가 잘 종료되었는 지만 확인하고 싶다면 Completable을 쓰면 된다. 아이템을 발행하지 않기 때문에 onNext()와 Single에서 쓰였던 onSuccess()는 쓰지 않고 `onComplete()`와 `onError()`만을 사용한다.

```
Completable.create(emitter -> {
    System.out.println("OK")
    emitter.onComplete();
}).subscribe(() -> System.out.println("Completed"));
```

```
OK
Completed
```

&nbsp;
### Maybe
---

![Maybe](/assets/img/post_images/rxjava3_maybe.png)
Maybe는 Single과 Completable을 합쳐둔 느낌이다. Single처럼 아이템을 하나만 발행할 수도 있고, Completable처럼 발행하지 않을 수도 있다. 따라서 아이템을 발행했을 때에는 onSuccess()를 호출하고, 발행하지 않을 때에는 onComplete()를 호출한다. onSuccess() 이후에 다시 onComplete()를 호출할 필요는 없다.

\
&nbsp;

***
#### 참고 내용 출처 :
* [https://blog.yena.io/studynote/](https://blog.yena.io/studynote/)
* [https://12bme.tistory.com/570](https://12bme.tistory.com/570)
* 스다 토모유키, 『RxJava 리액티브 프로그래밍』, 이승룔, (주)도서출판 길벗(2019)