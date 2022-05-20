---
title: RxJava(2) - Reactive Streams
author: Bean
date: 2022-01-10 22:42:00 +0800
categories: [Mobile frontend, Android]
tags: [RxJava]
layout: post
current: post
class: post-template
subclass: 'post'
navigation: True
cover:  assets/img/post_images/rxjava_cover.jpeg
---

앞선 글에서 RxJava가 Reactive Streams 사양을 구현한다고 하였다. 따라서 Reactive Streams이 뭔지 자세히 알면 RxJava를 더 쉽게 이해할 수 있다. 그래서 이번 글에서는 Reactive Streams을 더 알아보았다.

&nbsp;
## Reactive Streams란
&nbsp;

라이브러리나 프레임워크에 상관없이 데이터 스트림을 비동기로 다룰 수 있는 공통 메커니즘으로, 이 메커니즘을 편리하게 사용할 수 있는 인터페이스를 제공한다. 즉, Reactive Streams는 인터페이스만 제공하고 구현은 각 라이브러리와 프레임워크에서 한다.

* Reactive Streams: [https://www.reactive-streams.org/](https://www.reactive-streams.org/)

> The purpose of Reactive Streams is to provide a standard for asynchronous stream processing with non-blocking backpressure.

Reactive Stream 스펙 제일 위에 나오는 문구이다. Reactive Stream의 목적이 명확하게 나와있다. 해석해보면 **"Reactie Stream의 목적은 non-blocking backpressure를 이용하여 비동기 스트림 처리의 표준을 제공하는 것이다.”** 라고 되어 있다.

java의 RxJava, Spring5 Webflux의 Core에 있는 ProjectReactor 프로젝트 모두 reactive Stream을 사용하고 있다. 또한 Java9에 추가된 Flow 역시 reactvie stream 스펙을 채택하여 쓰고 있다. 따라서 비동기 프로젝트를 잘 이해하기 위해서는 기본 스펙이 되는 Reactive Stream에 대해서 이해가 필요하다.

&nbsp;
### Reactive Stream의 목적

계속적으로 들어오는 스트림 데이터를 효율적으로 처리하기 위해서는 비동기 시스템이 효과적이다. **비동기 처리를 하면서 가장 중요한 문제는 데이터를 받는 곳에서 데이터를 예측 가능한 범위 내에서 신중하게 제어할 수 있도록 해야한다는 것** 이다. Publisher는 Subscriber의 상태에 상관없이 데이터를 전달하는데만 집중하기 때문에 보내는 속도와 처리 속도가 다를 수 있다.

예를들어 데이터를 주는 곳에서 빠르게 데이터를 생성하여 1초에 50개의 데이터를 보내는데 데이터를 받는 곳의 수용 능력으로는 1초에 10개 정도 밖에 데이터를 처리하지 못한다면 다음과 같이 문제가 된다.

* Subscriber에 별도의 queue(버퍼)를 두고 처리하지 않고 대기중인 데이터를 저장할 수 있다.
* 하지만, queue의 사용 가능한 공간도 전부 금방 소모될 것이다.
* queue의 크기를 넘어가게 되면 데이터는 소실될 것이다.
* queue의 크기를 너무 크게 생성하면 OOM(Out Of Memory) 문제가 발생할 수 있다.

따라서, **Reactive Stream의 주된 목적은 비동기의 경계를 명확히하여 스트림 데이터의 교환을 효과적으로 관리하는 것**에 있다. BackPressure가 이를 달성할 수 있게 해주는 주요한 부분이다.

다시 말해 Reactive Stream은 다음의 스트림 지향 라이브러리에 대한 표준 및 사양이다.

1. 잠재적으로 무한한 숫자의 데이터 처리
2. 순서대로 처리
3. 컴포넌트간에 데이터를 비동기적으로 전달
4. backpressure를 이용한 데이터 흐름제어

&nbsp;
### BackPressure (배압)

앞서 Reactive Stream의 목적이 backpressure을 이용하고 비동기 스트림의 표준을 제공하는 것이라고 언급했다. BackPressure을 더 자세히 살펴보자. 배압은 한마디로 데이터 통지량을 제어하는 기능을 말한다. [리액티브 선언문 용어집](https://www.reactivemanifesto.org/ko/glossary) 에서는 BackPressure을 다음과 같이 설명하고 있다.

> 한 컴포넌트가 부하를 이겨내기 힘들 때, 시스템 전체가 합리적인 방법으로 대응해야 한다. 과부하 상태의 컴포넌트에서 치명적인 장애가 발생하거나 제어 없이 메시지를 유실해서는 안 된다. 컴포넌트가 대처할 수 없고 장애가 발생해선 안 되기 때문에 컴포넌트는 상류 컴포넌트들에 자신이 과부하 상태라는 것을 알려 부하를 줄이도록 해야 한다. 이러한 배압은 시스템이 부하로 인해 무너지지 않고 정상적으로 응답할 수 있게 하는 중요한 피드백 방법이다. 배압은 사용자에게까지 전달되어 응답성이 떨어질 수 있지만, 이 메커니즘은 부하에 대한 시스템의 복원력을 보장하고 시스템 자체가 부하를 분산할 다른 자원을 제공할 수 있는지 정보를 제공할 것이다.

앞서 데이터를 보내는 속도와 처리 속도가 다르면 문제가 된다고 했다. BackPressure로 이 문제를 어떻게 해결할 수 있을까?

보내는 쪽과 받는 쪽의 속도가 다른 문제는 **Publisher가 Subscriber에게 데이터를 Push 하던 기존의 방식을 Subscriber가 Publisher에게 자신이 처리할 수 있는 만큼의 데이터를 Request하는 방식으로 해결할 수 있다.** 필요한(처리할 수 있는) 만큼만 요청해서 Pull하는 것이다. 데이터 요청의 크기가 Subscriber에 의해서 결정되는 것이다. 이를 dynamic pull 방식이라 부르며, Back Pressure의 기본 원리이다.

\
&nbsp;
## Reactive Streams의 구성
&nbsp;

Reactive Stream은 데이터를 만들어 통지하는 **Publisher(생산자)** 와 통지된 데이터를 받아 처리하는 **Subscriber(소비자)** 로 구성된다. Subscriber가 Publisher를 **구독(subscribe)** 하면 Publisher가 통지한 데이터를 Subscriber가 받을 수 있다.

아래는 Reactive Streams API이다.

```java
public interface Publisher<T> {
   public void subscribe(Subscriber<? super T> s);
}

public interface Subscription {
   public void request(long n);
   public void cancel();
}

public interface Subscriber<T> {
   public void onSubscribe(Subscription s);
   public void onNext(T t);
   public void onError(Throwable t);
   public void onComplete();
}
```

* **Publisher** : 데이터를 통지하는 생산자
    * Subscriber를 받아들이는 subscribe 메서드 하나만 갖는다.
* **Subscriber** : 데이터를 받아 처리하는 소비자
    * Subscription을 등록하고 Subscription에서 오는 신호에 따라서 동작한다.
    * 데이터를 받아 처리할 수 있는 onNext, 에러를 처리하는 onError, 모든 데이터를 받아 완료되었을 때는 onComplete, 그리고 Publisher로부터 Subscription을 전달 받는 onSubscribe 메서드로 이루어진다.
* **Subscription** : Publisher와 Subscriber 사이에서 중계하는 역할
    * 데이터 개수를 요청하고 구독을 해지하는 인터페잏스
    * n개의 데이터를 요청하는 request와 구독을 취소하는 cancel을 갖는다.

&nbsp;
이를 토대로 다음과 같은 flow를 만들 수 있다.
  <div style="text-align: left">
    <img src="/assets/img/post_images/rxjava2-1.png" width="100%"/>
  </div>

1. Publisher에서 사용할 Subscription을 구현한다.
2. Publisher에서 전달(publishing)할 data를 만든다.
3. Publisher는 subscribe() 메서드를 통해 subscriber를 등록한다.
4. Subscriber는 onSubscribe() 메서드를 통해 Subscription을 등록하고 Publisher를 구독하기 시작한다. 이는 Publisher에 구현된 Subscription을 통해 이루어진다. 이렇게 하면 Publisher와 Subscriber는 Subscription을 통해 연결된 상태가 된다. onSubscribe() 내부에 Subscription의 request()를 요청하면 그때부터 data 구독이 시작된다.
5. Suscriber는 Subscription 메서드의 request() 또는 cancel()을 호출을 통해 data의 흐름을 제어할 수 있다.
6. Subscription의 request()에는 조건에 따라 Subscriber의 onNext(), onComplete() 또는 onError()를 호출합니다. 그러면 Subscriber의 해당 메서드의 로직에 따라 request() 또는 cancle()로 제어하게 된다.

&nbsp;

***

참고 내용 출처 :
  * [https://jongmin92.github.io/2019/11/05/Java/reactive-1/](https://jongmin92.github.io/2019/11/05/Java/reactive-1/)
  * [https://sabarada.tistory.com/98](https://sabarada.tistory.com/98)
  * 스다 토모유키, 『RxJava 리액티브 프로그래밍』, 이승룔, (주)도서출판 길벗(2019)
