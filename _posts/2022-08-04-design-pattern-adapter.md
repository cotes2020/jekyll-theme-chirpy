---
title: "디자인 패턴 - Adapter 패턴에 대해 알아보자"
date: 2022-08-04 00:56:00 +0900
categories: [디자인패턴]
tags: [디자인패턴, Adapter 패턴]
---

# 디자인 패턴 - Adapter 패턴에 대해 알아보자

## 0. 소개

이번 주는 Adapter 패턴에 대해 내가 발표를 하기로 했다. 발표하기 전 정리한 것들을 블로그에 남기고자 한다.

## 1. Adapter 패턴이란?

- **이미 제공되어 있는 것**과 **필요한 것** 사이의 차이를 없애주는 디자인 패턴이다.
- **특정 클래스 인터페이스**를 **클라이언트에서 요구하는 다른 인터페이스**로 변환한다.
- Wrappter 패턴이라고도 부른다.

<img width="473" alt="스크린샷 2022-08-08 오전 1 31 06" src="https://user-images.githubusercontent.com/64428916/183301438-20c25bd3-3f63-44ea-a1c6-fc784f710496.png">

- 클라이언트에서는 Target Interface를 호출하는 것처럼 보인다. 하지만 클라이언트의 요청을 받은 Adapter는 자신이 감싸고 있는 Adaptee에게 실질적인 처리를 위임한다. Adapter가 Adaptee를 감싸고 있는 것 때문에 Wrapper 패턴이라고도 불린다.

## 2. Adapter 패턴의 등장인물

<img width="613" alt="스크린샷 2022-08-08 오전 1 31 28" src="https://user-images.githubusercontent.com/64428916/183301454-0e641a16-f1c0-417e-bea0-6add07f961f1.png">

- Client

  - Target 역할의 메소드를 사용해서 일을 한다.

- Target

  - Client는 Target을 통해 Adaptee를 사용할 수 있다.
  - 지금 필요한 메소드를 결정한다.

- **Adapter**

  - Client가 사용하려는 Target Interface와 Adaptee 중간에서 둘을 연결해주는 역할이다.
  - Adaptee 역할의 메소드를 사용해서 어떻게든 Target 역할을 만족시키기 위한 것이 Adapter 패턴의 목적이다.

- Adaptee
  - 이미 준비되어 있는 메소드를 가지고 있는 역할이다.

## 3. 객체 어댑터와 클래스 어댑터

1. 객체 어댑터

 <img width="673" alt="스크린샷 2022-08-08 오전 1 32 51" src="https://user-images.githubusercontent.com/64428916/183301456-c32fc4a8-53a1-4152-bd18-4f4647eb50f4.png">

2. 클래스 어댑터

 <img width="673" alt="스크린샷 2022-08-08 오전 1 33 21" src="https://user-images.githubusercontent.com/64428916/183301458-8027a84c-8a29-4289-879d-3d0438a3df80.png">

- 객체 어댑터는 구성으로 Adaptee에 요청을 전달하는 반면, 클래스 어댑터는 Target과 Adaptee 모두 서브클래스로 만들어서 사용한다.

책에서 소개하는 예제로 코드를 살펴보자.

[객체 어댑터 - 위임을 사용한 Adapter 패턴]

<img width="648" alt="스크린샷 2022-08-08 오전 1 33 53" src="https://user-images.githubusercontent.com/64428916/183301485-29cb69c2-fe6a-4b34-b40c-501b8c82f9a2.png">

- Adapter는 Adaptee로 구성되어 있다. Adapter의 몸든 요청은 Adaptee에게 위임된다.
- `PrintBanner` 클래스는 `banner` 필드에서 `Banner` 클래스의 인터페이스를 가진다.
- `printWeak`, `printStrong` 메소드에서는 `banner` 필드를 매개로 `showWithParen`, `showWithAster` 메소드를 호출한다.
- 즉, 자신이 처리하는 것이 아니라 별도의 인스턴스에게 위임하고 있다.

**Print.java**

```java
public abstract class Print {
	public abstract void printWeak();
	public abstract void printStrong();
}
```

**PrintBanner.java**

```java
public class PrintBanner extends Print {
	private Banner banner;

	public PrintBanner(String string){
		this.banner = new Banner(string);
	}

	public void printWeak() {
		banner.showWithParen();
	}

	public void printString() {
		banner.showWithAster();
	}
}
```

**Banner.java**

```java
public class Banner {
	private String string;

	public Banner (String string) {
		this.string = string;
	}

	public void showWithParen() {
		System.out.println("(" + string + ")");
	}

	public void showWithAster() {
		System.out.println("*" + string + "*");

	}
}
```

[클래스 어댑터 - 상속을 사용한 Adapter 패턴]

<img width="648" alt="스크린샷 2022-08-08 오전 1 34 02" src="https://user-images.githubusercontent.com/64428916/183301487-1d6c2b65-1079-43f2-96c4-db5053d318cf.png">

- `PrintBanner` 클래스가 `Banner`클래스를 확장해서 `showWithParen`, `showWithAster` 메소드를 상속받는다.

**Print.java**

```java
public interface Print {
	public abstract void printWeak();
	public abstract void printStrong();
}
```

**PrintBanner.java**

```java
public class PrintBanner extends Banner implements Print {

	public PrintBanner(String string){
		super(string);
	}

	public void printWeak() {
		showWithParen();
	}

	public void printString() {
		showWithAster();
	}
}

```

[Main.java]

```java
public class Main {
	public static void main(String[] args) {
		Print p = new PrintBanner("Hello");
		p.printWeak();
		p.printStrong();
	}
}
```

- Main 클래스 내에서 `PrintBanner`의 인스턴스를 Print 인터페이스형의 변수로 대입한다.
- `PrintBanner` 클래스가 어떻게 실현되고 있는지는 모른다. 따라서 Main 클래스를 변경하지 않고도 `PrintBanner` 클래스 구현을 바꿀 수 있다.

## Question

1. 두 개의 인터페이스가 달라서 호환이 안되면 하나를 바꾸던지 둘 다 바꾸면 되는거 아닌가요?

- 과거 MS사에서 파일 읽기 코드를 개발한 적이 있다. 레거시 코드(옛날 버전의 코드들)가 너무 많아 다시 개발을 했었는데 갑자기 알 수 없는 곳에서 에러가 생겼다고 한다. 알고보니 파일을 읽을 때 네트워크 연결이 제대로 되어 있지 않았을 경우 문제가 생겼던 것이고 결국 어댑터를 사용하여 이전 코드를 사용했다고 한다. 이처럼 인터페이스를 바꾸는 것이 힘들 경우가 생긴다. 결국 어댑터 패턴은 프로그램의 완결성은 높지만 재사용성은 떨어질 때 사용하는 것이다.

2. 장점이 뭔가요?

- 호환되지 않는 인터페이스를 사용하는 클라이언트를 그대로 활용할 수 있다. 이를 통해 클라이언트와 구현된 인터페이스를 분리시킬 수 있으며, 향후 해당 인터페이스가 바뀌더라도 그 변경 내역은 adapter에 캡슐화 되기 때문에 클라이언트는 바뀔 필요가 없어진다.

## 참고문헌

[Java언어로 배우는 디자인 패턴 입문](http://www.yes24.com/Product/Goods/2918928)
