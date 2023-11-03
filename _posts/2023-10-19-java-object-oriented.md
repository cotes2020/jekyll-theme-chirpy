---
title: Section 08 객체지향 프로그래밍
date: 2023-10-19
categories: [blog]
tags: [java]
---


# 08 객체 지향 프로그래밍

구조적(structured)  프로그래밍이나 절차적(procedural)프로그래밍에서의 사고방식은
객체지향 프로그래밍 사고방식과 완전히 다릅니다

이번에는 객체의 측면에서 생각하는 법을 소개할 것입니다
클래스가 무엇인지, 객체와 객체의 상태, 동작이 무엇인지 다루고
객체지향의 기본 개념인 '캡슐화(encapsulation)'와 '추상화(abstraction)'도 배웁니다.




## Issue : 한 곳에서 다른 곳으로 비행하고 싶다

## 절차적 프로그래밍
절차적 프로그래밍은 절차를 중심으로 프로그래밍한다.

그러면 각 단계의 관점으로 생각한다.

'일단 공항에 가서, 체크인 창구 찾고, 체크인하고, 보안 검사받는다'

즉, 만들어야 하는 주요 **메서드**는 무엇이며 문제를 해결하기 위해 어떻게 결합할 것인지 위주로 생각한다.

절차적 또는 구조적 프로그래밍은 '절차' 함수의 관점에서 생각하는 것이다.

## 객체 지향 프로그래밍

'문제와 관련된 다양한 객체들을 중점적으로 보고 각자 가진 데이터는 무엇이며 해당 데이터에 할 수 있는 것은 무엇이 있을까?'식인 거죠.

1. 관련된 객체가 무엇이 있는가를 먼저 생각해본다.

    객체 지향적으로 생각한다는 것은 현재 가진 문제에 연관된 여러 요소를 확인하려 노력하는 것입니다

    처음으로 할 것은 비행기, 승무원, 탑승객, 파일럿 등을 확인하고

    연관된 것들을 알아냈으면, 그 데이터 중 내가 사용하고자 하는 것은 무엇인지 확인하는 것입니다.

    그리고 객체에 수행될 수 있는 행위가 무엇이 있는지 생각하는 것입니다.

    ```
    비행기(Aeroplane)
        airline 항공사, maker 제조사, type 에어버스가 있는가 없는가, position 현재위치 // data
        takeoff() 이륙하기, land() 착륙하기, cruise() 나아가기 //actions

    승무원(AirHostess)
        name, address
        wish() 인사하기, serve()

    탑승객(Passenger)
        name, address
        takeCab(), checkIn(), walk(), smile()
    ```

    객체가 **내포한 데이터**는 객체의 '상태 state'라고 합니다.
    객체의 상태는 시간에 따라 바뀔 수 있습니다.
    예를 들어, 비행기의 위치는 한시간 뒤 위치와 다를 수 있습니다.

    객체에 행할 수 있는 동작을 '동작 behaviour'이라고 합니다.
    

## 클래스란

클래스는 객체의 템플릿입니다.
객체는 클래스의 인스턴스입니다.


    ```
    class Planet{
        name, location, distanceFromSun // member data, state, fields, member variables 멤버변수

        revolve(), rotate() // actions, behaviour, methods (특정 객체에 수행할 수 있는 메서드)
    }

    Planet earth = new Planet();
    Planet venus = new Planet();
    ```

### 숙제 : 온라인 쇼핑 시스템을 **객체 지향 관점**으로 바라봐 보자

1. 연관된 여러 요소들을 생각해본다.
2. 그 요소들의 data 와 actions 를 생각해본다.
   
    ```
    사업체
        이름, 주소, 매출, 판매종목 등
        정보변경하기(), 순수익구하기(), 광고하기(), 고객관리하기()


    고객
        이름, 주소, 구매목록, 장바구니 등
        장바구니에담기(), 정보변경하기(), 구매하기(), 구매목록확인하기(), 로그인(), 로그아웃() 등

    물품
        이름, 가격, 원가, 재고 등
        재고갯수파악하기(), 판매갯수에따른가격계산하기(), 가격 변경하기() 등
    ```

### 상태 부여하기

| ducati | honda |
|speed - 80 | speed - 20|

오답노트 : 객체 멤버 변수가 명시적으로 초기화되지 않은 경우 객체 멤버 변수의 기본값은 참조 유형인 경우 null이고 숫자 유형(예: int, double 등)인 경우 0입니다.

### 캡슐화

MotorBike 클래스가 가진 문제점은 MotorBikeRunner 클래스가 직접적으로 MotorBike의 인스턴스 변수에 접근할 수 있다는 것인데, MotorBikeRunner는 또다른 별개의 클래스이기 때문에 좋지 않습니다.

독립적인 또 하나의 클래스인데 다른 클래스의 내부 변수에 접근하고 있습니다.

이것은 '캡슐화'라는 것을 파괴합니다.


'캡슐화'는 주인 클래스만이 특정 클래스의 데이터에 접근해야 한다는 개념입니다.
다른 클래스는 이 데이터에 접근하려면 해당 클래스의 동작 즉, 메서드를 통해야 합니다.


### 캡슐화 해야 하는 이유

일단 캡슐화는 데이터를 클래스 안에 안전히 가두는 법이라고 이해합시당.

#### 이클립스에서는 generate getters and setters 하면 멤버 변수의 게터세터를 자동으로 만들어줌.


### 캡슐화 1단계 
setter 를 쓰면 원하지 않는 방식으로 변수가 조작되는 것을 방지 할 수 있습니다.

### 캡슐화 2단계

게터와 세터를 넘어, 객체에 행할 수 있는 다양한 연산을 고안하고 

4줄짜리 코드를 **비즈니스로직을 캡슐화**함으로써 1줄로 줄일수 있다.

```java
package com.in28minutes.oops;

public class MotorBikeRunner {

  public static void main(String[] args) {
    MotorBike ducati = new MotorBike();
    MotorBike honda = new MotorBike();
    ducati.start();
    honda.start();
    ducati.setSpeed(100);
    int ducatiSpeed = ducati.getSpeed();
    ducatiSpeed += 100;
    ducati.setSpeed(ducatiSpeed);

    int hondaSpeed = honda.getSpeed();
    hondaSpeed += 100;
    honda.setSpeed(hondaSpeed);

    System.out.println(ducati.getSpeed());
    System.out.println(honda.getSpeed());
  }
}
```


```java
package com.in28minutes.oops;

public class MotorBikeRunner {

  public static void main(String[] args) {
    MotorBike ducati = new MotorBike();
    MotorBike honda = new MotorBike();

    ducati.setSpeed(100);
    ducati.increaseSpeed(200);
    honda.increaseSpeed(100);

    System.out.println(ducati.getSpeed());
    System.out.println(honda.getSpeed());
  }
}
```

### 생성자

1. 생성자는 클래스명과 정확하게 같아야합니다.
2. 리턴 타입을 지정해줄 필요 없습니다.
3. 객체지향 프로그래밍으로 객체를 만들때 해당 객체의 최초 상태를 설정할 수 있습니다.

오답노트 : 첫 번째 생성자(Book())에서 다른 생성자(Book(int numOfCopies))를 호출하기 전에 System.out.println("Parameterless Constructor is called");와 같은 다른 문장을 먼저 실행하면 안 됩니다. 생성자 호출은 항상 생성자 내부의 첫 번째 문장으로 와야 합니다.


### 기초적인 생성자의 중요한 점들

1. `new Cart();`를 호출하는 것은 생성자 메서드 호출과 거의 같습니다. 

2. 일단 기억할 것은 `new` 키워드로 새 객체를 만들 때 생성자가 호출되고, 
    생성자를 직접 만들지 않으면 자바 컴파일러가 디폴트 생성자를 제공하기 때문에 디폴트 생성자로 객체가 생성됩니다.

    이런 생성자를 제공하는 순간 자바 컴파일러는 디폴트 생성자를 제공하지 않아서 `MotorBike() {}`같은 인자 없는 생성자를 제공해야 합니다

    클래스안에 생성자를 하나도 만들지 않으면 자바컴파일러는 default constructor 를 알아서 제공합니다.

    하지만 생성자를 하나라도 만들기 시작하면 더이상 default constructor 를 제공하지 않습니다.

    그렇기 때문에 우리는 No Argument constructor 를 직접 만들어야 합니다.

    default constructor(No Argument constructor)

    ```java
    Book() {
        this.numOfCopies = 1000;
    }

    Book(int numOfCopies) {
    System.out.println("Constructor is called");
    this.numOfCopies = numOfCopies;
    }
    ```

3. 세 번째 중요한 점은 'this' 키워드로 타 생성자를 호출할 수 있다는 겁니다.

    'this' 키워드로 한 생성자에서 다른 생성자를 호출했었죠.

    ```java
    Book() {
        // 방법 1
        // this.numOfCopies = 1000;
        // 방법2 이게 더 낫습니다.
        this(1000);
    }

    Book(int numOfCopies) {
        System.out.println("Constructor is called");
        this.numOfCopies = numOfCopies;
    }
    ```

4. 생성자는 나중에 슈퍼 클래스의 복합성을 배울 때 또 다룰 예정입니다.
