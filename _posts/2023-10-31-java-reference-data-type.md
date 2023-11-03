---
title: Section 16 자바 프로그래밍의 참조 자료형
date: 2023-10-31
categories: [blog]
tags: [java]
---

<style>
  /* 기본 스타일 */
  .center-table {
    margin: 0 auto; /* 가운데 정렬을 위한 스타일 */
    text-align: center; 
    width : 50%;
    
  }
  .center-table td {
    width : 25%;
  }
    .center-table td:last-child {
    width : 30%;
  }
  div.content .table-wrapper>table {
    min-width: 90%;
    }
  .table-wrapper{
            width:100%;
            margin: 0;
            padding:0;
    }
  .table-wrap {
    display: flex;
    width: 100%;
    justify-content: space-around;
    flex-wrap: nowrap;
    flex-direction: row;           
  }

    .title-cell{
        background-color : orange;
    }

  @media (max-width: 768px) {
    .center-table {
      width: 100%; 
    }
    .table-wrap {
      flex-direction : column;
      justify-content: center; 
    }
  }
</style>



reference type,
primitive type

Planet jupiter = new Planet();
여기서 주피터는 참조변수(reference variable)가 됩니다.

int i = 5;
i 는 기본 변수(primitive variable) 입니다.

```java
class Animal {
    int id;
    Animal(int id){
        this.id = id;
    }
}


Animal dog = new Animal(12); 
Animal cat = new Animal(15); 
Animal nothing; // Stack 의 value 에는 null 이 나옵니다.

nothing = cat; // 이때 nothing 의 Stack-value 에는 1C 가 기록됩니다.

nothing.id = 10; // cat.id 도 10이 된다.

```


<div class="table-wrap">
<table class="center-table">
<tr><th colspan='3'  class="title-cell">Stack</th></tr>
  <tr>
    <th>Location</th>
    <th>Value</th>
    <th>variable-name</th>
  </tr>
  <tr>
    <td>A</td>
    <td>5</td>
    <td>i</td>
  </tr>
  <tr>
    <td>B</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>C</td>
    <td>1A</td>
    <td>dog</td>
  </tr>
  <tr>
    <td>D</td>
    <td>1C</td>
    <td>cat</td>
  </tr>
  <tr>
    <td>E</td>
    <td>1C</td>
    <td>nothing</td>
  </tr>
  <tr>
    <td>F</td>
    <td></td>
    <td></td>
  </tr>
</table>
		

<table class="center-table">
<tr><th colspan='2'  class="title-cell">Heap</th></tr>
  <tr>
    <th>Location</th>
    <th>Object</th>
  </tr>
  <tr>
    <td>1A</td>
    <td>Animal12</td>
  </tr>
  <tr>
    <td>1B</td>
    <td></td>
  </tr>
  <tr>
    <td>1C</td>
    <td>Animal15</td>
  </tr>
  <tr>
    <td>1D</td>
    <td></td>
  </tr>
  <tr>
    <td>1E</td>
    <td></td>
  </tr>
  <tr>
    <td>1F</td>
    <td></td>
  </tr>
</table>
</div>

Global Shared	
	

참조변수를 생성하면 메모리의 힙이라는 곳에 저장된다.
Animal dog = new Animal(12); 

dog 는 스택 어딘가에 변수이름으로 저장된다. 그리고 값에는 힙의 Location 이 주소가 저장된다. 그래서 참조변수라고 부르는 것이다.
dog 는 스택 어딘가에 변수이름으로 저장된다.

Stack 의 value 에는 실제 객체가 저장된 메모리의 경로(HEAP) 이 기록됩니다.


## String 자료형

다른 모든 객체는, 생성자를 사용해야 인스턴스를 만들 수 지만, String 은 그냥 리터럴로 생성 가능.

```java
// 다른 참조 자료형들은 생성자를 사용해야 함
BigDecimal bd = new BigDecimal(1.0); 

// 둘 다 가능
String str = new String("Test");  
String str = "Test";
```

## 문자 자료형 연습문제 미리 풀어보기

```java
    String str = "Hello World";
    int i= 0;
    while(i < str.length()){
        System.out.println(str.charAt(i));
        i++;
    }
```

## String API

안에 "" , '' 둘 다 넣어도 봐준다.
String 의 유틸리티 메소드라고도 불린다.

```java
String.charAt();
String.substring();
String.indexOf();
String.lastIndexOf();
String.contain();
String.startsWith();
String.endsWith();
String.isEmpty();
String.equals();
String.equalsIgnoreCase();
String.concat();
String.toUpperCase();
String.toLowerCase();
String.trim();
String.join();
String.replace();
```

### String 은 불변 자료형이다.

```java
String.concat();
String.toUpperCase();
String.toLowerCase();
String.trim();
```

```java

jshell> String str = "in28Minutes";
   ...> str.concat(" is awesome");
   ...> 
   ...> str;
str ==> "in28Minutes"
$2 ==> "in28Minutes is awesome"
str ==> "in28Minutes"

jshell> String str2 = "     hi      "
str2 ==> "     hi      "

jshell> str2.trim()
$5 ==> "hi"
```

concat 한다고 해도 str 은 바뀌지 않는다.

즉, 우리가 스트링을 변경하려고 할 때 우리가 하는 것은 특정 스트링을 변경하는 것이 아니라 새로운 스트링을 만드는 것이다.

예를 들어 {str.concat}을 실행하면 우리는 원래의 스트링은 영향을 받지 않고 그대로 남아 있게 됩니다 여기서도 스트링이 "in28minutes"로 남아있는 걸 확인했었죠.

스트링 클래스가 특정 값으로 한 번 생성된 이후에는 그 특정 값은 절대 바꿀 수 없습니다.

이러한 특성을 불변성(immutality)이라고 한다.

### String 객체의 + 연산

  + (결합)연산은 왼쪽에서 오른쪽으로 진행되기 때문에 아래와 같은 결과를 낸다.
  
  ```
	jshell> 1 + 2
	$1 ==> 3
	jshell> "1" + "2"
	$2 ==> "12"
	jshell> "1" + 2
	$3 ==> "12"
	jshell> "1" + 23
	$4 ==> "123"
	jshell> 1 + 23
	$5 ==> 24
	jshell> "1" + 2 + 3
	$6 ==> "123"
	jshell> 1 + 2 + "3"
	$7 ==> "33"
  ```

### String join 메서드

구분자 자리에 , 를 넣으면 편하다.

`String.join(CharSequence delimiter, CharSequence... elements)`

> delimiter 구분자
> 
> elements 요소

  ```sh
  jshell> String.join("," , "a" ,"b");
  $6 ==> "a,b"
  ```

### String replace 메서드

String.replace(CharSequence target, CharSequence replacement) 

> target (대체 대상)
>
> replacement (대체 문자열)

### 메서드 찾는법

- ide에서 String. + tab 한다.
- Jshell 에서는 str. String. 치면 나온다.
- 구글에 java doc 이라고 친다.


### StringBuffer 란?

String 은 수정이 불가능하다는 한계를 가지고 있었죠.

예를 들어 {"123" + "123" + "1234" + "123456"} 이런 스트링을 만들었을 때 얼마나 많은 스트링이 생성되는 걸까요.

```sh
jshell> "123" + "123" + "1234" + "123456"
$7 ==> "1231231234123456"
```

자, 전체를 결합한 스트링이 생성됐습니다

결합이 실행될 때, 각각의 숫자에 스트링 인스턴스를 생성하는 거예요. 

여기처럼 4개의 값이 있는 경우, 결합은 왼쪽부터 오른쪽으로 이루어지기 때문에 처음 두 값이 결합이 되어 "123123"이 되고 이 두 개의 결합이 5번째 인스턴스예요.

그다음에 이렇게 세 개의 값을 결합하는 게 여섯 번째 인스턴스, 다음은 일곱 번째가 되겠죠.

이 단순한 결합을 실행하면서 우리는 7개의 인스턴스를 생성하고 있는 거예요.

처음 네 개는 꼭 필요하죠, 기본적인 값이니까요.

하지만 그 외에 추가로 생성되는 객체 인스턴스가 세 개나 됩니다. 굉장히 비효율적인 방법이죠.
벌써 7개의 인스턴스가 생성되었는데 이건 매우 낭비입니다.

지금은 이렇게 네 가지 값이 있지만 만약 추가하는 스트링의 수가 200개라고 하면 불필요한 객체가 얼마나 많이 생기는 걸까요.

이런 걸 방지하기 위해 자바는 StringBuffer라는 또 다른 클래스를 제공합니다.

중요한 점은 StringBuffer는 **수정이 가능**합니다.


```sh
	jshell> StringBuffer sb = new StringBuffer("TEst");
	sb ==> "TEst"
	jshell> sb.append(" 123");
	$2 ==> "TEst 123"
	jshell> sb
	sb ==> "TEst 123"
	jshell> sb.setCharAt(1, 'e');
	sb ==> "Test 123"
```

값을 바꿀 수 있다는 것 결합을 많이 수행해야 하는 경우에는 꼭 스트링버퍼를 사용하시길 추천합니다.

### StringBuilder 란?

스트링 버퍼와 굉장히 유사합니다 하지만 최신 버전의 자바에만 제공이 됩니다.

스트링 버퍼는 동기화된 클래스인데요.

잠깐, 이게 무슨 말일까요?

스트링 버퍼는 멀티스레딩이 가능한 클래스예요. 

그러나 멀티스레딩이 가능하다는 것은 그것으로 인한 패널티 또한 있다는 뜻이 됩니다.

싱글 스레드 시나리오의 경우 멀티스레딩을 할 때 가지는 복잡한 사항들이 없는데 스트링 버퍼의 경우 속도가 매우 느려질 수 있습니다.

그럴 때 사용하는 게 바로 스트링 빌더입니다.


멀티스레딩 및 관련 사항들은 더 나중에 별도의 강의에서 배워보도록 하고 지금은, **멀티스레딩에 대해 전혀 걱정하지 않고 그냥 스트링 빌더를 사용하면 되겠습니다**.

다시 말해 스트링 버퍼 대신 스트링 빌더를 사용하면 돼요.

```sh
	jshell> StringBuilder sbldr = new StringBuffer("TEst");
	sbldr ==> "TEst"
	jshell>
```

이렇게 코드를 적고 괄호 안에 스트링을 넣으세요.


### StringBuffer vs StringBuilder 결론

나중에 스트링 버퍼와 스트링 빌더의 API를 보면 서로 다른 메소드를 볼 수 있는데 사실상 여러분이 기억해야 하는 건 이거예요.

굉장히 많은 스트링을 결합하거나 스트링 인스턴스를 많이 생성해야 한다면 대체할 수 있는 걸 찾으라는 거예요.

스트링 빌더든, 스트링 버퍼든 멀티스레딩에 대해 신경 쓰고 싶지 않다면 스트링빌더 클래스를 사용하는 게 맞고요 그럼에도 조금 안전하게 코드를 진행하고 싶을 경우 스트링 버퍼를 사용해 보세요.

## Wrapper Class

Q : 래퍼클래스가 왜 필요할까?<br/>
>    첫째, valueOf 로 다른 타입의 객체를 쓸 수 있게 해준다.<br/>
>    둘째, 추가적인 옵션(유틸리티 메서드)를 쓸 수 있게 해준당.<br/>
>    셋째, 기본 값들을 컬렉션에 저장할 수 있게 해준당.<br/>

| 기본 데이터 유형 | Wrapper 클래스 |
| ---------------- | -------------- |
| byte             | Byte           |
| short            | Short          |
| int              | Integer        |
| long             | Long           |
| float            | Float          |
| double           | Double         |
| char             | Character      |
| boolean          | Boolean        |



###  래퍼 클래스를 만드는 두 가지 방법

1. 생성자로 만들기
  new 키워드를 사용해 만들 수 있습니다.
  java9 부터 deprecated 됨.

2. valueOf 메서드로 만들기
  각각의 래퍼 클래스는 모두 valueOf 라는 정적 메소드를 가지고 있습니다.

```java
    Integer useNew1 = new Integer(1);
    Integer useNew2 = new Integer(1);

    Integer useValueOf1 = Integer.valueOf(1);
    Integer useValueOf2 = Integer.valueOf(1);
```

모든 래퍼 클래스는 불변성을 가집니다. (String 클랫 처럼)


래퍼클래스로 만들어지는 변수들은 참조변수입니다.

그런데 참조 변수에 대해 `==`는 **값**을 비교하지 않았죠.

**값이 메모리 어디에 저장**되었는지 비교했어요.

valueOf 를 사용해서 변수에 할당하면 Heap 안에 존재하는 기존의 객체를 다시 사용합니다.

위의 경우 valueOf() 메서드를 사용하면 많은 경우 캐시된(-128부터 127까지의 정수) 객체를 반환하여 메모리를 절약할 수 있습니다.

따라서 성능 및 메모리 관리를 고려할 때, 주로 `Integer.valueOf()` 메서드를 사용하는 것이 좋습니다.

> 래퍼 클래스는 기본 데이터 유형을 객체로 래핑하기 위해 사용되며, 이 객체들은 힙 메모리에 저장됩니다. <br/>
> 따라서 래퍼 클래스 타입의 변수는 해당 객체를 가리키는 참조 변수입니다. 


### 래퍼클래스 오토박싱

```java
    Integer useValueOf = Integer.valueOf(1);
    Integer useAutoBoxing = 1;
```

아래 변수도 valueOf 를 사용해서 만든건데 너무 길어서 자바 백그라운드에서 오토박싱 해줄게 하는거다. 

그래서 `==` 으로 비교해보면 true 가 나온다.

```sh
    jshell>     Integer useValueOf = Integer.valueOf(1);
       ...>     Integer useAutoBoxing = 1;
    useValueOf ==> 1
    useAutoBoxing ==> 1

    jshell> useValueOf == useAutoBoxing;
    $3 ==> true
```

### 래퍼클래스에 속한 상수들

```java
    Integer.MAX_VALUE;
    Integer.MIN_VALUE;

    // bit 단위로 나온다.
    Integer.SIZE;

    // byte 단위로 나온다.
    Integer.BYTE;
```

## Date 타입

자바 Date 가 성능이 구려서 자바8 버전 부터 Joda time framework 에서 가져와 구현했다.

```
LocalDate // 날짜 값을 가진다. Java.time 이라는 패키지 안에 속한다.

LocalTime // 시간 값을 가진다

LocalDateTime // 날짜 + 시간 값을 가진다.

```

```sh
    jshell> /imports
    |    import java.io.*
    |    import java.math.*
    |    import java.net.*
    |    import java.nio.file.*
    |    import java.util.*
    |    import java.util.concurrent.*
    |    import java.util.function.*
    |    import java.util.prefs.*
    |    import java.util.regex.*
    |    import java.util.stream.*

    jshell> import java.time.LocalDate;
    jshell> import java.time.LocalDateTime;
    jshell> import java.time.LocalTime;

    jshell> LocalDate now = LocalDate.now();
    now ==> 2023-11-02

    jshell> LocalDateTime now = LocalDateTime.now();
    now ==> 2023-11-02T11:26:28.417202

    jshell> LocalTime now = LocalTime.now();
    now ==> 11:26:37.098177
```

```sh
jshell> import java.time.*

jshell> LocalDate today = LocalDate.now();
today ==> 2023-11-02

jshell> today.getYear()
$3 ==> 2023

jshell> today.getDayOfWeek()
$4 ==> THURSDAY

jshell> today.getDayOfMonth()
$5 ==> 2

jshell> today.getDayOfYear()
$6 ==> 306


jshell> today.getMonth()
$7 ==> NOVEMBER

jshell> today.getMonthValue()
$8 ==> 11

jshell> today.isLeapYear()
$9 ==> false

jshell> today.lengthOfYear()
$10 ==> 365

jshell> today.lengthOfMonth()
$11 ==> 30

jshell> today.plusDays(100)
$12 ==> 2024-02-10

jshell> today.plusMonths(100)
$13 ==> 2032-03-02

jshell> today.plusYears(100)
$14 ==> 2123-11-02

jshell> today.minusYears(100)
$15 ==> 1923-11-02
```

중요한 것은 우리는 plus, minus 를 한다고 해서 today 변수의 값이 변하지는 않는 다는 겁니다.

또 하나의 불변성 클래스였습니다.

```sh
jshell> import java.time.*

jshell> LocalDateTime now = LocalDateTime.now();
now ==> 2023-11-02T11:47:14.114890

jshell> now.plus
plus(          plusDays(      plusHours(     plusMinutes(   plusMonths(    
plusNanos(     plusSeconds(   plusWeeks(     plusYears(   

jshell> now.get
get(              getChronology()   getClass()        getDayOfMonth()   
getDayOfWeek()    getDayOfYear()    getHour()         getLong(          
getMinute()       getMonth()        getMonthValue()   getNano()         
getSecond()       getYear()      

```


Modification

```sh
jshell> LocalDate yesterday = LocalDate.of(2023,11,02)
yesterday ==> 2023-11-02

jshell> LocalDate today = LocalDate.now();
today ==> 2023-11-02

jshell> today.withYear(2016);
$5 ==> 2016-11-02

jshell> today.withDayOfMonth(20);
$6 ==> 2023-11-20

jshell> today.withDayOfMonth(20);
$6 ==> 2023-11-20

jshell> today.withMonth(3);
$7 ==> 2023-03-02

jshell> today.withDayOfYear(3);
$8 ==> 2023-01-03
```

LocalDate,LocalDateTime, LocalTime 모두 불변한 자료형입니다. 
지금 배우는 모든 modification 메서드들은 새로운 객체로 리턴하는 메서드들입니다.



Comparison

```sh
jshell> import java.time.*

jshell> LocalDate yesterday = LocalDate.of(2023,11,01)
yesterday ==> 2023-11-01

jshell> LocalDate today = LocalDate.now()
today ==> 2023-11-02

jshell> today.isBefore(yesterday)
$4 ==> false

jshell> today.isAfter(yesterday)
$5 ==> true
```

Incorrect answer. Please try again.
Java의 참조 변수는 객체의 메모리 위치를 저장하는 데 사용되며, 이를 통해 객체의 속성 및 메서드에 액세스할 수 있습니다.



StringBuffer는 동기화, 즉 스레드 안전하므로 두 개의 스레드가 동시에 액세스할 수 없도록 보장합니다. 반면 StringBuilder는 비동기화, 즉 스레드 안전하지 않습니다. 즉, 두 개 이상의 스레드가 동시에 액세스할 수 있습니다.
