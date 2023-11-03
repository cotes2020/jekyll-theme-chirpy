---
title: Section 11 원시자료형
date: 2023-10-21
categories: [blog]
tags: [java]
---

# Primitive Data Types

## Integer
byte, short, char, int, long

## Float
float, double


### 정수 자료형의 래퍼클래스

자바의 래퍼 클래스에서 세부적으로 각자의 사이즈를 확인할 수 있습니다.

```java
Byte.SIZE
Byte.BYTES
Byte.MAX_VALUE
Byte.MIN_VALUE

Short.SIZE
Short.BYTES
Short.MAX_VALUE
Short.MIN_VALUE

Integer.SIZE
Integer.BYTES
Integer.MAX_VALUE
Integer.MIN_VALUE

Long.SIZE
Long.BYTES
Long.MAX_VALUE
Long.MIN_VALUE
```

### 정수 자료형의 형변환

모든 정수 유형의 리터럴은 기본적으로 int 유형이 디폴트입니다.

그래서 long 리터럴을 만드려면 l을 뒤에 붙여야 합니다. 

작은 유형에 큰 유형을 넣을 순 없습니다. 

작은 유형에 큰 유형을 넣으려고 하면 l 에서 i 로 손실변환이라고 나옵니다.

```sh
jshell> long l = 50000000l;
l ==> 50000000

jshell> int i = l;
|  Error:
|  incompatible types: possible lossy conversion from long to int
|  int i = l;
|          ^
```

그런데도 위험을 감수하고 싶다면 넣고 싶다면 '형변환 Casting'이라는 걸 합니다. 컴파일러와 충돌하지 않게 안심시켜 주는 겁니다.

이걸 Explicit cast (명시적 캐스팅, 강제 형 변환)라고 부릅니다.

```
jshell> int i = (int) l;
i ==> 50000000
```

하지만 l = i 하면 더 큰 용기에 넣는거니까 문제 없이 잘 작동합니다.
이걸 Implicit Cast (암시적 캐스팅, 자동 형 변환)라고 부릅니다.


### 정수 자료형 표현

```sh
jshell> int eight = 010;
eight ==> 8

jshell> int sixteen = 0x10;
sixteen ==> 16

jshell> int fifthteen = 0xf;
fifthteen ==> 15
```

자바 리터럴은 8진수와 16진수를 지원합니다.

0으로 시작하면 무조건 8진수(octal number)입니다.
0x으로 시작하면 무조건 16진수(hexadecimal number)입니다.

| 2진수 | binary number system | 0, 1 | 
| 10진수 | decimal number system | 0 to 9 |
| 8진수 | octal number system | 0 to 7 |
| 16진수 | hexadecimal number system | 0 to 9, A, B, C, D, E, F |

하지만 꼭 필요한 경우가 아니면 굳이 다른 진수 체계를 쓰지마세요ㅋㅋ


```
jshell> short s = 1234;
s ==> 1234

jshell> int i = 3456;
i ==> 3456

jshell> Short.MAX_VALUE;
$8 ==> 32767

jshell> s = i;
|  Error:
|  incompatible types: possible lossy conversion from int to short
|  s = i;
|      ^

jshell> s = (short) i;
s ==> 3456

```

short 는 16byte라서 32byte 짜리 자료를 넣을 수 없습니다. 그러므로 explicit casting 을 해줘야 합니다.


### 증감 연산자

post increment

pre increment


## double 이나 float 을 쓰면 안되는 이유


```sh
jshell> 34.56789876 + 34.2234
$1 ==> 68.79129875999999
```
뜬금없이 뒤에 99999... 이 생겼는데 부동소수값을 명확하게 표현하지 않아서 그렇습니다.

정확한 연산을 원한다면 BigDecimal 클래스를 쓰세요.

BigDecimal("문자열") 을 넣으면 됩니다. (double 값을 넣으면 정확하지 않아집니다.)


```sh
jshell> BigDecimal number1 = new BigDecimal("34.12345");
number1 ==> 34.12345

jshell> BigDecimal number2 = new BigDecimal("1.12345")
number2 ==> 1.

jshell> number1.add(number2)
$4 ==> 35.24690
```

중요한 점은 BigDecimal은 자바에서 변경 불가능한 값 immutable 이라는 겁니다.

jshell 에서 number1. tab 하면 가능한 메서드가 나옵니다.

재무 회계에서는 BigDecimal 을 쓰세요.


## 논리연산자

&&
|| 
^ // XOR 이라고 부른다.
!

&, | 는 나중에 다루겠습니다.


### 단축회로 연산자 &&, || 

수식 자체가 거짓(false) 이면 아예 그 문을 평가 하지 않음.

```
jshell> int j = 15;
   ...> int i = 10;
   ...> 
   ...> j > 15 && i++ > 5;
j ==> 15
i ==> 10
$3 ==> false

jshell> i
i ==> 10
```

비교연산이 false 여도 그 문을 평가 함.

```
jshell> int j = 15;
   ...> int i = 10;
   ...> 
   ...> j > 15 & i++ > 5;
j ==> 15
i ==> 10
$7 ==> false

jshell> i
i ==> 11
```

하지만 비교연산할때 i++ 하면 가독성에 안좋기 때문에 그냥 처음에 i++ 하고 복잡하지 않은 코드가 위 코드들 보다 낫습니다. 
코드는 심플하게 작성하고, 부수효과(side effect)가 없는지 확인하는 것이 좋습니다.

```
jshell> int i = 10;
i ==> 10

jshell> int j = 15;
j ==> 15

jshell> i++;
$3 ==> 10

jshell> j > 15 && i > 5;
$4 ==> false
```

## 관계 연산자

> < >= <= 

## 비교 연산자

== !=