---
title : 메모리 구조와 함수 호출 규약
categories : [Hacking, System]
tags : [Reversing, 메모리 구조, 함수 호출 규약]
---

## 메모리 구조
<hr style="border-top: 1px solid;"><br>

컴파일된 프로그램 메모리는 5개의 세그먼트로 나뉘어짐.

+ 텍스트<sub>text</sub>

+ 데이터<sub>data</sub>

+ bss<sub>block staretd symbol</sub>

+ 힙<sub>heap</sub>

+ 스택<sub>stack</sub>

<br>

여기서 세그먼트란 적재되는 데이터의 용도별로 메모리의 구획을 나눈 것.

각각 권한을 가지고 있으며 권한으로는 읽기,쓰기,실행이 있고, CPU는 권한이 부여된 행위만 할 수 있음.

<br>

![image](https://user-images.githubusercontent.com/52172169/151702408-54afc274-a506-4193-ad14-c7f57a5c5b91.png)

<br><br>
<hr style="border: 2px solid;">
<br><br>

### Code 영역
<hr style="border-top: 1px solid;"><br>

실행되는 프로그램의 코드 부분이 저장되는 영역 (변수는 저장하지 않음) , Text 영역이라고도 불림

이 부분에 저장된 내용을 하나씩 처리하며 프로그램이 실행됨

**코드만 저장하고 있으므로 쓰기가 금지되어 있음.** 또한 바뀌는 것이 없으므로 크기가 고정되어 있음.

<br>
<br>

### Data 영역
<hr style="border-top: 1px solid;"><br>

전역변수, 정적 변수 등이 저장되는 공간

프로그램 실행 시 프로그래머가 선언한 변수에 대한 메모리 공간이 할당되고 프로그램 종료 시 해제.

<br>

**데이터 영역**은 **초기화된 변수가 할당**되며, 쓰기 가능한 data segment와 쓰기가 불가능한, 오직 읽기만 가능한 ro(read-only)data segment로 나뉨.

  + data segment는 프로그램이 실행되면서 값이 변할 수 있는 전역 변수 등이 저장 
  
  + rodata segment는 프로그램이 실행되면서 값이 변하면 안되는 const 변수나 상수 문자열, 전역 상수 등이 저장

<br>

+ **초기화 되지 않은 변수**는 **BSS영역**에 할당됨.

  + BSS 세그먼트는 프로그램이 시작될 때, 모든 값이 0으로 초기화된다.

<br>

**두 세그먼트 모두 쓰기 가능**하지만 **크기는 고정**되며, 이 영역은 실행 대상은 아니므로 **실행 권한은 부여되지 않음**.

<br>
<br>

### Stack 영역
<hr style="border-top: 1px solid;"><br>

**지역 변수와 함수 호출 시 매개 변수가 저장되는 공간**

프로그램에서 어떤 함수를 호출할 경우 그 함수는 자신만의 변수 공간을 갖음. 그리고 그 함수의 코드는 다른 메모리에 위치해 있는 텍스트 세그먼트에 저장됨.

**함수가 호출될 때에는 컨텍스트와 EIP가 변경돼야 하므로 함수 호출 시 전달된 모든 인자와 EIP가 되돌아가야 할 주소와 함수에서 사용된 모든 지역 변수를 저장하는데 쓰임.**

스택 프레임이라는 곳에 다 같이 저장되며, 스택에는 여러 스택 프레임이 있음.

**Stack Segment는 스택 프레임이 있는 스택 데이터 구조로 돼있으며 ESP는 스택의 맨 끝 주소를 추적하는데 사용됨.**

Stack 영역은 다른 영역과 달리 위에서 아래로 쌓이는 구조로 되어있음 따라서 **높은 주소에서 낮은 주소로 증가**

<br>

이 영역은 CPU가 자유롭게 읽고 쓸 수 있어야 하므로 **읽기, 쓰기 권한**이 부여됨.

<br>
<br>

### Heap 영역
<hr style="border-top: 1px solid;"><br>

프로그래머가 동적 할당을 해 생성한 변수가 Heap 영역에 저장됨. 

**즉, 프로그래머가 직접 접근할 수 있는 메모리 세그먼트.**

따라서 크기가 고정되어 있지 않고 필요에 따라 크기가 커지거나 작아질 수 있음.

**힙은 낮은 메모리 주소에서 높은 메모리 주소 방향으로 증가.**

<br>

동일하게 읽기와 쓰기 권한이 부여됨.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 스택의 이해
<hr style="border-top: 1px solid;"><br>

![image](https://user-images.githubusercontent.com/52172169/151702428-055ba745-0096-423a-9775-4835c9a2fd83.png)

<br>

+ 프로그램 정보 -> 공유 라이브러리 등

+ .text -> 기계어 코드

+ .data -> 전역변수 중 초기화된 데이터

+ .got -> Global Offset Table

+ .bss -> 전역변수 중 초기화되지 않은 데이터

+ heap -> 동적 메모리 영역

+ stack -> 함수관련 정보 : 지역변수, 매개 변수, 리턴주소

+ 프로그램 정보 -> 환경변수, 기타

<br>

스택의 구조는 LIFO(Last-In-First-Out)

<br>

+ SFP<sub>Saved Frame Pointer</sub>, RET<sub>Return Address</sub>는 함수가 호출되면 기본적으로 스택에 쌓이는 값들임. 

    + SFP에는 이전 ebp의 값이 저장이 됨. 즉, EBP를 원래 값으로 되돌리는데 쓰임

    + RET에는 함수 호출이 끝난 다음에 실행되어야 할 코드의 주소값이 들어있음. 즉, 이전 스택 프레임의 함수 컨텍스를 복구하는 것.

<br>

예를 들어, main함수는 ```__libc_start_main```에서 호출되어서 RET에는 ```__libc_start_main```의 다음 실행할 코드의 주소가 들어있음. 

<br>

즉, **함수 실행이 끝나면 전체 스택 프레임이 스택에서 제거되고 EIP는 함수 호출 전에 하던 작업을 계속 실행할 수 있게 복귀 주소로 설정됨.**

호출된 함수에서 또 다른 함수가 호출되면 추가적인 스택 프레임이 생성되는 과정이 반복됨.

함수를 호출할 땐 ```call 프로시저``` 명령어로 호출이 되는데 구조적으로 아래와 같음.
: 출처는 <a href="https://learn.dreamhack.io/63#5" target="_blank">x86 Assembly🤖: Essential Part(2)</a>

<br>

![image](https://user-images.githubusercontent.com/52172169/177022632-7740c136-b5e2-4b31-b7e6-5a7e4bfbe964.png)

<br>

참고로 인자가 있는 함수가 호출되면 스택에는 아래와 같이 쌓이게 됨.

![image](https://user-images.githubusercontent.com/52172169/151170949-454dba57-f798-4790-ae5a-3c2dc1d72eb3.png)

<br>

```c
int main()
{
	int a = 10;	
	int b = 20;	
	printf("&a = %p, &b = %p", &a, &b); 
  
  // &a = 0xbffffa14, &b = 0xbffffa10 -> a가 더 높은 주소, b가 더 낮은 주소
}
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Calling Convention
<hr style="border-top: 1px solid;"><br>

함수를 호출하는 규약으로 스택을 이용하여 파라미터를 전달할 때 인자 전달 방법, 인자 전달 순서, 
전달된 파라미터가 해제되는 곳, 리턴 값 전달 등을 명시함.

+ 64bit - SYSV(SYSTEM V) 호출규약

+ 32bit - cdecl 등 등

<br>

추가 
: <a href="https://com24everyday.tistory.com/330" target="_blank">com24everyday.tistory.com/330</a>

<br>
<br>

### _cdecl
<hr style="border-top: 1px solid;"><br>

매개 변수 전달 방식 : 우측 -> 좌측 

즉, 마지막 인자에서 첫 번쨰 인자까지 거꾸로 스택에 push

C, C++에서의 default

함수를 호출한 곳에서 파라미터 해제를 담당.

함수의 리턴값은 EAX 레지스터에 저장.

<br><br>

### SYSV
<hr style="border-top: 1px solid;"><br>

SYSV에서 정의한 함수 호출 규약은 다음의 특징을 같다.

<br>

+ 6개의 인자를 RDI, RSI, RDX, RCX, R8, R9에 순서대로 저장하여 전달, 더 많은 인자를 사용해야 할 때는 스택을 추가로 이용한다.

+ Caller에서 인자 전달에 사용된 스택을 정리한다.

+ 함수의 반환 값은 RAX로 전달한다.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 추가 자료
<hr style="border-top: 1px solid;"><br>

메모리 구조 
: <a href="https://dongdd.tistory.com/36" target="_blank">https://dongdd.tistory.com/36</a>  
: <a href="https://bpsecblog.wordpress.com/gdb_memory/" target="_blank">bpsecblog.wordpress.com/gdb_memory/</a>

<br>

기초 지식 
: <a href="https://dongdd.tistory.com/79?category=779916" target="_blank">https://dongdd.tistory.com/79?category=779916</a>   

<br>

Dreamhack Calling Convention
: <a href="https://dreamhack.io/lecture/courses/54" target="_blank">Background: Calling Convention</a>

<br><br>
<hr style="border: 2px solid;">
<br><br>
