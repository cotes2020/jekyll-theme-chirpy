---
title: 환경 변수 설정 및 주소 구하기
categories: [Hacking, System]
tags: [Linux, BOF, 환경 변수]
---

## 환경 변수 설정 및 해제
<hr style="border-top: 1px solid;">

1회성 설정 방법

+ env : 환경 변수 조회 및 해제

+ env NAME=VALUE : NAME 환경 변수에 VALUE 값 지정

+ env -u NAME : NAME 환경 변수 제거

<br>

+ set NAME=VALUE : NAME 환경 변수에 VALUE 값 지정, bash 에선 set 생략 가능

+ unset NAME : NAME 환경 변수 해제

<br>

+ export NAME=VALUE : NAME 환경 변수 설정

영구 설정 방법은 참고자료에서 확인

<br>
<br>
<hr style="border: 2px solid;">
<br>
<br>

## 환경변수 주소 구하기
<hr style="border-top: 1px solid;">

```c
#include <stdio.h>
#include <unistd.h>

int main(int argc, char *argv[])
{
   printf("env : %p\n", getenv(argv[1]));
   return 0;
}
```

위 코드는 프로그램이 실행 중일 때 환경 변수가 어디에 있는지 주소를 출력해주는 프로그램이다.

환경 변수에서 중요한 부분은 **환경 변수가 스택에 위치한다는 점**과 셸에서 값을 설정할 수 있다는 점이다.

따라서 환경 변수의 주소를 구할 때, 프로그램 이름의 길이가 환경변수 위치에 영향을 준다. 

실행되는 프로그램의 이름도 스택 어디간에 위치해 다른 주소를 쉬프트한다.

만약, **프로그램 이름 길이가 한 바이트 늘어나면 환경 변수 주소는 두 바이트 줄어든다.**


<br>
<br>
<hr style="border: 2px solid;">
<br>
<br>

## Reference
<hr style="border-top: 1px solid;">

<a href="https://sites.google.com/site/sunitwarehouse/os/linux/0001" target="_blank">https://sites.google.com/site/sunitwarehouse/os/linux/0001</a>  
<a href="https://hashcode.co.kr/questions/1893/%EB%A6%AC%EB%88%85%EC%8A%A4-%ED%99%98%EA%B2%BD%EB%B3%80%EC%88%98-%EC%84%A4%EC%A0%95%ED%95%A0-%EB%95%8C-env-set-export-declare" target="_blank">set, env, export, declare 차이점</a>  

