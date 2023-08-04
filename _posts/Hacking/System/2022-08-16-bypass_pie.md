---
title: Bypass PIE
date: 2022-08-16 16:05 +0900
categories: [Hacking, System]
tags: [PIE Bypass]
---

## Bypass PIE
<hr style="border-top: 1px solid;"><br>

출처
: <a href="https://dreamhack.io/lecture/courses/113" target="_blank">Background: PIE</a>

<br>

ASLR이 적용되면 스택, 힙, 라이브러리의 주소가 고정되지 않아서 공격하기 어렵다.

하지만, main 함수의 주소는 매번 같다는 점을 이용해 고정된 주소의 코드 가젯을 이용한 ROP를 수행할 수 있었다.

따라서 PIE는 코드 영역의 주소로 랜덤화 하는 보호기법이다.

<br>

PIE 보호기법을 우회하기 위해서는 **코드 영역의 주소를 알아내야 한다.**

PIE 보호기법이 설정되어 있을 때, 코드 영역은 공유 라이브러리처럼 메모리에 로딩되기 때문에 libc.so.6 라이브러리 주소를 구하는 과정과 같이 특정 코드 영역의 주소를 알아낸다면 **코드 영역 베이스 주소를 구할 수 있다.**

코드 영역 베이스 주소를 구한다면, 오프셋 계산을 통해 코드나 데이터 영역의 주소를 구할 수 있다.

<br>

코드 베이스 주소를 구하기 어렵다면, 반환 주소의 일부 바이트만 덮는 공격을 고려해볼 수도 있다.

이러한 공격 기법을 Partial Overwrite라고 한다.

ASLR의 특성 상, 코드 영역의 주소도 하위 12비트(1바이트 + 4비트) 값은 항상 같다.

<br>

```
$ ./pie
buf_stack addr: 0x7ffc85ef3 7e0
buf_heap addr: 0x55617ffcb 260
libc_base addr: 0x7f0989d06 000
printf addr: 0x7f0989d6a f00
main addr: 0x55617f129 7ba

$ ./pie
buf_stack addr: 0x7ffe9088b 1c0
buf_heap addr: 0x55e0a6116 260
libc_base addr: 0x7f9172a7e 000
printf addr: 0x7f9172ae2 f00
main addr: 0x55e0a564a 7ba

$ ./pie
buf_stack addr: 0x7ffec6da1 fa0
buf_heap addr: 0x5590e4175 260
libc_base addr: 0x7fdea61f2 000
printf addr: 0x7fdea6256 f00
main addr: 0x5590e1faf 7ba
```

<br>

따라서 사용하려는 코드 가젯의 주소가 반환 주소와 하위 한 바이트만 다르다면, 이 값만 덮어서 원하는 코드를 실행시킬 수 있다고 한다.

그러나 만약 두 바이트 이상이 다른 주소로 실행 흐름을 옮기고자 한다면, ASLR로 뒤섞이는 주소를 맞춰야 하므로 브루트 포싱이 필요하며, 공격이 확률에 따라 성공하게 된다고 한다.

어렵네..

<br><br>
<hr style="border: 2px solid;">
<br><br>
