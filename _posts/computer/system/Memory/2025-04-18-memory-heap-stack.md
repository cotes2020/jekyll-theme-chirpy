---
title: "Memory Heap/Stack"
# description: ""
categories: [컴퓨터, 시스템]
tags: []
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2025-04-18. 18:07 # Init
# last_modified_at: 2025-04-18. 18:07
---

## Heap

---

주로 메모리 동적 할당  
i.e. malloc, calloc, realloc  

- Heap => Use CLR's Garbage Collector  
  - 힙에 더 이상 사용하지 않는 객체가 있으면, 그 객체를 쓰레기로 간주하고 수거  
  - 왜 자동으로 메모리 리턴을 해주는 스택 대신 힙을 쓰는가?  
  - Because 코드블럭이 끝나도 데이터를 유지하고 싶을 때  
  - So 프로그머가 원한다면 언제까지라도 데이터를 살릴 수 있는 또 다른 메모리 영역 Heap을 CLR이 제공  

## 관계

---

Heap과 Stack은 같은 영역의 공간을 사용함.  

Heap: 메모리 위쪽 주소부터 할당.  
Stack: 메모리 아래쪽 주소부터 할당.  

Heap Overflow, Stack Overflow.  

## Stack

---

@@ Assembly  

Assembly가 빈번하게, 저장되는 값 저장, Subroutine, Procedure 등  
Stack 자료 구조  
BasePointer 시작 위치를 BP 레지스터, StackPointer 제일 위 위치를 SP 레지스터  
Glow Down, 다른 메모리 영역과 달리 거꾸로 자란다  

- Stack => 쌓였다가, 코드 블럭이 끝나면 차례대로 풀림 (메모리에서 제거)  
