---
title: "Addressing Mode"
# description: ""
categories: [컴퓨터, 시스템]
tags: []
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2023-11-10. 09:21
# last_modified_at: 2023-11-26. 01:42
last_modified_at: 2024-08-29. 22:12
---

## 주소 지정 모드

---

- 주소 지정 모드 Addressing Mode
  - 주소 즉시 Address Immediate 주소 지정 모드
    - 주소 모드
      - 피연산자에 나타난 값을 주소로 해석함 (직접 간접, 절대 상대 주소 해당)
    - 즉시 모드
      - 피연산자에 나타난 값은 상수로 해석함
  - 직접 간접 Direct Indirect 주소 지정 모드
    - 직접 주소
      - 기계 명령어의 피연산자 부분이 주기억장치 접근 주소로 사용되는 경우
      - #주소 → 값 가져옴
    - 간접 주소
      - 피연산자 부분이 가르키는 곳에 저장된 기억장치 같이 2차(최종) 주소로 사용되는 경우
      - #주소 → 주소 → 값 가져옴 Like 포인터
  - 절대 상대 Absolute Relative 주소 지정 모드
    - 절대 주소
      - 피연산자에 나타난 주소가 그대로 주기억장치 접근 주소로 사용되는 경우
      - #주소 → 값 가져옴
    - 상대 주소
      - 피연산자에 나타난 주소에 제 3의 기준 값을 더한 값이 주기억장치 접근 주소로 사용되는 경우
      - #주소 + 기준 값 → 값 가져옴

- OP Coder 연산 코드, Operand 피연산자  
- I.E. Add_절대/상대_직접/간접, 상수  

## TODO

---

간단한 메모리 주소 모드

1. 참고
   - Reg[R]: 레지스터 R 안의 값
   - Mem[M]: 메모리 M 안의 값값

2. 표준모드 (R)
   - Mem[Reg[R]]
   - 레지스터 R은 메모리의 주소를 나타냄
   - C의 포인터 역참조 (Dereferencing)
   - movq (%rcx), %rax

3. 변위모드 D(R)
   - Mem[Reg[R]+D]
   - 레지스터 R은 메모리 구역의 시작 주소를 나타냄
   - 상수 변위 D는 오프셋을 나타냄
   - movq 8(%rbp), %rdx

```c
void Temp(SomeType a, SomeType b) {}
// a → %rdi, b → %rsi

__asm
{
    Temp:
        movq (%rdi), %rax
        movq (%rsi), %rdx
        movq %rdx, (%rdi)
        movq %rax, (%rsi)
    ret
}
```

- 완전한 메모리 주소 모드 (인덱스 주소모드)
  - 일반 형태
    - D(Rb, Ri, S) Mem[Reg[Rb]+S*Reg[Ri]+D]
    - D: 상수 변위(Displacement) 1,2,4 Bytes
    - Rb: Base Register: 16개 정수 레지스터 중 어떤 것
    - Ri: Index Register: %rap를 제외한 어떤 것
      - 의외로 %ebp를 사용할 수도 있음
    - s: 배울 (Scale): 1,2,4,8 (why there numbers?)
  - 특수 형태
    - (Rb, Ri) Mem[Reg[Rb]+Reg[Ri]]
    - D(Rb,Ri) Mem[Reg[Rb]+Reg[Ri]+D]
    - (Rb,Ri,S) Mem[Reg[Rb]+S*Reg[Ri]]

@ 여러 오퍼랜드 형태  

- 데이터의 이동: x86-64
  - 데이터 이동 move Source, Dest
    - Indtel은 move Dest, Source
  - 피연산자 (오퍼랜드 Operands)
    - Immediate 정수 상수 데이터
      - i.e. $0x400, $-533
      - C상수 같지만 '$'접두어 붙임
      - 1,2,4 Bytes로 인코드
    - Register: 16개 정수 레지스터 중 하나
      - i.e. %rax, %r13
      - 특별히 %rsp는 특수 용도로 예약되어 있음
      - 그 외 레지스터들은 특정 명령을 위한 특수용도를 가짐
    - Memory: 레지스터의 주어진 주소에서 연속된 8바이트 메모리
      - 간단 에 :%rax
      - 다양한 주소모드 (Address Modes) 제공

- movq 오퍼런드 조합
  - Imm
    - to Reg, movq $0x4, %rax
      - Like c temp = 0x4;
    - to Mem, movq $-146, (%rax)
      - Like c *p = -147;
  - Reg
    - to Reg, movq %rax, %rdx
      - Like c temp2 = temp1;
    - to Mem, movq %rax, (%rdx)
      - Like *p = temp;
  - Mem
    - to Reg. movq (%rax), %rdx
      - Like c temp = *p;
  - 하나의 명령으로 메모리에서 메모리로는 이동 불가

- 데이터 이동
  - movx, x in {b,w,l,q}
  - movq, 8Byte quad word
  - movl, 4Byte double word
  - movw, 2Byte word
  - movb, 1Byte byte
