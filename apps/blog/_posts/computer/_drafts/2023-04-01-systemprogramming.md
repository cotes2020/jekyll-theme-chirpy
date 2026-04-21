---
title: "시스템 프로그래밍"
# description: ""
categories: [컴퓨터, 🌑Computer-OS]
tags: []
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2023-04-01. 00:00 # ?
# last_modified_at: 2023-11-17. 09:33
# last_modified_at: 2023-11-26. 01:03
last_modified_at: 2024-08-29. 21:27
---

## 1

---

- 1.2 컴파일 시스템
  - 목적 프로그램
    - 재배치 가능 목적 프로그램 → 목적 파일
    - 실행 가능 목적 파일 → 실행 파일
  - Unix 컴파일
    - gcc -o hello hello.c
    - @ GNU Project
      - @ Free SW: "free" as in "free speech", not "free beer"

- 1.3 컴파일 시스템의 이해
  - 프로그램 성능 최적화
    - 기계어 수준 코드 이해
    - switch vs if-else, while vs for
  - 링킹 에러의 이해r
    - Link-Time Error, Compile-Time Error
  - 보안 약점 회피
    - 버퍼 오버플로 버그 Buffer Overflow Bugs
    - 인터넷과 네트워크 상의 보안 약점 - security holes

- 1.4 컴퓨터 시스템 - 하드웨어 구성
  - CPU
    - CPU operations
    - 적재Load - 작업Operate - 저장Store - 점프Jump
  - The process of Loading "hello" Code from KeyBoard... @
  - The process of Loading Executable File from Disk to MainMemory... @
  - The process of Printing Output Stream from Memory to Monitor

- 1.8 Computer System & Network

- 1.9 Hot Topics
  - Amdahl's law
    - 컴퓨터 시스템의 일부를 개선할 때 전체적으로 얼마 만큼의 최대 성능 향상이 있는지 계산하는 데 사용
    - 어떤 시스템을 개선하여 전체 작업 중 a%의 부분에서 k배의 성능이 향상되었을 때 전체 시스템에서 최대 성능 향상
    - → 뭘 최적화 시켜야 더 효율적인가?
  - 동시성 프로세스: 하나의 프로세서에서 다수의 프로세스 실행
  - 하이퍼 스레딩
    - 하나의 프로세서가 두 개의 논리적 프로세스처럼 작동하도록 함
    - 컴퓨터 처리속도 향상
    - i7에 적용
  - Abstraction of Computer System
    - Virtual Machine - OS
      - Processes
        - Instruction Set Architecture - Processor
        - Virtual Memory - Main Memory
          - Files - I/O Devices

## 2 Bits and Bytes

---

- 정보 표현: 비트
- 2진수/16진수
- 바이트 표현
- 불 대수
- C에서의 표기, C의 연산
  - 비트 단위 연산
  - 논리연산
  - 비트이동연산

### 2.1 정보의 저장

- 바이트 중심의 메모리 궝
  - 프로그램은 가상주소 Virtual Addresses로 표현됨
    - 개념상 아주 큰 바이트 배열
    - 각각의 바이트는 자신의 주소를 가짐
    - 모든 주소들 - 가상주소공간
  - 시스템은 특정 "프로세스"에 주소공간 Address Space을 제공
    - 프로세스-실행중인 프로그램
    - 프로그램은 자신의 데이터를 운영(다른 프로그램의 데이터는 다루지 않음)
  - 컴파일러+런타임시스템은 메모리 할당 Allocation
    - 각각의 프로그램 객체를 어디에 저장할 것인가?
      - 프로그램 개체: 프로그램 데이터, 명령, 제어정보 등
    - Multiple Mechanisms: static, stack, and heap
    - 어떠한 경우에도 단일 가상주소공간 내에 모두 할당

가상주소 = 논리주소  

- 16진수 표현: 바이트 값의 인코딩
  - Byte = 8 bits
    - Binary: 00000000 ~ 111111111
    - Decimal: 0 ~ 255
    - 2진수는 너무 길고, 10진수는 비트 패턴 변환이 어려움
    - 1바이트: Hexadecimal 00 ~ FF
      - 밑수 16 숫자 표현
      - Use Characters '0' ~ '9' ~ 'A' ~ 'F'
      - Write FA1D37B in C as
        - 0xFA1D37B, 0xfa1d37b
        - 0xC97B = 1100 1001 0111 1011

- 컴퓨터의 Words
  - 컴퓨터는 워드크기를 갖는다
  - 정수 데이터나 주소의 명목상 크기
  - 현재 대부분의 컴퓨터는 32bits(4 Bytes) 워드
    - 4GB 바이트로 주소를 한정
    - 많은 메모리르 사용하는 응용에겐 작을 수 있음
  - 최신 시스템은 64Bits (8Bytes) Word
    - 16 Exabytes의 잠재적 주소 공간
    - x86 - 64 컴퓨터 48bit 주소 지원: 256 Terabytes
  - 컴퓨터는 다수의 데이터 형식 지원
    - Typical 32-bit, Intel IA32, x86-64 별 데이터 크기
      - 특히 int, pointer

- 주소 지정과 바이트 순서
  - 다중 바이트 (e.g. intergers)에 표현되는 객체에 대하여, 다음 두 가지 사항 필요
    - 객체의 주소가 무엇인가?
    - 메모리 안에 바이트들을 어떤 순서로 배치 하는가:
  - 다중 바이트 객체
    - 연속된 바이트로 저장
    - 시작 주소: 사용된 바이트의 가장 작은 주소

- 워드 중심 메모리 구성
  - 주소는 바이트 위치 명세
    - 워드의 첫 바이트 주소
    - 연속된 워드의 주소는 다름
      - by 4 (32bit) ot 8 (64bit)
      - i.e. 32bit: 0000, 0004, 0008, 0012
      - i.e. 64bit: 0000, 0008

- Byte Ordering
  - 메모리에 바이트들으 정렬하는 방법
  - 규칙 Conventions
    - Big Endian: IBM, Sun
      - 최하위바이트 LSB가 가장 상위 주소에 배치
      - Come last
    - Little Endian: x86
      - 최하위바이트 LSB가 가장 하위 주소에 배치
      - Come First

@ 네트워크는 Big Endian 쓰기로  
@ 달걀 사진?  

- Byte Ordering Example
  - Example
    - 변수 x는 4Byte값 0x1234567
    - Address given by &x is 0x100
      - Big Endian: 01 23 45 67
      - Little Endian: 67 45 23 01

- Reading Byte - Reversed Listings
  - 역어셈블리 Disassembly
    - 이진 기계어 코드의 문자 표현
    - 기계 코드를 읽는 프로그램이 생성

@ 예시  

포인터 값들은 컴퓨터에 따라 다름  
Different compiler & machines assign different locations to objects  

- Two's complement representation (Covered Later)
- Solaris/SUN, Linux/x86-86, Linux/Alpha, IA32

- String의 표시
  - C에서 스트링 String
    - 문자들을 배열로 표시
    - 아스키 형식으로 인코드
      - 문자 집합을 표준 7-bit 인코딩
      - 문자 "0" = 코드 0x30
        - 10진수 숫자 n = 코드 0x30 + n
      - 문자열 String 은 null 값으로 종료
        - Final Character = 0x00
  - 호환성 문제
    - Byte Ordering 은 문제가 안됨
      - 데이터는 1 Byte 크기

- ASCII 코드 차트  

- Different Machines Follow Different Conventions
  - Word Size
  - Byte Ordering
  - Representations (Integer, Floating-point)

- Whren Prograaming, Be Aware Of...
  - Type Casting & Mixed Signed/Unsigned Expressions
  - Overflow
  - Error Propagation
  - Byte Ordering

정보의 표현과 처리  

### 2.1.5 코드의 표현

프로그램은 명령들을 그 순서에 맞게 부호화한 것이다.  

명령은 산술연산, 메모리 읽기/쓰기, 조건 분기등의 개별적 단순 연산으로 구성된다.  

명령은 바이트들로 부호화된다.  
→ Alpha, Sun, Mac은 4-Byte 명령들을 사용: RISC, Reduced Instruction Set Computer  
→ PC는 가별 길이 명령들 사용: CISC, Complex Instruction Set Compute  

서로 다른 컴퓨터들 → 서로 다른 부호화 방식
→ 이진코드는 대부분 호환성 없음

근본 개념 → 프로그램 역시 바이트의 연속 Byte Sequences

- C 함수 → 컴파일 → 기계어
  - Machine Code (Byte Representations)
    - Linux 32, Windows, Sun, Linux 64, ...
    - 서로 다른 컴퓨터들은 완전히 서로 다른 명령과 인코딩 방식 사용

- 명령의 표현
  - Sun은 2, 4-Byte Instructions 사용
  - PC는 길이가 1, 2, 3-Byte들을 갖는 명령들 사용
    - Windows / Linux는 완전한 바이너리 호환성 Binary Compatibility 을 제공 못함

### 2.1.6 불 대수 - Boolean Algebra

- 논리의 대수적 표현
  - True = 1, False = 0 으로 부호화
  - 집합 { 0, 1 } 에 대해서 정의

[Bit Wise Operate](/posts/bitwise-operator/)  

---

@ 221015  

show_bytes 실행 예제  

```c
int a = 15213;
printf("int a = 15213;\n");
show_bytes((pointer) &a, sizeof(int));
```

```c
int a = 15213;
0x11ffffcb8 0x6d → 0110 1101
0x11ffffcb9 0x3b → 0011 1011
0x11ffffcba 0x00
0x11ffffcbb 0x00
```

@ 221021  

2.2 정수의 표현  

### 2.2.1 C 정수 표현

- 여러 정수형 데이터 타입 지원
- 부호 없는 정수, unsigned 수식어
- 부호 있는 정우, Signed number representations
  - 비 대칭 범위를 가짐
  - 음수 범위가 양수범위보다 1 큼
  - 2의 보수(부호형) 인코딩
    - 부호 비트, 0 양수, 1 음수
  - 부호 여부에 따른 인코딩
    - 양수는 부호 여부 관계없이 똑같음
    - 음수는 부호 여부에 따라 다름

### 2.2.2 C 정수 변환, 캐스팅

- 비부호형과 부호형 간의 변환 Castings
  - T2B → B2U, B2U → T2B
  - 비트 패턴은 유지됨
  - 양수는 불변 (부호 여부 관계없이 똑같으니까)
  - 음수는 큰 양수 값으로 변화, (비부호형의 최대값 + 1 = 2^비트수)만큼의 변화
  - (int) or (unsigned)
- 상수 값 뒤에 U 접미사 붙이면 Unsigned
- 단일 수식(비교 연산 포함)에 부호형 비부호형 혼합시, 묵시적으로 부호형 비부호형으로 변환

### 2.2.3 C 확장, 절삭

- Zero Extension, 비부호 정수에 0 복제하여 확장하기
- Sign Extension, 부호 정수에 부호비트(MSB) 복제하여 확장하기
- 작은 정수 데이터 형에서 큰 데이터 형으로 변활할 때 수행

- 숫자 절삭으로 값이 변경될 수 있음 → 오버플로의 형태
- 비부호 숫자 x에 대하여, x를 k 비트 만큼 절삭 = x mod 2^k
- 부호 숫자, mod와 유사하게

### 2.3 정수산술연산

- 실제 합 w+1 bits 요구됨

- 비부호형 덧셈
  - Carry 출력 무시 → Modular

- 부호형 덧셈
  - MSB 버림, 나머지 비트들은 2의보수로서 정수를 다룸
  - 양수 음수 오버플로 시 +- 2^(w-1)

- 2의 보수 반전 (보수 & 증가)
  - ~x + 1 == -x (덧셈의 역원 additive inverse = 0)
  - ~x + x == 1... == -1
  - 0
  - ~0 = 1... == -1
  - ~0 + 1 = 0... == 0

- 곱셈
  - 비부호형: 2w까지 필요
    - i.e. 111 * 111 = 110001
  - 부호형
    - 최솟값(음수): 2w-1
      - i.e. 100 * 011 = 001100
    - 최댓값(양수): 2w (최솟값)^2 인 경우에만
      - i.e. 100 * 100 = 010000
  - 비부호, 실제곱 2*w, 상위 w 비트 무시, 모듈러 연산 적용됨
  - 부호, 실제곱 2*w, 상위 w 비트 무시, 비부호 결과와 하위 비트들은 동일
  - 상수를 사용한 곱셈
    - u << k = u * 2^k
    - u << 3 = u * 8
    - u << 5 - u << 3 = u * 24
    - 대부분 쉬프트와 덧셈이 곱셈보다 빠름
      - 컴파일러가 곱셈을 쉬프트 연산 코드로 자동 생성

```c
int mul12(int x)
{
    return x * 12;

    // 아래와 같이 컴파일 된다
    __asm
    {
        leal (&eax, %eax, 2), %eax
        sall $2, %eax
    }

    // 아래와 같은 의미
    // t ← x + x * 2
    // return t << 2
}
```

- 2의 거듭제곱 나눗셈
  - 결과 내림
    - i.e. 3.14 = 3, -3.14 = -4

- Modular 산술 연산 형태로 수행
  - 워드의 길이가 유한
  - 가능한 값의 범위가 제한
  - 연산 결과과 Overflow일수도 잇음
- 비부호형과 부호형(2의보수 방식)
  - 동일한 비트패턴을 가짐

## 3 프로그램의 컴퓨터 수준 표현1: 기초

---

- 개요
- 역사적 관심
- 프로그램의 인코딩
- 데이터 형식
- 정보의 접근

### 3.0 개요

- 인텔 x86 프로세서
  - 랩탑/데스크탑/서버 시장을 완전 장악
  - 프로세서 설계의 진화
    - 1978 소개된 8086까지의 역 호환성 제공
    - 시간의 흐름에 따라 더 많은 특징이 추가됨
  - CISC 구조 (Complex Instruction Set Computer)
    - 서로 다른 형식을 갖는 많은 명령들을 가짐
    - RISC의 성능과 차이 (Reduced Instruction Set Computers)
      - 인텔은 성능을 높이고 있다, 속도에 의함, 저전력

무어의 법칙  

### 3.1 역사적 관심

- Name, Date, Transistors, MHz
  - 8086, 1978, 29K, 5~10
    - 최초 16비트 프로세서, IBM PC & DOS의 기초
    - 1MB 주소공간
  - 386, 1985, 275K, 16-33
    - 최초 32비트 프로세서 (IA32)
    - 유닉스 실행 가능한 "Flat Addressing" (선형주소) 모델 추가
  - Pentium 4E, 2004, 124M, 2800-3800
    - 최초 64비트 프로세서 (x86-64)
  - Core2, 2006, 291M, 1060-3500
    - 최초 다중 코어 인텔 프로세서
  - Core i7, 2008,, 781M, 1700-3900
    - 4 코어

인텔의 Hyper-Threading 기법  

Dual-Core Processor, 인텔 x86 프로세서  
@ 사진

- x86 호환 기종 (Clones): Advanced Micro Devices (AMD)
  - 역사적으로
    - AMD는 인텔의 후발주자, 좀 느리나 저렴
  - 그 후,
    - DEC (Digital Equipment Corp.) 에서 정상급 회로 설계자 영입
    - 다른 회사 (ATI 등)들을 합병
    - Opteron 구축: Pentium 4에 강한 경쟁자로 부상
    - 자신들의 64비트 확장인 x86-64를 개발

@ QuadCore Opteron

- Intel의 64 bit
  - 2001, IA32에서 IA64로 근본적 변화 시도
    - 전체적으로 과거와 다른 구조 Itanium
    - IA32는 구형 코들만 실행됨 (상위 호환성 유지)
    - 성능은 실망적
  - 2003, AMD는 발전 방안에 참여
    - x86-64 (Now called "AMD64")
  - Intel은 IA64에 주안점을 둠
    - Hard to admit mistake or that AMD is better
  - 2004, 인텔은 IA32의 확장인 EM64T를 발표
    - Extended Memory 64-bit Technology
    - Almost identical to x86-64
  - All but low-end x86 processors support x86-64
    - But, lots of code stll runs in 32-bit mode

### 3.2 프로그램의 인코딩

- 정의
  - Architecture (Also ISA: Instruction set Architecture)
    - 어셈블리/기계어 코드 작성 또는 이해에 필요한 프로세서 설계 부분
    - 예: 명령어 집합 명세, 레지스터
  - Microarchitecture: implementation of the architecture
    - ISA가 프로세스 상에 구현되는 방법
    - 예: 캐쉬 크기와 코아 주파수
  - 코드 형태
    - 기계코드: 프로세서가 실행하는 바이트 수준 프로그램
    - 어셈블리 코드: 기계코드의 텍스트 버전
  - Example ISAs (intel)
    - 인텔: x86, IA32, Itanium, x86-64
    - ARM :거의 모든 이동전화에서 사용

- 어셈블리/기계코드 관점
  - 프로그래머가 볼 수 있느 상태
    - PC Program Counter 프로그램 카운터
      - 다음에 실행할 명령의 주소
      - IEP 라 칭함 (IA32) or RIP (x86-64)
    - 정수 레지스터 파일
      - 빈번히 사용되는 프로그램 데이터 저장
    - 조건코드 레지스터
      - 최근의 산술연사에 대한 상태정보 저장
      - 조건 분기를 위하여 사용됨 (진리값)
    - 메모리
      - Byte로 주소화된 배열
      - 코드와 사용자 데이터
      - 프로시저를 지원하는 스택

- C를 목적코드로 전환
  - i.e. p1.c p2.c
  - gcc -Og p1.c p2.c -o p
  - -o: 출력 파일명을 지정
  - -Og 기본적인 최적화 옵션 (new to recent ver of GGCC)
  - 파일 p에 결과 이진 파일 저장

text: C Program (p1.c, p2.c) (-S: ~.s 어셈블리 파일 생성)  
Compiler (gcc -Og -S)  
text: Asm program (p1.s,  p2.s)  
Assembler (gcc or as)  
binary: Object Program (p1.o, p2.o)  
Linker (gcc or ld), With Static Livraries (.a, .lib in windows)  
binary: Executable Progam (p)  

- 어셈블리의 특성: Data Types
  - 1,2,4, 8 바이트의 정수형 데이터
    - 데이터 갑
    - 주소 (미형식 포인터)
  - 부동소수점 데이터는 4,8,10 Bytes
  - 코드
    - 일련의 명령들을 인코딩하는 바이트 순서들
  - 집합체 (Aggregate: 배열, 구조체) 형식 없음
    - 다만 메모리에 연속으로 바이트들을 할당

- 어셈블리 특성: 연산
  - ALU에서 연산을 하려면 데이터가 필요
  - 레지스터 데이터나 메모리 데이터에서 데이터를 읽어 산술 함수 수행
  - 메모리와 레지스터 사이에 데이터 전송
    - 메모리에서 레지스터(CPU)로 데이터 이동 - Load (사이에 Bus)
    - 레지스터(CPU) 데이터를 메모리에 저장 - Store (사이에 Bus)
  - 전송 제어 Transfer Control
    - 프로시저까지 또는 프로시저에서 무조건 점프/무조건 분기 - to/from procecures, Like goto
    - 조건분기 Conditional Branch, Like if else

요즘 프로그래밍 언어
스트럭쳐드 프로그래밍  
goto가 없음

근데 어셈블리는 있다

gcc -o temp.c 실행 가능 목적 파일 (실행파일, object, 링크까지 끝난)  
gcc -c temp.c 재배치 가능 목적 파일 (목적파일, 링크는 없는 그냥 목적 파일)  

- 목적 코드 생성
  - Linux, gcc -Og -c temp.c
  - -c 옵션
    - as에 의한 어셈블까지만 수행
    - 링크는 미수행
  - temp.o 파일 산출
  - n바이트 안에 temp 목적코드 내장 (해보니까 n바이트라는 뜻)

- 목적코드
  - 어셈블러 (Assembler)
    - .s를 .o로 번역
    - 명령어 각각을 이진 인코딩
    - 거의 완벽한 실행 코드
    - 다른 파일들과의 코드 연결 (linkages)은 빠짐
  - 링커 (Linker)
    - 파일들 사이의 참조를 해결
    - 정적 런타임 라이브러리와 조합 (i.e. code for malloc, printf)
    - 일부 라이브러리들은 동적으로 연결 (Dynamically Linked), 프로그램 실행 시 링킹 발생

- 목적코드의 역 어셈블
  - Disassembled
  - Disassembler
    - objdump -d temp.o
    - 목적코드 조사에 유용한 도구
    - 일련의 명령들 비트 패턴을 분석
    - 어셈블리 코드와 유사한 해석 산출
    - a.out(complete executable) 이나 .o파일을 실행할 수 있음  

- 역어셈블의 다른 방법
  - Within gdb Debugger
  - gdb temp
  - disassemble temp
    - Disassemble procedure
  - x/`n`xb sumstore
    - Examine the n bytes starting at temp

- 역어셈블 할 수 있는 것은?
  - 실행코드로 번역될 수 있는 것
  - 역어셈블리는 바이트를 조사하고 어셈블리 소스를 재구성
  - Reverse engineeering forbidden by MS End User License Agrement

### 3.3 데이터 형식

- 데이터 형식
  - @@ bit byte word, word 기원
  - GAS (GNU 어셈블러)에서 "/"을 붙이는 데 문제 없음
    - FP도 "/"을 붙임
    - 왜냐하면, FP(부동소수점)는 정수와 다른 연산과 레지스터를 가짐

x86-64 c 데이터 형식 크기 (C, 인텔데이터 형식, 어셈블리-코드 접미사, 크기)

### 3.4 정보의 접근

@@ 메모리 주소 모드  

## _

---

*dest = t;  
movq %rax, (%rbs)  
0x40059e: 48 89 03  

C Code  
dest가 지정한 곳에 값 t를 저장  

Assembly Code  
> 8 바이트 값을 메모리로 이동  
>> x86-64 용어로 Quad words  
> Operands
>> t: 레지스터 %rax
>> dest: 레지스터 %rbx
>> *dest: 메모리 M[%rbx]

r = register  

Object Code  
> 3 바이트 명령
> 주소 0x40059e에 저장됨

(범용 레지스터)  
64bit 16개  
32bit 8개  

실행 가능 파일 생성  
실행파일 생성하려면 링커 필요  
One object file must contain main  
Combines with static run-time livraries (e.g., printf)  
Some libraries are dynamically linked (i.e. at execution)  
