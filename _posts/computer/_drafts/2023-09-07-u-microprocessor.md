---
title: "마이크로프로세서 과목"
# description: ""
categories: [컴퓨터, 🌑Computer-OS]
tags: []
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2023-09-08. 12:51
# last_modified_at: 2023-09-15. 12:53
# last_modified_at: 2023-10-20. 13:50
# last_modified_at: 2023-10-27. 12:37
# last_modified_at: 2023-11-03. 14:03
# last_modified_at: 2023-11-24. 12:12
# last_modified_at: 2023-12-08. 10:51
# last_modified_at: 2023-12-15. 12:08
last_modified_at: 2024-08-29. 22:11
---

## 마이크로프로세서

---

Micro-Processor (작은-연산장치)  
프로세서(CPU, GPU, ... 일반적으로 CPU)의 기능을 한 개 ~ 몇 개 이내의 칩으로 집약한 처리기  

메모리와 주변 장치와의 외부 연결을 위한 핀을 가지고 있다.  

메모리: 어드레스 버스, 데이터 버스, ...  
주변 장치: 전원 공급, 발진기, ...  

데이터 버스 크기, 어드레스 버스 크기, 연산 레지스터 크기, 범용 레지스터 수, 클록 스피드 등으로 성능이나 규모가 분류된다.  

강의에서는 AVR2560을 다룸  
AVR - Alf-Egil Bogen, Vergard Wollen, RISC  

## 범용 입/출력 포트 레지스터 - General Purpose IO Port Register

---

레지스터는 바이트 단위, 기능은 비트 단위  
AVR MCU 모든 레지스터는 메모리 앱 읽기/쓰기 방식 접근 가능, 일부는 격리 읽기/쓰기 접근 가능  

DDR: 데이터 방향 지정 레지스터  
→ 각 핀에 대해, 입력(0)으로 쓸지 출력(1)으로 쓸지 설정  
→ i.e. DDR == 0x01, 1핀 출력 + 7핀 입력  

PORT: 출력 용으로 설정된 핀에 대해, 데이터를 쓰는/입력하는 레지스터  
PIN: 입력 용으로 설정된 핀에 대해, 데이터를 읽는/출력하는 레지스터  

- 입/출력 레지스터
  - A ~ L 11개 (I 제외)
  - PIN 수: 11 * 8 = 88
  - 각각 DDRA, PORTA, PINA 명칭
    - \<avr/io.h\> 에 정의

- 연산
  - PORTF = 0xff: 모두 출력으로 설정
  - PORTF = PORTF | 0x01: 첫 핀만 출력으로 설정, 나머지 기존 그대로
  - PORTF = PROTF & 0xfe: 첫 핀만 입력으로 설정, 나머지 기존 그대로
  - v = PINF & 0x01: 맨 첫 핀의 입력값만 필터링

## 특수 기능 레지스터 - SFR - Special Function Register

---

범용 레지스터, 범용 입/출력 레지스터 → 데이터 보관/전달  
특수 기능 레지스터 → 설정: MCU 주요 장치들의 동작 방식, 주요 부우의 구성 요소 형태 조정  

마찬가지로, 레지스터는 바이트 단위, 기능은 비트 단위  
마찬가지로, AVR MCU 모든 레지스터는 메모리 앱 읽기/쓰기 방식 접근 가능, 일부는 격리 읽기/쓰기 접근 가능  

USART - Universal Synchronous and Asynchronous serial Receiver & transmitter  
UDRn: USART Data Register  

## 2차시

---

7372800  

범용 입출력 포트  

`#include <avr/io.h>`  
에 메모리 맵 대응이 되어잇다  

Like  
→ #define PORTE \#(unsigned char)0x180  

레지스터 3개  
PORTL 이 중심?  

PORTL 출력  
PINL 입력  

DDRL = RORTL을 제어하기 위한 주소?  
데이터를 입력할지 출력할지 여부 결정  
Direction Register  

`DDRL = 0xff // 1 출력`  
`DDRL = 0x00 // 0 입력`  

GPIU?  

@ 25p  

## 3차시

---

### USART

Universal Synchronous and Asynchronous serial Receiver and Transmitter  
Synchronous 보다 주로 Asynchronous를 씀  

### UDRn

USART Data register of the nth device  

하나의 8비트 레지스터를 I/O용으로 공유  
`UDR0 = x;`  
`x = UDR0;`  
I/O 동작에 따라 내부적으로 분리하여 인식  
I/O 동작을 위해서는 장치(디바이스) 초기화가 필요  

상대방에서 보내온 데이터가 저장되었다가 읽은 순간 소멸  
보낼 데이터를 쓰는 순간 전송이 시작됨  

### UBRRnL, UBRRnH

USART Baud Rate Registers of the nth device  
16비트 통신 속도 bps bit per second 조정  

UBRRnL Low바이트  
UBRRnH High바이트  

UCSRnA  
USART Control and Status Register A of the nth device  
Bit 1: U2Xn(Double the Transmission Speed of the nth device)  
설정된 통신 속도의 2배 송수신 여부  

UCSRnB  
USART Control and Status Register B of the nth device  
Bit 3: TXENn(Transmitter Enable of the nth device)  
송신 기능 활성화 여부  

### 비트 연산 매크로 (bit on/off)

sbi(byte b, int n): Set Bit  
8비트 변수 레지스터 b의 n번 비트를 1로 변경 (on)  
I.E. sbi(UCSR0A, U2X0): UCSR0A 레지스터의 U2XN(==1)번 비트를 1로 변경  

cbi(byte b, int n): Clear Bit  
8비트 변수 레지스터 b의 n번 비트를 0으로 변경 (off)  
I.E. cbi(UCSR0B, TXEN0): UCSR0B 레지스터의 TXEN0(==3)번 비트를 0으로 변경  

@ 비트 번호는 avr/io.h 에 다 정의되어 있음  

```c

#include <avr/io.h>
#include <compat/deprecated.h>

void uart_init()
{
    // 115.2Kbps
    UBRR0H = 0x00;
    UBRR0L = 0x07;

    // 통신 속도 2배
    sbi(UCSR0A, U2X0);

    // TX enable (Transmitter Enable, 송신 기능 활성화)
    sbi(UCSR0B, TXEN0);
}

void uart_putchar(char ch)
{
    if (ch == '\n')
        uart_putchar('\r');

    UDR0 = ch;
}

```

USART, 시리얼통신  

@  제어문자  

---

함수 선언(연결)은 헤더 파일을 이용함.  
소스 코드는 가급적 기능(함수) 단위로 작게 나누고, 파일명에 기능의 의미를 부여함.  
독립된 파일간의 함수 호출 시 함수 선언에 유의해야 함.  

임베디드 프로그램의 실행은 어떤 경우에도 어플리케이션의 범위를 벗어나서는 안됨.  

```c
int main()
{
    while(1) ; // must not return
    return 0;
}
```

## _

---

@ U 중간고사 출제: 상수선파에 대해 설명하고, 그 해결방법을 적으시오.  
@ U 중간고사 출제: 'DDRA, PORTA ~' 코드를 보고 아는대로 적으시오.  
@ U 중간고사 출제: 'uart_init' 코드를 보고 아는대로 적으시오.  
@ U 중간고사 출제: 코드 빈칸을 완성하시오, 프로그램을 완성하시오, Strtok  

@ U 기말고사 예상: 동적 웹 페이지 Dynamic Web Page, 이미 만들어진 텍스트 파일이 아니라, .exe 형태의 실행 파일을 실행시켜 그 결과를 받아 넘겨주는 경우  

@ U 기말고사 예상: IIS 등의 웹 서버는 요청된 URL이 지시하는 실행 파일을 실행시켜, 그 출력 결과를 브라우저에게 전달함.  
@ U 기말고사 예상: 따라서, 웹 서버와 실행 프로그램은 IPC (Inter-Process Communication) 수단을 사용하여 정보를 주고 받아야 함.  
@ U 기말고사 예상: 웹 서버와 실행 프로그램 사이의 IPC 수단은 공통적으로 정의되어야만 웹 환경에서 표준으로 사용될 수 있음 -> CGI (Commong Gateway Interface)  
@ U 기말고사 예상: 입력 IPC 수단으로 표준 입력(stdin)과 환경 변수 등 두 가지를 사용  
@ U 기말고사 예상: 출력 IPC 수단으로 표준 출력(stout)을 사용함 (printf())  

@ U 기말고사 예상: GET 쿼리는 CGI에서 'QUERY_STRING' 환경변수로 접근 가능, Windows는 getEnv함수로 환경 변수 접근 가능  
@ U 기말고사 예상: C, '\33' 같이 백슬래시 붙이면 8진수, '\33'은 ASCII로 ESC  
