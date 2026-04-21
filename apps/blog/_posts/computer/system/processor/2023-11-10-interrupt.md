---
title: "Interrupt"
# description: ""
categories: [컴퓨터, 시스템]
tags: []
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2023-11-10. 09:21
# last_modified_at: 2023-11-15. 16:060
last_modified_at: 2024-08-29. 21:28
---

## Interrupt 인터럽트

---

[참고 0](https://ko.wikipedia.org/wiki/%EC%9D%B8%ED%84%B0%EB%9F%BD%ED%8A%B8_%ED%95%B8%EB%93%A4%EB%9F%AC)  
[참고 1](https://ko.wikipedia.org/wiki/%EB%AA%85%EB%A0%B9_%EB%A0%88%EC%A7%80%EC%8A%A4%ED%84%B0)  

(가로채기, 처리 중간에 방해/중단되고 다른 일을 잠시 처리, 큰 틀에서 봤을 때 처리하는 일은 변함 없음)  

- #CPU에서 프로그램을 실행하고 있을 때 입출력 하드웨어 등의 장치에 예외상황이 발생하여 처리가 필요할 경우에 CPU에게 알려 처리할 수 있도록 하는 것
- CPU에 전달되는 사전 신호 Event Signal
- 사건 신호에는 여러 가지가 있으며, 주로 각각의 전용 회선으로 전달됨 (Source I.E. 키보드 마우스 등)
  - Vectored Interrupt
- 전용 회선이 없는 경우, 어떤 인터럽트가 발생했는지를 탐색해야 함
  - Interrupt Polling(Polling Interrupt)

- ISR, Interrupt Service Routine Or Interrupt Handler

## 인터럽트 우선 순위 Interrupt Priority

---

- 여러 개의 인터럽트가 동시에 발생한 경우 대응 처리 순서를 결정함
- 인터럽트에 대한 대응 처리 도중, 또 다른 인터럽트가 발생했을 때, 그 인터럽트를 보류 시킬 것인지 아니면 지금 즉시 처리할 것인지를 결정함
  - 새로 발생한 인터럽트의 우선 순위가 더 높으면 진행 중이던 대응 처리를 잠시 유보하고 새로운 인터럽트 처리를 먼저 처리한 후 재개함

## 인터럽트 사이클 Interrupt Cycle

---

CPU가 인터렙트 발생 여부를 체크(조사)하는 시기를 말한다.  

4단계의 [기계/명령 사이클](/posts/machine-instruction-cycle/)을 마칠 때마다, 다시말해 하나의 기계 명령어에 대한 처리를 마칠 때마다, 인터럽트 발생 여부를 조사한다.  
결국 기계/명령 사이클은, 인터럽트 사이클을 포함하여 총 5단계로 이루어져 있다고 볼 수 있다.  

## 인터럽트 유형

---

- 디바이스 인터럽트 Device Interrupt
  - 입출력 장치 등 CPU 외부 주변 기기에서 발생하는 인터럽트
    - 하드웨어 인터럽트 HW Interrupt 라고도 함

- 오류 인터럽트 Error Interrupt
  - CPU가 기계 명령어를 처리하는 도중에 발생하는 인터럽트
  - 잘못된 기계 명령어를 만나거나 0으로 나누는 등의 연산 불가 상황에서 발생
    - 예상하지 않는 오류 발생이란 뜻에서 예외 Exception라고 함

- 소프트웨어 인터럽트 SW Interrupt
  - 인터럽트를 발생시키는 기계 명령어, 즉 프로그램에 의해 발생된 인터럽트
  - 인터럽트 대응 처리를 테스트하거나, 시스템 콜을 위해 사용자 프로그램에서 운영체제로 진입하기 위해 사용
    - 프로그램의 흐름을 인위적으로 특별한 부분으로 빠져들게 하므로 트랩 Trap 이라고도 함

## != Polling 폴링

---

인터럽트는 폴링과 다르다.  

폴링은 하나의 장치 혹은 프로그램이 충돌 회피 또는 동기화 처리 등을 목적으로 다른 장치 혹은 프로그램의 상태를 주기적으로 검사하여 일정한 조건을 만족할 때 송신 등의 자료처리를 하는 방식  
대상을 주기적으로 감시하여 상황이 발생하면 해당처리 루틴을 실행해 처리하는 폴링과는 달리, 인터럽트는 상대가 CPU에게 일을 처리해 달라고 요청하는 수단.  

[Polling (폴링)](https://ko.wikipedia.org/wiki/%ED%8F%B4%EB%A7%81_(%EC%BB%B4%ED%93%A8%ED%84%B0_%EA%B3%BC%ED%95%99))
