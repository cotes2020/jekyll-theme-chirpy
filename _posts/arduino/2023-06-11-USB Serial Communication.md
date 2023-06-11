---
title: Section 01 아두이노 시리얼 통신 기능 이해
date: 2023-06-11 23:36:55 +0900
author: kkankkandev
categories: [Aduino, USB Serial Communication]
tags: [aduino, usb serial communication]     # TAG names should always be lowercase
---

# Chapter 07. USB Serial Communication

# 1. Arduino의 시리얼 통신 기능 이해

> Arduino의 중요한 특징 중 하나는 USB 시리얼 포트를 통해 직접 프로그램을 업로드할 수 있다는 점이다
> 

## Serial Port

<!-- ![Untitled 1](/assets/img/Untitled.png) -->
![사진1](https://private-user-images.githubusercontent.com/72260110/244950819-ff29f2ef-5e57-4237-90fc-03829bc13a87.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJrZXkiOiJrZXkxIiwiZXhwIjoxNjg2NTA3ODMzLCJuYmYiOjE2ODY1MDc1MzMsInBhdGgiOiIvNzIyNjAxMTAvMjQ0OTUwODE5LWZmMjlmMmVmLTVlNTctNDIzNy05MGZjLTAzODI5YmMxM2E4Ny5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBSVdOSllBWDRDU1ZFSDUzQSUyRjIwMjMwNjExJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDIzMDYxMVQxODE4NTNaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0xYzVkZDNhYzRhYjk1NmMyZTQ4NmMxZGE1MWRkMTZjMzcwY2I0Yzg2OTNmM2I5ZmNjZTcxMmZhYzM0OThiODcyJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.hPDO5u1ZdA43cOs0uGEJZRh4PfsrwoCRYxIj_NW71jk)
<!-- ![사진2](_site/assets/img/favicons/android-chrome-512x512.png) -->



- Arduino Uno의 0번과 1번 핀으로 시리얼 통신 수행 → 하드웨어 시리얼 포트
- 최근에는 시리얼 포트가 장착된 컴퓨터가 거의 없지만 USB 포트를 DB9 시리얼 포트로 변환하는 어댑터는 여전히 사용하고 있음
- Arduino Uno에 사용된 **ATmega328P** Micro Controller는 하나의 하드웨어 시리얼 포트만 가지고 있음
- Arduino Uno의 Hardware Serial Port는 **데이터 송신(TX, transmit)과 수신(RX, receive)에 디지털 0번과 1번 핀**을 사용

## Arduino가 USB Interface를 통해 통신할 수 있도록 사용하는 두 가지 방법

> 시리얼과 USB는 호환이되지 않는다.
> 
1. **별도의 집적 회로(IC)를 사용하는 방법**
    1. **Arduino Uno가** 시리얼과 USB사이의 변환을 위해 별도의 전용 Micro Controller IC를 사용
2. **USB 인터페이스를 내장하고 있는 마이크로 컨트롤러를 사용하는 방법**
    1. **Arduino 레오나르도**에 사용된 ATmega32U4 Micro Controller가 대표적인 예

## 1. 내장/외장 USB-Serial 변환장치를 사용하는 Arduino 보드

- USB-Serial 변환을 하기 위해서는 **FTDI**와 Silicon Labs에서 만든 칩(**CP210x**)이 흔히 사용되며 이들 칩은 Serial과 USB 사이의 변환 전용으로 사용 됨
- FTDI 칩이나 CP210 칩을 컴퓨터에 연결하면 컴퓨터에서는 DB9 포트와 같은 방법으로 제어할 수 잇는 ‘**가상 시리얼 포트**’가 나타남

![사진2](https://private-user-images.githubusercontent.com/72260110/244950804-f0cca405-1d26-47dc-a23f-e444c4a1e47f.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJrZXkiOiJrZXkxIiwiZXhwIjoxNjg2NTA3ODMzLCJuYmYiOjE2ODY1MDc1MzMsInBhdGgiOiIvNzIyNjAxMTAvMjQ0OTUwODA0LWYwY2NhNDA1LTFkMjYtNDdkYy1hMjNmLWU0NDRjNGExZTQ3Zi5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBSVdOSllBWDRDU1ZFSDUzQSUyRjIwMjMwNjExJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDIzMDYxMVQxODE4NTNaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1jY2UwYzk2Yzk1Y2EzYzFhNDI5NDI2YzU5NzU5Zjc5NGJiOWIzOGM4ODNlYTVjMzAxM2UxNTYyY2ZjZWNmOTE2JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.KXSre2KTNXcyWT8YPaJeXpvs_AJjslDAxZzM-VaWOtg)

<aside>
👨🏽‍🦯 마이크로컨트롤러가 동작할 때 USB를 통해 컴퓨터와 연결되어 있지 않아도 된다면 분리가 가능한 FTDI 프로그래머를 사용하는 것이 좋음

⇒ 제품의 단가를 낮출 수 있음
⇒ 제품의 크기를 작게 만들 수 있음

</aside>

### 보드 안에 FTDI 칩을 포함하고 있는 Arduino 보드

- Arduino Nano
- Arduino Extreme (단종)
- Arduino NG (단종)
- Arduino Diecimila (단종)
- Arduino Duemilanove (단종)
- 초기 Arduino Mega (단종)

### 외부 FTDI 케이블이나 브레이크아웃 보드를 사용하여 프로그래밍과 USB-시리얼 통신을 수행하는 Arduino 보드

- Arduino Mini
- Arduino Ethenet
- Arduino LilyPad
- Arduino Pro (단종)
- Arduino Pro Mini (단종)

## 2. 별도의 USB 기능을 포함하는 ATmega MicroController를 시리얼 변환기로 사용하는 Arduino 보드

### Arduino Uno

- USB-Serial 변환을 위해 FTDI 칩이 아닌 다른 IC(**ATmega16U2**)를 사용한 첫 번째 Arduino 보드.
    - (초기에는 ATmega8U2 사용)

### ATmega16U2 칩이 FTDI 칩과 다른 점

1. 윈도우에서 USB-Serial 변환 기능을 사용하기 위해서는 전용 드라이버를 설치해야 함 (Arduino IDE에 포함되어 있음)
2. 아두이노 우노라는 제품에 대한 생산자 ID(vendor ID)와 제품 ID(product ID)를 컴퓨터에 제공할 수 있음
    - FTDI 칩은 ‘일반(generic) USB-시리얼 장치’로 인식됨
    - 장치관리자에 ‘Arduino’로 표시됨
3. 별도의 펌웨어를 통해 USB-시리얼 변환 기능 이외의 기능 제공 가능
    - ATmega16U2에는 USB-Serial 변환 기능을 수행하기 위해 설치되어 있는 LUFA 라이브러리가 설치 되어있음
    - LUFA 기반 펌웨어가 아닌 다른 펌웨어로 교체 시, Arduino 보드가 **조이스틱, 키보드, MIDI** 장치 등 가상의 시리얼 포트와는 다른 장치로 컴퓨터에서 인식되도록 할 수 있음
    - LUFA 기반 펌웨어가 아닌 펌웨어를 사용하는 경우에는 Arduino에 프로그램을 업로드하기 위해 **AVRISP mkII**와 같은 별도의 프로그래머를 사용해야 함

## 3. USB 통신 기능이 내장된 MCU를 사용하는 아두이노 보드

### Arduino Leonardo

- 하나의 칩으로 메인 MCU와 USB 인터페이스 역할을 구현한 최초의 보드
- Arduono Leonardo를 포함한 여러 공식 아두이노 보드와 아두이노 호환 보드는 USB 통신 기능을 포함하고 있는 ATmega32U4 마이크로컨트롤러를 사용함

### USB 통신 기능을 내장한 마이크로컨트롤러를 사용하는 것의 장점

1. 가격
    1. 하나의 마이크로컨트롤러만 사용하면 되므로 단가를 줄일 수 있음
2. 범용성
    1. Arduino 보드를 키보드, 마우스, 조이스틱 등 시리얼 포트가 아닌 다른 USB 장치로 동작하도록 만들기가 쉬워짐 
3. 동시 통신
    1. USB를 통한 프로그램 업로드와 시리얼 통신을 위한 연결선을 분리해서 사용하므로 컴퓨터와 통신을 하면서 GPS와 같은 다른 시리얼 장치와 동시에 통신 가능

## 4. USB 호스트 기능이 있는 Arduino 보드

### USB 호스트 기능

- USB 장치를 아두이노 보드에 연결하여 사용할 수 있는 기능
- 연결된 USB 장치를 위한 드라이버가 있어야 함

### USB 호스트 기능이 있는 아두이노 보드

- Arduino 듀에, Arduino 제로, Arduino MKR100 등
- USB 호스트 라이브러리를 통해 장치 드라이버 제공

### AOA(Android Open Accessory) 프로토콜

- Arduino Mega ADK에서 사용
- Arduino와 Android 장치 사이의 통신 지원
- Android 장치에서 애플리케이션을 사용하여 아두이노의 입출력을 제어하는 것이 목적