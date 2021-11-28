---
title: bitstream tranceiver
author: HyungJoo Park
date: 2021-11-27 18:00:00 +0900
categories: [Exhibition,2021년]
tags: [post,hyungjoopark,arduino]
---

# Arduino SPI bitstream tranceiver

## 뭐하는 놈인가?


프로젝트를 하다보면 시스템 혹은 칩의 테스트를 위해 비트열을( ex)010010...) 출력해야 하는 상황이 꽤 자주 생긴다. 
보통은 테스트 회로를 만들 때 아두이노에 비트열을 직접 입력해서 코딩하는 경우가 많은데 이 경우, 비트를 수정할 때 마다 
코드를 수정해서 업로드 해야 하기 때문에 많이 번거롭다. 1인 개발일 때는 그렇게 번거로움이 크지 않을 수 있지만, 
코딩 담당, 테스트 담당이 따로 있거나 코딩을 여러명이서 하는 경우엔 많이 골치 아파질 수 있다.

서론이 길었는데 그래서 이 놈은 뭐하는 놈이냐면 비트열을 아두이노 코드에 직접 적는 것이 아닌 컴퓨터에 적어두고 전송할 수 있는 프로그램이다.
YAML형식의 텍스트 파일에 보내고 싶은 비트열을 적어두면 파이썬 프로그램이 그 파일을 읽어서 비트열을 뽑아 아두이노에 보내고 아누이노는 받은 비트열을 
SPI 통신 방식으로 외부에 출력한다. YAML파일에 비트열을 작성하는 방식에 따라 비트열의 길이, 비트열을 보내는 횟수를 조정할 수 있다.

## 예시

<img src="/assets/img/post/2021-11-27-HyungJoo_Park/EX_250khz.png" width="90%">

아두이노가 출력한 신호를 오실로스코프로 화면에 출력해 본 것이다. 초록색은 기준 클락으로 250khz로 진동하고 노란색이 비트스트림이다. 파란색은 EN신호로 신호가 출력중인지 아닌지를 나타낸다.

## 설치 및 사용법

<a href="https://niftylab.github.io/icscan_arduino/" target="_blank">docs</a> 참조

## 소스코드 링크
<a href="https://github.com/Park-Hyung-Joo/arduino_comm" target="_blank">source link</a>
