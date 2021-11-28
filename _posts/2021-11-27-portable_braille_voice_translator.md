---
title: 휴대용 점자 음성 번역기
author: JeongEun Kim, JeongHyun Kim, HanGyeol Yoon, SeongJin Lee, JeongEun Lee, InSeong Heo
date: 2021-11-27 00:35:00 +0900
categories: [Exhibition,2021년]
tags: [
    post,
    jeongeunkim,
    jeonghyunkim,
    hangyeolyoon,
    seongjinlee,
    jeongeunlee,
    inseongheo,
  ] # TAG names should always be lowercase, 띄어쓰기도 금지
---

# 휴대용 점자 음성번역기

## 프로젝트 목적 및 배경

사회적 약자의 불편한 점에 공감하여 프로젝트를 진행할 것을 계획하던 중, 시각장애인의 점자 문맹률이 높다는 사실을 접했다.
선진국의 경우 약 90%, 개발도상국의 경우 85% 정도의 점자문맹률이 나타난다. 교육시절도 부족하여, 시각장애인들이 점자를 배우는 것은 ‘ 또 다른 눈을 뜨는 것’이라 할 정도로 어렵다고 한다.
이러한 문제 상황에 해결을 돕고자 “시각장애인을 위한 휴대용 점자 인식 및 음성번역기“를 제작하기로 하였다. ‘점자 교육’, 그리고 ‘시각장애인의 생활의 질 개선’ 등의 효과를 기대하며 제품을 개발했다.

## 프로젝트 내용

### 카메라로 읽어온 이미지 전처리

- OpenCV(Python) 이용
- 광원체 > 점자 인식률을 높이기 위해서는 점자에 균일한 빛이 가해져야 한다. 균일함을 최대로 높혀줄 수 있는 칩 형태의 광원 4개를 달아주기로 결정

### 점들의 집합을 숫자열로 변환 후 한글 번역

- 점자 한 칸을 구성하는 점을 1부터 6까지의 숫자로 지정하여 숫자열로 표현
- 파이썬 딕셔너리로 숫자열을 자음, 모음으로 변환
- 결합함수를 통해 단어로 구성
  <img src="/assets/img/post/2021-11-27-portable_braille_voice_translator/braile.png">

### 음성송출

- TTS 사용하여 한글을 음성으로 송출
- 스피커로 송출

## 프로젝트 결과

### 실제 장치

<img src="/assets/img/post/2021-11-27-portable_braille_voice_translator/diagram.jpg">
<img src="/assets/img/post/2021-11-27-portable_braille_voice_translator/result.jpg">
첫번째로 전원 스위치를 눌러 전원을 켠 후, 카메라 셔터 스위치를 누르면
 사진을 찍어 점자 인식 과정을 진행하게 됩니다.
 
### 시연 영상
<video width="50%" controls>
    <source src="/assets/img/post/2021-11-27-portable_braille_voice_translator/demo_video.mp4">
    Sorry, your browser doesn't support embedded videos.
</video>
