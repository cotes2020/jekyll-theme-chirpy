---
title: Beacon을 이용한 위치추적과 자율주행 자동차
author: CHOI Hyunseo
date: 2021-11-28 18:00:00 +0900
categories: [Exhibition,2021년]
tags: [post,choihyunseo,beacon]     # TAG names should always be lowercase, 띄어쓰기도 금지 
---

------------------------------------------
# Beacon을 이용한 위치추적과 자율주행 자동차 - 쓰는중

## 개요 
**아두이노를 이용하여 자율주행 자동차와 컨트롤러를 제작하고, beacon으로 위치추적을 하였습니다.**
* Arduino UNO보드와 L298P보드, 초음파 센서를 이용하여 Level3 자율주행 자동차 제작
* Arduino UNO보드와 조이스틱을 이용하여 조이스틱 컨트롤러 제작, I2C LCD로 자동차 상태 표시
* HC-11을 이용하여 자동차와 컨트롤러 간의 RF통신 구현
* CLE-310 3개로 비콘을 구현하고 CLE-310 1개로 비콘들의 RSSI값을 SCAN하여 자동차의 위치를 추적

## 작품 제작

### 자율주행 자동차 제작

<img src="/assets/img/post/2021-11-28-LocationTracking-Beacon_AutonomousVehicle/vehicle1.jpg" width="90%"> <br>
**Arduino UNO보드와 L298P보드, 초음파 센서를 이용하여 Level3 자율주행 자동차 제작**
- 3.7 V, 2600 mAh인 18650 리튬 이온 전지 2개를 이용하여 7.4 V의 전원을 공급해준다.
- 하비 기어모터를 좌우에 각각 2개씩 연결한다.
- 스키드 조향 방식으로 방향 제어를 하며 제자리 선회도 가능하다.
- 초음파 센서를 이용해 거리 측정을 한다.
- 초록 LED로 on/off 상태를 나타낸다.
- HC-11을 이용하여 조이스틱 컨트롤러와 RF통신을 하여 조종할 수 있게 한다.
- CLE-310로 비콘들의 위치를 추적한다.

#### 소스코드
https://github.com/choi92/LocationTracking-Beacon_AutonomousVehicle/blob/main/barami21_vehicle_Beacon.ino]

### 조이스틱 컨트롤러 제작


