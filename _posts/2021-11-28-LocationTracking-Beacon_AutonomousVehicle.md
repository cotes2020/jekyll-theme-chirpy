---
title: Beacon을 이용한 위치추적과 자율주행 자동차
author: CHOI Hyunseo
date: 2021-11-28 18:00:00 +0900
categories: [Exhibition,2021년]
tags: [post,choihyunseo,beacon]     # TAG names should always be lowercase, 띄어쓰기도 금지 
---

------------------------------------------
# Beacon을 이용한 위치추적과 자율주행 자동차 

## 개요 
**아두이노를 이용하여 자율주행 자동차와 컨트롤러를 제작하고, beacon으로 위치추적을 하였습니다.**
* Arduino UNO보드와 L298P보드, 초음파 센서를 이용하여 Level3 자율주행 자동차 제작
* Arduino UNO보드와 조이스틱을 이용하여 조이스틱 컨트롤러 제작, I2C LCD로 자동차 상태 표시
* HC-11을 이용하여 자동차와 컨트롤러 간의 RF통신 구현
* BOT CLE310 3개로 beacon을 구현하고 HOST CLE310 1개로 beacon들의 RSSI값을 SCAN하여 자동차의 위치를 추적

### 개발 필요성 및 목적
4차 산업혁명 시대에 드론을 통한 무인 배달이 각광받고 있으나 드론으로는 대량의 물량을 수송할 수 없으며 드론이 가지 못하는 환경도 분명 존재한다. 그래서 대량의 물량을 수송할 수 있는 자율주행 자동차로 배달을 하는 시스템을 구현하고자 한다. <br>

보통 위치는 GPS로 추적하는 것이 일반적이지만 GPS는 오차 범위가 크므로 실내의 작은 공간에도 적용이 가능한 beacon을 이용하여 위치 추적을 하고자 한다.<br>

그리고 자율 배달 과정에서의 여러 변수들을 통제하기 위해 장애물 회피, 사람이 직접 조작이 가능한 시스템을 추가하고자 한다.<br>

---

## 작품 제작

### 자율주행 자동차 제작
<img src="/assets/img/post/2021-11-28-LocationTracking-Beacon_AutonomousVehicle/vehicle1.jpg" width="90%"> <br>
**Arduino UNO보드와 L298P보드, 초음파 센서를 이용하여 Level3 자율주행 자동차 제작** <br>
선행 연구인 <a href="https://github.com/choi92/Photovoltaic_Autonomous_Vehicle" target="_blank">"태양광 자율주행 자동차"</a> 의 자동차를 활용
- 3.7 V, 2600 mAh인 18650 리튬 이온 전지 2개를 이용하여 7.4 V의 전원을 공급해준다.
- 하비 기어모터를 좌우에 각각 2개씩 연결한다.
- L298P보드로 DC모터의 전류 방향을 바꿔서 회전 방향을 바꾼다.
- 스키드 조향 방식으로 방향 제어를 하며 제자리 선회도 가능하다.
- 초음파 센서를 이용해 거리 측정을 하여 전방 10cm 이내에 장애물이 있을 시 왼쪽으로 제자리 선회하여 장애물을 회피한다.
- 초록 LED로 on/off 상태를 나타낸다.
- HC-11을 이용하여 조이스틱 컨트롤러와 RF통신을 하여 조종할 수 있게 하고 위치를 보낸다.
- RF통신의 속도가 너무 빠르면 신호가 겹치므로 0.5초 간격으로 통신한다.
- HOST CLE310의 GPI를 HIGH로 해서 AT커맨드가 작동하도록 한다.
- HOST CLE310로 beacon들의 RSSI 값을 SCAN하여 parsing 후 자동차의 위치를 추적한다.

#### 소스 코드
<a href="https://github.com/choi92/LocationTracking-Beacon_AutonomousVehicle/blob/main/barami21_vehicle_Beacon.ino" target="_blank">자율주행 자동차 소스 코드 깃허브</a>

### 조이스틱 컨트롤러 제작
<img src="/assets/img/post/2021-11-28-LocationTracking-Beacon_AutonomousVehicle/joycon.jpg" width="90%"> <br>
**Arduino UNO보드와 조이스틱을 이용하여 조이스틱 컨트롤러 제작, I2C LCD로 자동차 상태 표시**
- 조이스틱의 신호를 I2C LCD에 표시하고 HC-11을 이용하여 자동차와 RF통신을 하여 조종할 수 있게 한다.
- HC-11을 이용하여 자동차와 RF통신을 하여 자동차의 위치를 받는다.
- RF통신의 속도가 너무 빠르면 신호가 겹치므로 0.5초 간격으로 통신한다.
- I2C LCD 첫 줄에는 자동차의 방향과 속도가 나오며 둘째 줄에는 자동차의 위치가 나온다.
- 버튼을 누르면 자동차가 위치추적을 하게 한다.

#### 알고리즘
<img src="/assets/img/post/2021-11-28-LocationTracking-Beacon_AutonomousVehicle/joycon2.png" width="90%"> <br>
- 먼저 좌표평면처럼 x값이 오른쪽으로 갈수록 커지고, y값이 위쪽으로 갈수록 커지게 한다.
- 0V ~ 5 V 사이의 값이 아날로그 형태로 0 ~ 1023으로 나타내어진다.
- x,y값이 470 ~ 550인 상태는 움직이지 않은 상태로 간주하고 정지 명령을 내린다.
- 속력에서 아날로그 값의 크기가 0 ~ 470정도인데 모터는 0 ~ 255의 PWM값을 조절할 수 있으므로 map함수로 변환시켜준다.
- 방향은 사진의 영역처럼 8방향으로 조종가능하게 한다.
- left와 right는 제자리 선회 방식으로 회전한다.
- forwardleft, forwardright, backwardleft, backwardright는 모터의 속도를 다르게하여 일반적인 방향전환을 한다. (50%->25%로 수정)

#### 3D 프린팅
<img src="/assets/img/post/2021-11-28-LocationTracking-Beacon_AutonomousVehicle/3d.jpg" width="90%"> <br>
<img src="/assets/img/post/2021-11-28-LocationTracking-Beacon_AutonomousVehicle/3d2.jpg" width="90%"> <br>
3D 모델링을 한 후 프린팅을 하여 컨트롤러의 케이스를 제작하려 했으나 프린터의 문제로 프린팅 실패


#### 소스 코드
<a href="https://github.com/choi92/LocationTracking-Beacon_AutonomousVehicle/blob/main/barami21_Joycontroller_Beacon.ino" target="_blank">조이스틱 컨트롤러 소스 코드 깃허브</a>

### Beacon 제작
<img src="/assets/img/post/2021-11-28-LocationTracking-Beacon_AutonomousVehicle/beacons.jpg" width="90%"> <br>
**BOT CLE310 3개로 beacon을 구현**
- CLE310 작동 전압에 맞게 1.5 V 전지 2개를 직렬 연결하여 3 V의 전원을 공급해준다.
- HOST CLE310이 SCAN할 수 있게 BOT으로 모드를 변경하고 GPI를 LOW로 한다.
- beacon들이 헷갈리지 않도록 pink, orange, yellow로 구분한다.

---

## 위치추적 알고리즘 설계
### Beacon으로 실험
<img src="/assets/img/post/2021-11-28-LocationTracking-Beacon_AutonomousVehicle/beacon1.jpg" width="90%"> <br>
HOST에 들어오는 RSSI 값을 보면서 beacon들의 위치를 계속 바꾸고, RSSI 값들을 기록

#### 실험 결과 
- 0.20 m 전후로 RSSI 값 -48 ~ -69
- 0.85 m 전후로 RSSI 값 -75 ~ -87 
- 5.15 m 전후로 RSSI 값 -87 ~ -93
- 다른 거리에서는 RSSI 값이 유동적으로 변하고 저 거리에서만 명확하게 구분 가능하다.
- 따라서 RSSI 값이 -90일 때를 기준으로 5.15 m 안에 있는지 밖에 있는지 판단한다.

### 구역 설정
<img src="/assets/img/post/2021-11-28-LocationTracking-Beacon_AutonomousVehicle/location3.jpg" width="90%"> <br>
- beacon 3개의 RSSI 값을 이용하여 반지름이 5.15 m인 원 3개 설정한다.
- 원 3개의 벤 다이어그램을 설정하여 구역을 7개로 나눈다.
- 벤 다이어그램 외부 구역은 U이다.
- RSSI 값으로 자동차가 어느 구역에 있는지 판단한다.

### 알고리즘 설계
- AT+SCAN 커맨드로 근처 5개의 장치의 RSSI 값을 선착순으로 수신한다.
- pink, orange, yellow의 어드레스가 각각 [74:F0:7D:1A:02:C8], [74:F0:7D:1A:02:A5], [74:F0:7D:1A:02:E8] 이므로 이를 이용하여 구분한다.
- beacon들의 어드레스를 이용하여 RSSI 값들을 각각 parsing한다. (자세한 소스 코드는 자동차 소스 코드 참조)
- RSSI 값이 -90일 때를 기준으로 반지름이 5.15 m인 원 안에 있는지 밖에 있는지 판단한다.
- 어느 원 안에 있는지를 판단하여 어느 구역에 있는지 벤 다이어그램에서 판단한다.

---

## 작품 구동

### 조이스틱 컨트롤러
#### 구동 모습
<a href="https://youtu.be/yJEVJwnj7W8" target="_blank">조이스틱 컨트롤러 구동 모습 유튜브</a>

### 자율주행 자동차
#### 주행 모습
<a href="https://youtu.be/aC3_Xr2P1Lc" target="_blank">자율주행 자동차 주행 모습 유튜브</a>

원래 조이스틱 컨트롤러로부터 속력도 수신하여 속력을 조절할 수 있게 하려 했으나 RF통신 특성상 여러 개의 신호를 수신하려면 통신 속도가 길어져 자동차에 적합하지 않다 판단하여 방향 조절만 가능

#### 장애물 회피
<a href="https://youtu.be/NmiGZ6Cq1X4" target="_blank">자율주행 자동차 장애물 회피 유튜브</a>

### Beacon을 이용한 위치 추적
<img src="/assets/img/post/2021-11-28-LocationTracking-Beacon_AutonomousVehicle/location22.png" width="90%"> <br>
넓은 공터에서 beacon들을 배치하고 자동차를 이동시켜 알고리즘대로 위치추적 <br>

<img src="/assets/img/post/2021-11-28-LocationTracking-Beacon_AutonomousVehicle/location.jpg" width="90%"> <br>
자동차의 위치추적 성공

---

## 의의
**RF통신 경험** - 다중 시리얼 입출력이랑 직접 통신 프로토콜 짜는 경험을 해봄으로서 통신을 배운다.

**위치추적 방안 배움** - ToA, TDoA, AoA, RSS 등의 위치추적 방안을 알아보고 각각의 장단점과 정확도를 배운다.

**Beacon의 정확도 배움** - Beacon의 정확도가 어느 정도인지 알 수 있게 되었고 추후 Beacon을 활용할 때 유용하게 이용할 것이다.

RF통신과 beacon을 새롭게 경험해보고 여러 알고리즘을 고안해보면서 전반적으로 배울 점이 많았다. <br>
그리고 자율주행 자동차를 조이스틱 컨트롤러로 성공적으로 조종하고 beacon을 통한 위치추적도 성공해서 전반적으로 만족스러운 결과가 나왔다.

---

## 개선할 사항
CLE310을 이용한 beacon의 정확도가 떨어져서 처음에 목표했던 삼각측량법을 이용한 정밀한 위치추적이 불가능하게 되었다. <br> 따라서 향후 정확도를 높여 삼각측량법이 가능하게 하여 정밀한 위치추적을 할 수 있도록 노력해야한다. <br>

원래 조이스틱 컨트롤러로부터 속력도 수신하여 자동차의 속력을 조절할 수 있게 하려 했으나 RF통신 특성상 여러 개의 신호를 수신하려면 통신 속도가 길어져 자동차에 적합하지 않다 판단하여 방향 조절만 가능하게 되었다.  <br> 따라서 향후 하나의 신호에 자동차의 방향과 속력을 담아 송신할 수 있는 알고리즘을 고안해야한다. <br>

시간상 하지 못했던 3D 프린팅을 통한 컨트롤러의 케이스 제작과 무선 충전 알고리즘 구현도 추후 성공할 수 있도록 노력해야한다.  
