---
title: 아두이노 RC카 자동주차 시스템
author: Kim Seunghwan
date: 2021-11-26 09:30:00 +0900
categories: [Exhibition,2021년]
tags: [post,kimseunghwan]     # TAG names should always be lowercase, 띄어쓰기도 금지 
---

------------------------------------------
# 목표
아두이노를 이용해 무선으로 조종 가능한 RC카를 만들고 빈 공간을 감지하여 자동으로 주차하는 기능을 개발한다.

-----
# 작품설명
RC카는 스키드 조향 방식으로 움직이며, 블루투스 모듈을 이용해 스마트폰 앱으로 조종할 수 있다. 자동주차 기능은 스마트폰이나 RC카에 설치한 스위치를 통해 켤 수 있으며, 자동주차 기능이 작동하는 동안에 공간을 찾아 움직이다가 차체에 부착된 초음파 센서들을 이용해 주차를 할 수 있는 공간이 감지되면 주차를 하도록 만들었다.

-----
# 주요 부품, 도구
 - PC 및 스마트폰
 - 아두이노 UNO 보드
 - 센서확장 쉴드
 - usb케이블
 - 블루투스 모듈 HM-10
 - 차량 프레임, 바퀴
 - 초음파 센서
 - 브레드보드
 - tact 스위치
 - LED
 - DC모터 드라이버
 - 배터리 및 배터리 홀더
 - RC카 제어 어플
어플은 직접 만들지 않고 구글 플레이 스토어에 등록되어 있는 어플을 받아 사용했다.

-----
# 주요 코드
전체 코드 중, 주차와 관련된 핵심 코드는 아래와 같다.
```cpp
void aaa(){
  ck=ck+1;
}

void loop() {
  if((ck%2)==0){
    digitalWrite(LED,LOW);
    if(bluetooth.available()){      
    char Blue_Val = bluetooth.read(); 
    control_SmartCar(Blue_Val);

    if(mode == 0){
      motor_role(R_Motor, L_Motor);
    }
    else if(mode == 1){
      Right_role(R_Motor, L_Motor);
    }
    else if(mode == 2){
      Left_role(R_Motor, L_Motor);
    }
    else if(mode == 4){
      left_rotation(R_Motor, L_Motor);
    }
    else if(mode == 5){
      right_rotation(R_Motor, L_Motor);
    }
    else{
      analogWrite(RightMotor_E_pin, 0);
      analogWrite(LeftMotor_E_pin, 0);
    }   
  }
  }
  else{
    digitalWrite(LED,HIGH);
    long duration1, RS1;
    digitalWrite(trigPin1, HIGH); 
    digitalWrite(trigPin1, LOW);
    duration1 = pulseIn(echoPin1, HIGH);
    RS1 = ((float)(340 * duration1) / 10000) / 2; 

    if ((RS1>25)){
      stopp(R_Motor, L_Motor);

      int filteredValue1=0;
      long duration2, RS2;
      for (int i =0; i<30; i++){
        digitalWrite(trigPin2, HIGH); 
        digitalWrite(trigPin2, LOW);
        duration2 = pulseIn(echoPin2, HIGH);
        RS2 = ((float)(340 * duration2) / 10000) / 2; 
        if((RS2)>70){
          RS2=filteredValue1/(i+1);
        }
        Serial.println(RS2);
        filteredValue1 += RS2;
        delayMicroseconds(30);
      }
      filteredValue1 /=30;
      
      if ((filteredValue1)>20){
      stopp(R_Motor, L_Motor);
      delay(1000);
      left_rotation(R_Motor, L_Motor);
      delay(530);
      stopp(R_Motor, L_Motor);
      delay(1000);
      int filteredValue2;
      for (int i =0; i<10; i++){
      long duration3, RS3;
      digitalWrite(trigPin3, HIGH); 
      digitalWrite(trigPin3, LOW);
      duration3 = pulseIn(echoPin1, HIGH);
      RS3 = ((float)(340 * duration1) / 10000) / 2; 
      filteredValue2 += RS3;
      delayMicroseconds(100);
    }
    filteredValue2 /= 10;
      
      R_Motor = LOW; 
      L_Motor = LOW;
      motor_role(R_Motor, L_Motor);
      int t = filteredValue2*18;
      delay(t);
      stopp(R_Motor,L_Motor);
        ck=ck+1;
    }
    else{
      delay(150);
      R_Motor = HIGH; 
      L_Motor = HIGH;
      motor_role(R_Motor, L_Motor);
    }
    }
    else{
      delay(30);
      R_Motor = HIGH; 
      L_Motor = HIGH;
      motor_role(R_Motor, L_Motor);
    }
    }
}
```
-----
# 시연 영상
1. 블루투스 원격 제어
<video controls width="90%">
    <source src="/assets/img/post/2021-11-26-selfparking_car/1.mp4">
</video>

2. 어플을 통해 자동주차 기능을 작동
<video controls width="90%">
    <source src="/assets/img/post/2021-11-26-selfparking_car/2.mp4">
</video>

3. 빈 공간이 감지되어도 주차 공간으로 충분하지 않을 경우 다른 주차공간을 찾도록 설계함
<video controls width="90%">
    <source src="/assets/img/post/2021-11-26-selfparking_car/3.mp4">
</video>

4. 차체에 부착된 스위치를 통해 자동주차 기능을 작동
<video controls width="90%">
    <source src="/assets/img/post/2021-11-26-selfparking_car/4.mp4">
</video>

5. 종합 시연 영상
<video controls width="90%">
    <source src="/assets/img/post/2021-11-26-selfparking_car/5.mp4">
</video>

-------
# 문제점, 개선방향
프로젝트를 진행하면서 여러 문제점이 발생했는데 정리하면 아래와 같다.
- 경로 이탈 문제 : RC카가 오래 작동할 경우 한쪽으로 치우치며 움직이거나 원하는 경로에서 조금씩 벗어나는 문제점이 발생했다. 원인은 크게 2가지로 보았는데 사용한 RC카 프레임이 다소 견고한 편이 아니다 보니 모터의 위치가 조금씩 흔들렸고, 모터의 회전 수도 좌우에서 차이가 조금 났었다.
- 배터리 전압 문제 : 배터리의 전압이 약해질수록 모터의 회전 강도가 약해지다 보니 수 차례 테스트를 거치는 동안 RC카의 움직임이 조금씩 둔해지는 결과가 나왔다. 새 배터리로 갈아줌으로써 당장은 문제를 해결할 수 있었지만 장기적으로 봤을 때, 다른 방식으로 개선해야 할 필요성을 느꼈다.
- 센서의 정보 처리 속도 : 처음에는 여러 개의 센서 값을 한 번에 받으려고 했으나 너무 반응이 느려진 탓에 RC카의 반응이 너무 늦었다. 이러한 문제를 해결하기 위해 센서들을 순차적으로 작동시켜 한 번에 하나의 센서 값만 받는 방식으로 변경하였다.
- 이외에도 주변 환경에 의한 변수(바닥의 마찰 등), 초음파 센서의 오차 등이 있었다.

계획을 구현해내는 과정에서는 고려하지 못했던 이러한 현실적인 변수로 인해 제작에 어려움이 있었고, 원래 계획보다 축소하여 작품을 만들게 되었다. 원래는 더 넓은 공간에서 다양한 상황이 주어졌을 때 주차기능이 작동하도록 하고 싶었으나 위의 문제점들 때문에 주차기능 자체를 오래 유지할 수 없었다. 다음에 개선할 때는 돈이 들고 복잡하더라도 조금 더 견고하게 하드웨어를 구성하고, 엔코더를 이용하는 등의 방식으로 지금보다 더 정밀하게 RC카를 제어해보고 싶다. 또한 아직은 잘 다룰 줄 모르지만 라즈베리 파이도 배워서 수치계산적인 면을 보완하고 더 복잡한 상황에서도 잘 작동하도록 만들고 싶다.

