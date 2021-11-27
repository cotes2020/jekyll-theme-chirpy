---
title: Thief Alarm
author:  Shin Minki
date: 2021-11-27 11:10:00 +0900
categories: [Exhibition,2021년]
tags: [post,minki] 
---

------------------------------------------

## 도둑질 방지 알람

### 작품 설명
아두이노 압력 센서와 LED를 이용한다. 원래는 책이나 패드 같은 물건으로 실험할려 했으나 제대로 작동이 되지 않아서 손으로 성능 체크하기로 했다. 
손으로 힘을 주었을때는 불이 들어오지 않았지만, 손을 놓았을때는 불이 들어왔다.

### 사용 부품 
아두이노 우노 , usb 케이블, 압력센서 FSR406, 점퍼 연결선

### 코딩 내용

```cpp


const int pressSensor = A1;
const int led = 9;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  pinMode(led,OUTPUT);
}

void loop() {
  // put your main code here, to run repeatedly:
  int value = analogRead(pressSensor);
  Serial.println(value);

  if(value <= 500){
    digitalWrite(led,HIGH);  
  }
  else{
    digitalWrite(led,LOW);  
  }
  
}

```
<img src="/assets/img/post/2021-11-27-thief-alarm/clip20211127_2350_43_792.png" width="50%"> 
<img src="/assets/img/post/2021-11-27-thief-alarm/clip20211127_2350_50_495.png" width="50%">

### 추가
 물건들 무게가 다 다르다보니 알람 기준을 손으로 힘조절하는 거밖에 답이 없었다. 회로를 구상하면서 아두이노에 대해 알 수 있게 되었고, 
아두이노를 어떻게 구현하는지 알 수 있게 되었다. 다음에는 원래 목표인 G메일과 연동해서 휴대폰으로 알 수 있게 하는 기능을 구현해보고 싶다. 


