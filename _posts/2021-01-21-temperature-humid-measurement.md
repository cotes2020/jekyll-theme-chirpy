---
title: 온습도 관리 프로그램
author: DongSeonKim
date: 2021-01-22 13:04:00 +0900
categories: [Exhibition,2020년]
tags: [post,dongseon]     # TAG names should always be lowercase, 띄어쓰기도 금지 
---

------------------------------------------

## 온 • 습도 탐지 및 관리시스템

### 작품 소개

작년 유난히 길었던 장마기간, 한여름에는 궂은 비에 집이 습도로 가득차기 마련이다. 방에 습기가 가득 차 있으면 곰팡이가 생기거나 벌레가 생기기 십상이기에 항상 관리를 해주어야 한다. 이를 위해 온도 및 습도를 측정하고 이를 인지하기 쉽게 LCD로 측정해볼 것이다. 더 나아가 데이터를 데스크톱으로 전송하여 데이터를 다뤄볼 예정이다.

 

### 작품 설명

아두이노를 활용하는 작품이다. DHT11을 이용하여 온습도 측정 및 데이터를 수집한다. 이를 LCD를 통해 표시하는 시스템이다. 마지막으로 전송된 데이터를 기반으로 활용할 수 있다.

 

### 부품

PC, 아두이노 UNO 보드, USB 케이블, DHT11, LCD

 

### 사용 라이브러리

DHT11 라이브러리, LCD 라이브러리(hd44780)

 

### 부품 설명

온습도 센서 모듈 (DHT11)

- 1개의 데이터 라인을 통해 온습도 값을 전송해주는 기본 센서
- 데이터 전송 포맷
- [8bit integral RH data] + [8bit decimal RH data] + [8bit integral T data] + [8bit decimal T data] + [8bit check sum]
- 온도 측정 범위 : 0~50' (오차범위 2')
- 습도 측정 범위 : 20~90% (오차범위 5%)



### 코딩 내용

```
#include <Wire.h>  

#include <hd44780.h>  

#include <hd44780ioClass/hd44780_I2Cexp.h>  

#include <DHT.h>  #include <DHT_U.h>     

#define DHTPIN 7  

#define DHTTYPE DHT11     

hd44780_I2Cexp lcd;  

DHT dht(DHTPIN, DHTTYPE);     

void setup() {   

Serial.begin(9600);   

lcd.begin(16, 2);   

dht.begin();  

}     

void loop() {   

float humi, temp;   

temp = dht.readTemperature();   

humi = dht.readHumidity();   

if(isnan(humi) || isnan(temp)){     

Serial.println("Failed to read from DHT sensor");     

return;   

}   

lcd.clear();   

lcd.setCursor (0, 0);   

lcd.print("Temp: ");  

lcd.print(temp);   

lcd.setCursor (0, 1);   

lcd.print("Humi: ");   

lcd.print(humi);   

delay(300);  

}  
```

 

### 실제 구상 사진

<img src="/assets/img/post/2021-01-21-temperature-humid-measurement/pic1.PNG" width="90%"> 
<img src="/assets/img/post/2021-01-21-temperature-humid-measurement/pic2.PNG" width="90%">

 

### 배운 점

시간적 한계에 의해서 온습도계를 이용하여 데이터를 수집하고 이를 LCD에 표시하는 것까지 진행해보았습니다. 회로를 구상해보면서 단순하지만 여러 센서를 병합하여 작품을 만드는 것을 경험해보았고, 어떻게 작동하는지 공부하고 배워볼 수 있었습니다. 이를 활용하여 더 복잡하거나 활용도 높은 작품들을 다음에 구상해보고 싶습니다.

 