---
title: 탈부착형 분실물 찾기 시스템
author: 29기 노우성, 29기 문상인
date: 2021-11-26 22:00:06 +0900
categories: [Exhibition,2021년]
tags: [post,sangin]     # TAG names should always be lowercase, 띄어쓰기도 금지
---

------------------------------------------


## 작품 소개
애플의 ‘나의 찾기’ 기능은 gps기능과 해당 장소에서는 알람 기능을 실행해준다는 면에서 편리하다.
하지만, ‘나의 찾기’ 기능은 애플기기에만 한정되고 gps 장치가 없는 귀중품에서는 구현될 수 없다는 단점이 있다.
이러한 문제를 해결하기 위해 아두이노의 gps, 피에조 부저를 이용하여 ‘나의 찾기’ 기능 탈부착형 gps 장치를 설계해보고자 한다.


## 작품 설명
아두이노를 활용하는 작품이다. Neo-6M을 통해 물체의 이동 경로를추적한다.
사용자는 물체를 찾기 위해 gps가 가리키는 위치로 이동할 것이다. 
물체에 접근했을 때 찾는 물체가 보이지 않으면 휴대폰 앱을 통해 'on' 버튼을 누르면 된다.  
이 신호는 HC-05를 통해 아두이노 시리얼 모니터에 입력된다. 
'on'을 입력 받으면 피에조 부저가 울리면서 물체의 위치를 소리를 통해 자세히 확인할 수 있다. 물체를 찾았으면 'off'버튼을 눌러 소리를 끄면 된다. 
'off'버튼을 누르기 전까지 소리는 꺼지지 않는다.


## 부품
HC-05(블루투스), NEO-6M(gps), 아두이노 핀헤더(1x4) 2.54mm(NEO-6M 납땜을 위해), 피에조 부저, NPN 트랜지스터, 점퍼케이블, 아두이노 MEGA 보드, PC, USB 케이블, 고정저항 1K옴


## 기타
NEO-6M을 통해 gps 결과값을 확인할려고 했으나, 첫 구매로 온 부품이 작동되지 않아 다시 제품을 구매하기로 했다.
두 번째로 구매한 부품은 납땜도 한 번에 하고 제대로 작동하는지 현장에서 확인하기 위해 세운전자상가에 직접 방문하여 구매하였다. 
이 경우 부품은 제대로 작동하였지만, 신호 송신만 되고 gps 결과값을 수신받지 못했다. 부품과 코딩 문제 때문에 시간이 많이 지체되었고, 제출 기간이 얼마 남지 않아 
더 이상의 시간 투자가 힘들었다. 따라서 NEO-6M을 이용한 프로젝트를 다루는 블로그를 찾아 제대로 작동했을 떄의 결과값이 어떻게 생기는지 눈으로 확인했다.
또한 본래 작품설계는 지갑에 붙일 수 있을 정도의 작은 크기로 제작하려고 하였으나, 아두이노를 통해 설계하면서 한계를 느꼈고,  미소소자에 대한 지식을 쌓은 후 소형화 작업을 수행하려한다.


## 코드

#include <SoftwareSerial.h> 
#include "PiezoSpeaker.h"
#include <TinyGPS.h> 

int PIEZOSPEAKER_5V_PIN_SIG=5; 

int RXPIN=6;
int TXPIN=5; 

int blueTx=10;    
int blueRx=11;    
SoftwareSerial mySerial(blueTx, blueRx);  
String myString="";
unsigned int piezoSpeaker_5vHoorayLength          = 6;                                                      
unsigned int piezoSpeaker_5vHoorayMelody[]        = {NOTE_C4, NOTE_E4, NOTE_G4, NOTE_C5, NOTE_G4, NOTE_C5}; 
unsigned int piezoSpeaker_5vHoorayNoteDurations[] = {8      , 8      , 8      , 4      , 8      , 4      };
PiezoSpeaker piezoSpeaker_5v(PIEZOSPEAKER_5V_PIN_SIG);

#define GPSBAUD 9600
TinyGPS gps;
SoftwareSerial uart_gps(RXPIN, TXPIN);
void getgps(TinyGPS &gps);

void setup() 
{
  Serial.begin(9600);  
  mySerial.begin(9600);
  uart_gps.begin(GPSBAUD);
  Serial.println("기기의 위치를 확인 중입니다");
}
void loop()
{
  while(uart_gps.available())     
  {
      int c = uart_gps.read();    
      if(gps.encode(c))      
      {
        getgps(gps);         
      }   
  }


  while(mySerial.available())
  {
    
    char myChar = (char)mySerial.read();
    myString+=myChar;
    delay(1000);
  }
if(!myString.equals(""))
  {
    Serial.println("input value :"+myString);
    if(myString=="on")  
      {
        piezoSpeaker_5v.playMelody(piezoSpeaker_5vHoorayLength, piezoSpeaker_5vHoorayMelody, piezoSpeaker_5vHoorayNoteDurations); 
    delay(500); 
    
      }
      
    if(myString=="onoff")
    {
      
      Serial.println("기기를 찾았습니다!!");
      myString=""; 
      delay(5);
      
      
    }
  }
}

void getgps(TinyGPS &gps) 
{
  gps.f_get_position(&latitude, &longitude);
  Serial.print("Lat/Long: "); 
  Serial.print(latitude,5); 
  Serial.print(", "); 
  Serial.println(longitude,5);
  
  int year;
  byte month, day, hour, minute, second, hundredths;
  gps.crack_datetime(&year,&month,&day,&hour,&minute,&second,&hundredths);
  Serial.print("Date: "); Serial.print(month, DEC); Serial.print("/"); 
  Serial.print(day, DEC); Serial.print("/"); Serial.print(year);
  Serial.print("  Time: "); Serial.print(hour, DEC); Serial.print(":"); 
  Serial.print(minute, DEC); Serial.print(":"); Serial.print(second, DEC); 
  Serial.print("."); Serial.println(hundredths, DEC); 
  Serial.print("Altitude (meters): "); Serial.println(gps.f_altitude());  
  Serial.print("Course (degrees): "); Serial.println(gps.f_course()); 
  Serial.print("Speed(kmph): "); Serial.println(gps.f_speed_kmph());
  Serial.println();
 
  unsigned long chars;
  unsigned short sentences, failed_checksum;
  gps.stats(&chars, &sentences, &failed_checksum);
  delay(10000);
}

## 작품 사진 및 영상

아두이노 회로 완성 사진
<img src="/assets/img/post/2021-11-26-Detachable-lost-and-found-system/circuit.JPG" width="90%">

NEO-6M을 통해 gps 결과값 수신하는 사진
<img src="/assets/img/post/2021-11-26-Detachable-lost-and-found-system/gps.JPG" width="90%">

on/off를 통해 피에조 부저 조작 영상
<source src="/assets/img/post/2021-11-26-Detachable-lost-and-found-system/HC-05,PiezoBuzzer.mp4"width="90%">



  
