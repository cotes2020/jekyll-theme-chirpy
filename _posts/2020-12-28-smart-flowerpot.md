---
title: 온습도 측정하는 스마트 화분
author: JiWon Yoon, MinJeong Kim
date: 2020-12-28 15:08:00 +0900
categories: [Exhibition,2020년]
tags: [post,jiwonyoon,minjeongkim,arduino,blynk]     # TAG names should always be lowercase, 띄어쓰기도 금지
---

# 목표

아두이노를 활용하여 온도와 습도를 측정하고 식물 성장에 알맞은 온습도와 비교해 'Blynk'라는 스마트폰 앱으로 현재 식물의 상태를 알 수 있다

# 작품 설명

DHT11을 통해 식물 성장 환경의 온습도를 측정하고 ESP-01 무선랜 모듈을 사용하여 사용자에게 식물의 현재 상태를 Blynk를 통해 전달해주는 시스템이다. 이를 통해 사용자는 Blynk 앱에서 실시간 온습도를 그래프 형식의 정보로 얻을 수 있다.

# 필요한 부품

PC, 스마트폰, 아두이노 UNO 보드, ESP-01 무선랜 모듈, USB 케이블, 점퍼 연결선, DHT11, 공유기 등

# 사용 라이브러리

DHT11 라이브러리, ESP826 라이브러리, Blynk 라이브러리

# 사용 IOT 서비스 

Blynk : 스마트폰 앱을 통해서 쉽게 IoT기기(아두이노, 라즈베리파이, ESP8266 등) 컨트롤러를 구성하고 원격으로 제어할 수 있는 서비스

# 추가 부품 설명 

DHT11 : 온도측정 영역은 0~50℃ (오차범위는 2℃), 습도측정 영역은 20~80% (오차범위는 5%)이며 데이터 수집 속도는 1Hz이다.

ESP 8266 : 와이파이 통신기능을 담당하는 마이크로컨트롤러

---- 

<div class="row">
    <div style="width: 50%">
        <figcaption>회로 구상도</figcaption>
        <img src="/assets/img/post/2020-12-28-smart-flowerpot/img1.png">
    </div>
    <div style="width: 50%">
        <figcaption>실제 구상</figcaption>
        <img src="/assets/img/post/2020-12-28-smart-flowerpot/img2.png">
    </div>
</div>

# 구상한 코드

```cpp
#define BLYNK_PRINT Serial
#include <ESP8266_Lib.h>
#include <BlynkSimpleShieldEsp8266.h>
#include <SoftwareSerial.h>
#include <DHT.h>

// 이메일로 얻은 token값을 입력
char auth[] = "AUTH_TOKEN";

// 공유기의 이름과 비밀번호를 입력
char ssid[] = "YourNetworkName";
char pass[] = "YourPassword";

// Uno 보드에서 Software Serial로 연결된 ESP-01을 이용
SoftwareSerial EspSerial(2, 3); // RX, TX
ESP8266 wifi(&EspSerial);

// ESP8266 baud rate
#define ESP8266_BAUD 38400

// 보드에 연결된 DHT 핀 번호
#define DHTPIN 7

// 사용하는 DHT 타입
#define DHTTYPE DHT11     // DHT 11

DHT dht(DHTPIN, DHTTYPE);
BlynkTimer timer;

// 앱에서 위젯의 읽기 빈도를 PUSH로 설정해놓고 Virtual Pin(5)로 초당 Arduino의 가동 시간을 전송
// 즉, Blynk 앱에 데이터를 보내는 빈도를 의미
void sendSensor()
{
  float h = dht.readHumidity();
  float t = dht.readTemperature(); // or dht.readTemperature(true) for Fahrenheit

  if (isnan(h) || isnan(t)) {
    Serial.println("Failed to read from DHT sensor!");
    return;
  }
  Blynk.virtualWrite(V7, t);
  Blynk.virtualWrite(V8, h);
}

// Blynk 앱과의 연동을 시작, BlynkTimer 구동시작
// 1초 간격으로 DHT11의 데이터를 수집
void setup()
{
  // Debug console
  Serial.begin(9600);

  // Set ESP8266 baud rate
  EspSerial.begin(ESP8266_BAUD);
  delay(10);

  Blynk.begin(auth, wifi, ssid, pass);
  dht.begin();
  timer.setInterval(1000, sendSensor);
}

// Blynk, timer 구동
void loop()
{
  Blynk.run();
  timer.run();
}
```

# 배운점

Blynk와 ESP8266을 활용해 보드에서 수집한 정보를 앱에 구현해내기 위한 프로젝트를 진행하며, 비록 성공하지는 못하였지만 아두이노 보드와 여러 가지 센서를 통해 어떻게 아이디어를 구현해낼 수 있는지, 그리고 그 아이디어를 IOT서비스를 통해 어떻게 앱으로 구현해내는 것인지에 관해 공부해볼 수 있었습니다. DHT11센서를 선택하기까지의 과정 속에서 Water level sensor를 이용해 습도 낮음, 보통, 높음을 RGB 모듈의 빨강,파랑,초록색으로 나타내보기도 하였으나 정확한 습도의 상태를 스마트폰 앱에 전달하는 시스템을 구축하는 데에 어려움이 있었습니다. 

# Water level sensor를 사용했을 때의 코드
```cpp
int a=0;
int RED = 12;
int GREEN = 11;
int BLUE = 10;

void setup() {
 pinMode(RED, OUTPUT);
 pinMode(GREEN, OUTPUT);
 pinMode(BLUE, OUTPUT);
 Serial.begin(9600);
}

void loop() {
  a=analogRead(A0);
  if (a>=100){
    digitalWrite(GREEN,HIGH);
    digitalWrite(BLUE,LOW);
    digitalWrite(RED,LOW);
  }
  else if(50<=a&&a<100){
    digitalWrite(BLUE,HIGH);
    digitalWrite(GREEN,LOW);
    digitalWrite(RED,LOW);
  }
  else {
    digitalWrite(RED,HIGH);
    digitalWrite(GREEN,LOW);
    digitalWrite(BLUE,LOW);
  }
  Serial.println(a);
  delay(600);
}
```

# RGB모듈과 수분센서 시연 영상 링크

https://www.youtube.com/watch?v=t-InZaT9B-g

이에 KOCW의 IOT 서비스에 관한 참고자료 (http://kocw.xcache.kinxcdn.com/KOCW/edu/document/cuk/wiesunghong1206/12.pdf) 의 예제에 나와있는대로, DHT11 습도 센서와 Blynk 서비스를 이용해 현재의 결과를 만들어 낼 수 있었습니다.

<div style="width: 50%; margin: 0 auto;"> 
    <img src="/assets/img/post/2020-12-28-smart-flowerpot/img3.png">
</div>

한편 위에 첨부한 사진처럼 시리얼 모니터에서 ‘ESP is not responding’이라는 결과가 출력되어 ESP8266 센서와 관련된 문제가 있음을 확인할 수 있었으나 정확한 원인을 찾지는 못하였습니다. 때문에 Blynk 앱과 연결에 성공하지는 못하였으나 실제로 아두이노 보드와 IOT서비스를 활용해 본 것은 유익한 경험이었습니다.