@@ -0,0 +1,50 @@
---
title: Create Affordable-Covid19-Detecting-System
author: WooYul Jung, Dongkun Ahn, Jeongeun Kim, Jinseo Heo
date: 2020-12-28 17:00:00 +0900
categories: [Exhibition,2020년]
tags: [post,jungwooyul,ahndongkun,kimjeongeun,heojinseo,opencv,flirlepton] 
---

------------------------------------------
# 작품 소개

최근 COVID-19의 확산으로 전세계적인 피해가 증가하고 있으며 특히 열악한 환경의 아프리카에서 그 피해가 극심하다. 이러한 가운데 시중에서 판매되는 COVID-19 탐지 원거리 체온계는 300만원을 호가하는 등 아프리카에서 감당하기 힘든 가격을 보인다. 따라서 아프리카에서도 COVID-19 감염확산을 늦추기 위해 적정기술의 일환으로 합리적인 가격으로 COVID-19을 조기에 감지하는 출입시스템을 개발하고자 한다.

# 작품 제작 순서

1) FLIR LEPTON 2.5 적외선 센서를 purethermal2 보드에 연결한다.
2) 파이썬 Lepton-SDK를 이용하여 80x60 pixel의  온도데이터를 구한다. 
2) 구한 온도데이터를 numpy 형태로 변환후 이미지화한다.
3) 이미지에 haar cascade를 이용하여 people-finding 기능을 구현한다 
4) 사람이 있다고 판단되면 온도를 5번 측정하여 평균값을 아두이노로 전달한다.
5) 3D프린터로 출력한 출입문 부품을 조립하여 출입문을 완성하고 모터를 부착한다.
6) 모터를 아두이노와 연결하고 아두이노에서 온도가 섭씨 37.5도 미만일 때 모터를 90도 회전하여 출입문을 가동한다.

# 필요한 부품

FLIR LEPTON 2.5 적외선 센서, purethermal2 보드, 아두이노 우노
<div class="row">
    <div style="width: 50%">
        <figcaption>얼굴 인식하는 모습</figcaption>
        <img src="/assets/img/post/2020-12-28-Create Affordable-Covid19-Detecting-System/img1.png">
    </div>
    <div style="width: 50%">
        <figcaption>3D프린터로 만든 문</figcaption>
        <img src="/assets/img/post/2020-12-28-Create Affordable-Covid19-Detecting-System/img2.png">
    </div>
</div>

# 사용 라이브러리

Lepton-SDK_PureThermal_Windows10_1.0.2
matplotlib.py, numpy.py, serial.py, cv2.py

# 개선 가능성

calibration 작업을 통해 FLIR LEPTON 2.5 적외선센서가 측정하는 체온의 표면온도의 오차를 줄일 수 있을 것이다. 
일반카메라 부품을 추가하고 opencv를 이용해 더 정확한 people-finding 기능을 구현할 수 있을 것이다.
마스크 쓴 사람이 인식이 잘 안된다.  

# 시연 링크

https://www.youtube.com/watch?v=t-InZaT9B-g

