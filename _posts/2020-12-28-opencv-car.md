---
title:  OpenCV 기반의 자율주행 모형차
author: Park Juntaek
date: 2020-12-28 14:22:00 +0900
categories: [Exhibition,2020년]
tags: [post,parkjuntaek]     # TAG names should always be lowercase, 띄어쓰기도 금지
---

작성자: 박준택

Pi Cam을 이용하여 트랙을 촬영하고 Jetson Nano에서 영상처리를 진행한다. 영상처리는 다음의 과정을 거친다

1. 차선이 아닌 불필요한 영역을 제거한다.
2. Sobel Operator를 이용하여 Binary Image를 얻는다.
3. Perspective Transform을 통해 차선을 위에서 본 것처럼 변환한다.
4. Sliding Window Algorithm을 적용하여 차선이 있는 것으로 예상되는 좌표들을 얻는다.
5. 얻은 좌표를 기반으로 차선의 기울기와 현재 차량이 중심으로부터 얼마나 떨어져 있는지 Offset을 구한다.

영상처리 이후에는 Jetson Nano에서 구한 차선의 기울기와 Offset을 Arduino Nano에 전송(UART)하고, Arduino는 수신한 값에 따라 서보모터를 조작하여 방향을 결정한다. 다음은 전체적인 개요에 대한 그림과 Sliding Window Algorithm의 진행과정이다.

----
<figure>
    <figcaption>전체 개요</figcaption>
    <img src="/assets/img/post/2020-12-28-opencv-car/img1.png">
</figure>
----
<figure>
    <figcaption>Sliding Window – Step1</figcaption>
    <img src="/assets/img/post/2020-12-28-opencv-car/img2.png">
</figure>
----
<figure>
    <figcaption>Sliding Window – Step2</figcaption>
    <img src="/assets/img/post/2020-12-28-opencv-car/img3.png">
</figure>
----
<figure>
    <figcaption>Sliding Window – Step3</figcaption>
    <img src="/assets/img/post/2020-12-28-opencv-car/img4.png">
</figure>
----


모형차의 물리적 한계(회전반경이 큼)에 의해 직각 커브는 한 번에 돌지 못하는 문제가 있었는데 모형차가 차선을 벗어나더라도 다시 차선을 찾아 들어오게끔 현재 인식하고 있는 차선의 위치(좌/우)와 과거에 인식했던 차선의 위치를 저장하고, 둘을 비교하여 합당한 판단을 내리게끔 코드를 추가하여 결과적으로 주어진 트랙에 대해서 시계/반시계 방향으로의 주행을 완료했다.


<video controls>

    <source src="/assets/img/post/2020-12-28-opencv-car/1.mp4">
    Sorry, your browser doesn't support embedded videos.
</video>

<video controls>

    <source src="/assets/img/post/2020-12-28-opencv-car/2.mp4">
    Sorry, your browser doesn't support embedded videos.
</video>
