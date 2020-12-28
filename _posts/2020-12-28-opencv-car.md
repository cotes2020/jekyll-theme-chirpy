---
title:  OpenCV 기반의 자율주행 모형차
author: Park Juntaek
date: 2020-12-28 14:22:00 +0900
categories: [Exhibition,2020년]
tags: [post,parkjuntaek]     # TAG names should always be lowercase, 띄어쓰기도 금지
---


이 게시글은 작성중입니다.

지금 비디오 재생 문제가 있는듯해서 알아보고 있습니다.

작성자: 박준택

Pi Cam을 이용하여 트랙을 촬영하고 Jetson Nano에서 영상처리를 진행한다. 영상처리는 다음의 과정을 거친다
차선이 아닌 불필요한 영역을 제거한다.

1. Sobel Operator를 이용하여 Binary Image를 얻는다.
2. Perspective Transform을 통해 차선을 위에서 본 것처럼 변환한다.
3. Sliding Window Algorithm을 적용하여 차선이 있는 것으로 예상되는 좌표들을 얻는다.
4. 얻은 좌표를 기반으로 차선의 기울기와 현재 차량이 중심으로부터 얼마나 떨어져 있는지 Offset을 구한다.

영상처리 이후에는 Jetson Nano에서 구한 차선의 기울기와 Offset을 Arduino Nano에 전송(UART)하고, Arduino는 수신한 값에 따라 서보모터를 조작하여 방향을 결정한다. 다음은 전체적인 개요에 대한 그림과 Sliding Window Algorithm의 진행과정이다.


<video controls>

    <source src="/assets/img/post/2020-12-28-opencv-car/1.mp4">
    Sorry, your browser doesn't support embedded videos.
</video>

<video controls>

    <source src="/assets/img/post/2020-12-28-opencv-car/2.mp4">
    Sorry, your browser doesn't support embedded videos.
</video>
