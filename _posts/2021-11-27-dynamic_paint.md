---
title: Dynamic Paint
author: Kwangsoo Seol, Kangseob Seo
date: 2021-11-27 00:10:00 +0900
categories: [Exhibition,2021년]
tags: [post,seolkwangsoo,seokangseob]     # TAG names should always be lowercase, 띄어쓰기도 금지 
---

------------------------------------------
# Dynamic Paint

### 작품 개요

RGB-D camera를 활용해서 주어진 거리 안에 있는 손가락을 추적해 따로 구현한 그림판에 그림을 그립니다.   
손가락이 움직이는 속도를 사용해 필압(선의 굵기)을 구현했습니다.

---
### 작품 구동 방식

1. RGB-D camera를 통해 실시간으로 image를 받아옵니다.   
2. 주어진 image에서 손을 인식하고, 손가락 끝의 좌표를 찾습니다.   
3. 손 끝의 속도를 통해 그릴 선의 굵기를 계산합니다.   
4. 따로 구현한 그림판에 이전 손 끝 좌표와 현재 손 끝 좌표를 잇는 선을 그립니다.

-----
### 구현 과정

##### Fingertip detection & tracking  

Google에서 만든 Mediapipe framework에 있는 function을 통해 검지손가락 끝의 좌표를 저장합니다.   
좌표를 update할 때, 새 좌표의 일부만 반영해 부드럽게 움직는 것처럼 보이도록 구현했습니다.   
손 떨림을 방지하기 위해 마지막으로 update한 좌표와 현재 좌표가 일정거리 이상인 경우에만 현재 좌표로 update합니다.   

<figure>
    <img src="/assets/img/post/2021-11-27-dynamic_paint/eq1.PNG" width="70%" height="70%"> 
</figure>

##### 경계면 구현 

검지손가락의 Depth 값을 사용해 camera와의 거리를 확인해 일정 거리 이상이면 인식하지 않게 구현했습니다.   

##### 그림판 구현  
OpenCV 내장 함수 사용해서 새로 만들었습니다.   

##### 필압 구현 
손가락의 속도가 빠를때는 굵기를 얇게, 느릴때는 굵게 구현했습니다.   
필압 수식은 exponential 함수를 통해 non-linear하게 구현했습니다.   

<figure>
    <img src="/assets/img/post/2021-11-27-dynamic_paint/eq2.PNG" width="70%" height="70%"> 
</figure>
Fingertip tracking part에서와 마찬가지로, 굵기가 바뀔 때, 현재 굵기에 일정비율을 더하는 방식으로 굵기가 부드럽게 변하도록 구현했습니다.
<figure>
    <img src="/assets/img/post/2021-11-27-dynamic_paint/eq3.PNG" width="70%" height="70%"> 
</figure>

##### Draw 방식  
중지를 통해 draw를 control하는 방식을 사용했습니다.   
중지가 펴져있으면 draw하고, 접혀있으면 draw하지 않습니다.

---


### 시연 영상

<video width="50%" height="50%" controls>
    <source src="/assets/img/post/2021-11-27-dynamic_paint/demonstration_video.mp4" width="50%" height="50%"> 
</video>

-----

##### 주저리주저리
초기엔 mouse control까지 하려고 생각해서 코드 다 짰는데 필요 없길래 지웠음   
원래 손가락 인식하는 model을 따로 학습시키려 하였으나 이미 많이 있길래 그 중에 하나 가져다 씀   
손가락 떨림 잡는거 손가락 속도로 하려다가 잘 안되길래 거리로 바꿈   
선 굵기는 화면과 손가락 사이의 거리에 따라서 바뀌는 방식으로 하려그랬는데 depth값이 그정도로 정확하진 않길래 속도로 바꿈   
