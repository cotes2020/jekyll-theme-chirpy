---
title: Virtual Trainer
author: Lee Sujong
date: 2021-11-27 01:10:00 +0900
categories: [Exhibition,2021년]
tags: [post,sujong,knn,mediapipe] 
---

------------------------------------------
# Virtual Trainer

### 작품 개요
운동을 도와주는 사람 없이도 도움을 받고 싶었다.

대표적으로 내가 제대로 자세를 잡고 있는지 궁금했다.

---

### 작품 설명

AWS에서 GPU 가상 컴퓨터를 구매해 streamlit 서버를 운영.(https://chosangnimiswatching.com)

사용자가 카메라를 켜서 운동을 하면 컴퓨터에서 관절 좌표를 분석한다.(mediapipe 사용)
<div class="row">
    <div style="width: 50%">
        <figcaption>mediapipe 적용</figcaption>
        <img src="/assets/img/post/2021-11-27-VirtualTrainer/test0.jpg">
    </div>
</div>

그 후 분석된 좌표를 이용해 어떤 자세에 가까운지 판단한다.(KNN 사용)

이 때 자세는 숄더프레스, 벤치프레스, 스쿼트, 데드리프트 4개가 가능하다.

KNN사용 시 특정 자세에 대한 확률이 임계값을 넘어서면, 자세를 확정짓는다.(최대수축지점)

이 상태에서 이완 시, 확률이 임계값보다 작아지면 횟수를 1 늘린다.

만약 임계값을 넘지 않는 상태가 5초 정도 진행되면, 쉬는 시간으로 간주하고 휴식시간을 측정한다.

-----
### 코드 설명

<div class="row">
    <div style="width: 100%">
        <figcaption>코드 구조</figcaption>
        <img src="/assets/img/post/2021-11-27-VirtualTrainer/Constructure.png">
    </div>
</div>

#### app.py

서버의 프론트엔드를 담당한다. 튜토리얼 및 횟수 설정 등 운동 시작 전 세팅이 가능하다.

또한 서버에서 카메라 영상을 받아 my_helper.py에 보내 분석을 요청하며

분석을 토대로 운동 종류를 판별한다.

확률값은 운동 4개와 휴식 동작 1개가 합쳐져 총 5개의 classification이 가능하다.

사용자가 처음 1회 운동 시 classification 확률값을 기반으로 4개의 운동에 대해 각각 횟수 증가를 측정하는데,

이 때 횟수가 올라간 종목을 선택한다.

#### my_helper.py

사람 인식과 관절 분석을 담당한다.

사람 인식은 mediapipe를 사용해 관절의 위치를 인식한다.

관절의 위치를 좌표로 옮긴 후, 이 좌표들의 군집과 가장 가까운 운동 종목을 확률로 보여준다.

이 확률은 총합 10을 기점으로 5개의 classification 확률을 나눠가진다.

return은 관절 좌표와 확률이다.

#### ShoulderP.py

숄더프레스의 횟수를 올린다. 얻어온 관절 좌표를 이용한다.

사용자의 코 높이가 기준이 되어 팔꿈치가 이 기준보다 높게 올라가면 횟수를 1개 늘린다.

이 기준에 다가갈수록 노란색 원이 커지다가 기준을 넘어가면 초록색 원으로 변한다.

이 때 한 쪽 팔만 넘어가면 초록색 원이 만들어지지 않는다.

#### Squat.py, BenchP.py, DeadL.py

알고리즘은 모두 같다. 어떤 확률값이 임계값을 넘어가면 초록색 원으로 변한다.

이 상태에서 자세를 복구하면(이완) 임계값보다 작아질 때 횟수를 1 늘린다.

그에 따라 원도 노란색으로 변하고 크기도 줄어든다.

#### Drawing

원을 그려주는 함수다. OpenCV를 이용해 이미지에 원을 그린다. 

-----
### 중간점검과 비교 시 바뀐 점

- 사용한 모델을 EfficientPose에서 mediapipe로 변경.
  - 이전 모델은 화질이 떨어지는 문제 발생. 이는 속도 유지를 위해 일부러 저화질로 변경하는 것으로 보임.
  - 모델 변경 후 화질이 떨어지는 문제가 해결됐고, 상대적으로 가벼운 모델이어서 속도도 빠르다.
- GPU 가상 컴퓨터 사용
- KNN으로 운동 종목을 판단

-------
### 결과물 확인 

<div class="row">
    <div style="width: 50%">
        <figcaption>메인서버</figcaption>
        <img src="/assets/img/post/2021-11-27-VirtualTrainer/main.png">
    </div>
</div>

숄더프레스 시연
<div class="row">
    <div style="width: 50%">
        <figcaption>초기 상태</figcaption></figcaption>
        <img src="/assets/img/post/2021-11-27-VirtualTrainer/first.png">
    </div>
</div>
<div class="row">
    <div style="width: 50%">
        <figcaption>숄더프레스 통과</figcaption>
        <img src="/assets/img/post/2021-11-27-VirtualTrainer/SPGreen.png">
    </div>
</div>
<div class="row">
    <div style="width: 50%">
        <figcaption>숄더프레스 중간</figcaption>
        <img src="/assets/img/post/2021-11-27-VirtualTrainer/SPYellow.png">
    </div>
</div>
<div class="row">
    <div style="width: 50%">
        <figcaption>숄더프레스 불균형</figcaption>
        <img src="/assets/img/post/2021-11-27-VirtualTrainer/SPNope.png">
    </div>
</div>

<video controls width="90%">
    <source src="/assets/img/post/2021-11-27-VirtualTrainer/ShoulderP.mp4">
</video>

스쿼트 시연
<div class="row">
    <div style="width: 50%">
        <figcaption>스쿼트 정면</figcaption>
        <img src="/assets/img/post/2021-11-27-VirtualTrainer/frontsq.png">
    </div>
</div><div class="row">
    <div style="width: 50%">
        <figcaption>스쿼트 측면</figcaption>
        <img src="/assets/img/post/2021-11-27-VirtualTrainer/sidesq.png">
    </div>
</div>
<video controls width="90%">
    <source src="/assets/img/post/2021-11-27-VirtualTrainer/squat.mp4">
</video>