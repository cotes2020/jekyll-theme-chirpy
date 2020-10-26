---
title: 첫 Kaggle 대회참여 - IEEE-CIS Fraud Detection
author: loveAlakazam
categories: [프로젝트][GCP]
tags: [인공지능 스피커][TTS][STT][Python3][Web-Scraping]
comments: false
---

<br>

# IEEE-CIS Fraud Detection 대회공지

[Kaggle IEEE-CIS Fraud Detection 대회](https://www.kaggle.com/c/ieee-fraud-detection)

<br>

![작성코드 URL](https://github.com/loveAlakazam/kaggle_IEEE_CIS_Fraud_Detection/blob/master/self_making_kernel/trial3/ieee_fraud_detection_inColab_myself.ipynb)

<BR>

> # 1. 활용 기술 스택

<BR>

- ### Language: `Python3`
- ### Environment: `Google Colab`
- ### Library: `pandas`, `matplotlib`, `seaborn`, `lightgbm`, `sklearn`

<br><br>

> # 2. 구현과정

<br>

- 1. 각 컬럼별 데이터 종류를 변경시켜 데이터프레임을 나타내는데 사용되는 메모리의 사이즈를 줄였습니다.
(예: np.int64 -> np.int8/ np.int16 /np.int32)

- 2. 데이터프레임 합치기

- 3. Object 타입 컬럼을 구성하는 클래스 중 이름이 겹치는 클래스들을 모아서 하나의 클래스(대표 클래스)로 합쳤습니다.

  - 오버피팅 발생을 줄이기 위해 하나의 컬럼을 구성하는 클래스의 개수들을 줄였습니다.

- 4. 라벨 인코딩(Label Encoding)
  - 머신러닝 모델링 과정에서, 모델링이 가능한 입력데이터의 타입은 *숫자타입*과 *Boolean타입* 입니다.

  - 모델링이 가능한 타입으로 데이터를 전처리 시켰습니다.


- 5. LGBM(Light Gradient Boostring Machine) 모델을 이용하여 모델링을 했습니다.

- 6. 훈련데이터 셋과 테스트 데이터 셋 모두 ROC-AUC 방식으로 평가했습니다.

- 7. 결과
  - 결과데이터셋 가채점 결과: **0.9246**
  - 최종순위: **4708**/6381



<br><br>

> # 3. 문제 해결과정

<br>

- ### [문제1] 머신러닝 알고리즘이 너무 낯설고, 이해하고 응용까지 어려웠습니다.

- ### [해결1] **파이썬 머신러닝 완벽가이드** 서재의 예제를 참고하여 이론을 배웠고, 실습예제를 타이핑하였습니다.
  - 학습내용
    - #### 분류(Classification) 알고리즘
      - 결정트리
      - Random Forest
      - GBM
      - Light GBM(LGBM)

    - #### 평가방식
      - 오차행렬
      - 정밀도와 재현율
      - F1스코어
      - ROC곡선과 AUC 스코어
    - #### Scikit Learn
      - 학습/ 테스트 데이터셋 분리
      - 교차검증
        - K-Fold
        - Grid Search CV
      - 데이터 전처리
        - Label Encoding


<BR>

- ### [문제2] 스스로 코드를 작성하기 어려웠습니다.

- ### [해결2] 그룹 스터디 회원이 작성한 소스코드와 오픈소스코드(캐글 필사)를 참고하여 틀을 잡았습니다.
  - 지난 대회의 오픈소스코드(캐글 필사)를 직접 따라치면서 EDA와 Feature Engineering, Esembling을 학습했습니다.

<br><br>

> # 4. 느낀점 및 보완점

<br>

- 통계와 수학에 대한 선행지식이 없으면 데이터분석을 이해하기가 어려움을 알았습니다.

- 캐글스터디를 통해서 학생신분뿐만 아니라 다양한 사람들(직장인, 개발자, 데이터분석 대회 유경험자, 대학원생)과 함께 데이터분석에 대해서 몰입할 수 있는 값진 시간이었습니다.
  - 여담으로, 정말 사람들이 데이터분석에 관심이 많고, 공부 의욕이 넘쳤습니다. 그래서 더 열심히 해야겠다는 다짐을 하기도 했습니다.

  - 정말 성실하게 데이터분석을 하시는 분들도 있었고, 전문가를 직접만나서 피드백을 받기도 했습니다.

<br><br>
