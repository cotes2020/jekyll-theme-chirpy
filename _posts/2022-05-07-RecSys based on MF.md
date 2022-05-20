---
title: Matrix Factorization(MF) 기반 추천시스템
author: Bean
date: 2022-05-07 16:32:00 +0800
categories: [추천시스템]
tags: [추천시스템]
layout: post
current: post
class: post-template
subclass: 'post'
navigation: True
cover:  assets/img/post_images/recsys_cover2.jpg
---

추천시스템의 기초적인 내용을 잘 알지 못한채로 추천 알고리즘을 구현하는 논문들을 몇개 읽다보니 구현의 기반이 되는 중요한 원리를 계속 놓치고 있는 듯한 기분이 들었다. 그래서 `Python을 이용한 개인화 추천시스템` 책을 사서 읽고, 또 다양한 블로그 포스팅을 보며 추천시스템에 대한 공부를 더 깊이 해보았다.

공부를 하다 보니 추천시스템에서 **Matrix Factorization**이 매우 중요하다는 것을 느꼈다. MF는 Collaborative Filtering의 한 종류로, Netflix Prize에서 처음 등장하여 엄청난 성능 향상을 보임으로써 추천 시스템 분야의 혁신을 일으켰다고 한다. 기존에 많이 사용되었던 컨텐츠 기반 추천 알고리즘은 Data Sparsity와 Scalibility에 취약했다.(사용자나 아이템이 늘어날수록 Sparsity가 증가하게 되는데, CF는 결측값을 임의로 채워서 추천하다보니 성능이 떨어진다. 반면 MF는 결측값을 아예 사용하지 않아 성능을 유지할 수 있다.)

MF는 이런 한계점을 극복하면서도 추천 속도가 빨라 현실 세계에서는 가장 많이 쓰인다. 딥러닝이 급부상한 요즘에는 MF의 원리를 딥러닝에 응용하여 성능을 한층 더 높였다고 한다.

그래서 이번 글에서는 Matrix Factorization 기반의 추천시스템을 자세히 정리해보았다.

&nbsp;
## 추천 알고리즘 분류
\
&nbsp;
먼저, MF는 추천 알고리즘의 한 종류인데, 추천 알고리즘 중 `모벨 기반 알고리즘`에 속한다. 이 모델 기반의 알고리즘 이외에도 메모리 기반의 추천 알고리즘도 있다. 이 둘의 차이점을 간단히 정리하면 아래와 같다.

### 메모리 기반 추천 알고리즘 (memory-based RS)

* 추천을 위한 데이터를 모두 메모리에 가지고 있으면서 추천이 필요할 때마가 이 데이터를 사용해서 계산을 해서 추천하는 방식
* 대표적으로 협업 필터링(Collaborative Filtering: CF) 기반 알고리즘을 들 수 있다.

### 모델 기반 추천 알고리즘 (model-based RS)

* 데이터로부터 추천을 위한 모델을 구성한 후에 이 모델만 저장하고, 실제 추천을 할 때에는 이 모델을 사용해서 추천을 하는 방식
* 모델을 생성할 때는 많은 계산이 요구되지만 한 번 모델을 생성해두면 이후부터는 더 빠른 추천을 제공할 수 있다.
* 이번 글에서 다루는 MF 방식, Deep-Learning 기반의 방식이 대표적이다.

&nbsp;
## Matrix Factorization(MF) 알고리즘 원리
\
&nbsp;
MF 알고리즘은 유저가 아이템에 대해 평가한 정보를 담고 있는 (user x item) 데이터 행렬을 (user x feature)의 유저 행렬과 (item x freature)의 아이템 행렬로 쪼개서 분석하는 방식을 의미한다.

좀 더 formal하게는 R: [user x item] 형태의 full-matrix(평가 데이터)를

* P : [ user x feature ]
* Q : [ item x feature ]
    의 두 행렬로 쪼개서 분석하는 방식이다.

<div style="text-align: left">
  <img src="/assets/img/post_images/mf3.png" width="100%"/>
</div>

\
&nbsp;
이 때 feature은 `latent factor` 로서 user와 item이 공통으로 공유하고 있는 특성이다.

feature을 더 잘 이해하기 위해서 예시로 영화 추천에서 feature의 개수가 2개인 경우를 생각해보자. feature의 개수가 2개이므로 user와 item의 특징을 2개의 요인으로 나타낼 수 있다.

만약 이 요인이 로맨스-미스터리 고전-판타지이고 값이 -1.0 \~ 1.0 사이라고 하면,

user 행렬 P와 item 행렬 Q는 다음과 같이 나타낼 수 있을 것이다.

**[ user 행렬 P ]**

| 사용자 \ 잠재요인 | 로맨스-미스터리 | 고전-판타지 |
| ---------- | -------- | ------ |
| user A | 0.63 | 0.23 |
| user B | 0.35 | 0.77 |

**[ item 행렬 P ]**

| 영화 \ 잠재요인 | 로맨스-미스터리 | 고전-판타지 |
| --------- | -------- | ------ |
| movie A | 0.12 | 0.82 |
| movie B | 0.45 | 0.34 |

이 표를 확인해보면 user A는 로맨스보다는 미스터리를, 판타지 보다는 고전을 좋아함을 알 수 있다. 또한 movie B는 로맨스와 미스터리 특성을 반반씩 가지고 있으며 고전의 특성을 조금 더 띄고 있다.

이처럼 영화의 특성과 user의 특정이 각각 2개의 잠재 요인으로 분리되었다. 그리고 이 잠재요인을 통해 어떤 영화가 어떤 user의 관심을 끌 지 예상해볼 수 있다.

&nbsp;
## MF 기반 추천 알고리즘 과정
&nbsp;

그렇다면 주어진 user, item의 관계를 P, Q로 어떻게 분해할 수 있을까? R을 P, Q로 분해하는 알고리즘을 간단히 나타내면 아래와 같다.

1. 잠재요인 개수 K를 정한다.
2. 주어진 K에 따라 임의의 수로 초기화된 두 행렬 P(m x k), Q(n x k)를 생성한다.
3. P, Q 행렬을 통해 예측값 $R(=P x Q^{T})$을 구하고 실제 값과 비교하여 오차를 줄이기 위해 P, Q 값을 수정한다.
4. 기준에 도달할 때까지 3을 반복한다.

여기서 오차를 줄이기 위해서 P, Q를 수정할 때 어떤 방식으로 하는 가가 중요한 이슈이다. 일반적으로는 기계학습에서 많이 사용되는 SGD(Stochastic Gradient Decent) 방법을 적용한다.

&nbsp;
## SGD(Stochastic Gradient Decent)를 사용한 MF 알고리즘
\
&nbsp;
SGD 방법을 적용해서 P, Q 행렬을 최적화하는 방법을 알아보자

### Objective Function
P, Q를 학습할 때 사용되는 목적함수는 다음과 같다. 이 목적 함수는 두가지 텀으로 나뉜다.
<div style="text-align: left">
  <img src="/assets/img/post_images/mf4.png"/>
</div>

* 오차 제곱 합
  + 기준으로 앞의 텀은 Squared Error이다. 실제 평점값과 예측 평점 값의 차이를 나타낸다. 이 때, **학습 데이터에서 실제 평점이 있는 경우에만 오차를 계산한다.** 이는 SVD와 다른 부분인데 뒤에 MF vs SVD에서 좀 더 자세히 다룰 것이다.
* 정규화
  + 뒤의 term은 과적합을 방지하는 정규화 텀이다. 다른 기계학습과 마찬가지로 모델이 학습함에 따라 파라미터의 값(weight)이 점점 커지는 데, 이 값이 너무 커지게 되면 학습 데이터에 과적합하게 된다. 이를 방지하기 위하여, 학습 파라미터인 p, q가 너무 커지지 않도록 규제를 해주어야 한다. 여기서는 L2 정규화를 사용하였다. 람다는 목적 함수에 정규화 텀의 영향력을 어느 정도로 줄 것인지 정하는 하이퍼 파라미터이다. 람다가 너무 작으면 정규화 효과가 적고, 너무 크면 파라미터가 제대로 학습되지 않아서 언더피팅(Underfitting)이 발생한다.

### Optimazation (SGD)
다름으로 목적함수를 최소화하는 최적화 기법으로 SGD(Stochastic Gradient Descent)을 사용한다. SGD에서는 파라미터 업데이트를 위해 목적함수를 p와 q로 편미분한다. 아래는 편미분 하는 수식을 보여주고 있고, 이렇게 도출된 Gradient를 현재 파라미터 값에서 빼줌으로써 학습을 진행하게 된다.

<div style="text-align: left">
  <img src="/assets/img/post_images/mf6.png"/>
</div>

&nbsp;
## MF 최적 파라미터 찾기
\
&nbsp;
쉽게 예상할 수 있듯이 잠재요인수 K와 iteration에 따라 예측의 정확도가 달라질 것이다. 최적의 K와 iteration은 다음의 방법으로 구해볼 수 있다.

1. K를 50~260까지 넓은 범위에서 10간격으로 RMSE를 계산해 최적값을 찾는다. 이 때, iteration으로 충분히 큰 수를 준다.
2. 찾은 최적값 K를 기준으로 이 숫자 전후 +- 10값에 대하여 더 작은 1의 간격으로 다시 RMSE값을 계산해 K의 최적값을 찾는다.
3. K를 찾은 최적값으로 고정 후 iterations의 1~300 범위에서 설정, 반복하여 iterations 최적값을 찾는다.

이 때, train/test set 분리와, SGD 실행 시 적용되는 난수에 따라 계산값이 달라질 수 있으므로 코드를 여러번 반복 실행 후 평균값으로 최적값을 설정해야 한다.

&nbsp;
## MF vs SVD
\
&nbsp;
MF와 SVD는 모두 데이터 분석과 기계학습에 널리 사용되고 있고, 유사한 점이 많아 추천시스템 용어로 같은 의미로 사용이 되곤 한다.
결론부터 말하자면 명백히 둘은 다른 기법이고, 실제로 SVD기법은 추천시스템에서 거의 사용되지 않는다.

먼저, SVD 방식에서는 행렬이 3개로 분할되지만 MF 방식에서는 2개로 분해된다.
또한, 추천시스템의 데이터셋에는 사용자가 평가하지 않은 null 값이 많이 존재하는데 MF와 SVD는 이 null값을 처리하는 방식에서 차이가 난다.

* (3개의 행렬로 분해되는) SVD방식에서는 null을 대체한 0값이 하나의 값으로 적용돼서 학습 후에도 0에 근사한 예측값이 도출된다.
즉, null에 대한 예측이 제대로 이루어지지 못한다.
  <div style="text-align: left">
    <img src="/assets/img/post_images/mf1.png" width="100%"/>
  </div>
* (2개의 행렬로 분해되는) MF방식은 모델을 학습하는 과정(SGD)에서 null(0)값을 제외하고 계산하는 구조이며, 이렇게 학습된 행렬 P,Q를 통해 null에 대한 예측도 정확하게 할 수 있다.
  <div style="text-align: left">
    <img src="/assets/img/post_images/mf2.png" width="100%"/>
  </div>

\
&nbsp;

***

참고 내용 출처 :
- [https://sungkee-book.tistory.com/12](https://sungkee-book.tistory.com/12)
- 임일, 『Python을 이용한 개인화 추천시스템』, 청람(2020)