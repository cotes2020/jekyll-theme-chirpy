---
title: PCA learning 구현 (from scratch)
author: Bean
date: 2022-03-24 09:32:00 +0800
categories: [AI, basic]
tags: []
layout: post
current: post
class: post-template
subclass: 'post'
navigation: True
cover:  assets/img/post_images/ai_cover2.jpg
---

## PCA란?
---
PCA는 분포된 데이터들의 주성분(Principal Component)를 찾아주는 방법이다.

### 주성분이란?
---
주성분은 그 방향으로 데이터들의 분산이 가장 큰 방향벡터를 의미한다.

좀더 구체적으로 보면 아래 그림과 같이 2차원 좌표평면에 n개의 점 데이터 (x1,y1), (x2,y2), ..., (xn,yn)들이 타원형으로 분포되어 있을 때

<div style="text-align: left">
   <img src="/assets/img/post_images/pca2.png" width="100%"/>
</div>

이 데이터들의 분포 특성을 2개의 벡터로 가장 잘 설명할 수 있는 방법 그림에서와 같이 e1, e2 두 개의 벡터로 데이터 분포를 설명하는 것이다. e1의 방향과 크기, 그리고 e2의 방향과 크기를 알면 이 데이터 분포가 어떤 형태인지를 가장 단순하면서도 효과적으로 파악할 수 있다.

PCA는 데이터 하나 하나에 대한 성분을 분석하는 것이 아니라, 여러 데이터들이 모여 하나의 분포를 이룰 때 이 분포의 주성분을 분석해 주는 방법이다.

위의 그림에서 e1 방향을 따라 데이터들의 분산(흩어진 정도)이 가장 크다. 그리고 e1에 수직이면서 그 다음으로 데이터들의 분산이 가장 큰 방향은 e2이다.
따라서 1차 주성분은 e1 벡터, 2차 주성분은 e2 벡터라고 할 수 있다.

### PCA 계산
---
PCA는 2차원 데이터 집합에 대해 PCA를 수행하면 2개의 서로 수직인 주성분 벡터를 반환하고, 3차원 점들에 대해 PCA를 수행하면 3개의 서로 수직인 주성분 벡터들을 반환한다.

이 PCA는 SVD decomposition Scikit-learn의 내장 라이브러리를 이용하여 구할 수도 있지만 이번에는 머신러닝 학습을 통하여 구해보았다. 이렇게 학습을 통해 PCA를 구하기 위해서는 Hebbian learning을 알아야 한다.

&nbsp;

## Hebbian learning
---

Hebbian learning은 1949년 캐나다 심리학자인 Donal Hebb가 제안한 Hebb의 학습 가설에 근거한 학습 규칙이다. 신경학적인 측면에서 사람이 어떻게 학습하는 가에 대한 비교적 간단한 학습 이론에 대해 살펴보자. 중심적인 아이디어는 다음과 같다.
> 두 개의 뉴런 A, B 가 서로 반복적이고 지속적으로 점화(firing)하여 어느 한쪽 또는 양쪽 모두에 어떤 변화를 야기한다면 상호간의 점화의 효율 (weight) 은 점점 커지게 된다.
이 간단한 규칙은 나중에 개발되는 많은 신경망 모델들의 학습 규칙의 토대가 된다.

이 Hebb의 학습 가설에 근거한 Hebbian 학습 규칙은 가장 오래되고 단순한 형태의 학습 규칙이다. 이 규칙은 **만약 연접(synapse) 양쪽의 뉴런이 동시에 또 반복적으로 활성화되었다면 그 두 뉴런 사이의 연결강도가 강화된다**는 관찰에 근거한다. 수학적으로는 다음 식으로 나타낼 수 있다.

$w_{ij}(t+1) = w_{ij}(t) + \eta y_{j}(t)x_{i}(t)$

그러나 Hebbian learning 식을 그대로 사용하면 nomalization 문제가 발생한다. (즉, w값이 발산한다.) 따라서 w 벡터의 크기를 1로 고정하거나, oja rule을 사용하여 nomalization을 방지하는 방향으로 풀이를 진행하였다. 각각의 코드는 아래와 같다.

## PCA learning 코드
---
```python
def pca_hebbian_learning(type):
  if (type == 1):
    S = S1
  else:
    S = S2
  X = np.array(S)

  # Normalizing X
  norm_X = X-X.mean(axis=0)
  norm_X = norm_X/X.std(axis=0)

  W = np.random.normal(scale=0.25, size=(2, 1))
  prev_W = np.ones((2, 1))

  learning_rate = 0.0001
  tolerance = 1e-8

  while np.linalg.norm(prev_W - W) > tolerance:
      prev_W = W.copy()

      Ys = np.dot(norm_X, W)
      W += learning_rate * np.sum(Ys*norm_X, axis=0).reshape((2, 1))
      # Normalizing W
      W = W / math.sqrt(W[0]**2 + W[1]**2)

  print("***Using hebbian learning***\n")
  print('eigenvector :\n', [W[0][0], W[1][0]])
  plot_first_pca(type, "hebbian learning", W)

def pca_oja(type):
  if (type == 1):
    S = S1
  else:
    S = S2
  X = np.array(S)

  # Normalizing X
  norm_X = X-X.mean(axis=0)
  norm_X = norm_X/X.std(axis=0)

  # Apply the Oja's rule
  W_oja = np.random.normal(scale=0.25, size=(2, 1))
  prev_W_oja = np.ones((2, 1))

  learning_rate = 0.0001
  tolerance = 1e-8

  while np.linalg.norm(prev_W_oja - W_oja) > tolerance:
      prev_W_oja = W_oja.copy()

      Ys = np.dot(norm_X, W_oja)
      W_oja += learning_rate * np.sum(Ys*norm_X - np.square(Ys)*W_oja.T, axis=0).reshape((2, 1))

  print("***Using Oja's rule***\n")
  print('eigenvector :\n', [W_oja[0][0], W_oja[1][0]])
  plot_first_pca(type, "Oja's rule", W_oja)
```

## PCA learning 결과
---
위의 코드로 학습을 시키면 다음의 결과를 확인할 수 있다.

<div style="text-align: left">
   <img src="/assets/img/post_images/pca1.png" width="100%"/>
</div>

\
&nbsp;

***

&nbsp;
참고 내용 출처 :
* [https://darkpgmr.tistory.com/110](https://darkpgmr.tistory.com/110)
* [http://www.aistudy.com/neural/hebbian_learning.htm](http://www.aistudy.com/neural/hebbian_learning.htm)