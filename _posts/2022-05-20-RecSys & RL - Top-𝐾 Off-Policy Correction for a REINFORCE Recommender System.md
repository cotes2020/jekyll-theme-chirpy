---
title: (작성중) RecSys & RL - Top-𝐾 Off-Policy Correction for a REINFORCE Recommender System
author: Beanie
date: 2022-05-20 15:03:00 +0800
categories: [RecSys]
tags: [RecSys, RL, paper]
layout: post
current: post
class: post-template
subclass: 'post'
navigation: True
cover:  assets/img/post_images/recsys_cover1.jpg
---

이전에 읽었던 `A Deep Reinforcement Learning Framework for News Recommendation` 논문은 알고리즘 구현 코드를 찾기가 어려웠다. 그래서 알고리즘 코드가 같이 있으면서 적당히 challenging한 논문을 다시 찾아보았고 이 `Top-𝐾 Off-Policy Correction for a REINFORCE Recommender System` 이 가장 괜찮아보여 읽어보았다. 이후 여러 샘플 코드를 참고하여 public dataset으로 직접 구현까지 해볼 예정이다.

## 주요 Contribution
&nbsp;

이 논문의 주요한 기여점은 다음과 같다.
* **REINFORCE Recommender**
    * Large, non-stationary state and action spaces
* **Off-Policy Candidate Generation**

    RL에서 추천시스템을 사용하는 경우, 데이터의 부족이 특히 문제가 된다. 고전적인 RL에서는 이 데이터 부족 문제를 해결하기 위해서 많은 양의 training data를 self-replay나 simulation을 통해 수집하였다.
    하지만 추천시스템 문제의 경우, 복잡하게 얽혀있는 추천시스템 환경 때문에 simulation을 통하여 데이터를 얻는 것이 불가능하다. 따라서 reward를 관찰하는 것 자체가 실제 유저에게 실제 추천을 제공하는 과정이 필요하기 때문에 탐색되지 않은 공간의 state, action space에 대하여 reward값을 쉽게 알아내기 어렵다.

    \
    &nbsp;

    따라서 본 논문에서는, **다른 추천시스템이나 과거의 policy에서 얻은 유저 feedback의 로그들을 활용하여 Off-policy learning을 수행** 한다. 이 때, 다른 policy에서 수집하였기 때문에 필연적으로 발생하는 bias를 해결하는 방식으로 학습한다.
* **Top-K Off-Policy Correction**
    * 실제 추천시스템 환경에서는 1개가 아닌 여러개의 추천을 동시에 제공해야 한다. 따라서 본 논문에서는 top-K recommender system을 위한 새로운 top-K off-policy correction을 정의한다.
* **Benefits in Live Experiments**
    * 아이템에 대한 유저 선호도는 계속 변함 -> 유저 state 값이 지속적으로 바뀐다.

&nbsp;
## Reinforce Recommender
&nbsp;
### MDP Modeling
추천시스템을 강화학습 세팅에 적합하게 맞춰보자. 강화학습을 위하여 환경을 Markov Decision Process(MDP)로 나타내면 다음과 같다.
* $S$ : 유저 state를 나타내는 연속적인 state space
* $A$ : 추천 아이템 후보군을 포함하고 있는 discrete한 action space
* $P$ : state transition probability ( $S \times A \times S \to \mathbb{R}$ )
* $R$ : 보상 함수 ( $S \times A \to \mathbb{R}$ ), 이 때, $ r(s, a) $ 는 유저 state s에서 action a를 할 때 바로 얻어지는 reward를 의미한다.
* $ \rho_{0} $ : 초기 상태 분포
* $ \gamma $ future reward에 대한 discount factor

### Policy Gradient

&nbsp;
## Model Architecture
&nbsp;

&nbsp;
## Off-Policy Correction
&nbsp;

&nbsp;
## Top-K Recommendation
&nbsp;

&nbsp;
## Exploration
&nbsp;

&nbsp;
## Experiment Results
&nbsp;