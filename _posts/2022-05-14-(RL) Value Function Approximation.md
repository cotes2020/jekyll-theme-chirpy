---
title: (Reinforcement learning) Value Function Approximation
author: Bean
date: 2022-05-13 12:23:00 +0800
categories: [AI, RL]
tags: [RL]
layout: post
current: post
class: post-template
subclass: 'post'
navigation: True
cover:  assets/img/post_images/ai_cover.jpg
---

이번 글에서는 강화학습 기본 개념 중 Value function approximation에 대해 다루고자 한다. 이 글은 David Silver의 강화학습 강의와 KAIST EE619 강화학습 수업에서 공부한 내용을 바탕으로 작성하였다.

## Tabular Methods
&nbsp;

규모가 작은 MDP 모델의 경우, state와 action을 Table에 저장하여 사용할 수 있다. 더 정확히는 value function을 사용하는 DP 문제의 경우, 모든 state s는 V(s) `vector`을 가질 것이다. 또한, action-value function을 사용하는 MC, TD의 경우 모든 state-action 쌍 (s, a)에 대하여 Q(s, a) `matrix`가 존재한다. 이렇게 행렬로 만들어 푸는 방식을 **Tabular Methods**라고 한다. 하지만 이런 Tabular Method는 state와 action의 차원이 무지하게 커지는 실생활 문제에 Generalization을 할 수 없다.

다음의 문제들을 생각해보자.
* Backgammon: $10^{20}$ states
* Compoter Go: $10^{170}$ states
* Robot: continuous state space

이런 경우, 특히나 continuous state space를 가지는 경우에는 특히나 Table만으로 핸들링이 불가능하고 따라서 새로운 방법이 필요하다.

&nbsp;
## Parameterizing value function
\
&nbsp;
그렇다면 실생활에 강화학습을 적용하기 위해서는 어떤 방법을 이용해야 할까? state와 action을 table로 구현하는 대신에, $ w$라는 새로운 변수를 통해 **value function**을 함수화함으로써 문제를 해결할 수 있다.
먼저, value function을 function approximation을 이용하여 추정해준다.

$$ \hat{V}_{w}(s) \approx V^{\pi}(s) ~~ or ~~\hat{Q}_{w}(s, a) \approx V^{\pi}(s, a) $$

다음으로, seen states(방문한 states)로 부터 unseen states(아직 방문하지 않은 states)로 일반화시킨다.
마지막으로 MC나 TD learning을 이용하여 파라미터 $w$를 학습하면 value function의 optimal한 추정치를 얻을 수 있다.

이를 그림으로 다시 확인해보면,
<div style="text-align: left">
   <img src="/assets/img/post_images/value_approximation.png" width="100%"/>
</div>
state가 함수의 input으로 넣고, $w$라는 parameter를 지나 action value function을 output으로 받는 과정이 된다.

&nbsp;
## On-Policy Prediction with Function Approximation
&nbsp;
### Loss Function: Mean-Square Value Error(MSVE)
On-Policy Prediction problem은 state value function을 추정해나가는 과정이다.
True value function $V^{\pi}(s)$ 와 Approximation function $\hat{V} _{w}(s)$ 가 있을 때, 두 함수가 최대한 같아지도록 $\hat{V} _{w}(s)$ 을 학습시키는 것이 목표이다.
이 때, 두 함수의 차이를 Mean-Square Value Error(MSVE)으로 가늠해볼 수 있다.

$$ J(w) = MSE(w) = E_{\pi} = \left [ V^{\pi}(s) - \hat{V}_{w}(s) \right ]^{2} \\ = \sum_{s\in S}^{}\mu^{\pi}(s)\left [ V^{\pi}(s) - \hat{V}_{w}(s) \right ]^{2}$$

여기서 $\mu^{\pi}(s)$ 는 $\pi$ 에 대한 on-policy distribution 이다.

### On-Policy Distribution
MSVE는 두 함수 간의 weighted $L_{2}$ distance 이다. 이떄 각 s에 대해서 weighting importance는 on-policy distribution $d^{\pi}(s)$ 에 의해 주어진다.
$d^{\pi}(s)$ 을 정의하는 한 가지 방법은

### Stochastic Gradient Descent
학습은 정의된 MSVE에 대해서 stochastic gradient descent 방법을 통해 진행해준다. Stochastic gradient의 목표는 MSE loss를 최소화하는 vector $w$ 를 찾는 것이다.
Stochastic Gradient Descent을 수행하는 과정은 다음과 같다.
1. Mean-Square Loss Function: 미분가능한 함수로 주어진다.
   * 만약 loss function이 미분가능하지 않다면, subgradient를 사용한다.
2. True Gradient: 먼저 loss function의 true gradient를 계산한다.
   * True gradient : $\sum_{s}^{}\mu^{\pi}(s) \left [ \left ( V^{\pi}(s) - \hat{V_{w}}(s) \right) \bigtriangledown \hat{V_{w}}(s) \right ]$
   * weight update : $ w_{t+1} = w_{t} + \alpha \sum_{s}^{}\mu^{\pi}(s) \left [ \left ( V^{\pi}(s) - \hat{V_{w}}(s) \right) \bigtriangledown \hat{V_{w}}(s) \right ] $
3. Sampling the Gradient: 이후 샘플을 통해 stochastic gradient 방법으로 true gradient를 추정한다. 이런 방식을 `sample the gradient` 라고 한다. 2번에서 True expectation $\sum_{s\in S}^{}\mu^{\pi}(s)$ 이 **$\mu^{\pi}(s)$ 로 부터 생성된 하나의 샘플 $s_{t}$ 로 계산한 sample mean** 으로 대체된다.
   * Stochastic gradient : $\left [ \left ( V^{\pi}(s_{t}) - \hat{V_{w}}(s_{t}) \right) \bigtriangledown \hat{V_{w}}(s_{t}) \right ]$
   * weight update : $ w_{t+1} = w_{t} + \alpha \left ( V^{\pi}(s_{t}) - \hat{V_{w}}(s_{t}) \right) \bigtriangledown \hat{V_{w}}(s_{t}) $

$\mu^{\pi}(s)$로 부터 샘플을 충분히 많이 추출한다면, sample-based SG는 true gradient 로 수렴한다.

### On-Policy MC, TD Learning
gradient를 계산하기 위해서, $s_{t}, V^{\pi}(s_{t})$ 의 true value가 필요하지만 이 값은 알기 힘들다. 따라서 $V^{\pi}(s_{t})$ 를 추정치인 MC, TD or TD( $\lambda$ ) 추청값으로 대체한다. 각각의 추정치로 대체하면,

* Gradient MC for Value Function Estimation: $ w_{t+1} = w_{t} + \alpha \left ( G_{t} - \hat{V_{w}}(s_{t}) \right) \bigtriangledown \hat{V_{w}}(s_{t}) $
* Semi-Gradient TD for Value Function Estimation: $ w_{t+1} = w_{t} + \alpha \left ( r_{t+1} + \gamma \hat{V_{w}}(s_{t+1}) - \hat{V_{w}}(s_{t}) \right) \bigtriangledown \hat{V_{w}}(s_{t}) $


&nbsp;
## On-Policy Control with Function Approximation
&nbsp;

### Prediction vs Control
On-Policy Control을 다루기에 앞서 Prediction과 Control의 차이를 복습해보자. Perdiction과 Control의 차이는 policy에 대한 목표와 관련이 있다.

먼저 RL에서 prediction task는 policy가 주어진 상황에서 이 policy가 얼마나 잘 동작하는 지를 측정하는 것을 목표로 한다. 따라서, 주어신 state에서 total reward를 예측하기 위해서 $\pi(\left.\begin{matrix}
a\end{matrix}\right|s)$ 가 고정되어 있다.
반면에 control task는 policy가 고정되어 있지 않으며 최적 policy를 찾는 것을 목표로 한다. 즉, 임의의 주어진 상태에서 total reward의 예측값을 가장 크게 하는 policy $\pi(\left.\begin{matrix}
a\end{matrix}\right|s)$ 를 찾는다.

### Control with Value Function Approximation
Control 과정은 다시 Policy evaluation 과정과 Policy improvement 과정으로 나뉜다. 이 때, model-free를 위해서 action-value function을 사용한다.

<div style="text-align: left">
   <img src="/assets/img/post_images/function_approx1.jpeg" width="80%"/>
</div>
* Policy evaluation : $\hat{Q}_{w}$ 를 $Q^{\pi}$로 추정
* Policy improvement : action value function에 $ \epsilon - greedy $ 한 action을 취함으로써 improve 진행

### Loss Function: Mean-Square Value Error(MSVE)
Prediction 때와 똑같이 MSVE를 구해보면,

$$ J(w) = MSE(w) = E_{\pi} = \left [ Q^{\pi}(s) - \hat{Q}_{w}(s) \right ]^{2} \\ = \sum_{s, a}^{}\mu^{\pi}(s, a)\left [ Q^{\pi}(s, a) - \hat{Q}_{w}(s, a) \right ]^{2} $$

$$ - \frac{1}{2} \bigtriangledown  J(w) = \sum_{s, a}^{}\mu^{\pi}(s, a)\left ( Q^{\pi}(s, a) - \hat{Q}_{w}(s, a) \right ) \bigtriangledown \hat{Q}_{w}(s, a) $$

### Stochastic Gradient Descent
policy $\pi$ 에서 샘플링한 $(s_{t}, a_{t})$에 대하여 MSVE 값을 이용해서 stochastic gradient descent or semi-gradient를 수행한다.

$$ \Delta w =  - \hat{\frac{1}{2}\bigtriangledown J(w)} = (Q^{\pi}(s_{t}, a_{t}) - \hat{Q}_{w_{t}}(s_t, a_{t})) \bigtriangledown \hat{Q}_{w_{t}}(s_t, a_{t}) $$

또한, Prediction 마찬가지로 true action value function $ Q^{\pi}(s_{t}, a_{t}) $ 을 알 수 없기 때문에 MC, TD target을 사용한다.
* For gradient MC:
   $ w_{t+1} = w_{t} + \alpha \left ( G_{t} - \hat{Q_{w}}(s_{t}, a_{t}) \right) \bigtriangledown \hat{Q_{w}}(s_{t}, a_{t}) $
* For semi-gradient TD:
   $ w_{t+1} = w_{t} + \alpha \left ( r_{t+1} + \gamma \hat{Q_{w}}(s_{t+1}, a_{t+1}) - \hat{Q_{w}}(s_{t}, a_{t+1}) \right) \bigtriangledown \hat{Q_{w}}(s_{t}, a_{t+1}) $

&nbsp;
## Off-Policy Prediction and Control with Function Approximation
&nbsp;

### Off-Policy TD Prediction ($V^{\pi}$(s)): Importance Sampling
Target policy $\pi$, Behavior policy $\beta$ 에 대하여 먼저 Tabular case를 다시 살펴보자.

$$ V(s_{t}) \leftarrow V(s_{t}) + \alpha \left (  \frac{\left.\begin{matrix} \pi(a_{t}\end{matrix}\right|s_{t})}{\left.\begin{matrix} \beta(a_{t}\end{matrix}\right|s_{t})} (r_{t+1} + \gamma V(s_{t+1})) - V(s_{t}) \right ) $$

$$ V(s_{t}) \leftarrow V(s_{t}) + \alpha  \frac{\left.\begin{matrix} \pi(a_{t}\end{matrix}\right|s_{t})}{\left.\begin{matrix} \beta(a_{t}\end{matrix}\right|s_{t})} (r_{t+1} + \gamma V(s_{t+1}) - V(s_{t}))  $$

이를 바탕으로 이전과 비슷한 방식으로 Stochastic Gradient Descent을 적용해 weight update 식을 구하면 다음과 같다.

$$ w_{t+1} = w_{t} + \alpha \left (  \frac{\left.\begin{matrix} \pi(a_{t}\end{matrix}\right|s_{t})}{\left.\begin{matrix} \beta(a_{t}\end{matrix}\right|s_{t})} (r_{t+1} + \gamma \hat{V}_{w_{t}}(s_{t+1})) - \hat{V}_{w_{t}}(s_{t}) \right ) \bigtriangledown \hat{V}_{w_{t}}(s_{t}) $$

$$ w_{t+1} = w_{t} + \alpha  \frac{\left.\begin{matrix} \pi(a_{t}\end{matrix}\right|s_{t})}{\left.\begin{matrix} \beta(a_{t}\end{matrix}\right|s_{t})} (r_{t+1} + \gamma \hat{V}_{w_{t}}(s_{t+1}) - \hat{V}_{w_{t}}(s_{t})) \bigtriangledown  \hat{V}_{w_{t}}(s_{t}) $$

&nbsp;
### Off-Policy TD Control (Action-Value Function)
Control의 경우에도 Tabular case를 다시 살펴보면,

$$ Q(s_{t}, a_{t}) \leftarrow Q(s_{t}, a_{t}) + \alpha\left ( r_{t+1} + \gamma \sum_{a}^{} \pi(\left.\begin{matrix}
a\end{matrix}\right|s_{t+1})Q(s_{t+1}, a) - Q(s_{t}, a_{t}) \right ) $$

Function approximatation의 경우로 발전시키면 다음과 같이 된다.

$$ w_{t+1} = w_{t} + \alpha\left ( r_{t+1} + \gamma \sum_{a}^{} \pi(\left.\begin{matrix}
a\end{matrix}\right|s_{t+1}) \hat{Q}_{w_{t}}(s_{t+1}, a) - \hat{Q}_{w_{t}}(s_{t}, a_{t}) \right ) \bigtriangledown  \hat{Q}_{w_{t}}(s_{t}, a_{t}) $$

&nbsp;
\
&nbsp;

---

참고 내용 출처 :
* KAIST EE619 Mathmatical Foundations of Reinforcement Learning
* [https://stats.stackexchange.com/questions/340462/what-is-predicted-and-controlled-in-reinforcement-learning](https://stats.stackexchange.com/questions/340462/what-is-predicted-and-controlled-in-reinforcement-learning)