---
title: Deep Q-Network(DQN) 구현
author: Beanie
date: 2022-05-16 12:02:00 +0800
categories: [AI, RL]
tags: [RL, coding]
layout: post
current: post
class: post-template
subclass: 'post'
navigation: True
cover:  assets/img/post_images/ai_cover.jpg
---

&nbsp;
## DQN 이란?
&nbsp;
이전에 [(Reinforcement learning) Value Function Approximation 포스팅]()에서 Value function approximation에 대하여 다루었다. 요약하면 강화학습에서 간단한 table 형태로 학습을 하게 되면 학습이 극도로 느려지는 문제가 있어 value function을 다양하게 근사하여 활용한다.

이러한 approximator로 Neural Network를 사용할 수도 있다. 이번 글에서 소개할 Deep Q-Networks(DQN)은 Q-learning 알고리즘의 Q(Action-value) 함수를 딥러닝으로 근사하는 알고리즘으로, DeepMind에서 발표한 [`Playing Atari with Deep Reinforcement`](https://arxiv.org/pdf/1312.5602.pdf) 라는 연구에서 제안되었다.

이 DQN 논문에서는 raw pixel을 input으로 받아, value function(≈future rewards)를 output으로 반환하는 Q-Learning의 parameter를 학습하는 Convolutional Neural Network(CNN)을 사용하였다. 즉, 고차원의 sensory input을 통해 control policies를 다이렉트로 학습하는 Deep learning model이라고 볼 수 있다.

그렇다면 이전에는 왜 강화학습에 딥러닝을 활용하지 못했을까? 사실 딥러닝과 강화학습이 고차원의 데이터를 활용하는 방법은 무척 다르다. 몇가지 예시로
* Deep-Learning 기반 방법들은 hand-labelled training dataset을 필요로 하지만, Reinforcement Learning에서는 오로지 delay와 노이즈가 포함된 스칼라값인 Reward만을 통해서 학습된다. 더욱이나 그 reward 조차 sparse, noisy, and delay한 성격을 가진다.
* Deep Learning에서는 data sample이 i.i.d하다는 가정이 있다. 하지만, Reinforcement Learning의 경우 현재의 state가 어디인 지에 따라 가능한 다음 state가 결정된다. 즉, 독립적이지 않고 종속적이며, state간의 correlation(상관관계)또한 크다.
* 에이전트가 학습함에 따라 policy가 달라지면서, 학습 데이터의 분산(distribution) 자체도 시간에 따라 변한다. 반면 딥러닝은 고정된 기본 분산(fixed underlying distribution)을 가정한다.

이러한 강화학습에 이유로 딥러닝 기법을 이전에는 바로 적용하지 못하였다.

`Playing Atari with Deep Reinforcement` 논문도 예외없이 이러한 문제에 직면했는데, 논문에서는 이 문제를 해결하기 위하여 **Experience replay** 와 **Target network** 라는 방법을 제안하였다. 이 두 특징적인 방법을 통하여 성공적으로 딥러닝을 강화학습에 적용하여 고차원 데이터를 처리할 수 있게 되었다.

### Experience replay
들어오는 입력을 순서대로 사용하면 데이터 간의 연관성이 너무 커지게 된다. 따라서 최근 n개의 데이터를 계속해서 저장하고, 네트워크를 학습할 때는 저장한 데이터 중 몇개를 무작위로 샘플하여 사용한다.

### Target network
Target network 설명은 [https://jsideas.net/dqn/](https://jsideas.net/dqn/)의 당나귀 예시가 이해하기 좋았다. 이 블로그의 설명을 차용하면

>일반적인 Q Learning은 당나귀 뒤에 올라타 낚시대로 당근을 드리우고 당나귀가 곧게 걷기를 바라는 것과 같다. 당근을 든 손을 곧게만 유지하면 당나귀가 직진할 것이라고 생각하지만, 실제로는 잘 안된다. 당나귀와 사람, 그리고 낚시대와 당근이 모두 연결되어 있기에, 당나귀가 움직이면 올라탄 사람도 흔들리고 그에 따라 당근도 흔들린다. 결국 영상에서처럼 당나귀는 직선으로 이동하는데 실패한다.

>당근의 위치를 Q 함수의 타겟 $(r+max_{a'}Q(s^{'}, a^{'}))$ 으로, 당나귀의 움직임을 추정치(Q)로 대입해보면 된다. 타겟과 추정치의 오차를 줄여야하는데, Q의 변화에 따라 타겟과 추정치가 모두 함께 변화하면 안정적인 학습(이동)이 어려워진다.

>DQN에서는 당나귀와 당근을 분리시키는 Fixed Q Targets 방법을 사용해서 문제를 해결한다. Q함수를 추정하는 네트워크(local network)와 Target을 설정하는데 사용하는 네트워크(target network)로 추정과 평가를 분리한다. 당나귀 등에서 내려서 낚시대를 드리우면, 당근의 위치는 더이상 당나귀의 움직임에 영향을 받지 않는다.

> 그리고 target network의 업데이트 주기를 local network보다 더 느리게 만듦으로써 목표가 자주 휘청이지 않도록 한다. DQN 구현에서는 local network가 4번 업데이트될 때 한번씩 target network의 파라미터를 local network의 파라미터를 사용해 soft update한다.

&nbsp;
## DQN 구현
&nbsp;

### Environment
먼저, 이번 DQN 구현은 제공받은 2개의 GridWorld 환경을 기반으로 진행하였다.

먼저 첫번째 GridWorld는 그림과 같다.

<div style="text-align: left">
  <img src="/assets/img/post_images/dqn4.png" width="50%"/>
</div>

그림에서 볼 수 있듯이 이 환경은 노란 동그라미, 빨간 네모, 녹색 네모로 구성되어 있다. 각각을 살펴보면,

* 노란 동그라미 : agent의 움직임을 나타낸다.
* 빨간 네모 : 폭탄이 위치해 있으며 agent가 해당 위치로 가면 reward -10을 얻고 다시 episode를 시작한다.
* 초록 네모 : 보물이 위치해 있으며 agent가 해당 위치로 가면 reward 20을 얻고 다시 episode를 시작한다.
* 빨간 네모, 초록 네모가 없는 공간을 노란 동그라미가 탐색할 때마다 reward -0.1을 얻는다.

두번째 GridWorld는 첫번째 GridWorld에 주황 네모가 추가되었다.

<div style="text-align: left">
  <img src="/assets/img/post_images/dqn5.png" width="45%"/>
</div>

* 주황 네모 : 음식 위치해 있으며 agent가 해당 위치로 가면 reward 10을 얻고 다시 episode를 시작한다.


### DQN class 구현

이러한 환경에서 잘 동작할 수 있는 DQN을 구현해보자.
DQN 구현은 **neural network 생성**, **action 선택**, **experience replay를 위하여 이전 experience를 메모리에 저장**, **메모리에 포함된 experience를 이용하여 학습**의 과정으로 나눠볼 수 있다.

* **neural network 생성** (`_construct_network()` 함수)

    <div style="text-align: left">
    <img src="/assets/img/post_images/dqn3.jpeg" width="80%"/>
    </div>

    이 DQN 모델에서 구현한 neural network는 위 다이어그램과 비슷하다. 먼저 input layer에서 n_features size의 입력을 받고 24 크기로 출력을 한다. 이 때, activation 함수로 relu 함수를 사용하였다. 또한 Hidden layer에서는 이전 레이어의 출력을 인풋으로 받아 24 크기로 출력을 한다. 이 때도 마찬가지로 activation 함수로 relu를 사용하였다. 마지막 output layer에서는 n_actions 크기로 출력을 하고 이 때, activation 함수를 linear로 설정하였다. 이렇게 구현한 모델의 코드는 아래와 같다.

    ```python
    model = Sequential()

    model.add(Dense(24, input_dim=self.n_features, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(self.n_actions, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
    ```

    이 모델에 n_features 크기의 state 벡터가 들어오면 모델의 prediction으로 얻은 예측된 reward 값과 true reward 값의 에러가 작아지도록 내부 weight 값을 조정한다. 이를 계속 반복하다보면 예측된 reward가 true reward와 같아지는 방향으로 (최적 policy를 찾는 방향으로) 학습되게 된다.

    좀 더 구체적으로 학습 과정을 `model.fit()`과 `model.predict()` 함수로 다시 나눌 수 있다. 각각을 살펴보면,
    * `model.fit(state, true_reward, epochs=1, verbose)`
        * 주어진 state에서 true reward를 예측하도록 학습한다.
    * `model.predict(state)`
        * unseen input에 대하여 reward를 예측한다.

    이 `model.fit()`과 `model.predict()`를 학습 과정에서 활용하며 최적 policy를 찾아나갔다.



* **action 선택** (`choose_action()` 함수)
    * e_greedy 확률로는 random 하게 action을 선택하고, 1 - (e-greedy) 확률로는 모델에서 예측한 reward가 가장 큰 action을 선택한다.

* **experience를 메모리에 저장** (`store_transition()` 함수)
    * experience replay 방식으로 experience를 재활용 위하여 학습을 메모리에 저장해둔다.

* **메모리의 experience를 이용하여 학습** (`learn()` 함수)


이 과정을 통합하여 구현된 전체 DQN Class 코드는 아래와 같다.

```python
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

EPISODES = 1000

np.random.seed(1)

class DeepQLearning:
   def __init__(
           self,
           n_actions,
           n_features,
           learning_rate=0.01,
           discount_factor=0.9,
           e_greedy=0.05,
           replace_target_iter=300,
           memory_size=500,
           batch_size=32
   ):
       self.n_actions = n_actions
       self.n_features = n_features
       self.learning_rate = learning_rate
       self.discount_factor = discount_factor
       self.memory = deque(maxlen=memory_size)
       self.batch_size = batch_size
       self.e_greedy = e_greedy
       self.replace_target_iter = replace_target_iter
       self.model = self._construct_network()

   def _construct_network(self):
       model = Sequential()

       model.add(Dense(24, input_dim=self.n_features, activation='relu'))
       model.add(Dense(24, activation='relu'))
       model.add(Dense(self.n_actions, activation='linear'))
       model.compile(loss='mse',
                     optimizer=Adam(lr=self.learning_rate))
       return model

   def store_transition(self, s, a, r, next_s, t):
       self.memory.append((s, a, r, next_s, t))

   def choose_action(self, state):
       # e_greedy 확률로는 random 하게 action을 선택하고, 1 - (e-greedy) 확률로는 모델에서 예측한 reward가 가장 큰 action을 선택한다.
       state = np.reshape(state, [1, self.n_features])

       # e_greedy 확률로 random 하게 action을 선택
       if np.random.rand() <= self.e_greedy:
           return random.randrange(self.n_actions)

       #  1 - (e-greedy) 확률로 모델을 통해 예측한 reward가 가장 큰 action을 선택
       act_values = self.model.predict(state)
       return np.argmax(act_values[0])  # returns action

   def learn(self):
       # 메모리에 저장된 experience에서 batch_size만큼 랜덤하게 선택
       minibatch = random.sample(self.memory, self.batch_size)
       # 선택된 각각의 experience에 대하여 모델이 실제 experience로 얻은 reward 방향으로 값을 잘 예측하도록 학습을 진행
       for state, action, reward, next_state, done in minibatch:
           state = np.reshape(state, [1, self.n_features])
           next_state = np.reshape(next_state, [1, self.n_features])
           target = reward
           if not done:
               target = (reward + self.discount_factor * np.amax(self.model.predict(next_state)[0]))
           target_f = self.model.predict(state)
           target_f[0][action] = target
           self.model.fit(state, target_f, epochs=1, verbose=0)
```

&nbsp;
## DQN Class 실행
&nbsp;

이제 이렇게 구현한 DQN Class를 활용하여 학습을 진행해보았다.
총 300개의 episode에 대하여 DQN을 학습시키고 average episode return을 그려보았다.
이 때, return 값이 새롭게 학습을 돌릴 때 마다 달라지기 때문에 300개의 episode로 학습하는 과정 전체를 5번 반복해서 episode reward의 min-max, mean를 나누어 plot하였다.

```python
import numpy as np

from env import Robot_Gridworld
from env2 import Robot_Gridworld as Robot_Gridworld2
from deep_q_learning import DeepQLearning
import matplotlib.pyplot as plt
from collections import deque

return_value = 0
gamma = 0.99
episode_return_list=[]
def update():
   global return_value
   step = 0
   reward_per_episode = []
   returns = deque(maxlen=100)

   for episode in range(300):

       state = env.reset()
       step_count = 0

       while True:
           return_value = 0
           env.render()
           action = dqn.choose_action(state)
           next_state, reward, terminal = env.step(action)
           return_value = reward + gamma * return_value

           step_count += 1
           dqn.store_transition(state, action, reward, next_state, terminal)

           if (step > 200) and (step % 5 == 0):
               dqn.learn()
           state = next_state

           if terminal == True:
               print(" {} End. Total steps : {}\n".format(episode + 1, step_count))
               break

           step += 1

       returns.append(return_value)
       reward_per_episode.append(np.mean(returns))
   episode_return_list.append(reward_per_episode)

   print('Game over.\n')
   env.destroy()


if __name__ == "__main__":
   for i in range(5):
       env = Robot_Gridworld()

       dqn = DeepQLearning(env.n_actions, env.n_features,
                           learning_rate=0.01,
                           discount_factor=0.9,
                           e_greedy=0.05,
                           replace_target_iter=200,
                           memory_size=2000
                           )


       env.after(100, update) #Basic module in tkinter
       env.mainloop() #Basic module in tkinter

   plt.figure()
   mean_return = [x.mean() for x in np.array(episode_return_list).T]
   min_return = [x.min() for x in np.array(episode_return_list).T]
   max_return = [x.max() for x in np.array(episode_return_list).T]
```

학습 결과는 다음과 같다.

* 첫번째 Gridworld (주황 네모 X)
    <div style="text-align: left">
    <img src="/assets/img/post_images/dqn1.png" width="100%"/>
    </div>

* 두번째 Gridworld (주황 네모 O)
    <div style="text-align: left">
    <img src="/assets/img/post_images/dqn2.png" width="100%"/>
    </div>

그래프를 비교해보면 두번째 Gridworld로 돌렸을 때, 보다 빨리, 적은 분산으로 최적 policy를 찾아가는 것으로 보인다. 하지만 두번째 Gridworld는 리워드 10을 주는 sub optimal한 경우가 추가 되었기 때문에 자칫하면 리워드 20을 주는 보물을 찾아가는 대신에 10을 주는 sub optimal에 빠질 수 있다. 그럴 경우 batch size를 키우거나, 학습이 진행되어 감에 따라 learning rate 서서히 감소시킴으로써 sub optimal에 빠지는 것을 방지할 수 있다.

\
&nbsp;

---

참고 내용 출처 :
* [http://wiki.hash.kr/index.php/DQN](http://wiki.hash.kr/index.php/DQN)
* [https://velog.io/@sjinu/%EA%B0%9C%EB%85%90%EC%A0%95%EB%A6%AC-7.-DQNDeep-Q-NEtwork](https://velog.io/@sjinu/%EA%B0%9C%EB%85%90%EC%A0%95%EB%A6%AC-7.-DQNDeep-Q-NEtwork)