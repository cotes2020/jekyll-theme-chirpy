---
title: Deep Q-Network(DQN) 구현
author: Bean
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

    좀 더 구체적으로 학습 과정을 model.fit()과 model.predict() 함수로 다시 나눌 수 있다. 각각을 살펴보면,
    model.fit(state, true_reward, epochs=1, verbose)
    주어진 state에서 true reward를 예측하도록 학습한다.
    model.predict(state)
    unseen input에 대하여 reward를 예측한다.


* **action 선택** (`choose_action()` 함수)

* **experience를 메모리에 저장** (`store_transition()` 함수)

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
       state = np.reshape(state, [1, self.n_features])
       if np.random.rand() <= self.e_greedy:
           return random.randrange(self.n_actions)
       act_values = self.model.predict(state)
       return np.argmax(act_values[0])  # returns action

   def learn(self):
       minibatch = random.sample(self.memory, self.batch_size)
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