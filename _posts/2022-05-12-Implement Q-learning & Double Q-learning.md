---
title: Q-learning & Double Q-learning 구현
author: Bean
date: 2022-05-12 12:23:00 +0800
categories: [AI, RL]
tags: [RL, coding]
layout: post
current: post
class: post-template
subclass: 'post'
navigation: True
cover:  assets/img/post_images/ai_cover.jpg
---

## Q-learning과 Double Q-learning
&nbsp;

### Q-learning
Q-learning 은 Off-policy learning 알고리즘 중 하나이며 Off-policy learning 알고리즘 중에서도 TD Control Off-policy 방식이다. Off-policy learning 알고리즘 중 Off-policy MC와 Off-policy TD가 있지만 Importance sampling 문제 때문에 사용이 어렵다. Q-learning는 Importance sampling이 필요없는 방법으로 보다 유용하다. Q-learning이 유일한 TD Control Off-policy 방식은 아니지만 유명하고 잘 알려져 있는 이유는 구현이 매우 쉽기 때문이다.

Q-learning은 Bellman optimality backup operation을 샘플 기반으로 추산하여 optimal action-value function Q(s, a)를 찾아간다. 그리고 이 추산치는 아래 수식에서 볼 수 있듯이, behavior policy과 target policy의 영향을 받지 않고 단지 action-value function에만 영향을 받는다.

$$ Q(s, a) \leftarrow  Q(s, a) +  (r + max Q(s', a')-Q(s,a)) $$

따라서 Importance sampling(behavior policy과 target policy 사이의 상관관계를 지워주기 위해서 사용)을 할 이유 자체가 사라진다.

### Double Q-learning
언뜻보면 On-policy TD Control인 SARSA나 다른 Off-policy learning 알고리즘에 비해서 장점만 있어 보이지만, Maximization bias라는 단점이 존재한다. Maximization bias는 Q-Learner가 가치함수 Q(s, a)를 실제 값보다 높게 평가하는 문제이다. 이는 MDP의 stochasticity 때문에 발생한다.

이런 Maximization bias를 방지하기 위한 방법들도 여러가지가 개발이 되었는 데 그 중 하나가 Double Q-learning이다. Double Q-learning은 Maximization bias를 single estimator의 문제점으로 규정하고, 이에 대한 대안으로 double estimator를 제안한다.

Double Q-learning은 서로 독립된 두개의 Q-functions Q1(s, a)와 Q2(s, a)를 만들고 각각 1/2의 확률로 업데이트를 해준다. 이런 방식으로 알고리즘이 동작하면 Maximization bias가 어느 정도 해결됨이 알려져있다.

&nbsp;
## Q-learning과 Double Q-learning 구현
&nbsp;

본격적으로 Q-learning과 Double Q-learning을 구현해보자. 구현은 제공받은 Gridworld Class와 Q-learning, Deep Q-learning class의 skeleton을 활용하여, Q-learning, Deep Q-learning class의 method와 Gridworld 및 policy를 visualization 하는 함수를 구현하였다.

### Gridworld visualization
학습을 돌린 Grid world의 환경을 확인하기 위하여 Grid world 인스턴스를 파라미터로 받는 `paint_maps()` 함수를 먼저 구현해주었다.

```python
def paint_maps(env, savefig=True):
   plt.figure(figsize=(env.row_max, env.col_max))
   plt.title('Grid World', fontsize=20)

   # Placing the initial state on a grid for illustration
   initials = np.zeros([env.row_max, env.col_max])
   initials[env.row_max - 1, 0] = 1

   # Placing the trap states on a grid for illustration
   traps = np.zeros([env.row_max, env.col_max])
   for t in env.terminal_states:
       if t != (0, env.col_max - 1):
           traps[t] = 2

   # Placing the terminal state on a grid for illustration
   terminals = np.zeros([env.row_max, env.col_max])
   terminals[(0, env.col_max - 1)] = 3

   # Make a discrete color bar with labels
   labels = ['States', 'Initial\nState', 'Trap\nStates', 'Terminal\nState']
   colors = {0: '#F9FFA4', 1: '#B4FF9F', 2: '#FFA1A1', 3: '#FFD59E'}

   cm = ListedColormap([colors[x] for x in colors.keys()])
   norm_bins = np.sort([*colors.keys()]) + 0.5
   norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
   ## Make normalizer and formatter
   norm = matplotlib.colors.BoundaryNorm(norm_bins, len(labels), clip=True)
   fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])

   diff = norm_bins[1:] - norm_bins[:-1]
   tickz = norm_bins[:-1] + diff / 2
   plt.imshow(initials + traps + terminals, cmap=cm, norm=norm)
   plt.colorbar(format=fmt, ticks=tickz)

   plt.xlim((-0.5, env.col_max - 0.5))
   plt.ylim((env.row_max - 0.5, -0.5))
   plt.yticks(np.linspace(env.row_max - 0.5, -0.5, env.row_max + 1))
   plt.xticks(np.linspace(-0.5, env.col_max - 0.5, env.col_max + 1))
   plt.grid(color='k')

   for loc in env.terminal_states:
       plt.text(loc[1], loc[0], 'X', ha='center', va='center', fontsize=40)
   plt.text(0, env.row_max - 1, 'O', ha='center', va='center', fontsize=40)

   if savefig:
       plt.savefig('./gridworld.png')
```

코드 실행 결과로 확인한 Gridworld는 다음과 같이 생겼다.
<div style="text-align: left" width="100%">
   <img src="/assets/img/post_images/gridworld.png" width="100%"/>
</div>


### Q-learning 구현
다음으로 Q-learning의 각 method를 구현하였다. 구현 상세 내용은 코드 내 주석을 통해 나타내었다.

```python
class Q_learning():
   def get_Q_table(self):
       return self.Q_table

   def action(self, state):
       prob = np.random.uniform(0.0, 1.0, 1)
       # epsilon의 확률로 random하게 action을 선택
       if prob <= self.epsilon:
           action_index = np.random.choice(range(4))
       # 1-epsilon의 확률로 해당 state에서 가장 높은 Q값을 추산하고 있는 action을 선택
       else:
           action_index = self.Q_table[state].argmax()

       return self.actions[action_index]

   def update(self, current_state, next_state, action, reward):
       s, a, r, ns = current_state, action, reward, next_state
       # action의 list index를 계산
       a_index = self.actions.index(a)
       # Q-Learning target : reward + gamma * (다음 state에서 추산하고 있는 가장 높은 Q값)
       td_target = r + self.gamma * self.Q_table[ns].max()
       # update : Q-Learning target에서 Q(s, a)를 뺀값을 업데이트 해준다.
       self.Q_table[s][a_index] += self.alpha * (td_target - self.Q_table[s][a_index])

   def get_max_Q_function(self):
       max_Q_table = np.zeros((self.env_row_max, self.env_col_max))
       # 가능한 모든 state를 돌면서 최대 Q값을 찾아 max_Q_table에 업데이트 해준다.
       for i in range(self.env_row_max):
           for j in range(self.env_col_max):
               max_Q_table[i, j] = self.Q_table[(i, j)].max()

       return max_Q_table
```

Q-learning 학습이 잘 되었는 지 직관적으로 확인하기 위하여 10000번 에피소드 마다의 학습된 policy를 도식화해보았다. 점점 더 나은 policy로 잘 가고 있는 것을 확인할 수 있다.

<div style="text-align: left" width="100%">
   <img src="/assets/img/post_images/q_learning_policy.png" width="100%"/>
</div>

이 policy를 그리기 위해 구현한 코드는 다음과 같다.

```python
def plot_policy(ax, env, pi):
   d_symbols = ['↑', '→', '↓', '←']
   colors = {0: '#F9FFA4', 1: '#B4FF9F', 2: '#FFA1A1', 3: '#FFD59E'}
   # Placing the initial state on a grid for illustration
   initials = np.zeros([env.row_max, env.col_max])
   initials[env.row_max - 1, 0] = 1

   # Placing the trap states on a grid for illustration
   traps = np.zeros([env.row_max, env.col_max])
   for t in env.terminal_states:
       if t != (0, env.col_max - 1):
           traps[t] = 2

   # Placing the terminal state on a grid for illustration
   terminals = np.zeros([env.row_max, env.col_max])
   terminals[(0, env.col_max - 1)] = 3

   cm = ListedColormap([colors[x] for x in colors.keys()])
   ax.imshow(initials+traps+terminals, interpolation='nearest', cmap=cm)

   ax.set_xticks(np.linspace(-0.5, env.col_max - 0.5, env.col_max + 1))
   ax.set_yticks(np.linspace(-0.5, env.col_max - 0.5, env.col_max + 1))
   for edge, spine in ax.spines.items():
       spine.set_visible(False)

   ax.set_xticklabels([])
   ax.set_yticklabels([])
   ax.tick_params(axis='x', colors='w')
   ax.tick_params(axis='y', colors='w')

   ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
   ax.tick_params(which="minor", bottom=False, left=False)

   for i in range(env.col_max):
       for j in range(env.row_max):
           direction = pi[(j, i)].argmax()
           direction = d_symbols[direction]
           ax.text(i, j, direction, ha="center", va="center", color="black", fontsize=15)
```

### Double Q-learning 구현
마찬가지로 Double Q-learning의 각 method를 구현하였다. 구현 상세 내용은 코드 내 주석을 통해 나타내었다.

```python
class Double_Q_learning():
   def get_Q_table(self):
       return self.Q_table

   def action(self, state):
       prob = np.random.uniform(0.0, 1.0, 1)
       # epsilon의 확률로 random하게 action을 선택
       if prob <= self.epsilon:
           action_index = np.random.choice(range(4))
       # 1-epsilon의 확률로 해당 state에서 가장 높은 Q값을 추산하고 있는 action을 선택
       # 이 때 기준이 되는 Q값은 Q1, Q2값의 합
       else:
           self.Q_table[state] = self.Q1[state] + self.Q2[state]
           action_index = self.Q_table[state].argmax()
       return self.actions[action_index]

   def update(self, current_state, next_state, action, reward):
       s, a, r, ns = current_state, action, reward, next_state
       # 0.5의 확률로 Q1을 업데이트
       if np.random.rand() < 0.5:
           # action의 list index를 계산
           a_index = self.actions.index(a)
           # Q1 table에서 next state에서 가장 큰 Q값을 가지고 있는 action의 index를 계산
           q1_a_index = self.Q1[ns].argmax()
           self.Q1[s][a_index] += self.alpha * (r + self.gamma * self.Q2[ns][q1_a_index] - self.Q1[s][a_index])
       # 나머지 0.5의 확률로 Q2를 업데이트
       else:
           # action의 list index를 계산
           a_index = self.actions.index(a)
           # Q2 table에서 next state에서 가장 큰 Q값을 가지고 있는 action의 index를 계산
           q2_a_index = self.Q2[ns].argmax()
           self.Q2[s][a_index] += self.alpha * (r + self.gamma * self.Q1[ns][q2_a_index] - self.Q2[s][a_index])

   def get_max_Q_function(self):
       max_Q_table = np.zeros((self.env_row_max, self.env_col_max))
       # 가능한 모든 state를 돌면서 최대 Q값을 찾아 max_Q_table에 업데이트 해준다. 이 때, Q1, Q2 중 Q2값을 기준으로 하였다.
       for i in range(self.env_row_max):
           for j in range(self.env_col_max):
               if (i, j) in self.Q2:
                   max_Q_table[i, j] = self.Q2[(i, j)].max()

       return max_Q_table
```

마찬가지로 Double Q-learning 학습이 잘 되었는 지 직관적으로 확인하기 위하여 10000 에피소드마다 policy를 그려보았다. Double Q-learning policy를 그리는 데 Q-learning policy를 그릴 때와 같은 코드를 사용하였다.

<div style="text-align: left" width="100%">
   <img src="/assets/img/post_images/double_q_learning_policy.png" width="100%"/>
</div>

&nbsp;
## 구현 결과 확인
&nbsp;

구현을 완료한 이후 학습된 두 모델의 max Q-function을 뽑아 plot 해보았다. 이 결과를 plot하는 데에는 제공받은 함수를 이용하였다.

<div style="text-align: left" width="100%">
   <img src="/assets/img/post_images/q-learning.png" width="100%"/>
</div>

이 그래프를 살펴보면 Q-learning의 max Q value과 Double Q-learning의 값이 최대 50정도까지 차이가 난다. 전반적으로 Q-learning에서 Q value값이 Double Q-learning에서의 값보다 overestimate 되었음을 확인할 수 있다.

