---
title: Bidirectional Associative Memory 구현
author: Bean
date: 2022-05-16 10:47:00 +0800
categories: [AI, basic]
tags: [AI]
layout: post
current: post
class: post-template
subclass: 'post'
navigation: True
cover:  assets/img/post_images/ai_cover2.jpg
---

## Bidirectional Associative Memory (BAM) 이란?
\
&nbsp;
BAM 은 Hopfield model을 확장한 모델이다.
여기서 또 Hopfield model이라는 것이 등장하는 데(..), 잠시 정리하고 넘어가자.

### Hopfield model
인공지능에서 가장 많이 사용되는 퍼셉트론(perceptron) 등은 학습 과정에서 weight가 점점 업데이트 되면서 최적 weight를 찾아간다. 이와 다르게 Hopfield network는 고정된 weight를 이용하여 정보를 연상한다.
Hopfield network는 학습 패턴에 대해 계산된 고정 가중치 행렬을 저장하고, 입력 패턴이 들어올 때마다 가중치 행렬을 이용하여 입력 패턴에 대한 학습 패턴을 연상한다. 홉필드 네트워크 알고리즘은 아래와 같다.

1) 학습 패턴에 양극화 연산을 적용

2) 학습 패턴에 대한 홉필드 네트워크의 가중치 행렬을 계산

3) 계산된 가중치 행렬을 저장

4) 입력 패턴이 들어오면 저장된 가중치 행렬을 이용하여 입력 패턴에 대한 학습 패턴을 연상

### BAM
BAM은 이런 Hopfield model을 양방향으로 패턴이 연상 가능하도록 확장한 것이다. 즉, A와 B 두개의 패턴이 있을 때, A로부터 B를 연상할 뿐 아니라, 반대로 B로부터 A를 연상할 수도 있다. 이런 양방향 특징 때문에 다양한 분야에 폭 넓게 응용된다.

여기서 계속 **연상** 이라는 말을 사용하고 있는데 **연상** 을 어떻게 이해하면 될까?

가벼운 예시로 우리는 밖에서 강아지를 만나면 '어? 강아지다!' 하는 생각을 한다. 강아지의 모습으로 부터 '강아지'라는 단어가 연상되었기 때문이다. 여기서 말하는 연상도 이와 유사하게 이해하면 된다. 확장하여 **양방향 연상** 은 강아지를 보고 '강아지'라는 단어를 연상하는 것에 더해 '강아지'라는 단어를 보고 강아지의 모습을 연상하는 것으로 이해해볼 수 있다.

그림으로 살펴보면, 아래 그림에서 BAM은 linear associator과 매우 유사하지만 자세히 보면 화살표 촉이 양쪽으로 연결되어 있는 것을 확인할 수 있다. 그리고 두 패턴의 집단(그림에서는 x, y로 표기됨)이 fully connected 되어 있음도 볼 수 있다. X에 포함된 요소들이 Y의 요소들과 완전히 연결되고, Y의 요소들도 X의 요소들로 완전히 연결된다.

<div style="text-align: left">
  <img src="/assets/img/post_images/bam1.png" width="100%"/>
</div>

이 때, 두 집합을 서로 연결하여 각각을 연상해내기 위해서는 두 집합의 상관관계를 알아야 한다. 이 상관관계를 구하는 식은 다음과 같다.

$$ w_{ji} = \sum_{s=1}^{S} x_{i}^{s} y_{j}^{s} $$

이 행렬은 BAM에서 매우 중요한데, 바로 BAM의 학습식이 된다! 바로 이 행렬이 각 집합으로 부터 나머지 집합을 연상하는 데 사용된다.

$$ y_{j} =  \left ( \sum_{i=1}^{I} w_{ji} x_{i} \right ), ~~x_{i} =  \left ( \sum_{j=1}^{J} w_{ji} y_{j} \right ) $$

\
&nbsp;
&nbsp;
## BAM 구현
&nbsp;

이제 이 내용을 기반으로 본격적으로 BAM을 구현해보자. class method 중 `update_weight()`는 위의 수식에서 $w_{ji}$ 를, `feedForward()`는 $y_{j}$, `feedBackward()`는 $x_{i}$ 를 구하는 식을 파이썬으로 구현한 함수이다.

### BAM class 구현

```python
import random as rand

class BAM:
  def __init__(self, n, p, random=False):
    self.row_count = n
    self.col_count = p
    self.weight_matrix = self.make_new_weight_matrix()

  def make_new_weight_matrix(self):
    return [ [0.0]*self.col_count for i in range(self.row_count) ]

  def update_weight(self, input, output):
    self.weight_matrix = self.make_new_weight_matrix()
    for i in range(len(input)):
      sample_in = input[i]
      sample_out = output[i]
      for r in range(self.row_count):
        for c in range(self.col_count):
          self.weight_matrix[r][c] += sample_in[r] * sample_out[c]

  # Generate x from y
  def feedForward(self, input):
    result = ( np.mat(input) * np.mat(self.weight_matrix) ).tolist()[0]
    result = map(lambda x: 1 if x>0 else -1, result)
    return list(result)

  # Generate y from x
  def feedBackward(self, output):
    result = ( np.mat(output) * np.matrix.transpose(np.mat(self.weight_matrix)) ).tolist()[0]
    result = map(lambda x: 1 if x>0 else -1, result)
    return list(result)

  def computeEnergy(self, input, output):
    e = 0.0
    for r in range(len(input)):
        for c in range(len(output)):
            e += self.weight_matrix[r][c] * input[r] * output[c]
    return -1 * e
```

&nbsp;

### 데이터 생성
이제 본격적으로 BAM으로 학습을 시켜보자. 학습을 위해서 데이터를 생성해준다.
학습 데이터셋은 S개의 $(x^{s}, y^{s})$ 로 이루어져 있다. 이 때 $x^{s}$ 는 1024개의 element, $y^{s}$ 512개의 element로 구성된다. 각각의 element는 50% 확률로 1, -1의 값을 가진다.

또한, 다양한 비교를 위하여 S가 각각 50, 100, 200일 때 데이터 D50, D100, D200 를 생성하였다.

```python
import numpy as np

# S=50
x50 = np.random.choice([-1, 1], (50, 1024))
y50 = np.random.choice([-1, 1], (50, 512))

# S=100
x100 = np.random.choice([-1, 1], (100, 1024))
y100 = np.random.choice([-1, 1], (100, 512))

# S=200
x200 = np.random.choice([-1, 1], (200, 1024))
y200 = np.random.choice([-1, 1], (200, 512))
```

&nbsp;

### BAM 학습 (노이즈가 없는 경우)
이제 정말로 BAM으로 연상을 시작해보자. S개의 데이터 각각을 input, output으로 하여 총 10번의 iteration 동안 `feedForward()`, `feedBackward()` 과정을 반복한 뒤, 계산된 output과 true output의 차이를 비교하였다.

```python
import matplotlib.pyplot as plt
import copy

bam = BAM(1024, 512)

def train_wo_noise(num_iteration, input, output):
    error_list = [0 for i in range(10)]
    bam.update_weight(input, output)
    for i in range(len(input)):
      test_input = input[i]
      test_output = output[i]
      for n in range(num_iteration):
        y = bam.feedForward(test_input)
        error_list[n] += sum([1 if test_output[i] != y[i] else 0 for i in range(len(y))])
        test_input = bam.feedBackward(y)

    # plot error
    plt.plot([e / 50 for e in error_list])
    plt.ylabel('number of different elements')
    plt.xlabel('epoch')
    plt.title('Average of all output errors')
    plt.show()

train_wo_noise(10, x50, y50)
```

위의 코드를 실행시키면 아래의 그래프가 출력된다.

<div style="text-align: left">
  <img src="/assets/img/post_images/bam2.png" width="100%"/>
</div>

noise가 없는 경우에는 에러 없이 output이 잘 연상되는 것을 볼 수 있다.


&nbsp;

### BAM 학습 (노이즈가 있는 경우)
이번에는 input에 노이즈를 주고 위의 과정을 다시 진행해보았다. 각 데이터셋마다 $x^{10}$ 에 노이즈를 준다. 노이즈를 주기 위해서 $x^{10}$ 의 특정 개수의 element에 -1 ~ 1사의의 랜덤한 값을 더하였다.
그리고 다양한 비교를 위하여 노이즈를 주는 element의 갯수를 0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100으로 나누어 진행해주었다.

```python
def make_noise_input(num_noise, input):
  randomlist = rand.sample(range(0, 1024 - 1), num_noise)
  for r in randomlist:
    if input[r] > 0:
      input[r] = -1
    else:
      input[r] = 1
  return input

def train_w_noise(num_noise_list, num_iteration, input, output, num_elements):
  total_error_list = []
  bam.update_weight(input, output)

  # noise 갯수 별로 반복해서 train
  for num_noise in num_noise_list:
    error_list = [0 for i in range(10)]
    # x^10에 대해 10개의 서로 다른 noisy input을 생성하고 실행
    for i in range(10):
      test_input = make_noise_input(num_noise, input[10])
      test_output = output[10]
      # num_iteration 수 만큼 반복하여 실행
      for n in range(num_iteration):
        y = bam.feedForward(test_input)
        error_list[n] += sum(1 for i, j in zip(test_output, y) if i != j)
        test_input = bam.feedBackward(y)
    total_error_list.append([e / 10 for e in error_list])

  # plot error
  for i in range(len(total_error_list)):
    plt.plot(total_error_list[i], label='{} noisy elements'.format(num_noise_list[i]))
  plt.ylabel('number of different elements')
  plt.xlabel('epoch')
  plt.title('[D{}] Average of all output errors with noisy input'.format(num_elements))
  plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  plt.show()

num_noise_list = [0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50]
train_w_noise(num_noise_list, 10, x50, y50)
train_w_noise(num_noise_list, 10, x100, y100)
train_w_noise(num_noise_list, 10, x200, y200)
```

코드를 실행시키면 아래의 그래프가 출력된다.

<div style="text-align: left">
  <img src="/assets/img/post_images/bam3.png" width="100%"/>
</div>

<div style="text-align: left">
  <img src="/assets/img/post_images/bam4.png" width="100%"/>
</div>

에러가 없던 noise 없는 경우와는 달리, noise element가 많을 수록 에러가 커진다. 이 에러는 iteration이 진행되는 동안 점점 감소되는 양상을 보인다.
또한, D200인 경우가 D50인 경우보다 학습에 어려움이 있음도 확인할 수 있다.


\
&nbsp;

---

참고 내용 출처 :
* [http://www.aistudy.co.kr/neural/BAM.htm](http://www.aistudy.co.kr/neural/BAM.htm)
* [https://untitledtblog.tistory.com/7](https://untitledtblog.tistory.com/7)