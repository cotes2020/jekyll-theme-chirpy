---
title: Competitive learning 구현 (from scratch)
author: Beanie
date: 2022-03-24 09:32:00 +0800
categories: [AI, basic]
tags: [AI, coding]
layout: post
current: post
class: post-template
subclass: 'post'
navigation: True
cover:  assets/img/post_images/ai_cover2.jpg
---

## Competitive learning이란?
&nbsp;

Competitive learning(경쟁 학습)은 비지도 학습이다. Competitive learning은 입력 벡터들을 군집화(clustering) 하는데 사용된다. 아래 같은 경쟁 학습 네트워크가 있다고 하자. 이 네트워크는 4 차원 입력 벡터들을 세 클러스터로 군집화 한다. 한 클러스터의 중심(centroid)은 가중 벡터(weight vector) 하나로 표현된다. 경쟁 학습은 한 입력 벡터에서 모든 가중 벡터까지의 거리를 구하고, 그 중 거리가 가장 짧은 가중 벡터 하나만 선택한다. 그 가중 벡터가 승자 노드이다. 오직 승자 노드의 가중 벡터만 갱신된다. 그래서 경쟁 학습 네트워크는 승자독식 네트워크(winner-take-all networks)라고도 불린다.

<div style="text-align: left" width="100%">
   <img src="/assets/img/post_images/competitive1.png" width="100%"/>
</div>

&nbsp;
## Competitive learning 구현
&nbsp;

먼저 learning에 사용될 데이터 샘플을 생성해준다.

```python
def makeSample(S):
  N1 = 500
  N2 = 500
  X_1, X_2, Y_1, Y_2 = [[], [], [], []]
  fig, ax = plt.subplots()

  while (len(X_1) < N1 or len(X_2) < N2):
    if (len(X_1) < N1):
      x1 = np.random.uniform(-20, 20)
      y1 = np.random.uniform(-20, 20)
      if (y1 > 0 and (x1+4)**2+y1**2 > 36 and (x1+4)**2+y1**2 < 100):
        X_1.append(x1)
        Y_1.append(y1)
        S.append([x1, y1])
    if (len(X_2) < N2):
      x2 = np.random.uniform(-20, 20)
      y2 = np.random.uniform(-20, 20)
      if (y2 < 3 and (x2-4)**2+(y2-3)**2 > 36 and (x2-4)**2+(y2-3)**2 < 100):
        X_2.append(x2)
        Y_2.append(y2)
        S.append([x2, y2])

  ax.scatter(X_1,Y_1, color='red', alpha=0.1)
  ax.scatter(X_2,Y_2, color='blue', alpha=0.1)
  ax.grid(True)
  plt.title("Samples for competitive learning")
  plt.show()
  return [X_1, X_2, Y_1, Y_2]
```

아래는 competitive learning 구현 코드이다.

```python
def competitive_learning(S, dimension) :
  n_iterations = 300
  learning_rate = 0.6

  # initialize w
  w = np.random.rand(dimension, 2)

  error_list = []
  for n in range(n_iterations):
    e = 0
    if (n > 0):
      learning_rate = 0.4 * learning_rate
    for s in S:
      # get current cluster
      distance = []
      for i in range(dimension) :
        distance.append(np.linalg.norm(s-w[i]))
      cur_cluster_index = np.argmin(distance, axis=0)
      # update w
      e += (s[0] - w[cur_cluster_index][0])**2 + (s[1] - w[cur_cluster_index][1])**2
      w[cur_cluster_index] = w[cur_cluster_index] + learning_rate * (s - w[cur_cluster_index])
    error_list.append(e / len(S1))

  plt.plot(range(len(error_list)), error_list)
  plt.title("Learning curve; "+str(dimension)+ ' cluster')
  plt.xlabel('$epoch$')
  plt.ylabel('$error$')
  plt.show()
  return w
```

```python
def plot_cluster_result(w, dimension) :
  w_x, w_y = [[], []]
  for i in range(dimension) :
    w_x.append(w[i][0])
    w_y.append(w[i][1])
  plt.scatter(w_x,w_y, color='black', label='cluster mean', alpha=1)

  for s in S:
      # get current cluster
      distance = []
      for i in range(dimension) :
        distance.append(np.linalg.norm(s-w[i]))
      cur_cluster_index = np.argmin(distance, axis=0)
      if (cur_cluster_index == 0):
        plt.scatter(s[0],s[1], color='red', alpha=0.1)
      elif (cur_cluster_index == 1):
        plt.scatter(s[0],s[1], color='green', alpha=0.1)
      elif (cur_cluster_index == 2):
        plt.scatter(s[0],s[1], color='blue', alpha=0.1)
      elif (cur_cluster_index == 3):
        plt.scatter(s[0],s[1], color='yellow', alpha=0.1)
      elif (cur_cluster_index == 4):
        plt.scatter(s[0],s[1], color='violet', alpha=0.1)
      elif (cur_cluster_index == 5):
        plt.scatter(s[0],s[1], color='lime', alpha=0.1)
      elif (cur_cluster_index == 6):
        plt.scatter(s[0],s[1], color='darkorange', alpha=0.1)
      elif (cur_cluster_index == 7):
        plt.scatter(s[0],s[1], color='darkcyan', alpha=0.1)

  plt.xlim(-15,15)
  plt.ylim(-15,15)
  plt.title("Clustered results; "+str(dimension)+ ' cluster')
  plt.legend()
  plt.grid(True)
```

&nbsp;
## Competitive learning 결과
&nbsp;

```python
S = []
XY = makeSample(S)
w = competitive_learning(S, 6)
plot_cluster_result(w, 6)

w = competitive_learning(S, 8)
plot_cluster_result(w, 8)
```
위 코드를 돌리고 결과를 확인하면 clustering이 잘되는 것을 확인할 수 있다.

### 6개의 군집으로 분류
<div style="text-align: left">
   <img src="/assets/img/post_images/competitive2.png" width="100%"/>
</div>

### 8개의 군집으로 분류
<div style="text-align: left">
   <img src="/assets/img/post_images/competitive3.png" width="100%"/>
</div>

\
&nbsp;

***
참고 내용 출처 :
* [https://roboticist.tistory.com/516](https://roboticist.tistory.com/516)