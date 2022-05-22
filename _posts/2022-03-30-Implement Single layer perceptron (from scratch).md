---
title: Implement Single layer perceptron (from scratch)
author: Beanie
date: 2022-03-30 09:32:00 +0800
categories: [AI, basic]
tags: [AI, coding]
layout: post
current: post
class: post-template
subclass: 'post'
navigation: True
cover:  assets/img/post_images/ai_cover2.jpg
---

이번 글에서는 아래 그림과 같은 네트워크의 Single layer perceptron을 python 코드로 구현한 코드를 담았다.

<div style="text-align: left">
   <img src="/assets/img/post_images/single.png" width="70%"/>
</div>

\
&nbsp;
전체 코드는 아래와 같다. Single layer perceptron은 구현이 매우 간단한데, 실제 값과 예측값의 오차를 적당한 비율로 wight parameter에 더해가는 식으로 학습해주면 된다.

```python
class SingleLayerPerceptron(object):

def __init__(self, eta=0.01, epochs=50):
    self.eta = eta
    self.epochs = epochs

def train(self, X, y):
    self.w_ = np.zeros(X.shape[1])
    self.b_ = 0.5
    self.errors_ = []

    for _ in range(self.epochs):
        errors = 0
        for xi, target in zip(X, y):
            update = self.eta * (target - self.predict(xi))
            self.w_ +=  update * xi
            self.b_ +=  update
            errors += int(update != 0.0)
        self.errors_.append(errors)
    return self

def net_input(self, X):
    return np.dot(X, self.w_) + self.b_

def predict(self, X):
    return np.tanh(self.net_input(X))
```

\
&nbsp;

***

이미지 출처 :
 * [https://www.tutorialspoint.com/tensorflow/tensorflow_single_layer_perceptron.htm](https://www.tutorialspoint.com/tensorflow/tensorflow_single_layer_perceptron.htm)
