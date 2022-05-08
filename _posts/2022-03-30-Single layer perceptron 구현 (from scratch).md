---
title: Implement Single layer perceptron (from scratch)
author:
  name: Bean
  link: https://github.com/beanie00
date: 2022-03-30 09:32:00 +0800
categories: [AI, basic]
tags: []
---

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