---
title: Implement Multi layer perceptron (from scratch)
author: Bean
date: 2022-04-06 09:32:00 +0800
categories: [AI, basic]
tags: []
layout: post
current: post
class: post-template
subclass: 'post'
navigation: True
cover:  assets/img/post_images/ai_cover2.jpg
---

이번 글에서는 아래 그림과 같은 네트워크의 2개의 hidden layer을 가진 Multi Layer Perceptron(MLP)을 python 코드로 구현한 코드를 담았다.

<div style="text-align: left">
   <img src="/assets/img/post_images/mlp.png" width="100%"/>
</div>

\
&nbsp;
바로 이전 글에서, Single layer perceptron은 오차를 적당한 비율로 weight vector에 더해주는 게 끝이었는데 MLP는 여러개의 layer가 있고, non-linear function도 추가되어 식이 더 복잡하다.

먼저 각각의 layer에 대해 feed forward 과정을 진행해주고, 이 값을 기반으로 backpropagation을 수행해준다.

Backpropagation은 아래의 계산된 Backpropagation error를 weight vector에 적당한 비율로 더해주며 진행되었다.

$$E\equiv \sum_{i=1}^{M}E_{i}=\frac{1}{2}\sum_{s=1}^{S}\sum_{i=1}^{M}(t_{i}^{s}-y_{i}^{s})^{2}$$

$$W_{ij}^{(123)}[n+1]=W_{ij}^{(123)}[n]-\eta ^{(123)}[n]\frac{\partial E}{\partial W_{ij}^{(123)}}[n]$$

각각의 레이어에서 error를 계산하면 다음과 같다.

$$\frac{\partial E}{\partial W_{ij}^{(3)}}=-\sum_{s=1}^{S}\delta _{i}^{(3)s}h_{j}^{(2)s}, \delta _{i}^{(3)s}\equiv -\frac{\partial E}{\partial \hat{y}_{i}^{s}}=f'(\hat{y}_{i}^{s})(t_{i}^{s}-y_{i}^{s}) $$

$$\frac{\partial E}{\partial W_{jk}^{(2)}}=-\sum_{s=1}^{S}\delta _{j}^{(2)s}h_{k}^{(1)s},
~~\delta _{i}^{(2)s}\equiv -\frac{\partial E}{\partial \hat{h}_{j}^{(2)s}}=f'(\hat{h}_{j}^{(2)s})\sum_{i=1}^{M}\delta _{i}^{(3)s}W_{ij}^{(3)}$$

$$\frac{\partial E}{\partial W_{kl}^{(1)}}=-\sum_{s=1}^{S}\delta _{j}^{(1)s}x_{l}^{s},
~~\delta _{i}^{(1)s}\equiv -\frac{\partial E}{\partial \hat{h}_{k}^{(1)s}}=f'(\hat{h}_{k}^{(1)s})\sum_{j=1}^{N_{2}}\delta _{i}^{(2)s}W_{jk}^{(2)}$$

\
&nbsp;
이 내용을 기반으로 Three layer perceptron을 구현해보았다.

```python
def ThreeLayerPerceptron_train(X_train, Y_train, p=20, q=10, eta=0.0015):
  import numpy as np
  import matplotlib.pyplot as plt

  # 0: Random initialize the relevant data
  w1 = 2*np.random.rand(p , X_train.shape[1]) - 0.5 # Layer 1
  b1 = np.random.rand(p)

  w2 = 2*np.random.rand(q , p) - 0.5  # Layer 2
  b2 = np.random.rand(q)

  wOut = 2*np.random.rand(q) - 0.5   # Output Layer
  bOut = np.random.rand(1)

  mu = []
  vec_y = []
  epoch_loss = []

  for n in range(0, 300):
    vec_y = []
    for I in range(0, X_train.shape[0]-1):

        # 1: input the data
        x = X_train[I]

        # 2: Start the algorithm

        # 2.1: Feed forward
        z1 = ReLU_act(np.dot(w1, x) + b1) # output layer 1
        z2 = ReLU_act(np.dot(w2, z1) + b2) # output layer 2
        y = sigmoid_act(np.dot(wOut, z2) + bOut) # Output of the Output layer


        #2.2: Compute the output layer's error
        delta_Out = 1/2 * (y-Y_train[I]) * sigmoid_act(y, der=True)

        #2.3: Backpropagate
        delta_2 = delta_Out * wOut * ReLU_act(z2, der=True) # Second Layer Error
        delta_1 = np.dot(delta_2, w2) * ReLU_act(z1, der=True) # First Layer Error

        # 3: Gradient descent
        wOut = wOut - eta*delta_Out*z2  # Outer Layer
        bOut = bOut - eta*delta_Out

        w2 = w2 - eta*np.kron(delta_2, z1).reshape(q,p) # Hidden Layer 2
        b2 = b2 -  eta*delta_2

        w1 = w1 - eta*np.kron(delta_1, x).reshape(p, x.shape[0])
        b1 = b1 - eta*delta_1

        # 4. Computation of the loss function
        mu.append((y-Y_train[I])**2)

    epoch_loss.append(np.mean(mu))


  plt.figure(figsize=(10,6))
  plt.scatter(np.arange(1, len(epoch_loss)+1), epoch_loss, alpha=1, s=10, label='error')
  plt.title('Averege Loss by epoch')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.show()

  return w1, b1, w2, b2, wOut, bOut, mu
```

```python
def ThreeLayerPerceptron_pred(X_test, w1, b1, w2, b2, wOut, bOut, mu):
  import numpy as np

  pred = []

  for I in range(0, X_test.shape[0]):
      # 1: input the data
      x = X_test[I]

      # 2.1: Feed forward
      z1 = ReLU_act(np.dot(w1, x) + b1) # output layer 1
      z2 = ReLU_act(np.dot(w2, z1) + b2) # output layer 2
      y = sigmoid_act(np.dot(wOut, z2) + bOut)  # Output of the Output layer

      # Append the prediction;
      # if y < 0.5 the output is zero, otherwise is 1
      pred.append( np.heaviside(y - 0.5, 1)[0] )


  return np.array(pred)
```

\
&nbsp;

***

#### 이미지 출처 :
 * [https://www.researchgate.net/figure/A-simple-MLP-with-two-hidden-layers_fig3_2225172302](https://www.researchgate.net/figure/A-simple-MLP-with-two-hidden-layers_fig3_222517230)
