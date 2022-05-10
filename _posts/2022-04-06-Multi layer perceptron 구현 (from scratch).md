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