---
title: Prevent Multi layer perceptron Overfitting
author: Bean
date: 2022-05-03 23:10:00 +0800
categories: [AI, basic]
tags: []
layout: post
current: post
class: post-template
subclass: 'post'
navigation: True
cover:  assets/img/post_images/ai_cover2.jpg
---

Looking at the existing multi-layer perceptron learning results, it can be seen that the training error continuously decreases as learning progresses. Decreasing error generally means the training is going well, but it may not always be the case. `Overfitting` may be occuring, where the model follows too closely to the training data, and may not be able to perform as well on data outside the training data set.

## Checking Validation error
&nbsp;

In order to check the overfitting, validation error should be checked together with training error. `training error` refers to errors that occurs during training with the training data, whereas `validation error` refers to an error that occurs by predicting new data (validation data). In order to obtain `validation error`, the data must first be divided into training dataset and test dataset. This can be done simply by using the `train_test_split` function of `sklearn.model_selection`.


```python
  X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size =0.2)
```

Next, let's calculate the validation error. The error can be calculated with the following function. For the validation data, the error between predicted value using the model trained so far and the actual value was calculated.

```python
def ThreeLayerPerceptron_validation_error(X_valid, y_valid, w1, b1, w2, b2, wOut, bOut):
    import numpy as np
    mu = []

    for I in range(0, X_valid.shape[0]):
        # 1: input the data
        x = X_valid[I]

        # 2.1: Feed forward
        z1 = ReLU_act(np.dot(w1, x) + b1) # output layer 1
        z2 = ReLU_act(np.dot(w2, z1) + b2) # output layer 2
        y = sigmoid_act(np.dot(wOut, z2) + bOut)  # Output of the Output layer

        # Append the prediction;
        pred = np.heaviside(y - 0.5, 1)[0]
        mu.append((y_valid[I]-pred)**2)

    return mu
```

Let's add this validation error to the train code. The same part as the previous code was omitted with ....

```python
def ThreeLayerPerceptron_train(X_train, y_train, X_valid, y_valid, p=20, q=10, eta=1e-3):
    ...

    for n in range(0, 300):
      vec_y = []
      for I in range(0, X_train.shape[0]-1):

          ...

          # 4. Computation of the loss function
          mu.append((y-y_train[I])**2)

      epoch_loss.append(np.mean(mu))
      loss = np.mean(ThreeLayerPerceptron_validation_error(X_valid, y_valid, w1, b1, w2, b2, wOut, bOut))
      validation_loss.append(loss)

      if early_stopping.validate(loss, n):
        break

    ...

    return w1, b1, w2, b2, wOut, bOut, mu
```

Plotting the training error and validation error after running training in this way, it can be seen that the training error continues to decrease as shown below, but the validation error increases after a certain section. This is where overfitting occurs.

<div style="text-align: left">
   <img src="/assets/img/post_images/overfitting1.png" width="100%"/>
</div>

&nbsp;

## Prevent overfitting adding Early stopping term and weight decay term
&nbsp;

Now that overfitting has been identified, let's try to prevent it in several ways.

### Early stopping
Early stopping refers to a method of terminating training before the validation error in the graph above enters an increasing trend.

Below is the early stopping implementation code. When the validation error continues to increase 10 times in a row compared to the previous validation error value, learning is terminated.

```python
class EarlyStopping():
  def __init__(self, patience=0):
    self._step = 0
    self._prev_loss = -1
    self.patience  = patience
  def validate(self, loss, epoch):
    if self._prev_loss < loss:
      self._step += 1
      if self._step > self.patience:
        print('Training process is stopped early in {}th epoch\n'.format(epoch))
        return True
    self._prev_loss = loss
```

### Weight decay

Overfitting can be avoided by applying weight decay. When learning, if the learning is carried out simply in the direction in which the loss function becomes smaller, the specific weight values ​​will rather increase and the result may deteriorate. Weight decay exerts a penalty when the weight becomes large in the loss function, so that the weight does not have a too large value during training. L1 regularization and L2 regularization are widely used as the penalty. In this implementation, the L2 method is used.

```python
def l2_penalty(w):
    return (w**2).sum() / 2
```

&nbsp;
## Final MLP training code
&nbsp;

The final MLP code implementation is as follows.


```python
def ThreeLayerPerceptron_train(X_train, y_train, X_valid, y_valid, p=20, q=10, eta=1e-3, weight_decay_lambda=0.1):
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
    validation_loss = []
    early_stopping = EarlyStopping(patience=10)

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
          delta_Out = 1/2 * (y-y_train[I]) * sigmoid_act(y, der=True)

          #2.3: Backpropagate
          delta_2 = delta_Out * wOut * ReLU_act(z2, der=True) # Second Layer Error
          delta_1 = np.dot(delta_2, w2) * ReLU_act(z1, der=True) # First Layer Error

          # 3: Gradient descent
          wOut = wOut*(1- eta*weight_decay_lambda) - eta*delta_Out*z2 # Outer Layer
          bOut = bOut*(1- eta*weight_decay_lambda) - eta*delta_Out

          w2 = w2*(1- eta*weight_decay_lambda) - eta*np.kron(delta_2, z1).reshape(q,p) # Hidden Layer 2
          b2 = b2*(1- eta*weight_decay_lambda) -  eta*delta_2

          w1 = w1*(1- eta*weight_decay_lambda) - eta*np.kron(delta_1, x).reshape(p, x.shape[0])
          b1 = b1*(1- eta*weight_decay_lambda) - eta*delta_1

          # 4. Computation of the loss function
          mu.append((y-y_train[I])**2)

      epoch_loss.append(np.mean(mu))
      loss = np.mean(ThreeLayerPerceptron_validation_error(X_valid, y_valid, w1, b1, w2, b2, wOut, bOut))
      validation_loss.append(loss)

      if early_stopping.validate(loss, n):
        break

    plt.figure(figsize=(10,6))
    plt.scatter(np.arange(1, len(epoch_loss)+1), epoch_loss, alpha=1, s=10, label='training error')
    plt.scatter(np.arange(1, len(validation_loss)+1), validation_loss, alpha=1, s=10, label='validation error')
    plt.legend()
    plt.title('Averege Loss by epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    print("\n")

    return w1, b1, w2, b2, wOut, bOut, mu
```

&nbsp;

***

#### references :
* [https://light-tree.tistory.com/216](https://light-tree.tistory.com/216)