---
title: Multi layer perceptron 과적합 방지
author:
  name: Bean
  link: https://github.com/beanie00
date: 2022-05-03 23:10:00 +0800
categories: [AI, basic]
tags: []
---

기존의 Multi layer perceptron 학습 코드를 보면 training error가 학습이 진행됨에 따라 지속적으로 감소함을 알 수 있다. error가 계속 감소한다는 것은 학습이 잘되고 있다는 뜻일 수 있지만 마냥 좋은 징조는 아니다. 파라미터가 training data의 특징을 너무 학습해 training data로는 성능이 좋지만 새로운 데이터에서는 오히려 안좋은 성능을 보이는 `과접합`이 일어날 수 있기 때문이다.

## Validation error 확인
---

과접합을 확인해보기 위해서는 training error외에 별도로 validation error를 확인해보면 된다. `training error`는 학습 데이터가 학습 상에서 발생하는 에러를 나타내는 반면 `validation error`는 학습 중인 파라미터로 학습 데이터 외에 새로운 데이터(validate data)를 예측해서 나오는 에러를 말한다. `validation error`를 구하기 위해서는 먼저 데이터를 training data와 test data로 분리해야 한다. 이는 `sklearn.model_selection`의 `train_test_split` 함수를 이용해 간단히 구할 수 있다.

```python
  X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size =0.2)
```

다음으로 validation error를 계산해보자. 다음의 함수로 에러를 계산할 수 있다. validation data에 대하여 현재까지 학습된 파라미터로 값을 예측한 뒤 예측 에러를 계산하였다.

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

이 validation error를 train 코드에 추가해보자. 이전 코드와 동일한 부분은 ...로 생략하였다.

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

이렇게 학습을 돌린 후 training error와 validation error를 plot 해보면, 아래와 같이 training error은 계속 감소하지만 validation error는 어느 구간 이후로 증가함을 확인할 수 있다. 이 부분이 과적합이 일어나는 부분이다.

<div style="text-align: left">
   <img src="/assets/img/post_images/overfitting1.png" />
</div>

&nbsp;

## Early stopping term과 weight decay term을 추가해 과적합을 방지하자.
---
과적합을 확인했으니 여러 방법으로 과적합을 방지해보자.

### Early stopping
---
Early stopping은 위 그래프에서 validation error가 증가 추세로 가기 전에 학습을 종료시키는 방법을 말한다.

아래는 Early stopping 구현 코드이다. 이전의 validation error값과 비교하여 10번 연속하여 validation error가 계속 증가하면 학습을 종료해주었다.

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
---

학습을 할 때, Loss function이 작아지는 방향으로만 단순하게 학습을 진행하면 오히려 특정 가중치 값들이 커지며 결과가 나빠질 수 있다. Weight decay는 학습 중 weight가 너무 큰 값을 가지지 않도록 Loss function에 weight가 커질 경우에 대한 패널티 항목을 집어넣는다. 이 패널티 항목으로 많이 쓰이는 것이 L1 Regularization과 L2 Regularization이다. Weight decay를 적용하면 overfitting을 벗어날 수 있다. 이번 구현에서는 L2 방식을 사용하였다.

```python
def l2_penalty(w):
    return (w**2).sum() / 2
```

&nbsp;
## 최종 MLP training 코드
---

이렇게 구현된 최종 MLP 코드는 다음과 같다.


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

#### 참고 내용 출처 :
* [https://light-tree.tistory.com/216](https://light-tree.tistory.com/216)