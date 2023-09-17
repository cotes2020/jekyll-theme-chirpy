---
title: Deep Learning과 TemsorFlow
date: 2023-06-13 13:15:55 +0900
author: kkankkandev
categories: [AI, Deep Learning]
comments: true
tags: [TensorFlow, Deep Learning]     # TAG names should always be lowercase
image:
  path: https://github.com/War-Oxi/Oxi/assets/72260110/110eebe0-5912-46b5-8b9e-631db9c8e05e

---

## Chapter 5. 딥러닝과 텐서플로

## 1. 딥러닝의 3가지 학습 방법

### 1.1. 지도학습

> 학습 데이터에 대하여 정답 쌍이 존재할 때 상관 관계를 모델링하는 것
> 

### 1.2. 비지도학습

> 학습 데이터만 있고 정답이 존재하지 않을 때 데이터의 숨겨진 패턴을 찾는 것
> 

### 1.3. 강화학습

> 특정 환경에서 행동에 대한 보상을 극대화하도록 학습하는 방법
> 

---

## 2. 학습을 효과적으로 실행할 수 있는 다양한 알고리즘과 규제 기법

### 2.1. ReLU함수

- 계산은 단순하고 성능은 더 좋은 활성 함수

### 2.2. 가중치 감쇠(weight decay) 기법

- **가중치를 작은 값으로 유지**하는 기법

### 2.3. 드롭아웃(dropout) 기법

- 임의로 일정 비율의 노드를 선택해 불능으로 놓고 학습하는 기법

---

## 3. Tensorflow

### 3.1. 텐서

> 딥러닝에서 사용하는 다차원 배열(multi-dimensional array)
> 

```python
import tensorflow as tf
import numpy as np

t=tf.random.uniform([2, 3], 0, 1)
n=np.random.uniform(0, 1, [2, 3])
print("tensorflow로 생성한 텐서:\n", t, "\n")
print("numpy로 생성한 ndarray:\n", n, "\n")

res = t+n
print("덧셈 결과 :\n", res)
```

```python
tensorflow로 생성한 텐서:
 tf.Tensor(
[[0.6083759  0.02200425 0.97783387]
 [0.57673144 0.08350623 0.14578283]], shape=(2, 3), dtype=float32) 

numpy로 생성한 ndarray:
 [[0.71571335 0.08340176 0.45973256]
 [0.95546853 0.75035322 0.1198678 ]] 

덧셈 결과 :
 tf.Tensor(
[[1.3240893  0.10540601 1.4375664 ]
 [1.5322     0.83385944 0.26565063]], shape=(2, 3), dtype=float32)
```

- tensorflow로 생성한 객체 ⇒ tf.Tensor형
- numpy로 생성한 객체 n ⇒ ndarray형

## 3.2. TensorFlow로 퍼셉트론 프로그래밍

### 3.2.1. 학습된  퍼셉트론의 동작 확인

```python
import tensorflow as tf

# OR 데이터 구축
x = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
y = [[-1], [1], [1], [1]]

# 퍼셉트론 신경망의 가중치를 설정
w = tf.Variable([[1.0], [1.0]])
b = tf.Variable(-0.5)

# 퍼셉트론 동작
s = tf.add(tf.matmul(x, w), b) 
o = tf.sign(s)

print(o)
```

> matmul 함수는 두 행렬을 곱해주는 함수이다.
> 

```python
### 결과
tf.Tensor(
[[-1.]
 [ 1.]
 [ 1.]
 [ 1.]], shape=(4, 1), dtype=float32)
```

### 3.2.2 퍼셉트론의 학습

```python
import tensorflow as tf

# OR 데이터 구축
x = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
y = [[-1], [1], [1], [1]]

# 가중치 초기화
w = tf.Variable(tf.random.uniform([2, 1], -0.5, 0.5)) #가중치에 [-0.5, 0.5] 사이의 난수 설정
b = tf.Variable(tf.zeros([1])) # 바이어스에 0 설정

# 옵티마이저
opt = tf.keras.optimizers.SGD(learning_rate=0.1) # 스토케스틱 경사 하강법(Stochastic Gradient Descent) 생성후 opt 객체에 저장

# 전방 계산
def forward():
    s = tf.add(tf.matmul(x, w), b)
    o = tf.tanh(s)
    return o

# 손실 함수 정의
def loss():
    o = forward()
    return tf.reduce_mean((y - o) ** 2)

# 500세대까지 학습(100세대마다 학습 정보 출력)
for i in range(500):
    opt.minimize(loss, var_list = [w, b])
    if(i % 100 == 0): print('loss at epoch', i, '=', loss().numpy())

# 학습된 퍼셉트론으로 OR 데이터를 예측
o = forward()
print(o)
```

```python
### 결과
loss at epoch 0 = 1.1765351
loss at epoch 100 = 0.098153956
loss at epoch 200 = 0.04385595
loss at epoch 300 = 0.02723797
loss at epoch 400 = 0.019496055
tf.Tensor(
[[-0.81489813]
 [ 0.8855341 ]
 [ 0.8854854 ]
 [ 0.9992482 ]], shape=(4, 1), dtype=float32)
```

### 3.3. Keras Programming

> TensorFlow로 프로그래밍을 하게 되면 신경망의 원리와 수식을 완전히 이해해야 하고 수식을 정확히 프로그램 코드로 표현해야 하기 떄문에 상당히 부담스럽다. 
케라스는 이런 부담을 덜어주기 위해 탄생하였다.
> 

#### 3.3.1 Keras를 사용한 퍼셉트론 프로그래밍

- model 클래스: Sequential과 functional API 모델 제작 방식 제공
    - Sequential - 층을 한 줄로 쌓는 데 사용
- layers 클래스: 다양한 종류의 층 제공
    - Dense - 인접한 두 층이 완전연결된 경우 사용
- optimizers 클래스: 다양한 종류의 옵티마이저 제공
    - SGD - 스토케스틱 경사 하강법(Stochastic Gradient Descent)

```python
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

# OR 데이터 구축
x = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
y = [[-1], [1], [1], [1]]

n_input = 2
n_output = 1

perceptron = Sequential()  # 층을 한 줄로 쌓는 데 사용
perceptron.add(Dense(units=n_output, activation='tanh',
                     input_shape=(n_input,), kernel_initializer='random_uniform',
                     bias_initializer='zeros'))  # Dense => 인접한 두 층이 완전연결된 경우 사용

perceptron.compile(loss='mse', optimizer=SGD(learning_rate=0.1), metrics=['mse'])  #mse(평균제곱오차) 손실함수
perceptron.fit(x, y, epochs=500, verbose=2)  # verbose => 학습 도중에 발생하는 정보를 출력하는 방식을 지정

res = perceptron.predict(x)
print(res)
```

## 4.  TensorFlow(Keras)로 다층 퍼셉트론 프로그래밍

### 4.1. MNIST 인식

텐서플로 프래그래밍: 다층 퍼셉트론으로 MNIST 인식

```python
# 다층 퍼셉트론으로 MNIST 인식
import numpy as np
import tensorflow as tf
from keras.datasets import mnist

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# MNIST를 읽어 와서 신경망에 입력할 형태로 변환
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)  # 텐서 모양 변환
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype(np.float32)/255.0  # ndarray 형으로 변환
x_test = x_test.astype(np.float32)/255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)  # 원핫 코드로 변환
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 신경망 구조 설계
n_input = 784    # 입력층
n_hidden = 1024  # 은닉층
n_output = 10    # 출력층

mlp = Sequential()
mlp.add(Dense(units=n_hidden, activation='tanh', input_shape=(n_input,),
              kernel_initializer='random_uniform', bias_initializer='zeros'))
mlp.add(Dense(units=n_output, activation='tanh',
              kernel_initializer='random_uniform', bias_initializer='zeros'))

# 신경망 학습
mlp.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
hist = mlp.fit(x_train, y_train, batch_size=128, epochs=30, validation_data=(x_test, y_test), verbose=2)

# 학습된 신경망으로 예측
res = mlp.evaluate(x_test, y_test, verbose=0)
print("정확률은", res[1]*100)
```

### 4.2. 학습 곡선 시각화

```python
# 다층 퍼셉트론으로 MNIST 인식
import numpy as np
import tensorflow as tf
from keras.datasets import mnist

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# MNIST를 읽어 와서 신경망에 입력할 형태로 변환
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)  # 텐서 모양 변환
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype(np.float32) / 255.0  # ndarray 형으로 변환
x_test = x_test.astype(np.float32) / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)  # 원핫 코드로 변환
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 신경망 구조 설계
n_input = 784    # 입력층
n_hidden = 1024  # 은닉층
n_output = 10    # 출력층

mlp = Sequential()
mlp.add(Dense(units=n_hidden, activation='tanh', input_shape=(n_input,),
              kernel_initializer='random_uniform', bias_initializer='zeros'))
mlp.add(Dense(units=n_output, activation='tanh',
              kernel_initializer='random_uniform', bias_initializer='zeros'))

mlp.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
# fit 함수의 결과는 hist에 저장된다.
hist = mlp.fit(x_train, y_train, batch_size=128, epochs=30, validation_data=(x_test, y_test), verbose=2)

res = mlp.evaluate(x_test, y_test, verbose=0)
print("정확률 => ", res[1] * 100)

import matplotlib.pyplot as plt

# 정확률 곡선
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.grid()
plt.show()

# 손실 함수 곡선
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.grid()
plt.show()
```

![Untitled](https://github.com/War-Oxi/Oxi/assets/72260110/110eebe0-5912-46b5-8b9e-631db9c8e05e)

![Untitled](https://github.com/War-Oxi/Oxi/assets/72260110/0a631ffe-f747-4286-acfa-9be99cb85dba)

## 5. 깊은 다층 퍼셉트론

> 다층 퍼셉트론에 은닉층을 더 많이 추가하면 깊은 다층 퍼셉트론(DMLP$_{deep MLP}$)이 된다.
> 

### 5.1. 오류 역전파 알고리즘

> 깊은 다층 퍼셉트론을 학습하려면 손실 함수를 정의하고 손실 함수의 최저점을 찾는 최적화 알고리즘을 고안해야 한다
> 
- 인공 신경망의 가중치(weight)와 편향(bias)을 조정하기 위해 오차를 역으로 전파하여 각각의 가중치와 편향에 대한 기여도를 계산하는 방법

## 6. 딥러닝의 학습 전략

### 6.1. 그레이디언트 소멸(vanishing gradient) 문제와 해결책

> 미분 이론의 연쇄 법칙(chain rule)에 따르면 I번째 층의 그레이디언트는 오른쪽에 있는 $I+1$번째 층의 크레이디언트에 자신의 층에서 발생한 그레이디언트를 곱하여 구한다.
따라서 그레이디언트가 작으면 왼쪽으로 진행하면서 그레이디언트가 점점 작아지는 현상이 발생한다.
이처럼 왼쪽으로 갈수록 그레이디언트가 기하급수적으로 작아지는 현상을 그레이디언트 소멸(vanishing gradient)이라고 한다.
> 
- 그레이디언트 소멸 현상이 발생하면 오른쪽에 있는 층의 가중치는 갱신이 제대로 일어나지만 왼쪽으로 갈수록 갱신이 매우 더딤
- 결국 전체 신경망 학습이 매우 느려져서 수 주 또는 수 개월을 학습해도 수렴에 도달하지 못하는 문제가 발생한다.

#### 6.1.1 병렬 처리로 해결

> 더 빠른 컴퓨터를 사용
> 

ex) CoLab의 TPU 사용

#### 6.1.2 ReLU 함수 사용

> 입력이 0을 넘으면 그 입력을 그대로 출력하고, 0 이하면 0을 출력하는 함수
> 

시그모이드 함수에 비해 ReLU함수는 그레이디언트 소멸이 발생할 가능성이 낮다.
### 6.2 과잉 적합과 과잉 회피 전략

#### 6.2.1. 과소 적합(underfitting)

> 데이터에 비해 작은 용량의 모델을 사용해 오차가 많아지는 현상
> 

#### 6.2.2 과잉 적합(overfitting)

> - 모델의 용량이 데이터 복잡도에 비해 너무 커서 발생.
- 너무 큰 모델이 데이터에 과도하게 적응하여 일반화 능력을 잃는 현상
> 
- 과잉 적합을 해소하는 전략 중 가장 확실한 방법은 데이터의 양을 늘리는 것이다. ⇒ 데이터 증대(data augmentation)

## 7. 딥러닝이 사용하는 손실 함수

> 신경망 모델의 예측값과 실제 값 사이의 차이를 측정하는 함수
> 
- 회귀 문제에서는 평균 제곱 오차(Mean Squared Error, MSE)나 평균 절대 오차(Mean Absolute Error, MAE)를 사용
- 분류 문제에서는 교차 엔트로피 손실(Cross-Entropy Loss)나 로그 손실(Log Loss) 등이 사용
- 손실 함수를 최소화하는 것이 학습의 목표이며, 이를 위해 역전파 알고리즘 등을 사용하여 가중치를 조정

### 7.1. 평균제곱오차(Mean Squared Error, MSE)

- Mean Squared Error에서는 오차가 더 크고 그레이디언트는 더 작은 상황이 발생할 수 있음
- 이러한 불공정성을 해결하기 위해 딥러닝은 주로 교차 엔트로피를 사용.

### 7.2 교차 엔트로피(cross entropy)

> 교차 엔트로피는 두 확률 분포가 다른 정도를 측정한다.
> 
- MSE(평균제곱오차)가 안고 있는 불공정성 문제를 해결.

### 7.2.1 엔트로피 함수

> 확률 분포의 무작위성, 불확실성을 측정하는 함수
> 

## 8. 딥러닝이 사용하는 옵티마이저

> 신경망 학습은 손실 함수의 최저점을 찾아가는 과정이다.
신경망에서는 SGD(스토케스틱 경사 하강법) 알고리즘으로 최저점을 찾는다.
> 
- SGD와 같은 최적화 알고리즘을 옵티마이저(optimizer)라고 부른다
- 신경망 학습에 이용되는 데이터는 잡음과 변화가 아주 심하므로 표준에 해당하는 SGD 옵티마이저는 종종 한계를 드러낸다.
- SGD 옵티마니저의 한계를 극복하기 위해 momentum과 adaptive learning rate(적응적 학습률)이라는 두 가지 아이디어를 사용한다.

### 8.1. 모멘텀을 적용한 옵티마이저

> 이전 미니배치에서 얻었던 방향 정보를 같이 고려해 잡음을 줄이는 효과
> 

### 8.2 적응적 학습률을 적용한 옵티마이저

> 학습률(learning rate)이라는 하이퍼 매개변수 $p$가 있다.
> 
- 너무 크게 설정하면 최저점을 지나치는 현상이 나타나고 너무 작게 설정하면 최저점에 수렴하는 데 시간이 너무 많이 걸린다.
- 적응적 학습률을 적용한 옵티마이저에는 Adagrad, RMSprop, Adam 등이 있다.

#### 8.2.1. Adagrad

> 이전 그레이디언트를 누적한 정보를 이용하여 학습률을 적응적으로 설정하는 기법
> 

#### 8.2.2. RMSprop

> 이전 그레이디언트를 누적할 때 오래된 것의 영향을 줄이는 정책을 사용하여 AdaGrad를 개선한 기법
> 

#### 8.2.3. Adam

> RMSprop에 모멘텀을 적용하여 RMSporp를 개선한 기법
> 

---

## 용어 정리

- 손실함수란?
    - 신경망 모델의 예측값과 실제 값 사이의 차이를 측정하는 함수
    - 회귀 문제에서는 평균 제곱 오차(Mean Squared Error, MSE)나 평균 절대 오차(Mean Absolute Error, MAE)를 사용
    - 분류 문제에서는 교차 엔트로피 손실(Cross-Entropy Loss)나 로그 손실(Log Loss) 등이 사용
    - 손실 함수를 최소화하는 것이 학습의 목표이며, 이를 위해 역전파 알고리즘 등을 사용하여 가중치를 조정
- 활성함수란?
    - 신경망의 각 뉴런에서 입력값을 변환하여 출력값을 계산하는 함수
    - 대표적인 활성화 함수로는 시그모이드 함수(Sigmoid function), 하이퍼볼릭 탄젠트 함수(Tanh function), 렐루 함수(Rectified Linear Unit, ReLU) 등이 있음
- 최적화함수란?
    - 손실 함수를 최소화하기 위해 신경망의 가중치를 조정하는 방법을 결정하는 함수
    - 주요한 최적화 함수로는 확률적 경사 하강법(Stochastic Gradient Descent, SGD)와 그 변형인 모멘텀(Momentum), 아다그라드(Adagrad), 알엠에스프롭(RMSprop), 아담(Adam) 등이 있음
    - 손실 함수의 그래디언트(gradient)를 사용하여 가중치를 업데이트하며, 빠른 수렴과 최적의 모델 파라미터를 찾는 데 도움을 줌
- 원핫 코드(One-Hot Code)란?
    - 주로 머신 러닝에서 범주형 변수를 처리할 때 사용되는 방법
    - 범주형 데이터를 표현하는 방법
    - 각각의 범주에 대해 이진(0 또는 1) 값을 가지는 벡터로 표현
    - 원핫 코드를 통해 컴퓨터 알고리즘은 범주형 데이터를 다루기 쉽게 됨
- 오류 역전파 알고리즘이란?
    - 인공 신경망에서 학습을 위해 사용되는 기법
    - 입력과 출력 사이의 오차를 사용하여 신경망의 가중치를 조정
- SGD(Stochastic Gradient Descent)란?
    - 딥러닝 모델의 학습에 주로 사용되는 최적화 알고리즘
- Adam이란? ⇒ Optimizer
    - 학습률(learning rate)을 조정하면서 각 가중치의 업데이트 속도를 조절하여 최적화 과정을 수행
- perceptron.compile()의 verbose 매개변수
    - verbose = 0: 학습 과정의 상세 정보를 출력하지 않습니다. 즉, 출력이 없습니다.
    - verbose = 1: 기본값으로, 학습 과정에서 진행 막대(progress bar)와 함께 로그 정보를 출력합니다. 각 에포크(epoch)마다 손실(loss) 및 정확도(accuracy) 등의 정보가 표시됩니다.
    - verbose = 2: 학습 과정에서 진행 막대(progress bar) 없이 간단한 로그 정보만 출력됩니다. 예를 들어, 각 에포크의 완료 상태만을 표시합니다.
- 퍼셉트론이란?
    - 인공 신경망의 한 종류로, 이진 분류(binary classification)를 위한 선형 분류기(linear classifier)
    - 퍼셉트론은 프랑크 로젠블라트(Frank Rosenblatt)에 의해 1957년에 제안 됨
    - 입력 벡터를 받아 가중치와 곱하고, 이를 활성화 함수를 통해 결과를 출력하는 구조
    - 가중치와 입력값의 선형 조합을 구하는 부분을 선형 함수(Linear function)라고 함
    - 선형 함수의 결과를 활성화 함수(Activation function)로 전달하여 최종 출력을 계산
- 단층 퍼셉트론이란?
    - 하나의 은닉층만을 갖고 있으며, 선형 분리 가능한 문제를 해결
- 다층 퍼셉트론이란?
    - 여러 개의 은닉층을 가지고 있어 복잡한 비선형 문제를 해결
- 시그모이드 함수란?
- tanh 시그모이드 함수란?
- 로지스틱 시그모이드 함수란?
- softmax 함수란?