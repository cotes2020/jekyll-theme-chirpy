---
title: "모두를 위한 딥러닝 2 - Lab8-1: Perceptron"
author: Kwon
date: 2022-05-11T23:00:00 +0900
categories: [pytorch, study]
tags: [perceptron]
math: true
mermaid: false
---

[모두를 위한 딥러닝](https://deeplearningzerotoall.github.io/season2/lec_pytorch.html) Lab8-1: Perceptron 강의를 본 후 공부를 목적으로 작성한 게시물입니다.

***

## Neuron

먼저 퍼셉트론의 컨셉이 된 뉴런에 대해 알아보자. 뉴런은 동물의 신경계를 구성하는 세포로 다음과 같은 형태이다.

![동물의 뉴런](/posting_imgs/lab8-1-1.svg)

뉴런은 자극을 전기 신호를 전달하는 통로라고 할 수 있는데, 이때 강도가 어느 정도(threshold)를 넘어서는 신호만을 전달한다.

***

## Perceptron

퍼셉트론은 이런 뉴런의 작동 방식을 반영하여 만든 인공신경망의 한 종류로, 다수의 입력을 받아 하나의 출력을 내보내는 구조이다.

![퍼셉트론의 구조](/posting_imgs/lab8-1-2.png)

처음 퍼셉트론이 등장했을 때는 AND, OR 문제를 아주 잘 해결하였다. 다음 그래프를 보면 AND, OR 문제 모두 선형으로 잘 분리되는 것을 볼 수 있다.

![AND(좌)와 OR(우)의 분류 형태](/posting_imgs/lab8-1-3.jpg)

그래서 더 복잡한 문제도 풀어낼 수 있지 않을까 하며 기대를 받았었는데, XOR 문제가 단일 퍼셉트론으로 해결할 수 없다는 것이 증명되면서 퍼셉트론 연구에 암흑기가 도래하게 되었다.

![단일 선형으로 해결할 수 없는 XOR](/posting_imgs/lab8-1-4.jpg)

위 그림을 보면 어떤 직선을 그어도 하나의 직선으로는 XOR 문제를 명확하게 가를 수 없는 것을 볼 수 있다.
실제로도 그런지 코드로 학습해 보면서 한번 확인 해 보자.

### XOR train code with single preceptron

모델은 선형에 시그모이드를 활성화 함수로 사용하도록 하고, 이진분류이므로 손실함수로 `BECLoss`를 사용한다.

```python
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 시드 고정
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(device)
Y = torch.FloatTensor([[0], [1], [1], [0]]).to(device)
```

먼저 XOR에 대한 데이터를 정의해 준다.

```python
linear = torch.nn.Linear(2, 1, bias=True)
sigmoid = torch.nn.Sigmoid()

# Sequential로 여러 모듈을 묶어 하나의 레이어로 사용
model = torch.nn.Sequential(linear, sigmoid).to(device)

criterion = torch.nn.BCELoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1)

'''output
0 0.7273974418640137
100 0.6931475400924683
200 0.6931471824645996
300 0.6931471824645996
400 0.6931471824645996
500 0.6931471824645996
600 0.6931471824645996
...
9700 0.6931471824645996
9800 0.6931471824645996
9900 0.6931471824645996
10000 0.6931471824645996
'''
```

100epoch 부터 loss가 잘 감소하지 않더니 200부터는 아예 감소하지 않는다. 확실히 확인해 보기 위해 정확도를 출력해 보면

```python
with torch.no_grad():
    hypothesis = model(X)
    predicted = (hypothesis > 0.5).float()
    accuracy = (predicted == Y).float().mean()
    print('\nHypothesis: ', hypothesis.detach().cpu().numpy(), '\nCorrect: ', predicted.detach().cpu().numpy(), '\nAccuracy: ', accuracy.item())

'''output
Hypothesis:  [[0.5]
 [0.5]
 [0.5]
 [0.5]] 
Correct:  [[0.]
 [0.]
 [0.]
 [0.]] 
Accuracy:  0.5
'''
```

모든 hypothesis 결과가 0.5로 나오는 것을 확인할 수 있다. 이처럼 단일 퍼셉트론으로는 XOR을 제대로 분류할 수 없다.

다음 포스팅에서는 XOR을 해결한 Multi Layer Percetron에 대해 알아보겠다.

***

#### Image source

* Neuron: [https://commons.wikimedia.org/wiki/File:Neuron.svg](https://commons.wikimedia.org/wiki/File:Neuron.svg)

* Perceptron: [https://commons.wikimedia.org/wiki/File:Rosenblattperceptron.png](https://commons.wikimedia.org/wiki/File:Rosenblattperceptron.png)