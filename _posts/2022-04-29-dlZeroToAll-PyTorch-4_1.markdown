---
title: "Cross Entropy(교차 엔트로피)"
author: Kwon
date: 2022-04-29T14:00:00+0900
categories: [pytorch, study]
tags: [multivariate-linear-regressoin]
math: true
mermaid: false
---
[모두를 위한 딥러닝](https://deeplearningzerotoall.github.io/season2/lec_pytorch.html) Lab 4_1: Multivariate Linear Regression 강의를 본 후 공부를 목적으로 작성한 게시물입니다.

***
## Theoretical Overview
이전 포스팅까지의 regression은 하나의 변수를 통해 예측을 하는 것이었다. 하지만 직관적으로 생각해 보더라도 여러 변수를 가지고 더 많은 정보를 통해 예측하면 더 좋은 결과가 나올 것 같다.
**Multivariate Linear Regression**은 이처럼 여러 변수를 input으로 사용하여 예측하는 것을 말한다.

Hypothesis와 Cost는 앞에서 사용했던 것들을 사용하되 변수가 늘어난 것에 맞춰 수정해야 한다. 수정해서 나온 hypothesis와 cost는 다음과 같다.

\\[ H(x_1, x_2, x_3) = x_1w_1 + x_2w_2 + x_3w_3 + b \\]
\\[ cost(W, b) = \frac{1}{m} \sum^m_{i=1} \left( H(x^{(i)}) - y^{(i)} \right)^2 \\]

변수의 수에 따라 hypothesis가 변한 것을 확인할 수 있다. 이때 각각의 변수에 대해 $W$가 따로 주어진다. 

***
## Import
{% highlight python %}
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
{% endhighlight %}

***
## Data and Hypothesis
### Data
데이터는 다음과 같이 3개의 퀴즈 점수(x)와 최종 시험 점수(y)를 포함하는 셋을 사용할 것이다.

| Quiz 1 (x1) | Quiz 2 (x2) | Quiz 3 (x3) | Final (y) |
|:-----------:|:-----------:|:-----------:|:---------:|
| 73 | 80 | 75 | 152 |
| 93 | 88 | 93 | 185 |
| 89 | 91 | 80 | 180 |
| 96 | 98 | 100 | 196 |
| 73 | 66 | 70 | 142 |
{:.inner-borders}

{% highlight python %}
x1_train = torch.FloatTnsor([[73], [93], [89], [96], [73]])
x2_train = torch.FloatTensor([[80], [88], [91], [98], [66]])
x3_train = torch.FloatTensor([[75], [93], [90], [100], [70]])

y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
{% endhighlight %}
<br>
### Hypothesis Function
위 데이터에 대해 hypothesis를 코드로 표현하면 다음과 같이 표현할 수 있다.

{% highlight python %}
hypothesis = x1_train * w1 + x2_train * w2 + x3_train * w3 + b
{% endhighlight %}

하지만 input 변수의 개수가 100개, 1000개 일 때도 위와 같이 hypothesis와 데이터셋을 정의할 수는 없는 노릇이다. 이걸 간단하게 만들기 위해 $x_1w_1 + x_2w_2 + x_3w_3$를 행렬곱으로 바꿔 생각해 보자.

각 변수와 가중치의 곱은 아래와 같은 행렬 곱과 같다.

\\[
\begin{pmatrix}
x_1 & x_2 & x_3
\end{pmatrix}
\cdot
\begin{pmatrix}
w_1 \\\\\\
w_2 \\\\\\
w_3 \\\\\\
\end{pmatrix}
=
\begin{pmatrix}
x_1w_1 + x_2w_2 + x_3w_3
\end{pmatrix}
\\]

그러므로 [lab1](https://https://qja1998.github.io/2022/04/14/dlZeroToAll-PyTorch-1/)에서 나왔던 `matmul()`을 사용하여 변수와 가중치의 곱을 간단하게 구할 수 있다.

{% highlight python %}
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

hypothesis = x_train.matmul(W) # W의 차원은 [변수의 개수, 1]로 맞춰 줄 것
{% endhighlight %}

이때 주의할 것은 행렬곱 연산이 가능하도록 `W`의 차원을 변수에 개수에 따라 맞춰줘야 한다는 것이다.


***
## Train
Multivariate Linear Regression의 hypothesis까지 모두 알아봤으니 학습하는 전체적인 코드를 작성해 보자.

[lab3](https://https://qja1998.github.io/2022/04/18/dlZeroToAll-PyTorch-3/)의 **Train with `optim`**의 코드와 크게 다르지 않지만
`W = torch.zeros((3, 1), requires_grad=True)`에서 가중치의 차원을 변수의 개수에 따라 맞춰준 것만 차이가 난다.

{% highlight python %}
# 모델 초기화
W = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs + 1):
    
    # H(x) 계산
    hypothesis = x_train.matmul(W) + b # or .mm or @

    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 로그 출력
    print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'.format(
        epoch, nb_epochs, hypothesis.squeeze().detach(), cost.item()
    ))

'''output
Epoch    0/20 hypothesis: tensor([0., 0., 0., 0., 0.]) Cost: 29661.800781
Epoch    1/20 hypothesis: tensor([67.2578, 80.8397, 79.6523, 86.7394, 61.6605]) Cost: 9298.520508
Epoch    2/20 hypothesis: tensor([104.9128, 126.0990, 124.2466, 135.3015,  96.1821]) Cost: 2915.712402
Epoch    3/20 hypothesis: tensor([125.9942, 151.4381, 149.2133, 162.4896, 115.5097]) Cost: 915.040649
Epoch    4/20 hypothesis: tensor([137.7967, 165.6247, 163.1911, 177.7112, 126.3307]) Cost: 287.936157
Epoch    5/20 hypothesis: tensor([144.4044, 173.5674, 171.0168, 186.2332, 132.3891]) Cost: 91.371010
Epoch    6/20 hypothesis: tensor([148.1035, 178.0143, 175.3980, 191.0042, 135.7812]) Cost: 29.758249
Epoch    7/20 hypothesis: tensor([150.1744, 180.5042, 177.8509, 193.6753, 137.6805]) Cost: 10.445281
Epoch    8/20 hypothesis: tensor([151.3336, 181.8983, 179.2240, 195.1707, 138.7440]) Cost: 4.391237
Epoch    9/20 hypothesis: tensor([151.9824, 182.6789, 179.9928, 196.0079, 139.3396]) Cost: 2.493121
Epoch   10/20 hypothesis: tensor([152.3454, 183.1161, 180.4231, 196.4765, 139.6732]) Cost: 1.897688
Epoch   11/20 hypothesis: tensor([152.5485, 183.3609, 180.6640, 196.7389, 139.8602]) Cost: 1.710555
Epoch   12/20 hypothesis: tensor([152.6620, 183.4982, 180.7988, 196.8857, 139.9651]) Cost: 1.651412
Epoch   13/20 hypothesis: tensor([152.7253, 183.5752, 180.8742, 196.9678, 140.0240]) Cost: 1.632369
Epoch   14/20 hypothesis: tensor([152.7606, 183.6184, 180.9164, 197.0138, 140.0571]) Cost: 1.625924
Epoch   15/20 hypothesis: tensor([152.7802, 183.6427, 180.9399, 197.0395, 140.0759]) Cost: 1.623420
Epoch   16/20 hypothesis: tensor([152.7909, 183.6565, 180.9530, 197.0538, 140.0865]) Cost: 1.622141
Epoch   17/20 hypothesis: tensor([152.7968, 183.6643, 180.9603, 197.0618, 140.0927]) Cost: 1.621262
Epoch   18/20 hypothesis: tensor([152.7999, 183.6688, 180.9644, 197.0661, 140.0963]) Cost: 1.620501
Epoch   19/20 hypothesis: tensor([152.8014, 183.6715, 180.9665, 197.0686, 140.0985]) Cost: 1.619764
Epoch   20/20 hypothesis: tensor([152.8020, 183.6731, 180.9677, 197.0699, 140.0999]) Cost: 1.619046
'''
{% endhighlight %}

hypothesis의 출력값과 cost를 보면 정답에 점점 근접하는 것을 확인할 수 있다. 이 경우에도 lab3에서 언급한 것과 같이 learning rate에 따라 cost의 극소점을 지나쳐 발산할 수도 있다.

***
## High-level Implementation with `nn.Module`
PyTorch의 `nn.Module`을 상속하여 모델을 생성하면 hypothesis 정의, cost 계산 등의 학습에 필요한 여러 요소들을 편하게 만들고 사용할 수 있다.

### Model
`nn.Module`을 이용하여 모델을 생설할 때는 다음과 같은 형식을 따라야 한다.

1. `nn.Module`를 상속해야 한다.
2. `__init__`과 `forward()`를 override 해야한다.
   * `__init__`: 모델에 사용될 module을 정의. `super().__init__()`으로 `nn.Module`의 속성으로 초기화
   * `forward()`: **Hypothesis**가 들어갈 곳. (Hypothesis에 input을 넣어 predict하는 것을 forward 연산이라고 함)

{% highlight python %}
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)
{% endhighlight %}

이렇게 모델을 정의하면 우리가 직접 W, b를 명시하여 hypothesis를 정의해 주지 않아도 된다. gradient 계산인 `backward()`는 PyTorch가 알아서 해주기 때문에 신경쓰지 않아도 된다.

{% highlight python %}
# 기존의 모델 초기화
W = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

hypothesis = x_train.matmul(W) + b

# nn.Module을 활용한 모델 초기화
class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)
{% endhighlight %}

Multivariate Linear Regression Model을 생성할 때도 `nn.Linear()`의 차원을 조절해 주는 것으로 간단히 모델을 생성할 수 있다.
이렇게 모델의 수정이 자유로운 것은 `nn.Module`의 확장성을 잘 보여주는 부분이라고 생각한다.

<br>

### Cost (`nn.functional`)
cost도 `torch.nn.functional`를 이용해서 간단히 정의할 수 있다.

{% highlight python %}
cost = F.mse_loss(prediction, y_train)
{% endhighlight %}

계속 사용하던 MSE를 위와 같이 정의하면 추후에 다른 loss를 사용하려고 할 때 loss 함수만 변경해 주면 되기 때문에 훨씬 편리하다.

<br>

### Training with `nn.Module`
앞서 새롭게 정의했던 모델과 cost를 이용하여 전체 학습 코드를 작성하면 다음과 같다.
나머지 부분은 크게 차이가 없고, hypothesis와 cost를 정의하는 부분이 더 간결해진 모습을 볼 수 있다.

{% highlight python %}
# 데이터
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
# 모델 초기화
model = MultivariateLinearRegressionModel()
# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs+1):
    
    # H(x) 계산
    prediction = model(x_train)
    
    # cost 계산
    cost = F.mse_loss(prediction, y_train)
    
    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    # 로그 출력
    print('Epoch {:4d}/{} Cost: {:.6f}'.format(
        epoch, nb_epochs, cost.item()
    ))

'''output
Epoch    0/20 Cost: 31667.599609
Epoch    1/20 Cost: 9926.266602
Epoch    2/20 Cost: 3111.514160
Epoch    3/20 Cost: 975.451172
Epoch    4/20 Cost: 305.908539
Epoch    5/20 Cost: 96.042679
Epoch    6/20 Cost: 30.260782
Epoch    7/20 Cost: 9.641659
Epoch    8/20 Cost: 3.178671
Epoch    9/20 Cost: 1.152871
Epoch   10/20 Cost: 0.517862
Epoch   11/20 Cost: 0.318802
Epoch   12/20 Cost: 0.256388
Epoch   13/20 Cost: 0.236816
Epoch   14/20 Cost: 0.230657
Epoch   15/20 Cost: 0.228718
Epoch   16/20 Cost: 0.228094
Epoch   17/20 Cost: 0.227880
Epoch   18/20 Cost: 0.227803
Epoch   19/20 Cost: 0.227759
Epoch   20/20 Cost: 0.227729
'''
{% endhighlight %}
