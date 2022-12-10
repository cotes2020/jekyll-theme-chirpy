---
title: "모두를 위한 딥러닝 2 - Lab4_2: Loading Data"
author: Kwon
date: 2022-04-30T00:00:00 +0900
categories: [pytorch, study]
tags: [multivariate-linear-regressoin]
math: true
mermaid: false
---

[모두를 위한 딥러닝](https://deeplearningzerotoall.github.io/season2/lec_pytorch.html) Lab 4_2: Loading Data 강의를 본 후 공부를 목적으로 작성한 게시물입니다.

***
## Data in the Real World
[lab4_1](/posts/dlZeroToAll-PyTorch-4_1/)에서 다루었던 Multivariate Linear Regression에서는 학습 데이터로 3개의 차원을 가진 5개의 샘플을 사용했었다.
하지만 실제 우리가 다루려고 하는 데이터는 그 크기가 그리 만만하지 않다. 본 강의에서 예를 들었던 **ImageNet**의 경우 1400만개 이상을 이미지 데이터 셋을 포함하고 있으며 그 용량이 120GB 이상이다.
이 데이터를 한번에 학습을 진행한다는 것은 하드웨어 상의 용량 문제도 있겠지만, 하드웨어적인 문제가 해결된다 하더라도 느리고 Gradient Descent 시에 cost 연산에 대한 computing power 부담이 너무 크다.

이런 문제점을 해결하기 위해 등장한 것이 Minibatch이다. 데이터가 너무 크니까 **'데이터의 일부로 학습하면 어떨까?'**라는 생각에 나온 개념이다.

***
## Minibatch Gradient Descent
앞에서 잠간 언급한 것처럼, 대용량 데이터를 균일하게 나눠 gradient dscent를 하는 것을 Minibatch Gradient Descent라고 한다.
이렇게 학습을 하는 경우 업데이트가 빠르고 하드웨어적인 부담도 덜 수 있다.
하지만 어디까지나 전체가 아닌 일부 데이터를 써서 학습하는 것이기 때문에 잘못된 방향으로 학습이 될 수도 있고,
결과적으로 어느정도 수렴하더라도 그 과정이 아래 그림과 같이 좀 거칠게 Gradient descent 될 수도 있다. (아래 그림은 이해를 위한 임의의 그래프이며, 실제 loss의 그래프가 아닙니다.)

![Minibatch Gradient Decsent](/posting_imgs/lab4_2-1.png)

![Gradient Decsent](/posting_imgs/lab4_2-2.png)

***
## Dataset
### PyTorch Dataset
`torch.utils.data.Dataset`을 상속하여 사용자 정의 데이터 셋을 만들어 사용할 수 있다. 이 방식으로 데이터 셋을 정의할 경우 다음과 같은 형식을 따라야 한다.
1. `torch.utils.data.Dataset`을 상속해야 한다.
2. `__init__`, `__len__` 그리고 `__getitem__`을 override 해야 한다.
   * `__init__`   : 데이터 셋을 만들 때(`Dataset`객체가 생성될 때)만 실행됨. 실제도 데이터를 정의하는 곳.
   * `__len__`    : 데이터 셋의 총 데이터 수를 반환함.
   * `__getitem__`: 주어진 인덱스(`idx`)에 따라 그 인덱스에 맞는 x, y 데이터를 반환함.

lab4_1에서 사용했던 데이터로 데이터 셋을 만드는 과정은 다음과 같다.

```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self):
        self.x_data = [[73, 80, 75],
                       [93, 88, 93], 
                       [89, 91, 90],
                       [96, 98, 100],
                       [73, 66, 70]]
        self.y_data = [[152], [185], [180], [196], [142]]
        
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.x_data[idx]), torch.FloatTensor(self.y_data[idx])
    
dataset = CustomDataset()
```

원하는 데이터를 넣어 데이터 셋 class를 만든 후 인스턴스를 생성해 준다.

### PyTorch DataLoader
`torch.utils.data.DataLoader`를 사용하면 `Dataset` 객체의 데이터에 대해 minibatch를 자동으로 제공하여 batch에 대한 반복을 쉽게 할 수 있도록 해준다.

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)
```

위와 같이 설정하면 각 minibatch들의 크기를 2로 나누어 제공한다. `batch_size`는 통상적으로 2의 제곱수(16, 32, 64, 128, 256...)로 설정하는데 이는 GPU memory가 2의 거듭제곱이기 때문이다.

경우에 따라 66개의 batch나 120개의 batch나 (둘 다 $2^6$과 $2^7$ 사이의 범위이다. 예를 들자면 그렇다는 것) 필요한 시간이 동일한 경우가 발생할 수 있다. 그러므로 최대의 효율을 위해 GPU memory에 크기를 맞춰주는 것이다. 또 하나 일반적으로 알려진 것으로는 batch size가 작을수록 시간은 오래 걸리고, 효과가 좋다는 것이다.

shuffle 옵션은 데이터를 epoch마다 섞어서 모델이 데이터의 순서를 외우는 것을 방지할 수 있도록 해 준다. 대부분의 경우 데이터의 순서를 외우지 못하도록 학습하는 것이 좋으므로 Ture로 해 놓는 것이 좋다.

***
## Full Code with `Dataset` and `DataLoader`
`Dataset` 과 `DataLoader`를 이용하여 학습하는 코드를 작성해보자.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(3, 1)
    
    def forward(self, x):
        return self.model(x)

# 모델 초기화
model = MultivariateLinearRegressionModel()
# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=1e-5)
```

모델, optimizer는 lab4_1과 동일하게 정의해주고

```python
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CustomDataset(Dataset):
    def __init__(self):
        self.x_data = [[73, 80, 75],
                       [93, 88, 93], 
                       [89, 91, 90],
                       [96, 98, 100],
                       [73, 66, 70]]
        self.y_data = [[152], [185], [180], [196], [142]]
        
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.x_data[idx]), torch.FloatTensor(self.y_data[idx])
    
dataset = CustomDataset()

dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)
```

데이터 셋은 오늘 포스팅에 나온 방법으로 정의를 해준다.

이들을 가지고 학습하는 코드는 이렇게 작성할 수 있다.

```python
epochs = 20
for epoch in range(epochs + 1):
    for batch_idx, smaples in enumerate(dataloader):
        x_train, y_train = smaples
        
        prediction = model(x_train)
        
        cost = F.mse_loss(prediction, y_train)
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        # 5 epoch 마다 로그 출력
        if epoch%5 == 0:
            print('Epoch {:4d}/{} Batch {}/{} Cost {:6f}'.format(
                epoch, epochs, batch_idx, len(dataloader), cost.item()
            ))

'''output
Epoch    0/20 Batch 0/3 Cost 390.795288
Epoch    0/20 Batch 1/3 Cost 55.004257
Epoch    0/20 Batch 2/3 Cost 18.470478
Epoch    5/20 Batch 0/3 Cost 8.500160
Epoch    5/20 Batch 1/3 Cost 3.519112
Epoch    5/20 Batch 2/3 Cost 2.370358
Epoch   10/20 Batch 0/3 Cost 2.786815
Epoch   10/20 Batch 1/3 Cost 4.166077
Epoch   10/20 Batch 2/3 Cost 5.060166
Epoch   15/20 Batch 0/3 Cost 4.609153
Epoch   15/20 Batch 1/3 Cost 6.680350
Epoch   15/20 Batch 2/3 Cost 4.476605
Epoch   20/20 Batch 0/3 Cost 4.082047
Epoch   20/20 Batch 1/3 Cost 2.758399
Epoch   20/20 Batch 2/3 Cost 4.820268
'''
```

한 epoch 안에서 minibatch로 3번 학습하는 것을 확인할 수 있다. 다만 앞서 batch size가 작을수록 오래걸리고 효과는 좋다고 했는데 이번의 경우에는 성능이 그렇게 좋아 보이지 않는다(lab4_1에서의 최종 cost는 0.227729).

그 이유는 batch size가 너무 작기 때문이다. 우리가 minibatch를 사용하여 학습을 할 수 있는 이유는 minibatch가 전체 데이터 셋을 대표할 수 있다고 가정하기 때문이다. 하지만 이번 상황을 보면 batch size가 2 밖에 되지 않는다.
이정도 크기로는 아무리 생각해도 전체 데이터의 분포를 대표하기에는 무리가 있어 보인다. 이런 이유 때문에 성능이 떨어진 것이라고 할 수 있다.

그러므로 batch size를 설정할 때에는 충분히 전체 분포를 반영할 수 있도록 설정하는 주의를 기울여야 한다.
