---
title: "모두를 위한 딥러닝 2 - Lab10-4: ImageFolder"
author: Kwon
date: 2022-05-23T01:00:00 +0900
categories: [pytorch, study]
tags: [imagefolder, cnn, transform]
math: true
mermaid: false
---
[모두를 위한 딥러닝](https://deeplearningzerotoall.github.io/season2/lec_pytorch.html) Lab10-4: ImageFolder 강의를 본 후 공부를 목적으로 작성한 게시물입니다.

***

## ImageFolder

`torchvision.datasets`에 있는 ImageFolder는 directory에 따라 category를 자동으로 labeling 하여 데이터로 만들어 준다. 우리가 찍은 사진을 학습하는데 사용할 때 아주 좋은 기능이다.

이번 강의에서는 미리 찍어서 제공해 준 회색, 빨간색 의자 사진 분류를 해볼 것이다.

먼저 category에 맞게 directory를 생성해주어야 ImageFolder를 사용할 수 있으므로, directory는 다음과 같은 구조여야 한다.

![](/posting_imgs/lab10-4-1.png)

```py
import torchvision
from torchvision import transforms

from torch.utils.data import DataLoader

from matplotlib.pyplot import imshow
%matplotlib inline

trans = transforms.Compose([
    transforms.Resize((64,128))
])

train_data = torchvision.datasets.ImageFolder(root='custom_data/origin_data', transform=trans)
```

원본 데이터가 있는 곳을 root로 잡고 `Compose`를 통해 적용할 `transforms`들을 묶어 넣어준다. 원본 데이터가 265x512로 너무 커서 64x128로 바꾸어주는 과정을 거친다.
여기서는 하나의 `transforms`을 사용하지만 어러개를 사용해야할 때 `Compose`로 묶어 사용할 수 있다.

이렇게 불러온 데이터들을 정리하여 train data로 만들어준다. directory 상 gray가 더 빠르므로 label 0, red가 label 1이다.

```py
for num, value in enumerate(train_data):
    data, label = value
    print(num, data, label)
    
    if(label == 0):
        data.save('custom_data/train_data/gray/%d_%d.jpeg'%(num, label))
    else:
        data.save('custom_data/train_data/red/%d_%d.jpeg'%(num, label))
```

***

## Train

### Imports and Data

```py
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device =='cuda':
    torch.cuda.manual_seed_all(777)

trans = transforms.Compose([
    transforms.ToTensor()
])

train_data = torchvision.datasets.ImageFolder(root='./custom_data/train_data', transform=trans)
data_loader = DataLoader(dataset = train_data, batch_size = 8, shuffle = True, num_workers=2)
```

앞서 만들어놓은 train data를 `ImageFolder`로 불러와서 사용한다. 물론 자동으로 label을 붙여 데이터를 생성해준다.

### Model and Loss/Optimizer

두 번의 CNN layer를 거치고 FC layer를 하나 통과시키는 [lab10-2](https://qja1998.github.io/2022/05/21/dlZeroToAll-PyTorch-10-3/)에서 사용한 것과 data shape말고는 거의 같은 모델이다.

```py
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,6,5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6,16,5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(16*13*29, 120),
            nn.ReLU(),
            nn.Linear(120,2)
        )
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.shape[0], -1)
        out = self.layer3(out)
        return out
```

이렇게 만든 model은 꼭 테스트 하는 과정을 거쳐야 한다고 한다. 테스트는 넣으려는 데이터와 shape이 같은 Tensor를 생성하여 통과시켜보는 것을 말한다.

```py
#testing 
net = CNN().to(device)
test_input = (torch.Tensor(3,3,64,128)).to(device)
test_out = net(test_input)
```

optimizer와 loss도 역시 동일하다.

```py
optimizer = optim.Adam(net.parameters(), lr=0.00005)
loss_func = nn.CrossEntropyLoss().to(device)
```

### Train model

학습 역시 이전과 다르지 않개 진행한다.

```py
total_batch = len(data_loader)

epochs = 7
for epoch in range(epochs):
    avg_cost = 0.0
    for num, data in enumerate(data_loader):
        imgs, labels = data
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        out = net(imgs)
        loss = loss_func(out, labels)
        loss.backward()
        optimizer.step()
        
        avg_cost += loss / total_batch
        
    print('[Epoch:{}] cost = {}'.format(epoch+1, avg_cost))
print('Learning Finished!')

'''output
[Epoch:1] cost = 0.6341210007667542
[Epoch:2] cost = 0.3761218190193176
[Epoch:3] cost = 0.1116236224770546
[Epoch:4] cost = 0.03525366261601448
[Epoch:5] cost = 0.016341226175427437
[Epoch:6] cost = 0.009176642633974552
[Epoch:7] cost = 0.005688846111297607
Learning Finished!
'''
```

### Save model

이렇게 학습한 모델을 매번 다시 학습하는 것은 너무 비효율적이다. 그래서 모델을 저장해 준다.

```py
torch.save(net.state_dict(), "./model/model.pth")
```

불러오는 것은 다음과 같이 새로운 CNN 객체를 생성하여 넣어주면 된다.

```py
new_net = CNN().to(device)
new_net.load_state_dict(torch.load('./model/model.pth'))
```

기존의 모델과 동일한 것을 확인할 수 있다.

```py
print(net.layer1[0])
print(new_net.layer1[0])

print(net.layer1[0].weight[0][0][0])
print(new_net.layer1[0].weight[0][0][0])

net.layer1[0].weight[0] == new_net.layer1[0].weight[0]

'''output
Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
tensor([-0.0914,  0.0032, -0.0170, -0.0211,  0.0933], device='cuda:0',
       grad_fn=<SelectBackward>)
tensor([-0.0914,  0.0032, -0.0170, -0.0211,  0.0933], device='cuda:0',
       grad_fn=<SelectBackward>)
tensor([[[True, True, True, True, True],
         [True, True, True, True, True],
         [True, True, True, True, True],
         [True, True, True, True, True],
         [True, True, True, True, True]],

        [[True, True, True, True, True],
         [True, True, True, True, True],
         [True, True, True, True, True],
         [True, True, True, True, True],
         [True, True, True, True, True]],

        [[True, True, True, True, True],
         [True, True, True, True, True],
         [True, True, True, True, True],
         [True, True, True, True, True],
         [True, True, True, True, True]]], device='cuda:0')
'''
```

### Test

test할 때도 train data와 똑같이 처리하여 사용하면 된다.

```py
trans=torchvision.transforms.Compose([
    transforms.Resize((64,128)),
    transforms.ToTensor()
])
test_data = torchvision.datasets.ImageFolder(root='./custom_data/test_data', transform=trans)

test_set = DataLoader(dataset = test_data, batch_size = len(test_data))
```

이렇게 하면 역시 label이 붙은 채로 data가 생성되게 된다.

test 결과는 다음과 같다.

```py
with torch.no_grad():
    for num, data in enumerate(test_set):
        imgs, label = data
        imgs = imgs.to(device)
        label = label.to(device)
        
        prediction = net(imgs)
        
        correct_prediction = torch.argmax(prediction, 1) == label
        
        accuracy = correct_prediction.float().mean()
        print('Accuracy:', accuracy.item())
```