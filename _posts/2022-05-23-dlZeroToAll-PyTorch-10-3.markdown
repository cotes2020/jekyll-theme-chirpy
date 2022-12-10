---
title: "모두를 위한 딥러닝 2 - Lab10-3: Visdom"
author: Kwon
date: 2022-05-23T00:00:00 +0900
categories: [pytorch, study]
tags: [visdom, visualization]
math: true
mermaid: false
---
[모두를 위한 딥러닝](https://deeplearningzerotoall.github.io/season2/lec_pytorch.html) Lab10-3: Visdom 강의를 본 후 공부를 목적으로 작성한 게시물입니다.

***

## Visdom

Visdom은 Meta 사(facebook)에서 제공하는 PyTorch에서 사용할 수 있는 시각화 도구이다. 실시간으로 데이터를 시각화하면서 바뀌는 점을 확인할 수 있다는 장점이 있다.

### Install

터미널에서 pip를 이용하여 설치할 수 있다. 설치가 완료된 후에는 visdom server를 실행해 주어야 사용이 가능하다.

```python
> pip isntall visdom
> python -m visdom.server
# You can navigate to http://localhost:PORT
```

서버를 실행한 후에 나오는 localhost 주소를 통해 visdom 화면을 확인할 수 있다.

#### AttributeError

필자의 경우 `AttributeError: module 'brotli' has no attribute 'error'`라는 에러가 발생해서 아래 코드를 통해 `brotli` module을 추가서 설치하여 해결하였다.

```python
conda install -c conda-forge brotlipy
```

### Text

먼저 설치에 문제가 없는지 확인하는 겸 간단한 text를 출력해보자. `vis.text()`사용하여 출력할 수 있다.

```python
vis.text("Hello, world!",env="main")
```

![](/posting_imgs/lab10-3-1.png)

위와같이 새로운 창에 Hello, world가 출력되는 것을 확인할 수 있다.

### Image

이번에는 이미지를 출력해보자

무작위 픽셀로 생성한 200x200 이미지와 3개의 28x28 이미지를 만들고 출력해보면 다음과 같다. 이 때는 하나의 이미지는 `vis.image()`를, 여러 이미지는 `vis.images()`를 사용한다.

```python
a=torch.randn(3,200,200)
vis.image(a)
vis.images(torch.Tensor(3,3,28,28))
```

![](/posting_imgs/lab10-3-2.png)

다음은 조금 더 이미지 다운 데이터인 mnist와 CIFAR10를 출력하려 한다.

```python
MNIST = dsets.MNIST(root="./MNIST_data",train = True,transform=torchvision.transforms.ToTensor(), download=True)
cifar10 = dsets.CIFAR10(root="./cifar10",train = True, transform=torchvision.transforms.ToTensor(),download=True)

#CIFAR10
data = cifar10.__getitem__(0)
print(data[0].shape)
vis.images(data[0],env="main")

# MNIST
data = MNIST.__getitem__(0)
print(data[0].shape)
vis.images(data[0],env="main")
```

![](/posting_imgs/lab10-3-3.png)

두꺼비(?)와 숫자 5가 잘 나온다. 또한 이런 이미지들도 당연히 `vis.images()`를 통해 한번에 많은 이미지도 출력할 수 있다.

```py
data_loader = torch.utils.data.DataLoader(dataset = MNIST,
                                          batch_size = 32,
                                          shuffle = False)
for num, value in enumerate(data_loader):
    value = value[0]
    print(value.shape)
    vis.images(value)
    break
```

![](/posting_imgs/lab10-3-4.png)

굳이 for문을 쓰지 않고도 다음과 같이 iter 객체를 사용하여 출력할 수도 있다.

```py
img = next(iter(data_loader))[0]
vis.images(img)
```

지금까지 띄운 창들을 모두 끄고싶으면 다음 코드를 실행해서 끌 수 있다.

```py
vis.close(env="main")
```

이렇게 하면 main에 띄워진 것들을 모두 끌 수 있다.

### Line Plot

Lint Plot은 `vis.line()`에 X, Y 데이터를 넣어 선형 그래프를 그릴 수 있다.

```py
Y_data = torch.randn(5)
plt = vis.line (Y=Y_data)

X_data = torch.Tensor([1,2,3,4,5])
plt = vis.line(Y=Y_data, X=X_data)
```

![](/posting_imgs/lab10-3-5.png)

가장 간단한 예제로 Y 데이터만 설정해 준 것인데, 이 경우에 X축은 무조건 0과 1 사이를 나눠 point를 생성한다.
만약 X 값들을 다르게 만들어주고 싶으면 새로운 tensor를 만들어 넣어주면 된다.

기존의 plot에 point를 추가할 수도 있다. `vis.line()`에 새로 넣을 데이터와 그 데이터를 추가할 plot을 넣어주고, `update='append'`로 설정한다.

```py
Y_append = torch.randn(1)
X_append = torch.Tensor([6])

vis.line(Y=Y_append, X=X_append, win=plt, update='append')
```
![](/posting_imgs/lab10-3-6.png)

두개의 다른 그래프를 비교하기 위해 겹쳐 그리고 싶으면 (n, 2) shape의 데이터와 그에 맞는 X 데이터를 넣어주면 된다.

```py
num = torch.Tensor(list(range(0,10)))
num = num.view(-1,1)
num = torch.cat((num,num),dim=1)

plt = vis.line(Y=torch.randn(10,2), X = num)
```

![](/posting_imgs/lab10-3-7.png)

X는 0-9이 2줄 저장되어있는 데이터, Y는 (10,2)의 랜덤 데이터이다.

그래프에 대한 범례는 `showlegend=True`로 보이게 할 수 있고, `legend = []`를 통해 직접 지정해 줄 수 있다. defualt는 그낭 정수로 나온다.

```py
plt = vis.line(Y=Y_data, X=X_data, opts = dict(title='Test', showlegend=True))
plt = vis.line(Y=Y_data, X=X_data, opts = dict(title='Test', legend = ['1번'],showlegend=True))
plt = vis.line(Y=torch.randn(10,2), X = num, opts=dict(title='Test', legend=['1번','2번'],showlegend=True))
```

![](/posting_imgs/lab10-3-8.png)

<br>

마지막으로, 가장 재밌게 봤던 기능인데 아까 나온 append 기능을 통해 실행이 반복될 때마다 plot을 자동으로 업데이트하도록 코드를 작성할 수 있다.

```py
def loss_tracker(loss_plot, loss_value, num):
    '''num, loss_value, are Tensor'''
    vis.line(X=num,
             Y=loss_value,
             win = loss_plot,
             update='append'
             )

plt = vis.line(Y=torch.Tensor(1).zero_())

for i in range(500):
    loss = torch.randn(1) + i
    loss_tracker(plt, loss, torch.Tensor([i]))
```

![](/posting_imgs/lab10-3-9.gif)

실제로 학습할 때 loss를 넣어 학습을 모니터링하는데 유용하게 쓸 수 있다.

[이전 포스팅](/posts/dlZeroToAll-PyTorch-10-3/)에서 학습할 때 사용했던 코드에서 `loss_tracker`를 추가하여 실행하면 다음과 같이 모니터링이 가능하다.

```py
total_batch = len(data_loader)

for epoch in range(training_epochs):
    avg_cost = 0
    
    for X, Y in data_loader:
        X = X.to(device)
        Y = Y.to(device)
        
        optimizer.zero_grad()
        hypothesis = model(X)
        
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()
        
        avg_cost += cost / total_batch
    
    print('[Epoch:{}] cost = {}'.format(epoch+1, avg_cost))
    # tracking
    loss_tracker(loss_plt, torch.Tensor([avg_cost]), torch.Tensor([epoch]))
print('Learning Finished!')
```

![](/posting_imgs/lab10-3-10.gif)

loss가 어떻게 변하고 있는지 확인하는데 매우 적합한 기능인것 같다.