---
title: "모두를 위한 딥러닝 2 - Lab11-1: RNN Baisics"
author: Kwon
date: 2022-06-06T00:00:00 +0900
categories: [pytorch, study]
tags: [rnn]
math: true
mermaid: false
---

[모두를 위한 딥러닝](https://deeplearningzerotoall.github.io/season2/lec_pytorch.html) Lab11-1: RNN Baisics 강의를 본 후 공부를 목적으로 작성한 게시물입니다.

***

## with PyTorch

PyTorch에서 RNN은 in/output size만 잘 맞춰주면 바로 사용이 가능하다.
**"h, e, l, o"** 4개의 알파벳으로 이루어진 데이터셋을 통해 2차원의 output(class가 2개)을 내는 RNN을 만들어볼 것이다.

```py
import torch
import numpy as np

torch.manual_seed(0)

input_size = 4
hidden_size = 2

# one-hot encoding
h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]
input_data_np = np.array([[h, e, l, l, o], [e, o, l, l, l], [l, l, e, e, l]], dtype=np.float32)

# transform as torch tensor
input_data = torch.Tensor(input_data_np)
``` 

위와 같이 one-hot encoding하여 데이터를 만들들고 Tensor로 바꿔준다.
알파벳에 그냥 숫자(index)를 붙여 사용하지 않고 굳이 one-hot encoding을 하는 이유는 숫자 크기에 따라 network가 의미를 부여할 수 있기 때문이다.
실제로는 그저 알파벳일 뿐이지만 더 큰 숫자를 할당 받은 알파벳을 더 중요하게 생각하면서 학습할 수 있다는 것이다.

다시 돌아오면 알파벳의 종류가 4가지이기 때문에 input data의 한 차원은 4로 shape이 `(-, -, 4)`가 되고,
단어의 길이(**sequence length**)가 5이므로 shape은 `(-, 5, 4),`
마지막으로 data의 개수가 3개(**batch size**)이기 때문에 shape이 `(3, 5, 4)`가 된다.

다음으로 RNN layer를 만들 차례이다. `torch.nn.RNN`을 통해 RNN layer를 만들 수 있으며 이때 in/output size를 지정해주어야 한다.
input size은 알파벳의 종류(4)가 되고 output size는 우리가 원하는 class의 개수(2)가 된다.
sequence length와 abtch size는 input과 동일하며, data만 잘 만들어서 넣어줬다면 PyTorch에서 알아서 처리해주기 때문에 따로 입력할 필요가 없다.

```py
rnn = torch.nn.RNN(input_size, hidden_size)

outputs, _status = rnn(input_data)
print(outputs)
print(outputs.size())

'''output
tensor([[[-0.7497, -0.6135],
         [-0.5282, -0.2473],
         [-0.9136, -0.4269],
         [-0.9136, -0.4269],
         [-0.9028,  0.1180]],

        [[-0.5753, -0.0070],
         [-0.9052,  0.2597],
         [-0.9173, -0.1989],
         [-0.9173, -0.1989],
         [-0.8996, -0.2725]],

        [[-0.9077, -0.3205],
         [-0.8944, -0.2902],
         [-0.5134, -0.0288],
         [-0.5134, -0.0288],
         [-0.9127, -0.2222]]], grad_fn=<StackBackward>)
torch.Size([3, 5, 2])
'''
```

한 가지 이상하다고 생각할 수 있는 부분이 있다. 분명 output size을 입력한다고 했는데 `hidden_size`라고 정의하여 넣었다.
이는 RNN의 내부 구조를 보면 알 수 있다.

![](/posting_imgs/lab11-1-1.png)

빨간 박스 부분을 보면 hidden으로 넘어가는 부분과 output으로 나가는 data가 결국 같은 data에서 나눠지는 것을 볼 수 있다.
그러므로 hidden size와 output size는 같다. 이 때문에 `hidden_size`라고 정의하여 넣은 것이다.

### Data shape in RNN

앞서 나온 shape을 정리해 보면 다음과 같다.

![](/posting_imgs/lab11-1-2.png)

(batch_size, sequence_length, dimension) 순으로 데이터가 구성되며,  앞서 언급한 대로 in/output size만 잘 넣어주면 나머지는 PyTorch가 데이터에 맞게 처리해준다.