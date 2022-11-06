---
title: "모두를 위한 딥러닝 2 - Lab11-2: RNN - hihello / charseq"
author: Kwon
date: 2022-06-06T01:00:00 +0900
categories: [pytorch, study]
tags: [rnn]
math: true
mermaid: false
---

[모두를 위한 딥러닝](https://deeplearningzerotoall.github.io/season2/lec_pytorch.html) Lab11-2: RNN - hihello / charseq 강의를 본 후 공부를 목적으로 작성한 게시물입니다.

***

## 'hihello' problem

hihello 문제는 같은 문자들이 다음 문자가 다른 경우 이를 예측하는 문제를 말한다.
hihello에서 'h'와 'l'은 2번씩 등장하지만 어디에 문자가 위치하느냐에 따라 다음에 올 문자가 달라진다.
이런 경우가 RNN의 **hidden state**가 빛을 발휘하는 경우이다. 알파벳 만으로 판별할 수 없지만 순서를 기억하여 뒷 문자를 예측하는데 도움이 되기 때문이다.

### with Code

[lab11-1](https://qja1998.github.io/2022/05/26/dlZeroToAll-PyTorch-11-1/)의 코드를 확장하여 일반화한 것이다.
전체적인 구조는 거의 동일하기 때문에 추가된 부분에 초점을 맞춰 살펴보려 한다.

```py
char_set = ['h', 'i', 'e', 'l', 'o']

# hyper parameters
input_size = len(char_set)
hidden_size = len(char_set)
learning_rate = 0.1

# data setting
x_data = [[0, 1, 0, 2, 3, 3]]
x_one_hot = [[[1, 0, 0, 0, 0],
              [0, 1, 0, 0, 0],
              [1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 1, 0],
              [0, 0, 0, 1, 0]]]
y_data = [[1, 0, 2, 3, 3, 4]]

X = torch.FloatTensor(x_one_hot)
Y = torch.LongTensor(y_data)
```

마찬가지로 one-hot encoding하여 Tensor로 바꾼다. 다만 각 알파벳 변수에 배열을 저장하는 방식이 아니라 `char_set`에 저장된 알파벳을 `x_data`의 값을 인덱스로 불러오는 방식이다.
one-hot encoding은 `x_data`에 적용하여 학습한다.

데이터를 자세히 보면 input(x)은 마지막 문자가 없고 target(y)은 첫 문자가 없는 것을 볼 수 있는데,
이건 각 차시의 RNN이 다음 문자를 출력하기 때문이다. input에

다음은 모델과 loss, optimizer를 만들어준다. 마찬가지로 `torch.nn.RNN`를 사용하여 정의한다.

```py
rnn = torch.nn.RNN(input_size, hidden_size, batch_first=True)  # batch_first guarantees the order of output = (B, S, F)

# loss & optimizer setting
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn.parameters(), learning_rate)
```

```py
# start training
for i in range(100):
    optimizer.zero_grad()
    outputs, _status = rnn(X)
    loss = criterion(outputs.view(-1, input_size), Y.view(-1))
    loss.backward()
    optimizer.step()

    result = outputs.data.numpy().argmax(axis=2)
    result_str = ''.join([char_set[c] for c in np.squeeze(result)])
    print(i, "loss: ", loss.item(), "prediction: ", result, "true Y: ", y_data, "prediction str: ", result_str)

'''output
0 loss:  1.7802648544311523 prediction:  [[1 1 1 1 1 1]] true Y:  [[1, 0, 2, 3, 3, 4]] prediction str:  iiiiii
1 loss:  1.4931954145431519 prediction:  [[1 4 1 1 4 4]] true Y:  [[1, 0, 2, 3, 3, 4]] prediction str:  ioiioo
2 loss:  1.3337129354476929 prediction:  [[1 3 2 3 1 4]] true Y:  [[1, 0, 2, 3, 3, 4]] prediction str:  ilelio
3 loss:  1.215295433998108 prediction:  [[2 3 2 3 3 3]] true Y:  [[1, 0, 2, 3, 3, 4]] prediction str:  elelll
4 loss:  1.1131411790847778 prediction:  [[2 3 2 3 3 3]] true Y:  [[1, 0, 2, 3, 3, 4]] prediction str:  elelll
5 loss:  1.0241888761520386 prediction:  [[2 3 2 3 3 4]] true Y:  [[1, 0, 2, 3, 3, 4]] prediction str:  elello
6 loss:  0.9573155045509338 prediction:  [[2 3 2 3 3 4]] true Y:  [[1, 0, 2, 3, 3, 4]] prediction str:  elello
7 loss:  0.9102011322975159 prediction:  [[2 0 2 3 3 4]] true Y:  [[1, 0, 2, 3, 3, 4]] prediction str:  ehello
...
96 loss:  0.5322802066802979 prediction:  [[1 3 2 3 3 4]] true Y:  [[1, 0, 2, 3, 3, 4]] prediction str:  ilello
97 loss:  0.5321123003959656 prediction:  [[1 3 2 3 3 4]] true Y:  [[1, 0, 2, 3, 3, 4]] prediction str:  ilello
98 loss:  0.5319531559944153 prediction:  [[1 3 2 3 3 4]] true Y:  [[1, 0, 2, 3, 3, 4]] prediction str:  ilello
99 loss:  0.5317898392677307 prediction:  [[1 3 2 3 3 4]] true Y:  [[1, 0, 2, 3, 3, 4]] prediction str:  ilello
```

학습을 진행하는 것에 크게 특이한 점은 없고 결과를 낼때 `argmax`를 통해 one-hot vector를 index 값으로 바꿔줘야 한다. 이렇게 바꾼 output은 `''.join([char_set[c] for c in np.squeeze(result)])`를 통해 실제 단어로 바꿔 출력할 수 있다.

결과를 보면 처음에는 이상한 단어들이 나오다가 마지막에 다와서는 첫 문자를 제외한 'ilello'가 제대로 나온 것을 확인할 수 있다.

***

## Charseq

지금까지 했던 것을 다시 한번 일반화 시켜 임의의 문장도 학습할 수 있도록 한다.

### Data

```py
sample = " if you want you"

# make dictionary
char_set = list(set(sample))
char_dic = {c: i for i, c in enumerate(char_set)}
print(char_dic)

'''output
<torch._C.Generator at 0x22912756f30>
{'o': 0, 'n': 1, 'a': 2, ' ': 3, 'w': 4, 'i': 5, 'y': 6, 't': 7, 'u': 8, 'f': 9}
'''
```

set을 통해 중복된 문자를 제거하고, 문자와 문자의 index를 담는 dictionary를 만들어 사용한다.

```py
# hyper parameters
dic_size = len(char_dic)
hidden_size = len(char_dic)
learning_rate = 0.1

# data setting
sample_idx = [char_dic[c] for c in sample]
x_data = [sample_idx[:-1]]
x_one_hot = [np.eye(dic_size)[x] for x in x_data]
y_data = [sample_idx[1:]]

# transform as torch tensor variable
X = torch.FloatTensor(x_one_hot)
Y = torch.LongTensor(y_data)
```

one-hot encodeng을 identity matrix(단위행렬)를 통해 진행한다. `np.eye(size)`를 통해 만들 수 있는 단위행렬은 주대각선(좌상우하)의 원소가 모두 1이고 나머지는 모두 0인 정사각 행렬이다.
앞서 뽑아냈던 문자들의 index를 사용하여 단위행렬의 한 줄을 뽑아내면 그것이 곧 해당 문자의 one-hot vetor가 되기 때문에 손쉽게 one-hot encodeng을 할 수 있다.

이후 데이터의 길이에 맞춰 각 size를 정의하고 x에서는 맨 뒤 문자, y에서는 맨 앞 문자를 빼서 학습할 수 있도록 한다.
그리고 만들어진 데이터를 Tensor로 바꿔준다.

### Train Result

모델과 학습은 다르지 않으므로 결과만 한번 살펴보자

```py
'''output
0 loss:  2.4069371223449707 prediction:  [[7 7 0 7 8 5 8 7 8 7 8 0 7 8 5]] true Y:  [[5, 9, 3, 6, 0, 8, 3, 4, 2, 1, 7, 3, 6, 0, 8]] prediction str:  ttotuiututuotui
1 loss:  2.1236345767974854 prediction:  [[1 0 0 1 0 8 0 1 8 8 8 1 1 0 8]] true Y:  [[5, 9, 3, 6, 0, 8, 3, 4, 2, 1, 7, 3, 6, 0, 8]] prediction str:  noonouonuuunnou
2 loss:  1.8809428215026855 prediction:  [[6 0 3 6 0 8 3 6 0 8 7 3 6 0 8]] true Y:  [[5, 9, 3, 6, 0, 8, 3, 4, 2, 1, 7, 3, 6, 0, 8]] prediction str:  yo you yout you
3 loss:  1.71848464012146 prediction:  [[6 0 3 6 0 8 3 6 4 5 7 3 6 0 8]] true Y:  [[5, 9, 3, 6, 0, 8, 3, 4, 2, 1, 7, 3, 6, 0, 8]] prediction str:  yo you ywit you
4 loss:  1.5743740797042847 prediction:  [[6 0 3 6 0 8 3 6 2 5 7 3 6 0 8]] true Y:  [[5, 9, 3, 6, 0, 8, 3, 4, 2, 1, 7, 3, 6, 0, 8]] prediction str:  yo you yait you
5 loss:  1.4554158449172974 prediction:  [[6 9 3 6 0 8 3 6 8 5 7 3 6 0 8]] true Y:  [[5, 9, 3, 6, 0, 8, 3, 4, 2, 1, 7, 3, 6, 0, 8]] prediction str:  yf you yuit you
6 loss:  1.3661972284317017 prediction:  [[5 9 3 6 0 8 3 6 2 5 7 3 6 2 8]] true Y:  [[5, 9, 3, 6, 0, 8, 3, 4, 2, 1, 7, 3, 6, 0, 8]] prediction str:  if you yait yau
7 loss:  1.2864983081817627 prediction:  [[5 9 3 6 2 8 3 6 2 1 7 3 6 2 8]] true Y:  [[5, 9, 3, 6, 0, 8, 3, 4, 2, 1, 7, 3, 6, 0, 8]] prediction str:  if yau yant yau
8 loss:  1.2224119901657104 prediction:  [[5 9 3 6 2 8 3 6 2 1 7 3 6 2 8]] true Y:  [[5, 9, 3, 6, 0, 8, 3, 4, 2, 1, 7, 3, 6, 0, 8]] prediction str:  if yau yant yau
...
46 loss:  0.8302408456802368 prediction:  [[5 9 3 6 0 8 3 4 2 1 7 3 6 0 8]] true Y:  [[5, 9, 3, 6, 0, 8, 3, 4, 2, 1, 7, 3, 6, 0, 8]] prediction str:  if you want you
47 loss:  0.8290660381317139 prediction:  [[5 9 3 6 0 8 3 4 2 1 7 3 6 0 8]] true Y:  [[5, 9, 3, 6, 0, 8, 3, 4, 2, 1, 7, 3, 6, 0, 8]] prediction str:  if you want you
48 loss:  0.8275652527809143 prediction:  [[5 9 3 6 0 8 3 4 2 1 7 3 6 0 8]] true Y:  [[5, 9, 3, 6, 0, 8, 3, 4, 2, 1, 7, 3, 6, 0, 8]] prediction str:  if you want you
49 loss:  0.8264601230621338 prediction:  [[5 9 3 6 0 8 3 4 2 1 7 3 6 0 8]] true Y:  [[5, 9, 3, 6, 0, 8, 3, 4, 2, 1, 7, 3, 6, 0, 8]] prediction str:  if you want you
'''
```

마찬가지로 학습의 막바지로 갈수록 학습했던 문장이 잘 나오는 것을 확인할 수 있다.