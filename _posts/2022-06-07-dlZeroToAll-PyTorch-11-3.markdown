---
title: "모두를 위한 딥러닝 2 - Lab11-3: RNN - longseq"
author: Kwon
date: 2022-06-07T00:00:00 +0900
categories: [pytorch, study]
tags: [rnn]
math: true
mermaid: false
---

[모두를 위한 딥러닝](https://deeplearningzerotoall.github.io/season2/lec_pytorch.html) Lab11-3: RNN - hihello / charseq 강의를 본 후 공부를 목적으로 작성한 게시물입니다.

***

## Longseq

앞서 살펴보았던 RNN 예제들은 모두 한 단어나 짧은 문장에 대해 RNN을 학습시키는 내용들이었다.
하지만 우리가 다루고 싶은 데이터는 더 긴 문장이거나 내용을 가질 가능성이 높다.
이런 상황에서는 그 데이터 전체를 넣어 RNN을 학습시키기에는 들어가는 데이터의 길이도 매번 다를 뿐더러 그 크기가 너무 커서 학습이 불가능할 수도 있다.
그래서 일정한 크기의 window를 사용하여 RNN에 잘라서 넣어준다.

아래 문장을 보자.

```py
sentence = ("if you want to build a ship, don't drum up people together to ")
```

이 문장의 마지막 문자만 잘라서 바로 넣기 보다는 크기 10의 window로 잘라 넣으려고 한다.
이때 window를 오른쪽으로 한칸씩 밀어가면서 data를 만든다. 위 문장을 자르면 다음과 같다.

```
   x_data    ->    y_data

"if you wan" -> "f you want"
"f you want" -> " you want "
" you want " -> "you want t"
"you want t" -> "ou want to"
"ou want to" -> "u want to "
```

이렇게 하면 일정한 크기로 데이터를 잘라 학습을 할 수 있다. (x_data로 y_data를 학습하여 예측)

***

## with Code

### Imports

```py
import torch
import torch.optim as optim
import numpy as np

torch.manual_seed(0)
```

### Data

앞서 본 문장과 더불어 총 3개의 문장을 학습에 사용해 본다.

```py
sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")
```

[lab11-2](/posts/dlZeroToAll-PyTorch-11-2/)의 charseq 예제에서 봤던 것과 같은 방법으로 one-hot encoding에 사용할 dictionary를 생성한다.

```py
# make dictionary
char_set = list(set(sentence))
char_dic = {c: i for i, c in enumerate(char_set)}

# hyper parameters
dic_size = len(char_dic)
hidden_size = len(char_dic)
sequence_length = 10  # Any arbitrary number
learning_rate = 0.1
```

이후 window를 이용하여 자르는 방식으로 data를 만들고 one-hot encoding 한다.

```py
# data setting
x_data = []
y_data = []

# window를 오른쪽으로 움직이면서 자름
for i in range(0, len(sentence) - sequence_length):
    x_str = sentence[i:i + sequence_length]
    y_str = sentence[i + 1: i + sequence_length + 1]
    print(i, x_str, '->', y_str)

    x_data.append([char_dic[c] for c in x_str])  # x str to index (dict 사용)
    y_data.append([char_dic[c] for c in y_str])  # y str to index

x_one_hot = [np.eye(dic_size)[x] for x in x_data]

X = torch.FloatTensor(x_one_hot)
Y = torch.LongTensor(y_data)

'''output
0 if you wan -> f you want
1 f you want ->  you want 
2  you want  -> you want t
3 you want t -> ou want to
4 ou want to -> u want to 
...
166 ty of the  -> y of the s
167 y of the s ->  of the se
168  of the se -> of the sea
169 of the sea -> f the sea.
'''
```

### Model

문장이 더 길고 복잡하기 때문에 기존 한 층의 RNN으로는 학습이 잘 안 될 수 있다. 그래서 RNN 층을 더 쌓고 마지막에 fully connected layer를 연결하여 더 복잡한 모델을 만들어 사용해 볼 것이다.

![](/posting_imgs/lab11-3-1.png)

```py
# declare RNN + FC
class Net(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, layers):
        super(Net, self).__init__()
        self.rnn = torch.nn.RNN(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x):
        x, _status = self.rnn(x)
        x = self.fc(x)
        return x

net = Net(dic_size, hidden_size, 2)
```

`torch.nn.RNN`을 이용하여 RNN을 생성할 때 `num_layers`를 layer 수만큼 설정해 주는 것으로 원하는 층의 RNN을 생성할 수 있다.
RNN과 FC를 정의하고 foward에서 연결시켜주는 모습이다.

### Train

```py
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), learning_rate)

# start training
for i in range(100):
    optimizer.zero_grad()
    outputs = net(X)
    loss = criterion(outputs.view(-1, dic_size), Y.view(-1))
    loss.backward()
    optimizer.step()

    results = outputs.argmax(dim=2)
    predict_str = ""
    for j, result in enumerate(results):
        # print(i, j, ''.join([char_set[t] for t in result]), loss.item())
        if j == 0:
            predict_str += ''.join([char_set[t] for t in result])
        else:
            predict_str += char_set[result[-1]]

    print(predict_str)

'''output
hswmsshsmmmwsisshsmmhwwshhhhwmsssswmhmshhsmshismhhsismshmhmsismisshhswhissmwmmmmhssswwishhhwshmhhsmshsmwwmmhismhmsssmhmshmhmshmhhmshmhhsissimmhsismshmwwmmmhsshhhshmwsmmuismshmwwmm
                                                                                                                                                                                   
      t t  ttt ttt  t  ttt t ttt t t t t   t  t t ttt   t t  t t t tt t  ttt  tt t ttt t t   tt t t tt t t t t ttt ttt t ttt t tt t tt t  tt t t  ttt tt t t t t   t t t   t t  ttt
  b.o  b.   o  o             o                                        o        o                           o             o                     o   o       o   o                   
e ae  as      a    a aa  a       a        a   a  a      a  ata aa   aa   aa  a              a     a  aa   a                aa   aa   a   a aa   a  a  a     aa     a           aa  
e tl  ee teeeettlteeetl tleeeee eeeee eleet etteeetleeteeeeeeletleeeeeeeeeeeeteeoeestee eletteeeeeletteeeeteeeeeeeetelteeetleseteleteleeeetteteeoeteeee eeeeleeee eeeeeeteeeeeell e
e to  ot oo oot tooo ot ouoto o ootoo ouoou ooootto  oootu ootootoo oo ooo ooo ouoo ooo ooooo oouoto otoo uoouo ooo uoooo oto oootoo oooo  utoo oo ttot ooo oto ooo ooo  oooooouo o
e t o     ott      ttttt    u  tu ttt ot utt             tt t  t   tttttt   ttt   t     ut ut   tt      u   u t   t tt t   t        t ot  tt  u  t    t ut    ut  o  u   o  o  t  t
  t  t     t   t    t  t  tttt t  tt       t    t ttt  et t        t tt   t   t    tt      t e  t  tt t  t     tt        t t t   t  t    tt t tt  tt   t   t t t   t   t         t 
            t tt         t       tt              t         t t                 t t                                   t t   tt                      t           t   t         e  t  
           t   o    t             t        o              tt          o       t    ot                                t      t                  o  t                               t
e  o  t    oh  o  o o  o  oo oot  to o  oo o    o oooo  o o  to  o h too  oo  oo   oto o  oo oo to oo    o o  ooto o     o t o o      to  o  oto  oh o o   o   ooo  o  ot  tooo  o 
e  o  t  t th  o  o t  oeto too t to t tot o    o oo o  h o   o  o h too  to  th  toto    tt oeoto oo to o oo ooto       oot  oo to e to to  tto  th o o   o  ohoo toto t  to etht 
theo lt  thth te lo t  oete tollthto t tetto    e otte taet  to  o t too  too th ttoto it tt teeto ee to e eo eott e   t tet t o to e lo to  tto tth o o   e  thee tetoath totetht 
thto tt to to to te t to cettoelthto   to toe tteto  e ae t  to te   to   te  th ttoto e  tehe  to    tet to  eoto  th  t  tht este e lo to  eto tto eto      eo e neeeaoh toe thes
thto  t  t to lt t  t to t  to  t to   totto     to       t  lo      to t t   th ttot   tt t    to    t   to   oto  th  d  th    toe  to t    to tto  to  s    o   n   tt  tot th s
t to  t  o to to to ttto to to  t to t totto    oto t   o t  to eo   to   to  th ttoto  tttt    to o  to  too  oto  th  d  t   o to   to to   to tto  tod  o  to     t tt  to  th  
thto  th t to to t  t  o to to  t to t totto     to       t  to      wo   tot th  thto  t tt    to    to  too   to  t      t   o to   to to   uo tth  to   o  to     to t  to  th  
thto  th t to to t  t    to to  t to   totto     to     m to to      wo   tot th  thto  t tth   to    to  to    to  t      th  o to   to to   to uth  to      to     to t  to  th  
thto ethet to to to tht  to to dthto   totto     to    em tonto e    wo   tot th  thto  d tth   to    to  to    to  t      th  o toe  to to e to ethi to     eto   n t  o  to  th  
thto eth t to lo 'o tht eto to 'thto   tttt  kl eth    em thnto e    to   tot th  thto  t tthe  to    to  to    to  th n  ethe p toe  to to e to  thi        ett e n th t  toe the 
tuto eto t to tt 'e tht epo to 'ththt  tmtt  kl eth     n th to e  e to   tot th  toto  t tthe  totho tot to l etot to oe ethe t toi  to to e to  thi   d     tt e   thta  toe the 
tuto lao t todto 'e thtoeco tos't tos  totto kl  to    en to to ee d to   tod to  toto  d 't em totoo tot to  p tosoto nipetoe p toio to lo t to  toio od  o  to e   t tou toemtoe 
tuto  to t to tp  e tos  to to 't to   tootoo l eto    en to to ee o wo   tot to  toao  t tthem tosoo tod to    so  t so noth np toe  to to deso ltoemtod sm  to     t eaa toem oe 
'utos to t to los e tot epo to 't tosl to tn  le tos   en to lo  h n wo   tot to  toaon t dt er tos   tot tor , wod tos enotossa toe  to lo   sod tosm  d sm  tn e   theoa toe to  
...
g you want to build a ship, don't drum up people together to collect wood and don't assign them tasks and work, but rather teach them to long for the sndless immensity of the sea.
f you want to build a ship, don't drum up people together to collect wood and don't assign them tasks and work, but rather teach them to long for the sndless immensity of the sea.
f you want to build a ship, don't drum up people together to collect wood and don't assign them tasks and work, but rather teach them to long for the sndless immensity of the sea.
f you want to build a ship, don't drum up people together to collect wood and don't assign them tasks and work, but rather teach them to long for the sndless immensity of the sea.
'''
```

loss와 optimizer를 정의하고 학습을 진행한다.
처음에는 상당히 이상한 문장들이 출력된다. 하지만 학습이 진행됨에 따라 원래의 문장에 가깝게 문장들이 출력되는 것을 확인할 수 있으며, 이는 학습이 잘 되었다고 볼 수 있다.