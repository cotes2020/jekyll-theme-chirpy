---
title: "ëª¨ë‘ë¥? ?œ„?•œ ?”¥?Ÿ¬?‹ 2 - Lab3: Minimizing Cost"
author: Kwon
date: 2022-04-18T14:00:00+0900
categories: [pytorch, study]
tags: [linear-regressoin, cost, gradient-descent]
math: true
mermaid: false
---

[ëª¨ë‘ë¥? ?œ„?•œ ?”¥?Ÿ¬?‹](https://deeplearningzerotoall.github.io/season2/lec_pytorch.html) Lab 3: Minimizing Cost ê°•ì˜ë¥? ë³? ?›„ ê³µë??ë¥? ëª©ì ?œ¼ë¡? ?‘?„±?•œ ê²Œì‹œë¬¼ì…?‹ˆ?‹¤.

***
## Theoretical Overview
?´ë²ˆì—?Š” Grdient descent?— ????•´ ì¡°ê¸ˆ ?” ì§‘ì¤‘? ?œ¼ë¡? ?•Œ?•„ë³´ê¸° ?œ„?•´ Hypothesisë¥? ì¡°ê¸ˆ ?” ?‹¨?ˆœ?•˜ê²? $ H(x) = Wx $ë¡? ë°”ê¾¸?–´ ?‚´?´ë³´ì.

cost?Š” ?˜‘ê°™ì´ MSE(Mean Square Error)ë¥? ?‚¬?š©?•˜ê³? ?°?´?„°?„ ?´? „ê³? ê°™ì?? ê³µë?? ?‹œê°? - ?„±?  ?°?´?„°ë¥? ?‚¬?š©?•œ?‹¤. ([Lab2 ?¬?Š¤?Œ… ì°¸ì¡°](http://qja1998.github.io/2022/04/17/dlZeroToAll-PyTorch-2/))

\\[ MSE = cost(W) = \frac{1}{m} \sum^m_{i=1} \left( H(x^{(i)}) - y^{(i)} \right)^2 \\]

***

## Import
{% highlight python %}
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
{% endhighlight %}

***
## Cost by W
W?˜ ë³??™”?— ?”°ë¥? cost ê·¸ë˜?”„ë¥? ê·¸ë ¤ë³´ë©´ ?‹¤?Œê³? ê°™ì?? 2ì°? ê³¡ì„ ?´ ê·¸ë ¤ì§„ë‹¤. ê·¸ëŸ¬ë¯?ë¡? costê°? ê°??¥ ?‘??? ? ??? ê¸°ìš¸ê¸?(ë¯¸ë¶„ê°?)ê°? 0?¸ ê·¹ì†Œ? ?´?‹¤.

{% highlight python %}
W_l = np.linspace(-5, 7, 1000)
cost_l = []
for W in W_l:
    hypothesis = W * x_train
    cost = torch.mean((hypothesis - y_train) ** 2)

    cost_l.append(cost.item())

plt.plot(W_l, cost_l)
plt.xlabel('$W$')
plt.ylabel('Cost')
plt.show()
{% endhighlight %}
![](/posting_imgs/images/lab3-1.png)
<br>

***
## Gradient Descent by Hand
costê°? ê°??¥ ?‘??? ? ?„ ì°¾ëŠ” ê²ƒì´ ?š°ë¦¬ì˜ ëª©í‘œ?¸?°, ?´ê²ƒì„ cost?˜ ë¯¸ë¶„ê°’ì„ ?´?š©?•˜?Š” ë°©ì‹?œ¼ë¡? ?‹¬?„±?•˜? ¤ê³? ?•œ?‹¤.

cost?Š” ?‹¤?Œê³? ê°™ìœ¼ë¯?ë¡?
<br><br>
\\[ MSE = cost(W) = \frac{1}{m} \sum^m_{i=1} \left( H(x^{(i)}) - y^{(i)} \right)^2 \\]

$W$?— ????•´ ë¯¸ë¶„?•˜ë©? ?•„?˜??? ê°™ì?? ê²°ê³¼ë¥? ?–»?„ ?ˆ˜ ?ˆ?‹¤.

\\[ \nabla W = \frac{\partial cost}{\partial W} = \frac{2}{m} \sum^m_{i=1} \left( Wx^{(i)} - y^{(i)} \right)x^{(i)} \\]

?´? ‡ê²? êµ¬í•œ gradient?Š” ?‹¤?Œ ?‹ê³? ê°™ì´ ?•™?Šµ?— ? ?š©?•˜ê²? ?œ?‹¤.

\\[ W := W - \alpha \nabla W \,\,\left(\alpha = learning\,\,rate\right)\\]

?´?Ÿ° ?˜•?ƒœë¡? ?•™?Šµ?„ ?•˜?Š” ?´?œ ë¥? ?•œë²? ?•Œ?•„ë³´ì.
<br><br>
?•„?˜?˜ ?‘ gif?Š” ê°ê° ê·¹ì†Œ? ?˜ ì¢Œìš°?—?„œ ê·¹ì†Œ? ?— ? ‘ê·¼í•  ?•Œ ? ‘?„ ?˜ ë³??™”ë¥? ?‚˜????‚¸ ê²ƒì´?‹¤.
<br><br>
![](/posting_imgs/images/lab3-2.gif)
<br><br>
ë¨¼ì?? ?™¼ìª½ì—?„œ ? ‘ê·¼í•˜?Š” ê²½ìš° ê¸°ìš¸ê¸?(gradient)ê°? ?Œ?ˆ˜?´ê³? ê·¹ì†Œ? ?œ¼ë¡? ?„?‹¬?•˜ê¸? ?œ„?•´?„œ?Š” $W$ê°? ì»¤ì ¸?•¼ ?•œ?‹¤. ê·¸ëŸ¬ë¯?ë¡? ?Œ?ˆ˜?¸ ê¸°ìš¸ê¸°ë?? ë¹¼ì£¼?–´ ê·¹ì†Œ? ?— ê°?ê¹ê²Œ ?„?‹¬?•  ?ˆ˜ ?ˆ?‹¤.
<br><br>
![](/posting_imgs/images/lab3-3.gif)
<br><br>
?‹¤?Œ?œ¼ë¡? ?˜¤ë¥¸ìª½?—?„œ ? ‘ê·¼í•˜?Š” ê²½ìš° ê¸°ìš¸ê¸°ê?? ?–‘?ˆ˜?´ê³? ê·¹ì†Œ? ?œ¼ë¡? ?„?‹¬?•˜ê¸? ?œ„?•´?„œ?Š” $W$ê°? ?‘?•„? ¸?•¼ ?•œ?‹¤. ?´ ?•Œ?Š” ?–‘?ˆ˜?¸ ê¸°ìš¸ê¸°ë?? ë¹¼ì£¼?–´ ê·¹ì†Œ? ?— ê°?ê¹ê²Œ ?„?‹¬?•  ?ˆ˜ ?ˆ?‹¤.

ê²°êµ­ ?´ ?‘˜?‹¤ ë¹¼ì•¼?•˜ë¯?ë¡? ëª¨ë‘ ë§Œì¡±?•˜?Š” ?‹?´ $ W := W - \alpha \nabla W $, ê¸°ìš¸ê¸°ì˜ ëº„ì…ˆ?œ¼ë¡? ì£¼ì–´ì§??Š” ê²ƒì´?‹¤. ?´ ?•Œ $learning\,\,rate$?¸ $\alpha$?Š” ë§? ê·¸ë??ë¡? ?•™?Šµë¥?(?•œ ë²ˆì— ?•™?Šµ?„ ?–¼ë§ˆë‚˜ ?•  ê²ƒì¸ê°?)?„ ?‚˜????‚´?Š” ê²ƒì´ë¯?ë¡? ?ƒ?™©?— ë§ê²Œ ìµœì ?™” ?•˜?—¬ ?‚¬?š©?•œ?‹¤.

?‹¤ë§?, ?•™?Šµë¥ ì´ ?„ˆë¬? ?‘?œ¼ë©? ?ˆ˜? ´?´ ?Š¦?–´ì§?ê³?, ?„ˆë¬? ?¬ë©? ì§„ë™?•˜ë©? ë°œì‚°?•´ ë²„ë¦¬ê¸? ?•Œë¬¸ì— ? ? ˆ?•œ ë²”ìœ„?˜ ê°’ì„ ?‚¬?š©?•´?•¼ ?•œ?‹¤.

![](/posting_imgs/images/lab3-4.jpg)

?´?–´?„œ ?•?„  ?‹?“¤?„ ì½”ë“œë¡? ?‘œ?˜„?•˜ë©? ?‹¤?Œê³? ê°™ë‹¤.

\\[ \nabla W = \frac{\partial cost}{\partial W} = \frac{2}{m} \sum^m_{i=1} \left( Wx^{(i)} - y^{(i)} \right)x^{(i)} \\]

{% highlight python %}
gradient = torch.sum((W * x_train - y_train) * x_train)
print(gradient)

''' output
tensor(-14.)
'''
{% endhighlight %}

\\[ W := W - \alpha \nabla W \,\,\left(\alpha = learning\,\,rate\right)\\]

{% highlight python %}
lr = 0.1
W -= lr * gradient
print(W)

''' output
tensor(1.4000)
'''
{% endhighlight %}

***
## Training
?•?„œ êµ¬í˜„?–ˆ?˜ ê²ƒë“¤?„ ?™œ?š©?•˜?—¬ ?‹¤? œë¡? ?•™?Šµ?•˜?Š” ì½”ë“œë¥? ?‘?„±?•´ ë³´ë©´ ?‹¤?Œê³? ê°™ë‹¤.

{% highlight python %}
# ?°?´?„°
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])
# ëª¨ë¸ ì´ˆê¸°?™”
W = torch.zeros(1)
# learning rate ?„¤? •
lr = 0.1

nb_epochs = 10
for epoch in range(nb_epochs + 1):
    
    # H(x) ê³„ì‚°
    hypothesis = x_train * W
    
    # cost gradient ê³„ì‚°
    cost = torch.mean((hypothesis - y_train) ** 2)
    gradient = torch.sum((W * x_train - y_train) * x_train)

    print('Epoch {:4d}/{} W: {:.3f}, Cost: {:.6f}'.format(
        epoch, nb_epochs, W.item(), cost.item()
    ))

    # gradientë¡? H(x) ê°œì„ 
    W -= lr * gradient

''' output
Epoch    0/10 W: 0.000, Cost: 4.666667
Epoch    1/10 W: 1.400, Cost: 0.746666
Epoch    2/10 W: 0.840, Cost: 0.119467
Epoch    3/10 W: 1.064, Cost: 0.019115
Epoch    4/10 W: 0.974, Cost: 0.003058
Epoch    5/10 W: 1.010, Cost: 0.000489
Epoch    6/10 W: 0.996, Cost: 0.000078
Epoch    7/10 W: 1.002, Cost: 0.000013
Epoch    8/10 W: 0.999, Cost: 0.000002
Epoch    9/10 W: 1.000, Cost: 0.000000
Epoch   10/10 W: 1.000, Cost: 0.000000
'''
{% endhighlight %}

**Hypothesis output ê³„ì‚° -> cost??? gradient ê³„ì‚° -> gradientë¡? hypothesis(weight) ê°œì„ **

?œ„??? ê°™ì?? ?ˆœ?„œë¡? ì´? 10 epoch ?•™?Šµ?•˜?Š” ì½”ë“œ?´?‹¤. ?•™?Šµ?„ ?•œë²? ?•  ?•Œë§ˆë‹¤ costê°? ì¤„ì–´?“¤ê³?, ?š°ë¦¬ê?? ?ƒê°í•œ ?´?ƒ? ?¸ $W$?¸ 1?— ? ?  ê°?ê¹Œì›Œì§??Š” ê²ƒì„ ?™•?¸?•  ?ˆ˜ ?ˆ?‹¤.

***
## Training with `optim`

**Training**?—?„œ ?–ˆ?˜ ê²ƒì²˜?Ÿ¼ ?š°ë¦¬ê?? gradientë¥? ê³„ì‚°?•˜?Š” ì½”ë“œë¥? ì§ì ‘ ?‘?„±?•˜?—¬ ?‚¬?š©?•  ?ˆ˜?„ ?ˆì§?ë§? PyTorch?—?„œ ? œê³µí•˜?Š” `optim`?„ ?´?š©?•˜?—¬ ê°„ë‹¨?•˜ê²? êµ¬í˜„?•  ?ˆ˜?„ ?ˆ?‹¤.

{% highlight python %}
# ?°?´?„°
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])
# ëª¨ë¸ ì´ˆê¸°?™”
W = torch.zeros(1, requires_grad=True)
# optimizer ?„¤? •
optimizer = optim.SGD([W], lr=0.15)

nb_epochs = 10
for epoch in range(nb_epochs + 1):
    
    # H(x) ê³„ì‚°
    hypothesis = x_train * W
    
    # cost ê³„ì‚°
    cost = torch.mean((hypothesis - y_train) ** 2)

    print('Epoch {:4d}/{} W: {:.3f} Cost: {:.6f}'.format(
        epoch, nb_epochs, W.item(), cost.item()
    ))

    # costë¡? H(x) ê°œì„ 
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

''' output
Epoch    0/10 W: 0.000 Cost: 4.666667
Epoch    1/10 W: 1.400 Cost: 0.746667
Epoch    2/10 W: 0.840 Cost: 0.119467
Epoch    3/10 W: 1.064 Cost: 0.019115
Epoch    4/10 W: 0.974 Cost: 0.003058
Epoch    5/10 W: 1.010 Cost: 0.000489
Epoch    6/10 W: 0.996 Cost: 0.000078
Epoch    7/10 W: 1.002 Cost: 0.000013
Epoch    8/10 W: 0.999 Cost: 0.000002
Epoch    9/10 W: 1.000 Cost: 0.000000
Epoch   10/10 W: 1.000 Cost: 0.000000
'''
{% endhighlight %}

`optim.SGD`ê°? ?š°ë¦¬ê?? ë§Œë“¤?–´?„œ êµ¬í˜„?–ˆ?˜ gradient?— ????•œ ì²˜ë¦¬ë¥? ?•´ì£¼ê³  ?ˆ?Š” ê²ƒì„ ë³? ?ˆ˜ ?ˆ?‹¤.
{% highlight python %}
optimizer.zero_grad() # gradient ì´ˆê¸°?™”
cost.backward()       # gradient ê³„ì‚°
optimizer.step()      # ê³„ì‚°?œ gradientë¥? ?”°?¼ W, bë¥? ê°œì„ 
{% endhighlight %}
???ë²? ê°•ì˜?—?„œ?„ ?“±?¥?–ˆ?˜ ?œ„ 3ê°œì˜ ë¬¶ìŒ ì½”ë“œë¥? ?†µ?•´ gradient?— ????•œ ê³„ì‚°ê³? ê·¸ì— ?”°ë¥? ?•™?Šµ?´ ?´ë£¨ì–´ì§?ê³? ?ˆ?‹¤.

?´ ?•Œ?„ ë§ˆì°¬ê°?ì§?ë¡? $W$??? costë¥? ë³´ë©´ ?˜ ?•™?Šµ?´ ?˜?Š” ê²ƒì„ ë³? ?ˆ˜ ?ˆ?‹¤.