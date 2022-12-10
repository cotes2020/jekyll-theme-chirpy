---
title: "Cross Entropy(교차 엔트로피)"
author: Kwon
date: 2022-04-28T00:10:00+0900
categories: [background, loss]
tags: [entropy, kl-divergence, cross-entropy]
math: true
mermaid: false
---

### Related Post

1. [Entropy](/posts/entropy/)
2. [KL Divergence](/posts/KL-divergence/)
3. **Cross Entropy**

***
## Cross Entropy

교차 엔트로피의 의미는 이름에서 찾아볼 수 있다. 먼저 교차 엔트로피의 식을 한번 보자.

\\[ H\left( P, Q \right) = -\sum^n_{i=1} p\left( x_i \right) \log{ q\left( x_i\right ) } \\]

엔트로피 식에 $P$와 $Q$의 밀도함수들이 **교차**해서 들어가 있다. 그런 의미에서 교차 엔트로피라는 이름이 붙은 것이다.

이 식의 의미는 [KL divergence 포스팅](/posts/KL-divergence/)에서 확인할 수 있듯이 확률분포 $P$를 근사하는 $Q$를 $P$ 대신 사용하여 샘플링했을 때의 엔트로피를 말한다.

그런데 어떻게 이런 의미를 가지는 교차 엔트로피가 그 자체로 loss의 역할을 할 수 있는 걸까? 실제로 loss를 구하기 위해서는 원래 분포의 엔트로피와의 차이를 구해야 할 것이다. 이는 쿨백-라이블러 발산의 의미와 동일하다.
쿨백-라이블러 발산의 식은 다음과 같았다.

\\[ KL\left( P\|\|Q \right) = H\left( P, Q \right) - H\left( P \right) \\]

분명 교차 엔트로피 말고도 원래 확률분포의 엔트로피인 $ H\left( P \right) $ 항이 존재한다. 하지만 우리가 loss를 사용하게 되는 실제 classification problem에 대해서 한번 생각해 보자.

classification problem에서 우리는 한 data에 대해 정답이 주어져 있다고 가정한다. 이 말은 **이미 정해져 있는 정답에 속하는 것이 불확실하지 않다**는 말이기 때문에 $P = 1$이고 그에 따라 $ H\left( P \right) = 0$이다.
그러므로 $ H\left( P \right) $ 항을 무시하고 다음과 같이 적을 수 있다.

\\[ KL\left( P\|\|Q \right) = H\left( P, Q \right) =  -\sum^n_{i=1} p\left( x_i \right) \log{ q\left( x_i\right ) } \\]

이런 이유 때문에 교차 엔트로피 자체가 loss로 기능할 수 있는 것이다.