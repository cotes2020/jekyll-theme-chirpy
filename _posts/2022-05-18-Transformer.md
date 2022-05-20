---
title: Transformer
author: Bean
date: 2022-05-16 12:02:00 +0800
categories: [AI, basic]
tags: [AI]
layout: post
current: post
class: post-template
subclass: 'post'
navigation: True
cover:  assets/img/post_images/ai_cover2.jpg
---

**Transformer은 recurrence나 convolution없이 attention만을 사용하는 새로운 sequence transduction model**이다.

## Background
&nbsp;

기존의 seq2seq 모델은 인코더-디코더 구조로 구성되어 있었다. 인코더에서는 입력 시퀀스(ex> 문장)를 하나의 벡터 표현으로 압축하고, 디코더는 이 벡터 표현을 통해 출력 시퀀스를 만들어냈다. 하지만 이 구조에서는 인코더가 입력 시퀀스를 압축하는 과정에서 정보가 일부 손실된다는 단점이 존재했는데, 그래서 이를 보정하기 위하여 어텐션이 사용되었다. 이 때 어텐션은 인코더 쪽의 hidden variable과 디코더 쪽의 hidden variable간의 유사성을 비교하였다. 이 어텐션을 이용하면 입력과 출력 시퀀스의 distance에 대한 고려 없이 둘간의 의존성을 모델링할 수 있다.

본 논문에서는 이렇게 어텐션을 RNN의 단점을 보정하는 역할로서 사용하는 것이 아니라 **어텐션** 만을 사용해 인코더-디코더를 만들어보는 것을 제안한다.

&nbsp;
## 모델 아키텍쳐
&nbsp;
<div style="text-align: left">
  <img src="/assets/img/post_images/transformer1.jpeg" width="100%"/>
</div>

### Encoder and Decoder Stacks

#### Encoder

트랜스포머는 RNN 없이 인코더-디코더를 구성한다. RNN이 있는 구조에서는 인코더와 디코더에 각각 존재하는 RNN 한 개가 t개의 time step를 가지는 구조였다면, 트랜스포머는 인코더, 디코더 자체가 N개(본 논문에서는 6개)씩 존재한다. 각 6개의 layer은 다시 2개의 sub-layer을 가진다. 여기서 첫 번째 sub-layer은 **multi-head self-attention mechanism**이고, 두번째 sub-layer은 **position-wise fully connected feed-forward 네트워크** 이다. 또한, Encoder에서 layer nomalization을 수행한 이후에 각 sub-layers에서 residual connection을 해준다.

#### Decoder
Encoder와 마찬가지로 6개의 layer을 stack하였고, 동일한 2개의 sub-layer을 가지며 이에 떠해 multi head attention을 수행하는 3번째 sub-layer가 추가된다. 또한 마찬가지로 layer nomalization 이후 각 sub-layer에서 residual connection을 해준다.

반면 Encoder와는 달리 self-attention을 수행하는 sub-layer에 조금 수정을 가했는데,

### Attention

Attention 함수는 query와 key-value pair (key, value) 의 집합을 output으로 매핑한다. 좀 더 자세히 말하면, output은 value들의 weighted sum으로 계산되는 데, 이 때, 각 value에 대응되는 weight 값은 query와 대응되는 key간의 compatibility function으로 계산된다.

<div style="text-align: left">
  <img src="/assets/img/post_images/transformer2.jpeg" width="100%"/>
</div>

#### Scaled Dot-Product Attention

Compatibility 함수로 대표적으로 **scaled dot product** 를 사용한다. Scaled dot product에서 각 value의 weight은 다음 식으로 계산된다.

$$ Attention(Q, K, V) = softmax( \frac{QK^{T}}{\sqrt{d_{k}}} ) V $$

이 때, 식을 확인해보면 $ \frac{1}{\sqrt{d_{k}}} $ 를 추가로 곱해주는 것을 볼 수 있다. 이는 $ d_{k} $ 가 큰 수 일 때, dot product가 너무 큰 값을 가져 softmax function이 매우 작은 gradient 값을 가지게 된다. 따라서 dot product를 $ d_{k} $ 로 나눠 이를 보정해준다.

#### Multi-Head Attention
어텐션을 큰 dimension에 대하여 한 번에 적용하면 학습 속도가 오래 걸린다. 따라서 본 논문에서는 어텐션을 사용할 때 한 번에 적용하는 것이 아니라 여러개로 분할하여 병렬로 수행한 뒤 이를 다시 하나로 합치는 방법을 제안하였다. 즉, 전체 dimension을 h로 나눠서 병렬로 attention을 h번 적용한다.

1. query와 key, value를 각각 $ d_{k}, d_{k}, d_{v} $ 차원에 linear projection한다.
2. 이렇게 projection 된 버전의 query, key, value를 가지고 Scaled dot product attention을 수행한다.
3. 이 과정을 h번 반복한다.
4. 구해진 h개의 벡터를 합치고 다시 project 하여 최종 값을 얻는다.

### Position-wise Feed-Forward Networks
sub layer에 적용되는 attention에 추가로, 인코더, 디코더의 각 레이어는 각 position에 동일하게 적용되는 fully connected feed-forward network를 가진다. 이 네트워크는 ReLu activation을 사이에 둔 두개의 linear transformation으로 구성된다.

$$ FFN(x) = max(0, xW_{1} + b_{1})W_{2} + b_{2} $$

### Embeddings and Softmax
다른 sequence transduction 모델과 유사하게, 입력 토큰들을 출력 토큰들로 변환해주는 데 학습된 embedding을 사용한다. 또한, 디코더 출력을 다음 토큰의 확률로 바꿔주는데 학습된 transformation과 softmax 함수를 사용한다.

### Positional Encoding

트랜스포머 모델에서는 recurrence도, convolution도 사용하지 않는다. 따라서 모델이 시퀀스의 순서 정보를 처리하기 위해서는 시퀀스에 있는 각 토큰의 상대 위치 혹은 절대 위치 정보를 추가로 알려줘야 한다. 이를 위해서 인코더, 디코더에 들어가기 전에 각각의 하위 레이어인 임베딩 레이어에 **positional encoding**이 추가하여 sequence에 순서를 준다.

positional encoding는 다양한 방법으로 구할 수 있는데, 논문에서는 다양한 주파수의 사인, 코사인 함수를 사용하여 이 값을 계산한다.

이처럼 positional encoding을 사용하면 시퀀스의 순서 정보를 보존할 수 있게 된다. 이후 인코더와 디코더에서는 같은 단어를 입력 받아도 순서 정보에 따라 다른 벡터 값을 처리하게 된다.

&nbsp;
## Training
&nbsp;

### Training Data and Batching
학습에는 4.5백만개의 setence pair로 구성된 standard WMT 2014 English-German dataset을 사용하였다 또한, English-French 학습에는 보다 큰 (36M) WMT 2014 English-Frensh dataset을 사용하였다. sentence pair은 길이에 따라 batch로 묶었고, 각 batch는 약 25000개의 source token과 약 25000개의 target token으로 구성되어 있다.

### Optimizer
Adam optimizer을 사용하였다.

### Regularization
정규화 방식으로 Residual dropout을 사용하였다.

&nbsp;
## Result
&nbsp;

### Machine Translation
### Model Variance
