---
title: LoRA(Low-Rank Adaptation) 정리
author: Juwon Oh
date: 2022-06-05 09:11:00 +0800
categories: [NLP]
tags: [NLP, LoRA]
pin: true
---

## LoRA의 문제의식

- **기존 Large-Scale Pretrained Model FineTuning의 문제**
    - **FineTuning**
        - 정의: pretrained model을 특정한 task나 domain에 맞게 weight를 조정하는 방법.
        - 단점
            - pretrained model을 만들기에는 **너무 많은 resource**가 필요
            - 다양한 **downstream task**들에 pretrained model을 적용하고 싶은 경우.
                - downstream task: 문서 분류, 개체명 인식, 기계 번역 등의 사용자가 모델을 사용해서 수행하려고 하는 특정한 작업.
        - 문제: 모델을 FineTuning하기 위해서는 **여전히 너무 많은 리소스가** 필요하다.
- **문제의식: pretrained model 전체를 FineTuning하는 건 Over-parameterization이다.**
    - **해결 방안: 특정 task에 필요한 중요한 layer만 tuning하면 되지 않을까?**
        - **Instinsic dimension**: pretrained model을 특정 task에 맞게 finetuning할 때 필요한 정보    
- **Low-Rank space**: LoRA는 intrinsic dimension을 찾기 위해 **Pretrained model을 Low-Rank space로 변환.**
    - Rank: **선형대수학의 Rank와 동일**한 개념. → 일반적으로 column vector의 갯수. matrix에서 의미가 있는 기준들의 수.
    - LoRA에서 Low-Rank의 의미: **Large-Scale Pretrained Model에서 핵심정보를 가진 parameter만을 뽑아내서 사용한다는 것.**

## LoRA의 원리

- **기본 원리: pre-trained weight의 정보를 low-rank로 optimize할 수 있음.**
    - **정의: 저차원으로 표현된 보다 더 중요한 matrix만 FineTuning하는 방법.**
    - **차원 축소 기법과 유사함:** PCA나 SVD와 유사하게 **전체 weight matrix를 가장 잘 표현하는 low rank R**을 찾는 것. AutoEncoder에서는 latent variable이라고 생각하면 됨.
        - AutoEncoder 구조와 유사
    - GPT-3와 같은 큰 모델도 원본(full rank)의 정보를 r=1,2 정도의 저차원에서도 표현가능함.
- **Lora FineTuning의 의미**
    - **수식**
        
        ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/733b4406-19a6-4ac4-9f99-129d82fa9509/Untitled.png)
        
        - 공통점: **Auto regressive한 language model**을 **의미**한다.
            
            ex) GPT, T5 예측을 하면서 확률적인 모델을 학습함. 
            
    - Full FineTuning 과정
        - 학습 방법: auto-regressive 한 방식으로 학습. → 뒤의 단어를 모르는 상태로 순차적으로 단어를 생성함.
            - **P_Φ(Pi)는 모델의 weight**를 의미함.
        - 학습 과정: 특정 단어 다음에 나올 단어( y_{t}|x, y_{<t}) ****를 잘 예측하는 weight Φ를 찾는 게 목적. → **pretrained model의 weight 전체가 update된다.**
    - **LoRA FineTuning 과정**
        - 기존 pretrained 모델의 weight = (P_Φ**0**)
        - LoRA finetuning의 **핵심**: 기존 모델에 **θ만큼의 정보를 추가하는 것.**
            - pretrained 모델의 weight인 **Φ0는 FineTuning 과정에서 바뀌지 않음.**
            - **θ: FineTuning할 때 우리가 update하는 weight**
                
                $$
                \Delta \Phi(\theta)(y_t|x,y_{<t})
                $$
                
                ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b651f062-8d89-4149-8cab-e348e562a6a5/Untitled.png)
                
    
    사진 설명: BA는 위의 수식을 표현한 것이며 backpropagation을 통해서 학습하는 부분이다.

## LoRA의 학습 방법

- **LoRA FineTuning의 구조**
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/24a60777-007e-4530-9f89-a1e919fb618e/Untitled.png)
    
    - input: X(d * x dimension)
    - Pretrained Weights: d x d의 정사각행렬 → 13000 * 13000 = 16900000
    - **LoRA의 의미: Pretrained Weights(d x d)의 matrix를 A matrix(d x r)과 B Matrix(r x d)로 압축하여 finetuning 하자.**
        - 예시: d가 13000, r이 2인 경우
            - A의 param수 = 13000 * 2
            - B의 param 수 = 2 * 13000
            - **전체 param 수 → 52000**
        - 전체 pretrained model의 parameter에서 **중요한 parameter를 선택해서 FineTuning 할 수 있다. → FineTuning에 필요한 시간복잡도와 공간복잡도가 획기적으로 감소함.**
- **학습 과정**
    - **핵심: feed forward 연산에서는 W_0와 BA가 같이 사용된다.하지만 기울기를 update 하는 backpropation에서는 BA만 update된다.**
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f19ba83d-ee13-45a2-8aeb-46965e29c867/Untitled.png)
    
    1. **feedforward**
        - **input x은 W_0와 BA를 거쳐서 hiden layer h를 학습**한다.
            
            $$
            h = W_o * x + \Delta W_x = W_{0} * x + BAx
            $$
            
        - h는 마지막 출력 값 이전의 layer.
        - 주의: **w_0는 기울기를 update하지 않는다.**
    2. **Backpropagation**
        - **학습이 되는건 BA**
            - AB의 **초기 값**: **A는 표준정규분포, B는 0으로 initialization**. → 맨 처음에 B가 0이기에 feed forward 과정에서 AB의 영향은 없음.
        - **h(hidden layer)와 loss function을 기반**으로 **AB의 weight를 update** →
            - **의미: w_0를 거친 결과가 들어있음. pretrain model(W_0)의 정보를 간접적으로 받음.**
        - 의문: BA(r * r) 면 x(d*1)과 곱이 가능한가?
            - 추가된 것의 의미: B(r*d) * A(d*r) = BA (r * r) * x(d, 1)
        - **Low rank Adaptation:** **pretrained model 옆에 simple layer를 가지고 기울기를 update하면, low rank를 가진 layer로 모델을 FineTuning 할 수 있음.**
    3. **지속적인 학습**
        - 학습이 진행되면서 h는 학습된 BA의 영향도 받는다.
- **주의 사항**
    - LoRA는 원본 **모델을 압축하는 게 아니라, 모델을 FineTuning할 때 사용하는 방법**이다.
    - **r은 hyper-parameter이고, hyper param tuning을 해야 함.**
        - **적당한 r을 찾는게 중요한 issue이다.**
    - **데이터 셋의 구성은 Down stream task마다 달라질 수 있음.**
        
         ex) 요약 task: X: 원본 문서, Y: 요약문 → loss function도 마찬가지로 정의
        
    - Vison에서 Sqeexe Excitaion branch와 유사한 기법

## LoRA의 장점

- Transformers에 적용하기
    - Transformers는 qkv vector를 사용.
    - LoRA 적용: 동일한 방식으로 qk를 기반으로 weight를 update할 때 사용할 수 있음.
    - 일반적으로 LLM에 사용되는 Transformer에서도 적용할 수 있음.
        
        ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/360e0348-b192-4ecf-b30e-6061f5e03e40/Untitled.png)
        
    
    $ W_q = W_0*q + BA * q $
    $ W_k= W_{0} * k + BA *k $

- **No Additional Inference Latency**
    - 다른 task를 수행하는 model들을 **serving할 때 LoRA는 BA만 바꾸어 학습하면 된다.**
        - LoRA finetuning에서 **W= W_0(pretrained model) + BA**
        - **BA는 특정 downstream task를 위한 weight**
            - **학습:** **BA만 학습하면 되기에 학습이 더 효율적이다.**
            - **서빙:** pretrained model이 downstream model 전반을 잘 수행할 수 있다면, W_0는 메모리에 올리고, **BA만 바꿔서 서빙하는 방식으로 서빙 할 수 있음.**
                
                → 구현하는데는 난이도가 있을수도 있음.

## 정리

- LoRA의 **원리**: 기존 **Pre-trained model은 학습**하지 않고, **BA만 학습을 시킴.**
- **LoRA를 사용할 때도 원본 PreTrained model이 필요.**
- Transformer기반 LLM에 **LoRA를 적용시키면 VRAM 사용량과 서빙속도를 빠르게 할 수 있음.**
    - GPT-3 175B의 경우 VRAM 1.2TB를 LoRA를 사용하면 **350GB로 줄일 수 있음.**
        - r=4의 경우 350GB를 35MB로 줄일 수 있음.
    - Pretrained model로 GPT-3를 두고, 여러 **Downstream task를 LoRA로 처리하면 25% 정도 속도 향상이 있음.**