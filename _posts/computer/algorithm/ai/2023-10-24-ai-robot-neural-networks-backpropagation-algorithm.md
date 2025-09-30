---
title: "Neural Networks & Backpropagation Algorithm - 신경망 & 역전파 알고리듬"
# description: ""
categories: [컴퓨터, 알고리즘, AI]
tags: [AI]
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2023-10-25. 10:07
last_modified_at: 2023-12-08. 10:44
---

10, 11차시  

## Neural Networks & Backpropagation Algorithm - 신경망 & 역전파 알고리듬

---

신경망과 역전파 알고리듬  
뉴런과 시냅스  

배우는 것 (구조, 학습 알고리듬)  
학습 알고리듬: Error Backpropagation Learning Algorithm 역전파 학습 알고리듬을 통한  
Architectures 구조: Feed-Forward 전진 전파 / Multi-Layer Neural Network 다층 신경망(뉴런적인) 네트워크  

역전파 알고리듬이 신경망 연구에 부활을 일으켰다?  
@ Resurgence 부활  

Topologies 이상 = Architectures 구조  

@ U 중간고사 출제: 신경망의 일반화를 설명하시오.  
→ 신경망의 가장 큰 특징: 일반화  

→ @ 일반화에 의해 갈라진  
→ @ 기호주의: 논리/수학적인 방법으로  
→ @ 연결주의: 감각/계산적인 방법으로  
→ @ 지금은 서로 잘 섞인  

→ @ vs 추상화-개체지향  
→ @ 책을 많이 읽어라, 같은 생각만해서, 생각이 경직되니까  

@ U 중간고사 출제: 주어진 신경망에 대해, 주어진 입력과 가중치에 따른 최종 출력 계산  
→ Feed-Forward ~  

이러한 신경망 구조가 어떻게 되냐? (시험 x)  
→ @ 이러한 신경망 구조가 어떻게 되냐?  
→ @ 다음 층의 뉴런과 모두 연결된다
→ @ 맨 위, 맨 아래 뉴런 빼고 모두 생략해서 그리기  

## Neuron, Artificial Neuron, Neural Networks - 뉴런, 인공 뉴런, 신경망

---

@ AI의 매개변수가 ~개다  
@ 인간의 시냅스가 ~개다  

@ Neuron - 뉴런: 입력 N개, 출력은 1개  
@ (출력 - 여러 갈래로 나눠지지만, 값은 똑같은 하나의 값)  

Weight - W - 가중치  
→ Like 시냅스  
→ Like 수도꼭지, 신호량 통제/조절  

1. Weighted Sum (X<sub>n</sub>), 입력들을 각가의 가중치를 곱해 X를 구하고 더함
2. Activation Function - 활성 함수를 통해 최종 값 y<sub>n</sub> 계산
   - Weighted Sum (X<sub>n</sub>)가 겁나 크니까 작게 찌그러뜨리기/조절하기

Neural Network - 다층 신경망  

@ 층(레이어) 256개? 몇 백개?  
@ 뇌도 레이어가 나뉨 (뉴런 집중 or 옅은)  

중간층(= 구 은닉층)  

@ 입력층의 뉴런은 모양이 다르다 !?  
@ → 모든 뉴런은 입력을 받아서 `계산을 하고` 뿌리는데,  
@ → 입력층 뉴런은 입력을 받아서 `뿌리기만` 함  
@ → 반대로 보면, 입력을 받을 때 가중치 W 곱한 값을 가져오는데, 입력층은 그냥 가져옴  
@ → So, (엄밀히 말하면) 입력층 != 뉴런  

i번째 뉴런과 j번째 뉴런을 잇는 가중치 Wij  

I.E. 대화 인공지능에 질문을 하면, 뭔가 처리되고, 답변이 옴  
무튼 사용하는 입장에서는 뉴럴 네트워크 안의 과정/원리를 모른채 씀  

우리는 (입력, 출력) Pair만 전달하고, 그 가중치 W를 찾는 과정을 AI가  
우리가 원하는 값을 찾기 위해서 직접 가중치 W를 조정해야 하는 거면 안쓰지  

출력 계산 방향: →  
학습 방향: ←  

## Backpropagation - 역전파

---

Feed-Forward  
→ 입력을 계산해서 출력을 앞으로 → 앞으로 →  
→ Back Backpropagation (Forward propagation)  

Feed-Back  
입력을 계산해서 출력을 뒤로 ← 뒤로 ←  

RNN, 순환 신경망  
→ 한 층에서 다음 층으로, 뉴런이 출력을 할 때  
→ 자기 자신에게도 출력값을 입력함 (Feedback)  

Backpropagation - (오류) 역전파 (학습) 알고리듬  
→ 오류: 우리가 AI에게 준 (입력, 출력) 페어랑 출력이 다른 거  
→ 오류를 이용해 다시 가중치를 수정하는 알고리듬  

중간-출력층 사이의 가중치 조정  
Wij = Wjk + △Wjk  
△델타Wjk를 아는 값과 (이상-에러 오차)를 이용하여 계산  

입력-중간층 사이의 가중치 조정  
Wij = Wjk + △Wjk  
△델타Wjk를 아는 값과 (중간-출력층 사이의 가중치 조정 과정에서 구한 값)을 이용하여 계산  

→ 입력-중간층 사이 가중치를 고치기 위해서,  
→ 중간-출력층 사이 가중치를 먼저 고치고 나온 값을 이용  
→ = 앞에서 계산한 값을 역으로 (뒤로) 전파한다  

## ~

---

여기서 문제,  
여러 층을 거쳐 옮겨오다보면 제대로 동작 안하는 경우가  
→ 해결하기 위해 나온게 딥러닝  

알고리듬을 보면 그냥 더하기만 있음 (곱하기도 더하기)  
→ 간단한 계산, 그래서 한 번에 많은 단순 계산을 할 수 있는 GPU 사용  

답을 아는 경우  
답을 모르면 강화학습, 비지도 학습?  

[Neural Network 모의 실험 - 참고](https://playground.tensorflow.org/)  

Sample Project - Game NPC AI  
Health, Has-Knife, Has-Gun, Enemies 상태에 따른  
Attack, Hide, Wander, Run 행동 결정 알고리듬  
