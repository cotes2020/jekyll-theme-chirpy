---
title: "Adaptive Resonance Theory - ART1"
# description: ""
categories: [컴퓨터, 알고리즘, AI]
tags: [AI]
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2023-10-25. 09:59
# last_modified_at: 2023-10-31. 13:30
last_modified_at: 2023-12-08. 10:30
---

5, 6차시  
@ 50분 정도의 집중력  

## 서론

---

Resonance 공명  

모든 물체는 고유한 진동수를 가진다  
진동수: 1초에 진동하는 수, Hz  

공명: 같이 운다  
→ 같은 진동수의 물체가 주변에서 진동하면, 함께 진동한다  
→ 진동수가 다르면 공명하지 않는다  
→ 깨구락지의 공명: 케로케로케로...  

Adaptive 적응형  
@ LOL 적응형 능력치: `Adaptive` Force  

## Adaptive Resonance Theory - ART1

---

적응형 공명  
→ 기준과 공명하는 것들을 찾아 그룹화 (반복)  
→ Recommender System - 추천  

Adaptive Resonance Theory - ART1  
→ Clustering Algorithms - 군집화 알고리듬  
→ Unsupervised Learning Algorithm - 비지도 학습 알고리즘 (With Biological Motivations)  

Supervise 지도  
@ Supervised User  

지도 학습:  
답이 이미 존재하는, 누가 지도해주는 (Supervised)  
@ 분류, 누가 누구다  

비지도 학습:  
답이 없는 (비정량적인? Like 취향 - 나누는 기준이 정해진 게 없음)  
@ 덩어리 만들기, 누가 비슷하다  

강화 학습 - 말 그대로  

Clustering Algorithm  
클러스터(덩어리), 군집화 → 분류, 클래스화(레이블)  
유사성에 의해 덩어리를 나눈다  
@ Like 하얀건 종이요, 검은건 글씨다  

사람이 뭔가 배울 때 이미 알고 있는 것과 비슷한 것을 찾아 연관시키고(덩어리),  
만약 그게 없다면 이해를 위한 새로운 생각(덩어리)을 만든다  
에서 모티브를 얻은  

ART1  
→ Feature Vector: 뭐가 어떻다 하는 특징 데이터 (테이블)  
→ Feature Vector의 1, 0의 개수가 비슷해야 비슷한 데이터  

@ U 중간고사 출제: 적응형 공명 이론에 대한 클러스터링 결과를 보고, 특정 요소(i.e. 고객)에게 특정 요소(i.e. 물건)를 추천하는 과정을 설명하시오.  

같은 분류의 요소들끼리 데이터 Vector를 합 (Sum Vector)  
해당 분류의 요소들마다 Sum Vector에서 가장 큰 수 순서대로 없는 요소를 추천  

## 과정

---

1. Create Initial Prototype Vector
2. For each Example Vector, Continue
3. Example Close to Prototype?
   - Passes Vigilance Test?
   - More Prototypes?
4. Place Example in current prototype vector

Create Initial Prototype Vector  
→ 임의의 요소 선택 (첫 번째든 랜덤이든)  
→ P<sub>0</sub> = E<sub>0</sub>  

For each Example Vector  
→ Close to Prototype?  

β - 그냥 1.0 (일단) 몰라도 된다  
d - 전체 물건의 개수  
E - 산 물건  

→ Proximity Test - 유사도 테스트  
P∩E (1에 대해서만) / P 전체물건 + β > E 산물건(E 1의수) / E 전체물건 + β  
@ PDF 8p Eq1 3/4 > 4/8 (오타)  

→ Vigilance Test? (덩어리로 묶을 지, 한 번 더 평가)  
P∩E / E 산물건(E 1의수) \< p (Vigilance Parameter)  

true?  

Place Example in Current Prototype Vector (마킹? 라벨링? 덩어리?)  

남은 것들에 대해 새로운 프로토타입을 만들어 반복  

P = P∩E  
1110110  
1110010  
→ 1110010  

덩어리에 크게 의미 없는 데이터를 제거  
(교재 3.4 밑에 Finally 부터 나오는 내용)  

Using ART1 for Personalization(Recommend System)  
Sum Vector를 이용하여  

## K-means Algorithm  

---

K-means Algorithm  
덩어리가 K개 있다고 못박아두고 분류  

먼저 K를 정하고
→ (이 데이터에는 K개의 덩어리가 있다고 가정)  
→ (K를 어떻게 정하냐는 다른 논제)  
→ (비지도학습 알고리듬이지만, 답이 정해져 있는 것처럼 - K)  

- Loop ~
  1. K개의 임의의 점들을 기준으로, 가까운 것들을 모아 K개의 덩어리로 만듦
  2. 이후 각 덩어리의 중심으로 기존 K개의 임의의 점들을 재위치
  3. 만약, 이전과 달라진 점이 없다면? 분류 완료 End

실제로는 점이 아니라 테이블  
3차원 이상으로 넘어가면 이해하기 힘듦  
못 찾을 수도 있음  
