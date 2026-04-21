---
title: "Ant Algorithms - 개미 알고리듬"
# description: ""
categories: [컴퓨터, 알고리즘, AI]
tags: [AI]
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2023-10-25. 10:01
# last_modified_at: 2023-10-31. 13:30
last_modified_at: 2023-12-08. 10:44
---

7, 8차시  

## Ant Algorithms - 개미 알고리듬

---

Ant Algorithm - 개미 알고리듬  
개미와 `Stigmergy` - 개미들의 소통방법(냄새/페로몬을 이용한)  

{% include embed/youtube.html id='81GQNPJip2Y' %}

{% include embed/youtube.html id='emRXBr5JvoY' %}

{% include embed/youtube.html id='V1GeNm2D2DU' %}

→ 개미는 가는 길마다 페로몬을 뿌림 (Like 헨젤과 그레텔)  
→ 개미는 페로몬이 많은 쪽을 따라감  
→ 정답인지는 모르지만 최적 경로에 가까울수록 (짧을 수록) 더 많이 왔다갔다 할 수 있음  
→ 최종적으로 가장 많은 개미들이 다니는 경로 = 최적 경로  

Nest, Food, Obstacle  

개미는 Network을 따라 이동, 냄새가 강한 쪽으로 이동  
컴퓨터 개미는 냄새를 식으로 맡음  
(강의 자료에 있는 식은 오류가 많음)  

최적의 길을 찾아내는  
변화에도 금방 적응하는  

Traveling Salesman Problem - TSP  
= 해밀턴 패턴, 모든 정점을 중복없이 한 번씩만 방문  

[개미 군집 최적화 ACO - 참고](https://www.mql5.com/ko/articles/11602)  

@ U 중간고사 출제:  
인공 개미 집단의 시간 변화에 따른 Behavior-행태를 나타낸 그림 3장  
각각 어떤 원리에 의해 어떤 일이 일어나고 있는지 (그림 사이사이 일어난 일)  
→ (Stigmergy)-페로몬을 남기면서 서로 소통 ~  
→ 페로몬에 의해 잘못된 경로를 만들기도 하지만, 페로몬에 의해 결국 최적 경로를 찾아낸다  
→ 어떤 조건에서든 최적 경로를 찾아낸다  

(개미 알고리듬)-이 원리를 어떻게 응용할 수 있는지  
→ 최단 경로를 찾는 ~  

## 식

---

ρ - 초매개변수, (일단) 무시해도 되는  

α - 페로몬 지수, β - 거리 지수  
→ 어떤 걸 더 중심적으로 생각하느냐에 따라 조정  

타겟 경우의 값 / 현재 위치에서 갈 수 있는 모든 경우의 수 값 합  

τ(r, u) - r, u 사이 페로몬 양  
η(r, u) - r, u 사이 거리  

만약 페로몬 양이 똑같다면, 거리가 더 짧은 쪽으로  
(때문에 거리는 역수 모양)  
