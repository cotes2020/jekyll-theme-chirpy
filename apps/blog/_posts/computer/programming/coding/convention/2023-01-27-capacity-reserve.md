---
title: "Capacity Reserve"
# description: ""
categories: [컴퓨터, 프로그래밍, Convention]
tags: []
image: "/assets/img/background/kururu-lab.jpg"

date: 2023-01-27. 07:37
# last_modified_at: 2023-01-27. 07:37
---

{% include embed/youtube.html id = "9er81n6NyuM" %}

Capacity: the maximum amount that something can contain.  
Reserve: 예약  
Budget: 예산, 지출 예상 비용  

## Capacity Reserve

---

입력 크기를 알고 있다면, Capacity를 예약할 것.  
좀 더 넉넉하게 하더라도, Capacity를 예약할 것.  

예를 들어 C++ STL Vector, C# List 같은 동적 배열.  
Size/Count가 Capacity를 초과하여 크기를 늘려야 할 때,  
더 큰 크기의 새로운 메모리 할당, 기존 요소 복사, 기존 메모리 Free 하는 비용을 생각해야한다.  

## 여담

---

고성능 프로그램에서는 이런 리소스 Budget을 확실히 정해 모든 걸 배열로 처리하는 것도 가능  
이러면 프로그램을 오래 돌리고 나서야 리소스 부족 문제가 발견되는 걸 막을 수 있다,  
프로그램이 시작하는 순간 이 최대 리소스를 사용하고 있느니.  

- Placement New 기법  
  - 정적 메모리 풀을 만들어 놓고 할당 받아 쓰는 방식  
  - 대규모 탄막이나 파티클이 등장할 경우 프레임 드랍을 막기 위해  
