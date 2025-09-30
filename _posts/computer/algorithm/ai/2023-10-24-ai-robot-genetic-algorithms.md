---
title: "Genetic Algorithms - 유전 알고리듬"
# description: ""
categories: [컴퓨터, 알고리즘, AI]
tags: [AI]
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2023-10-25. 10:05
# last_modified_at: 2023-10-31. 13:30
# last_modified_at: 2023-12-07. 10:24
# last_modified_at: 2023-12-08. 10:41
last_modified_at: 2023-11-11. 12:41
---

N차시  

@ 용불용설  
@ 루카 LUCA  
@ En~: ~이 되게 하다  
@ Takes Place: 발생하다  

@ 유전자 알고리듬 (X, 잘못된 표현)  

@ 답을 찾아야 하는데, 답을 모름  
@ 더 좋은 상태들을 섞어서 답을 찾겠다는 것  

## Genetic Algorithms - 유전 알고리듬

---

계산 문제 해결에 쓰는  
Simulated Evolution 모의 진화, 진화 흉내  
를 알아본다  

유전 알고리듬은 `탐색 알고리듬`이다  

주어진 문제를 해결하기 위해  
Encoded 암호화된 Candidate 후보 Solutions 의 Population 모집단  
에서 동작하는 알고리듬이다  

간단한 알고리듬의  
Sequence of Instructions 명령어 열을 진화시키는 진화적인 계산을 통해  
유전 알고리듬의 Use를 Illustrate 보여준다  
→ Stack Machine  

## Biological Inspiration

---

Phenomenon 현상 of Natural Evolution를 흉내내는  
Optimization 최적화 알고리듬이다  

자연 진화에 있어서,  
Species 종은, 복잡한 환경/치열한 경쟁 속에서 생존하기 위해  
적응하는 방법을 Search 탐색한다  

탐색, Chromosomes(염색체, DNA)에서 일어나는 변화와 그 효과는  
생존과 Reproduction 재생산에 대한 `Graded 점수`로 매겨진다  

Natural Selection 자연 선택  
~  

## Genetic Algorithm High-Level Flow

---

@ U 중간고사 출제: [8-Queen](/posts/n-queen/) 문제에 유전 알고리듬을 적용하는 과정  

1. Initialization
   - 문제 정의
   - 인코딩 (컴퓨터가 처리할 수 있는 형태로, 자료구조 지식 요구)
   - 적합도 함수 디자인
2. Evaluation
   - 적합도 계산 (적합도 함수 사용)
3. Selection
   - 선택 연산
4. Recombination
   - Generic Operation 유전 연산
     - Crossover 교차
     - Mutation 돌연변이

Initialization - 문제 정의, 인코딩, 적합도 함수 디자인  
→ 사람이 관여하는 단계  

Evaluation - 적합도 평가(계산) 단계  
→ .E. PPT 라면 맛 척도 7, 10, 9, 5  

Selection - 선택 연산 단계  
→ 모든 맛 척도를 더하고, 그 비율을 원판에 표시, (다트 게임처럼) 랜덤으로 선택  
→ 비율이 낮은 요소도 꽤 높은 확률로 선택될 수 있음 = 자연선택  
→ 좋은 것만 선택하지 않음  

Recombination - Genetic Operation 유전 연산 단계  
→ Crossover 교차, Mutation 돌연변이  

Crossover - 교차, 부분 교체: 부모 유전자 섞이듯  
Mutation - 돌연변이: 아주 드물게, 부모에게 없는 성질을 주기 위해  
→ (교체보다 나빠질 확률이 높음, 교체보다 비율을 적게)  

@ 대부분 나빠지는 경우가 많은데,  
@ 공학적(수학적)인 문제의 경우, 모집단의 평균 점수는 높아짐  

## In Example

---

MSG, 계란, 파, 김  
라면에 재료 들어가는 경우의 수  
@ 실제로는 재료 종류가 많겠죠  

→ 2^4 - 1 = 15  
→ (아무것도 안넣는 경우 제외, 0000)  

1 = 0001, 2 = 0010, 3 = 0011, ...  
레시피 Like DNA  
레시피 번호를 통해 어떤 재료가 들어가있는 지 알 수 있음 (10진수 → 2진수)  

Initialization  
→ 데이터를 기반으로 맛을 평가할 수 있는 함수 (적합도 함수)  
→ 모집단 만들기: 임의의 모집단 요소 수 (초매개변수)  

Evaluation 적합도 평가: 적합도 함수 계산 ~  
Selection 선택 연산: 누적막대~  

Recombination - Genetic Operation 유전 연산  
→ 교차: 일반적으로 모두 교차 X, (70%? - 초매개변수)  
→ 돌연변이: 염색체 단위가 아니라 유전자 단위로, 아주 드물게  

## Sample Problem - Stack Machine

---

@ U 중간고사 출제: Stack Machine, 주어진 명령어와 스택을 보고, 최종적으로 프로그램이 어떤 문제를 푸는지  

숫자가 아니라 기호를 다루는 문제를 최적화하는 문제를 다뤄보자  
→ 명령어들의 열  

스택 머신 (in VM)  
Zero-Address  

- 0 DUP: A → A A, Duplicate
- 1 SWAP: A B → B A
- 2 MUL: 2 3 → 6, Multiply
- 3 ADD: 2 3 → 5
- 4 OVER: A B → B A B, 위에서 두 번째에 있는 요소 DUP
- 5 NOP: No-Operation, Filler

Solution Encoding  
문제의 솔루션을 인코딩 (문제 자체가 아니라)  
→ 명령어 (0 ~ 5)를 연속된 바이트 문자열로 표현  

Fitness Evaluation - 적합도 평가  
→ 임의의 스택, 프로그램이 End하거나 (Solve?), END 명령에 도달할 때까지  
→ 그 다음 적합도 계산 (현재 스택 상태와 목적 함수 결과 차이)  

Recombination - 재생산  
→ Crossover 연산자, 특정 지점에서 부모의 꼬리를 SWAP  
→ Mutation 돌연변이, 명령 랜덤 재할당  

I.E.  

x^8 → DUP MUL DUP MUL DUP MUL  

DUP: x → x x  
MUL: → x^2  
DUP: → x^2 x^2  
MUL: → x^4  
DUP: → x^4 x^4  
MUL: → x^8  
