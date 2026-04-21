---
title: "PS-Algorithm | 코딩 테스트를 위한 알고리듬, 자료 구조"
# description: ""
categories: [컴퓨터, 알고리즘, PS]
tags: [PS, 알고리듬, 자료구조]
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

# 알고리듬 과목
# date: 2023-09-07. 10:29
# last_modified_at: 2023-09-07. 10:29
# last_modified_at: 2023-09-12. 13:31
# last_modified_at: 2023-09-14. 10:30
# last_modified_at: 2023-09-19. 13:37
# last_modified_at: 2023-09-21. 10:24
# last_modified_at: 2023-09-26. 13:33
# last_modified_at: 2023-10-05. 10:32
# last_modified_at: 2023-10-10. 13:32
# last_modified_at: 2023-10-17. 13:30 ?
# last_modified_at: 2023-10-19. 10:38
# last_modified_at: 2023-10-26. 10:33
# last_modified_at: 2023-10-31. 13:31
# last_modified_at: 2024-02-17. 14:18
# last_modified_at: 2024-02-19. 18:02
# last_modified_at: 2024-02-19. 19:35
# last_modified_at: 2024-08-29. 21:34

# Data Structure
# date: 2024-02-18. 11:34
# last_modified_at: 2024-02-19. 17:52
# last_modified_at: 2024-02-21. 20:54
# last_modified_at: 2024-08-29. 22:00

date: 2024-11-13. 05:10 # Init
# last_modified_at: 2025-03-04. 23:41 # 읽을 것 하나
last_modified_at: 2025-09-24. 22:15 # E 전략
---

2024-11-13. 05:10: 글 계승, 병합.  
`2023-09-07-U-Algorithm: 알고리듬 과목`,  
`2024-02-18-Data-Structure: Data Structure`  

## 머리말

---

코딩 테스트를 위한 알고리듬, 자료 구조.  

## 주제

---

알고리듬과 자료구조.  

이는 특정 분야의 흥미로운 문제를 풀거나, 코드 최적화를 위한 것.  
단순히 서로 다른 Alg-, Dat-Str를 사용하는 것만으로 성능과 결과가 크게 달라질 수 있다.  

언어를 잘 사용하는 것도 물론 중요하지만,  
그 내부 동작과 여러 Alg-/Dat-Str 간의 차이점을 이해하고 써야 의미가 있다.  

### Algorithm | 알고리듬

> **범용적 절차 / 명령어 집합**  
> 어떤 일을 하기 위해 미리 만들어둔, 이미 만들어진 명령의 집합.  
> 사실 모든 코드는 이런 의미에서 알고리듬.  

### Data Structure | 자료 구조

> **데이터를 저장하고 조작하는 방법**

## 전략

---

### 정리

- 목표: 게임 코테 안정적 통과
- 설계:
  - 초기 설계:
    - 알고리듬 개념 복습
      - 시간 복잡도
    - C++ 환경 설정
      - C#으로도 풀자. 알고리듬 복습 + 언어 이해 + 양쪽 언어 비교
    - 일주일에 한 문제는 풀기
    - 가이드 라인 설정
      - 바킹독
      - 코드트리

### 목표

- 초기
  - 과거 기준: 실랜디
    - ~= 백준 골드2, ~= 클래스 5, 플머스 1~3
    - Bfs dfs, 위상정렬, 기초적인 자료구조 등
  - 백준 Class 난이도 별로
- 후기
  - 최근 기준: 골랜디
    - ~= 백준 플래, 플머스 \>= 3, 넥토는 5도 보임
  - 어떻게 푸는 지 바로 생각날 정도로, 외울 정도로, 어려운 게 있으면 더 풀고
  - 프로그래머스: 42895, 기출
  - 코드포스 (블루)

### 문제와 방법

- 1:
  - Q: 공부랑 실전 문제 괴리가 있음
  - A: 코테 보는 곳 지원해서 경험 쌓기 or 대회

- 2:
  - Q: 처음 보는 유형이나 잘 모르는 것
  - A: 답 보기
    - 왜 이렇게 짰는지 **이해**가 가장 중요
    - 여러 번 돌려봐도 괜찮음
    - 잘 못 풀면 그 알고리듬 집중적으로 공부하면 돼

- 3:
  - Q: 히든 케이스
    - 플머스/리트코드: 케이스 전부 공개
    - 실제: 히든 케이스
      - 예외 처리, 실수 -> 치명적
  - A: 처음부터 설계 잘하고, 검증도 잘하기

- 4:
  - Q: 환경
  - A: HackerRank 해커랭크
    - 테스트 케이스 어디서 틀렸는지 보려고 프린트하면 안나오는데, return으로 그냥 값 넘겨주면 출력돼서 케이스 어떻게 나오는지 알 수 있다? (히든 케이스)

- 5:
  - Q: 문제
  - A: 6문제, 약 3시간 30분
    - 4.5솔 -> 위험

### 유형

- 참고
  - \*: 강조
  - \!: 넥토 출현
  - \.: 넥토 출현 (불확실)
  - \?: 크게 언급 없는 것 같은
- 분야: (바킹독)
  - 못해도 11강, 그리디, 탐색 (완전 탐색/BFS/DFS), 그래프, DP, 백트래킹/시뮬레이션
  - 좀 더? (크루스칼, 다익스트라/최소신장트리, 트라이)
- 하:
  - 구현 & 자료구조:
    - C++ 자료구조-컨테이너
    - 해싱: Hash, Set, Map으로 간단히 풀 수 있는
    - 우선 순위 큐 .
    - 파싱, 스택, 힙, 트리
    - **문자열 조합/정렬** !
  - 그 외 단골
    - **냅색**
    - **lis, lcs**
    - **이분탐색** .
  - **투포인터, 윈도우 슬라이드** !!.
  - DFS/BFS ?.
  - 시뮬레이션 (스택, 큐, 덱) ?
- 중
  - **위상정렬**
    - 노드와 엮을 수 있는 계층적 문제
  - **MST** / 크루스칼 + 프림
  - 그리디
  - 백트래킹
- 상
  - **DP** !!
    - 가끔 수학과 엮여서 출제
    - 거의 항상 나옴
  - **트라이** / KMP
  - 다익스트라
  - 플로이드
- 모름
  - **그래프** !!
    - **노드 같은거 섞인 최단경로** .
  - 세그먼트 트리
  - A*
- X
  - 라운드로빈 (스케쥴링)
  - 삼항트리
- [22 넥토](https://thereisnotruth.github.io/daily/221007_daily_1/)

## 메모

---

### 즐겨찾기

- 사이트
  - [솔브닥](https://solved.ac/profile/mascari4615)
  - [백준](https://www.acmicpc.net/)
  - [프로그래머스](https://programmers.co.kr/)
  - [엣코더](https://atcoder.jp/home)
    - [엣코더 분석/추천](https://kenkoooo.com/atcoder#/table/)
  - [릿코드](https://leetcode.com/)
  - [코드포스](https://codeforces.com/)
  - [코드트리](https://www.codetree.ai/ko/trail-info)

- 강의
  - [알고리즘 - 바킹독](https://blog.encrypted.gg/)
  - [알고리즘 - 라이](https://blog.naver.com/prologue/PrologueList.naver?blogId=kks227)

- _
  - [C# vs C++](https://moguwai.tistory.com/entry/C%EA%B3%BC-C%EC%9D%98-%EB%AC%B8%EB%B2%95%EC%A0%81%EC%9D%B8-%EC%B0%A8%EC%9D%B4%EC%A0%90)
  - [코딩 테스트 및 알고리즘 문제해결 공부 방법](https://www.slideshare.net/SuhyunPark23/kucc-2022-4)
  - [알고리즘 공부 방법/순서](https://baactree.tistory.com/14)
  - [[알고리즘] 아호 코라식(Aho-Corasick) 알고리즘](https://pangtrue.tistory.com/305)
  - [[알고리즘] KMP(Knuth-Morris-Pratt) 알고리즘](https://pangtrue.tistory.com/303?category=724827)
  - [[자료구조] Trie(트라이)](https://pangtrue.tistory.com/331?category=724827)
  - [메모이제이션](https://namu.wiki/w/%EB%A9%94%EB%AA%A8%EC%9D%B4%EC%A0%9C%EC%9D%B4%EC%85%98)
  - [C++ 익명함수 사용법](https://progl.tistory.com/5)
  - [누적합](https://book.acmicpc.net/algorithm/prefix-sum)

- 읽어볼 것
  - [1](https://www.acmicpc.net/board/view/34613)

### 참고

- [[바킹독의 실전 알고리즘] 0x03강 - 배열](https://youtu.be/mBeyFsHqzHg?si=8rGdOuR6HleGFKgG)
- Hello Coding 알고리즘
- 알고리즘 도감

### **TODO:**

- 복합 자료구조
- 연결리스트 배열 (A로 시작하는 이름 연결리스트, B로 시작하는 이름 연결리스트, ... )
