---
title: LinkedHashMap
author: jimin
date: 2022-11-24 00:00:00 +0900
categories: [Java, 자료구조]
tags: [Map] #tags는 반드시 소문자!!!
pin: false
---

# LinkedHashMap

# LinkedHashMap 이란?

- Map 인터페이스를 구현한 Map 구현체 중 하나
- LinkedList로 저장되어 순서가 있다.

# 내용 특징

1. 내부적으로 LinkedList를 사용하므로 값을 출력할 때 순서대로 출력되어 나온다.
2. 순서 말고는 HashMap과 다른 것이 없음
3. 즉, 삽입 순서를 보장하는 HashMap이다.
4. get = O(1), containsKey = O(1), next = O(1) 

# 참고 사이트

- [https://genie247.tistory.com/entry/MapHashMapTreeMapHashtable-차이점](https://genie247.tistory.com/entry/MapHashMapTreeMapHashtable-%EC%B0%A8%EC%9D%B4%EC%A0%90)