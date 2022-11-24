---
title: LinkedList
author: jimin
date: 2022-11-24 00:00:00 +0900
categories: [Java, 자료구조]
tags: [List] #tags는 반드시 소문자!!!
pin: false
---

# LinkedList란?

- 각 노드가 ‘데이터’와 ‘포인터’를 가지고 한 줄로 연결되어 있는 방식의 자료구조
- 자바의 List 인터페이스를 상속받은 여러 클래스 중 하나
- Collection 프레임워크의 일부, java.util 패키지에 소속

# 내용 특징

1. 데이터를 담고 있는 노드들이 앞,뒤로 연결
2. 데이터 삽입 시 목표 노드의 앞,뒤의 링크만 변경
3. ArrayList에 비해 데이터의 삽입, 삭제가 용이 → 기본적으로 삽입삭제의 시간복잡도는 O(1)로 ArrayList에 비해 용이하지만 만약 삽입삭제 위치를 모른다면 위치탐색을 해야하므로 O(n)이 될 수 있음 
4. 탐색 시 순차적으로 탐색해야 하므로 탐색속도가 떨어진다
5. 메소드 별 시간복잡도
    
    ![LinkedList의 시간복잡도](/assets/img/postpic/Java/%EC%9E%90%EB%A3%8C%EA%B5%AC%EC%A1%B0/LinkedList/LinkedList%EC%82%AC%EC%A7%841.png)
    

# 참고 사이트

- [https://coding-factory.tistory.com/552](https://coding-factory.tistory.com/552)
- [https://www.grepiu.com/post/9](https://www.grepiu.com/post/9)