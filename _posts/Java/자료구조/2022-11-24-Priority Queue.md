---
title: Priority Queue
author: jimin
date: 2022-11-24 00:00:00 +0900
categories: [Java, 자료구조]
tags: [Queue, 우선순위큐]
pin: false
---

# Priority Queue란?

- **`PriorityQueue`**란 **`우선순위 큐`**로써 **일반적인 큐의 구조 FIFO(First In First Out)를 가지면서**, 데이터가 **들어온 순서대로 데이터가 나가는 것이 아닌 우선순위를 먼저 결정**하고 그 **우선순위가 높은 데이터가 먼저 나가는 자료구조**이다.
- **`PriorityQueue`를 사용하기 위해선** 우선순위 큐에 **저장할 객체는 필수적으로 `Comparable Interface`를 구현해야한다.(중요)**
- Queue를 implements 하는 **class**

# Priority Queue 특징

1. **높은 우선순위의 요소를 먼저 꺼내서 처리하는 구조**이다.
    
    *우선순위 큐에 들어가는 원소는 비교가 가능한 기준이 있어야한다.*
    
2. **내부 요소는 힙으로 구성되어 이진트리 구조**로 이루어져 있다.
3. 따라서 우선순위 큐를 이용한 정렬은 O(nlogn)이다.
4. **우선순위를 중요시해야 하는 상황에서 주로 쓰인다.**

# 실제 사용

Gillog class (Comparable Interface구현함)

```java
class Gillog implements Comparable<Gillog> {

    private int writeRowNumber;
    private String content;

    public Gillog(int writeRowNumber, String content) {
        this.writeRowNumber = writeRowNumber;
        this.content = content;
    }

    public int getWriteRowNumber() {
        return this.writeRowNumber;
    }

    public String getContent() {
        return this.content;
    }

    @Override
    public int compareTo(Gillog gillog) {

        if (this.writeRowNumber > gillog.getWriteRowNumber())
            return 1;
        else if (this.writeRowNumber < gillog.getWriteRowNumber())
            return -1;
        return 0;
    }
}
```
main

```java
public static void main(String[] args) {

        PriorityQueue<Gillog> priorityQueue = new PriorityQueue<>();

        priorityQueue.add(new Gillog(3650, "10년후 글"));
        priorityQueue.add(new Gillog(31, "한달 후 글"));
        priorityQueue.add(new Gillog(1, "첫번째 글"));
        priorityQueue.add(new Gillog(365, "1년후 글"));

        while (!priorityQueue.isEmpty()) {
            Gillog gilLog = priorityQueue.poll();
            System.out.println("글 넘버 : " + gilLog.getWriteRowNumber() + " 글 내용 : " + gilLog.getContent());
        }
    }
```



# 참고 사이트

- [https://velog.io/@gillog/Java-Priority-Queue우선-순위-큐](https://velog.io/@gillog/Java-Priority-Queue%EC%9A%B0%EC%84%A0-%EC%88%9C%EC%9C%84-%ED%81%90)
- [https://velog.io/@april_5/자료구조-우선순위-큐Priority-Queue](https://velog.io/@april_5/%EC%9E%90%EB%A3%8C%EA%B5%AC%EC%A1%B0-%EC%9A%B0%EC%84%A0%EC%88%9C%EC%9C%84-%ED%81%90Priority-Queue)