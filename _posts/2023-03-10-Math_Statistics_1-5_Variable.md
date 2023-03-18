---
title: "[기초통계] 변수"
date: 2023-01-03 18:32:00 +0900
# date: 2023-03-06 08:43:59
author: "HYUNHO NOH"
categories: [Math]
tag: [Statistics, Sapientia a Dei]
---
<!-- author_profile: false
toc: true
sidebar:
    nav: "docs"
--- -->

# 정의

하나의 개념을 대표하며 그것의 속성 값을 나타내는 (변하는) 수

- 예시1)
    
    김대리는 고객 대상으로 상품과 서비스에 대한 만족도 조사를 하였다. 여기서 변수는 **고객 만족도**이고, 조사한 변수의 값은 고객마다 다르기 때문에 값이 변화한다. 이를 변수라고 함
    
- 예시2)
    - 변수: 교수님 강의 만족도는 어느 정도 입니까?
    - 속성 - 다음 속성 중 하나 선택
        - 매우 만족(5)
        - 만족(4)
        - 보통(3)
        - 불만족(2)
        - 매울 불만족(1)

![image1](https://github.com/nhh2907/nhh2907.github.io/blob/master/image/2023-03-15_Math_Statistics_1-5_Variable/1.jpeg)

<img src = "https://github.com/nhh2907/nhh2907.github.io/blob/master/image/2023-03-15_Math_Statistics_1-5_Variable/1.jpeg">


# 변수 구성 요소

참고) 1에서 4의 순서로 구성되어 있음

1. 변수 : 예시로 “교육 정도”
2. 속성 : 속성1(대졸), 속성2(고졸), 속성3(중졸), ….
3. 값 : 각 속성을 수로 변경
4. 관계 : 값들 사이에 **순서 유무**
    - 만약 순서가 있으면 값들 사이에 **등간척도가 존재 유무**

# 종류

<img src = "https://github.com/nhh2907/nhh2907.github.io/blob/master/image/2023-03-15_Math_Statistics_1-5_Variable/4.png">

By numerical variables, we mean variables for which **the values are numbers**. And, by categorical variables, we mean variables for which **the values are categories**:

- 범주형(Categorical)/이산형(Discrete) 변수
    - 명목(**N**ominal) 변수/척도
    - 순서(**O**rdinal) 변수/척도
- 수치형(Numerical)/연속형(Continuous) 변수
    - 구간(**I**nterval) 변수/척도
    - 비율(**R**atio) 변수/척도

# 각 변수 종류에 대한 설명

### (1) 명목(Nominal) 변수 - Unordered List

A variable that has two or more categories, without any implied ordering

- 각 **속성(범주)**간 순위 없음
- 범주에 할당된 값은 범주 이름을 대신할 뿐 의미는 없음
- 예)
    - Gender(변수) - Male, Female
    - Marital Status(변수) - Unmarried, Married, Divorcee
    - State(변수) - New Delhi, Haryana, Illinois, Michigan

### (2) 순위(Ordinal) 변수 - Ordered List

A variable that has two or more categories, with clear ordering

- 각 **속성(범주)**간 순위가 있다
- 범주에 할당된 값은 범주의 이름뿐만 아니라 순위를 나타냄
- 순위 사이에 **등간성은 없다**
    - 즉, “매우 좋음”과 “좋음”과의 간격 크기와 “나쁨”과 “매우 나쁨”의 **간격 크기가 같다고 단언할 수 없다**
- 예)
    - Scale - Strongly Disagree, Disagree, Neutral, Agree, Strongly Agree
    - Rating - Very low, Low, Medium, Great, Very great

### (3) 구간(Interval) 변수 - Ordered List + Intervals without a zero point

An interval variable is similar to an ordinal variable, except that the intervals between the values of the interval variable **are equally spaced**. In other words, it has order and equal intervals.

- 각 **속성(값)** 사이에 **등간성이 있음**
- 할당된 값은 임의 단위로서 **비율**이나 **절대영(”0”)의 의미가 없음**
    - 따라서 덧셈은 가능하나 곱셈은 안됨
- 예)
    - Temperature in Celsius - Temperature of 30°C is higher than 20°C, and temperature of 20°C is higher than 10°C. The size of these intervals is the same.
    - Annual Income in Dollars - Three people who make $5,000, $10,000 and $15,000. The second person makes $5,000 more than the first person and $5,000 less than the third person, and the size of these intervals is the same

### (4) 비율(Ratio) 변수 - Ordered List + Intervals with a zero point

It is interval data with a natural zero point. When the variable equals 0.0, there is none of that variable

- 측정된 범주 사이에 **등간성이 있음**
- 할당된 값은 임의 단위로서 **비율**과 **절대영(”0”)의 의미가 있음**
    - 따라서 덧셈과 곱셈 모두 가능
- 예)
    - Height
    - Weight
    - Age
    - Time
    - Temperature in Kelvin - It is a ratio variable, as 0.0 Kelvin really does mean 'no temperature

# 주의

ordinal을 ratio로 무작정 바꾸는 경우, ordinal 값 사이에 존재하는 간극의 크기가 명확하지 않을 수 있기 때문에 위험함 

예를 들어 “매우불만족”과 “보통” 사이의 거리가 “보통”과 “매우 만족” 사이의 거리와 같다는 위험한 전제를 깔고 있는 겁니다.

# 질적변수(Qualitative Variable)와 양적변수(Quantitative Variable)

<aside>

※ 변수의 속성이 **수**인지 **수가 아닌지**로 구분

</aside>

<img src = "https://github.com/nhh2907/nhh2907.github.io/blob/master/image/2023-03-15_Math_Statistics_1-5_Variable/2.png">

### 질적변수(Qualitative Variable)
    
Sometimes referred to as “categorical” variables, these are variables that take on (속성값으로) names or labels and can fit into categories. Examples include: 

변수의 값이 비수치적이며 특정 카테고리에 포함 시키도록 하는 변수

- 예)
    - 아이들 장난감의 색상
    - 특정 집단 구성원의 성별
    - 종교
        
        <img src = "https://github.com/nhh2907/nhh2907.github.io/blob/master/image/2023-03-15_Math_Statistics_1-5_Variable/3.png">
            
### 양적변수(Quantitative Variable)
    
변수의 값을 숫자로 나타낼 수 있는 변수

- 예)
    - 사람의 키나 몸무게
    - 한 가정의 소득
    - 고속도로 통행량

# Reference

[Qualitative vs. Quantitative Variables: What's the Difference? (statology.org)](https://www.statology.org/qualitative-vs-quantitative-variables/)  
[Basic Statistics : Types of Variables (listendata.com)](https://www.listendata.com/2014/04/basic-statistics-types-of-variables.html)  
[Sapientia a Dei](https://www.youtube.com/watch?v=otvjWhlefnc&t=362s)  
[통계학 변수 - 네이버 백과](https://terms.naver.com/entry.naver?docId=727342&cid=42140&categoryId=42140)