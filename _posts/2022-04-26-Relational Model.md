---
layout: post
title: "Relational Model"
excerpt: "Database 수업 정리(1)"

categories: [Database]
tags:
- [Database, SQL]
use_math: true

#permalink: /Database/Relational-Model/

toc: true
toc_sticky: true

date: 2022-04-25
last_modified_at: 2022-04-25
---
## Structure of Relational Databases

![image](https://user-images.githubusercontent.com/63302432/165148932-5f1dc7f9-67c3-4f7a-a0ed-c0f48b282788.png)


---

## Formal Definition of Relation

- **attributes**
    - $A_1, A_2,\ ...\ ,A_n$(n == Attribute or column의 개수)
        - ex. ID, name, dept_name, salary
- **domains**
    - $D_1, D_2,\ ...\ ,D_n$
        - ex. $height \in [100.0,\ 300.0]$
    - `atomic`: attribute에 해당하는 값이 나눌 수 없고 하나의 단일 값만 가지고 있어야 됨
        - 예시
            
            
            | address |
            | --- |
            | 인천시 미추홀구 ... |
            
            - 인천시 / 미추홀구로 나눠서 사용 → atomic (X)
            - 나눠서 사용하지 않을 때 `atomic`하다고 한다.
    - `null`: 값을 이용할 수 없을 때, 혹은 값이 없을 때 special value인 **null**이라고 한다.
- **relation r**
    - $D_1, D_2,\ ...\ ,D_n$이 주어졌을 때  $D_1 \times D_2\times ...\ \times D_n$의 subset을 말한다.
        
        → $\times$는 cartesian product를 말하며 $D_1, D_2,\ ...\ ,D_n$로 만들 수 있는 모든 조합을 말한다.
        
    - n-tuples($A_1, A_2,\ ...\ ,A_n$)는 $A_i \in D_i$ 를 만족한다.
    - 예시
        
        $D_1 = \{ a,\ b,\ c,\ d \},\ D_2 =  \{1,\ 2,\ 3\}$을 만족할 때
        
        $D_1 \times D_2 = \{ (a, 1),\ (a, 2),\ (a, 3),\ ... (d, 3)\}$ 으로 12개의 순서쌍이 만들어지고 여기서
        
        $(a,\ 1)$  한 쌍을 pair 라고 부른다.
        
        이때 relation r = $\{ (a,\ 2),\ (b,\ 3),\ (d,\ 1),\ (d,\ 2)\}$ 라고 할 수 있다.
        
        이를 tuple로 나타낸다면 
        
        | $A_1$ | $A_2$ |
        | --- | --- |
        | a | 2 |
        | b | 3 |
        | d | 1 |
        | d | 2 |
        
        로 나타낼 수 있으며 각각의 row를 tuple이라고 부른다.
        
        $r \subseteq D_1 \times D_2$
        

---

## Keys

- $r(A_1,\ A_2,\ ... ,\ A_n)$  → $r(R)$ 이라고 하자.
- $K \subseteq R$ 이라고 가정하자.
- **Super Key**
    - 어떤 key를 지정했을 때 이 key를 통해 모든 tuples를 구분할 수 있다면 key가 된다. 
    그 키들의 집합을 `Super Key` 라고 한다.
        
        ![image](https://user-images.githubusercontent.com/63302432/165149257-cb3c1129-ad1a-46c8-bed9-3b78e3d8ed2b.png)
        
    - 이 Database를 보면 {ID}, {ID, name} 뿐만 아니라 모든 attribute를 가지고 있는 것도 key가 될 수 있다는 것을 볼 수 있다.
- **Candidate Key**
    - 이렇게 Superkey에서 key를 만들 수 있는 최소한의 요건을 만족하는 최소 개수의 attribute를 
    `minimal한 키` 라고 부르는데 이를 Candidate Key라고 말한다.
    - some of super key == candidate key
    - all of candidate key == super key
    - 이 키는 앞으로도 중복이 안 될 것이라는 확신이 필요하다.
- **Primary Key**
    - candidate key 중에서 하나만 선택된 key를 말한다.
- **Foregin Key**
    
    ![image](https://user-images.githubusercontent.com/63302432/165149368-7409d771-72c2-497b-b287-ef5dd0d0ed36.png)
    
    - instructor에서 dept_name은 department의 dept_name을 참조한 attribute이다.
        
        이렇게 외부 relation에서 참조한 key를 Foregin Key라고 부른다.
        
    - 참조한 relation의 해당 attribute에서 없는 것을 가지고 있으면 안 된다. (참조해서만 사용해야된다.)

---

## Relational Query Languages

### - Fucntional vs imperative vs declarative

### “Pure” languages

- Relational algebra → functional
    - functional: $c = a\ + b$ → $(Assign(Add(a,\ b),\ c))$ 처럼 순서가 있도록 표현하는 것.

### Relational operators

- 한 개 또는 두 개의 relations(input)만 받는다.
- relational operators의 결과는 항상 하나의 relation이다. (결과 값은 하나만 나와야 된다.)

### Relational Algebra

- **Six basic opeations**
    
    
    | $\sigma$ (sigma) | Select |
    | --- | --- |
    | $\Pi$ (Pi) | Project |
    | $\cup$ | Union |
    | $-$ | Set difference |
    | $\times$   | Cartesian product |
    | $\rho$ (rho) | Rename |
    
    $Intersection(\cap)$ 교집합은 Union과 Set difference의 조합으로 만들 수 있어서 basic operation에  들어가지 않는다.
    
    ---
    
- **Selection($\sigma$)**
    
    select tuples with A = B and D > 5 in relation r
    
    → $\sigma_{A=B\ and\ D>5}(r)$
    
- **Project($\Pi$)**
    - DB 안에서 특정 attributes만 뽑아낼 때
    - Selecting attributes ID and salary from the instructor relation
        
        → $\Pi_{ID,\ salary}(instructor)$
        
- **Cartesian Product($\times$)**
    
    ![image](https://user-images.githubusercontent.com/63302432/165149425-bccba781-3a59-4822-ba40-d97b338da4c4.png)
    
- **Natural Join($\Join$)**
    
    ![image](https://user-images.githubusercontent.com/63302432/165149470-cd448aaa-275e-42cc-b429-98413de97e05.png)
    
    ---
    
    instructor 와 department를 natural join 한 결과 값
    
    ≠ {instructor $\Join_{\ inst.dept\_name\ =\ dept.dept\_name}$department}
    
    == $\{\Pi_{R\cup U}(\sigma_{inst.dept\_name=dept.dept\_name}(inst\times dept))\}$
    
    ---
    
    ![image](https://user-images.githubusercontent.com/63302432/165149490-be3b4b42-a765-4272-a6d8-b2492a3c3021.png)

    
- **Theta Join($\Join_{\theta}$)**
    - $r \Join_\theta s == \sigma_{\theta}(r\times s)$
    - attribute에 대한 중복을 제거하지 않음
- **Union($\cup$)**
    - 중복 제외 합치기
- **Set difference($-$)**
- **Rename($\rho)$**
    - 이름 변경/지정
    - $\rho_{x(A_1, A_2, ...,A_n)}(E)$
    - $\rho_{cs\_inst}(\sigma_{dept\_name=cs}(instructor))$
    - $\rho_{cs\_inst(A,B)}(\Pi_{name, sal}(\sigma_{dept\_name=cs}(instructor))$
        
        → instructor DB에서 부서 이름이 cs인 교수들의 name과 sal을 ‘cs_inst’ realtion의 A, B attribute로 지정하겠다!
        
    - **self join($\rho_i$)**
        - $\Pi_{i.ID}(\sigma_{i.salary>j.salary}(\rho_i(instructor)\times\sigma_{j.id=12121}(\rho_j(instructor)))$ 
           $==$ 
        $\Pi_{i.ID}(\rho_{i}(instructor)\Join_{i.salary>j.salary}\rho(_{j}(instructor))$(?)
        - instructor에 있는 id가 ‘12121’인 것과 모든 id를 비교해(cartesian p-duct) i의 salary와 j의 salary를 비교해 i의 salary가 높은 경우만 추출

        
---

