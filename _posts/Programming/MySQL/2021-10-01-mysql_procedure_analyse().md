---
title : MySQL procedure analyse()
categories: [Programming, MySQL]
tags : [MySQL_procedure_analyse(), SQL Injection]
---

## MySQL procedure analyse()
<hr style="border-top: 1px solid;"><br>

+ MySQL에는 LIMIT 함수가 있고 옵션으로 procedure을 사용 가능하며 procedure name에는 analyse()가 있음.

<br>

```
limit 함수에는 아래와 같이 두 개의 옵션(procedure, into)을 사용 가능함.

 [LIMIT {[offset,] row_count | row_count OFFSET offset}]
    [PROCEDURE procedure_name(argument_list)]
    [INTO OUTFILE 'file_name'
        [CHARACTER SET charset_name]
        export_options
      | INTO DUMPFILE 'file_name'
      | INTO var_name [, var_name]]
    [FOR UPDATE | LOCK IN SHARE MODE]]

출처: https://tempuss.tistory.com/entry/Limit-절에서의-SQL-Injection [Tempus]
```

<br>

+ analyse()은 질의 결과를 검사하고 테이블 크기를 줄이는 데 도움이 될 수 있는 각 열에 대해 최적의 데이터 유형을 제안하는 결과의 분석을 반환함.

+ select 문 뒤에 추가해서 사용함.

자세히 
: <a href="https://dev.mysql.com/doc/refman/5.6/en/procedure-analyse.html" target="_blank">dev.mysql.com/doc/refman/5.6/en/procedure-analyse.html</a>  

<br>

```php
mysql> select * from user where id='admin' limit 1 procedure analyse();

| Field_name| Min_value | Max_value | Min_length | Max_length | Empties_or_zeros | Nulls | Avg_value_or_avg_length | Std | Optimal_fieldtype|

| user.user.id| admin| admin | 5 | 5 |0 | 0 | 5.0000 | NULL | ENUM('admin') NOT NULL |


출처: https://tempuss.tistory.com/entry/Limit-절에서의-SQL-Injection [Tempus]
```

<br>

+ field_name은 ```DB명.테이블명.컬럼명```으로 표시함.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 응용 기법
<hr style="border-top: 1px solid;"><br>

PROCEDURE ANALYSE()를 이용한 SQL Ijection
: <a href="https://www.hides.kr/284" target="_blank">www.hides.kr/284</a> 

<br>

MySQL XML Functions
: <a href="https://dev.mysql.com/doc/refman/5.6/en/xml-functions.html" target="_blank">dev.mysql.com/doc/refman/5.6/en/xml-functions.html</a>

<br><br>
<hr style="border: 2px solid;">
<br><br>
