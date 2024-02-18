---
title : MySQL 사용법 in Linux (Incomplete)
categories: [Programming, MySQL]
tags : [MySQL, Incomplete]
---

## root 계정 생성
<hr style="border-top: 1px solid;"><br>

```mysqladmin -u {root} password {password}```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 로그인
<hr style="border-top: 1px solid;"><br>

```sql
mysql [OPTIONS] [database]

-p, --password[=name]
                      Password to use when connecting to server. If password is
                      not given it's asked from the tty.

-u, --user=name     User for login if not current user.
```

<br>

예를 들어, test db에 접속할 때
: ```mysql -u {root} -p test```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Database 생성 및 삭제
<hr style="border-top: 1px solid;"><br>

+ 생성
  + ```CREATE DATABASE {db name} CHARACTER SET utf8 COLLATE utf8_general_ci;```

<br>

+ 삭제
  + ```DROP DATABASE {db name};```

<br>

+ 열람
  + ```SHOW DATABASES;```

<br>

+ 선택
  + ```USE {db name}```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 테이블 생성
<hr style="border-top: 1px solid;"><br>

```sql
CREATE TABLE table_name (
    칼럼명1 data_type,
    칼럼명2 data_type
)
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 테이블 값 조회
<hr style="border-top: 1px solid;"><br>


<br><br>
<hr style="border: 2px solid;">
<br><br>

## 테이블 값 추가
<hr style="border-top: 1px solid;"><br>


<br><br>
<hr style="border: 2px solid;">
<br><br>

## 테이블 값 수정
<hr style="border-top: 1px solid;"><br>


<br><br>
<hr style="border: 2px solid;">
<br><br>

## 테이블 삭제
<hr style="border-top: 1px solid;"><br>


<br><br>
<hr style="border: 2px solid;">
<br><br>
