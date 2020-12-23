---
title: mysql timezone 수정하기
author: juyoung
date: 2020-12-16 16:08:00 +0800
categories: [project, portpolio]
tags: [project]
---

# 1. mysql shell을 실행한다.  
저는 MySQL 8.0 Command Line Client - Unicode 을 실행했습니다.  

# 2. mysql> select @@global.time_zone, @@session.time_zone; 명령을 실행한다.

```console

mysql> select @@global.time_zone, @@session.time_zone;
+--------------------+---------------------+
| @@global.time_zone | @@session.time_zone |
+--------------------+---------------------+
| SYSTEM             | SYSTEM              |
+--------------------+---------------------+
1 row in set (0.00 sec)
```  

# 3. SET GLOBAL time_zone='Asia/Seoul';  
SET time_zone='Asia/Seoul'; 명령을 실행한다가 아래의 에러 발생!  

ERROR 1298 (HY000): Unknown or incorrect time zone: 'Asia/Seoul'  

[[MySQL] mysql server timezone 한국으로 설정하기.](https://jwkim96.tistory.com/23)이곳에 설명된 대로 실행하다가


```language
https://dev.mysql.com/downloads/timezones.html에서
윈도우 유저는 Non POSIX with leap seconds 파일을 다운로드 후 압축을 풀고, 
약 47,000줄 정도의 sql을 mysql 스키마에 실행한다. (use mysql)
```
이게 대체 뭔소린지 알 수가 없어서 포기하고...

# 4. SET GLOBAL time_zone='+09:00';  
SET time_zone='+09:00'; 명령을 실행한다  

# 5. select @@global.time_zone, @@session.time_zone; 
이 명령어로 다시 확인해보면 이런 화면이 보인다  


```console 

mysql> select @@global.time_zone, @@session.time_zone;
+--------------------+---------------------+
| @@global.time_zone | @@session.time_zone |
+--------------------+---------------------+
| +09:00         | +09:00           |
+--------------------+---------------------+
1 row in set (0.00 sec)
```

# 6. 그러나 이 설정은 mysql를 재실행하면 다시 초기화되므로 영구적으로 적용시키켜야 한다.  
 
my.ini 파일에 [mysqld] 섹션 맨 아래에 default-time-zone을 추가한다  

파일경로: C:\ProgramData\MySQL\MySQL Server 8.0\my.ini  

```language
# If the value of this variable is greater than 0, a replica synchronizes its master.info file to disk.
# (using fdatasync()) after every sync_master_info events.
sync_master_info=10000

# If the value of this variable is greater than 0, the MySQL server synchronizes its relay log to disk.
# (using fdatasync()) after every sync_relay_log writes to the relay log.
sync_relay_log=10000

# If the value of this variable is greater than 0, a replica synchronizes its relay-log.info file to disk.
# (using fdatasync()) after every sync_relay_log_info transactions.
sync_relay_log_info=10000

# Load mysql plugins at start."plugin_x ; plugin_y".
# plugin_load

# The TCP/IP Port the MySQL Server X Protocol will listen on.
loose_mysqlx_port=33060

#Default time-zone setting - (기본 time-zone 설정)//맨 마지막에 이 부분을 더해준다.
default-time-zone='+09:00'
```  

# 7. 마지막으로 mysql서버를 재시작 해준다.  

windows관리도구 서비스 프로그램 실행 > 서비스 창에서 mysql80 오른쪽 클릭> 중지 후 시작 버튼 누르기
![windows관리도구 서비스 프로그램](/assets/img/service.jpg)
![windows관리도구 서비스 프로그램](/assets/img/mysql_restart.png)  

참고:   
[ MySQL Timezone 접속문제 해결](https://junho85.pe.kr/1483),  
[[MySQL] mysql server timezone 한국으로 설정하기.](https://jwkim96.tistory.com/23)