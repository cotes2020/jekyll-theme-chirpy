---
title : MySQL information_schema
categories: [Programming, MySQL]
tags : [MySQL, information_schema, table_schema, table_name, column_name, processlist, user_privilleges, version check]
---

## Information_schema
<hr style="border-top: 1px solid;"><br>

데이터의 집합 즉, 메타데이터(데이터에 대한 데이터)들을 종류별로 묶어 테이블을 만들었고, 이 테이블을 모아 db를 만들었는데 이것이 information_schema라고 함.

여기에는 MySQL 서버가 운영하는 모든 다른 db에 대한 정보를 저장하는 장소라고 함.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 테이블 역할 정리
<hr style="border-top: 1px solid;"><br>

```
CHARACTER_SETS
COLLATIONS
COLLATION_CHARACTER_SET_APPLICABILITY
COLUMNS
COLUMN_PRIVILEGES
ENGINES
EVENTS
FILES
GLOBAL_STATUS
GLOBAL_VARIABLES
INNODB_BUFFER_PAGE
INNODB_BUFFER_PAGE_LRU
INNODB_BUFFER_POOL_STATS
INNODB_CMP
INNODB_CMPMEM
INNODB_CMPMEM_RESET
INNODB_CMP_PER_INDEX
INNODB_CMP_PER_INDEX_RESET
INNODB_CMP_RESET
INNODB_FT_BEING_DELETED
INNODB_FT_CONFIG
INNODB_FT_DEFAULT_STOPWORD
INNODB_FT_DELETED
INNODB_FT_INDEX_CACHE
INNODB_FT_INDEX_TABLE
INNODB_LOCKS
INNODB_LOCK_WAITS
INNODB_METRICS
INNODB_SYS_COLUMNS
INNODB_SYS_DATAFILES
INNODB_SYS_FIELDS
INNODB_SYS_FOREIGN
INNODB_SYS_FOREIGN_COLS
INNODB_SYS_INDEXES
INNODB_SYS_TABLES
INNODB_SYS_TABLESPACES
INNODB_SYS_TABLESTATS
INNODB_SYS_VIRTUAL
INNODB_TEMP_TABLE_INFO
INNODB_TRX
KEY_COLUMN_USAGE
OPTIMIZER_TRACE
PARAMETERS
PARTITIONS
PLUGINS
PROCESSLIST
PROFILING
REFERENTIAL_CONSTRAINTS
ROUTINES
SCHEMATA
```

<br>

이 중에서 써먹을만한 부분만 찾아봄.

<br><br>

### SCHEMATA
<hr style="border-top: 1px solid;"><br>

SCHEMATA 테이블에는 SCHEMA_NAME 컬럼이 존재함.

SCHEMA_NAME에는 database 목록이 들어있음.

<br>

서버 내에 존재하는 db 목록들 출력
: ```select schema_name from information_schema.schemata```

<br><br>

### COLUMNS
<hr style="border-top: 1px solid;"><br>

+ TABLE_SCHEMA : 해당 db 이름

+ TABLE_NAME : 해당 db 안의 테이블 이름

+ COLUMN_NAME : 해당 db 안의 테이블의 컬럼 이름

+ COLUMN_TYPE : 컬럼 타입과 길이 표시 ex) int(30)

<br>

test db에 있는 테이블과 컬럼 목록 출력
: ```select table_name, column_name from information_schema.columns where table_schema = 'test'```

<br><br>

### PROCESSLIST
<hr style="border-top: 1px solid;"><br>

+ USER : 유저 이름

+ HOST : 호스트 이름

+ INFO : 현재 입력한 쿼리문

<br>

select * from processlist
: info에 select * from processlist limit 0,25가 들어있음
: limit 0,25는 자동으로 붙음

<br><br>

### USER_PRIVILLEGES
<hr style="border-top: 1px solid;"><br>

+ GRANTEE : 사용자 명

+ PRIVILEGE_TYPE : 권한 타입 ex) select, insert 등

+ IS_GRANTABLE : YES or NO 둘 중 하나

<br><br>
<hr style="border: 2px solid;">
<br><br>

## mysql version check
<hr style="border-top: 1px solid;"><br>

version check
: ```select @@version```

<br>

<a href="https://poqw.tistory.com/24" target="_blank">poqw.tistory.com/24</a>에서 더 자세히 확인 가능

<br><br>
<hr style="border: 2px solid;">
<br><br>
