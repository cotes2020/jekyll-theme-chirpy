---
title: "Stored Procedure SQL Injection"
date: 2024-04-12 19:13:00 +0900
author: aestera
categories: [Web]
tags: [Web, Database, SQL Injection]
math: true
---

### **0x1. Intro**

**프로시저**(procedure)의 사전적 의미는 ***'어떤 일을 하는 공식적이거나 인정된 방식인 일련의 행동'*** 이다. 이 포스팅에서는 DB서버에 저장된 프로시저인 **Stored Procedure**에 대해 알아보고 이의 **SQL Injection** 가능성에 대해 살펴보겠다.

***
### **0x2. Stored Procedure란?**

여러 SQL을 사용하기 위한 쿼리들의 집합. DB내부에 저장된 **사용자 지정 함수**와 비슷하다고 생각하면 편할 것 같다.  
- SQL Server에서 제공되는 프로그래밍 기능. 일련의 쿼리를 하나의 모듈화 시켜 사용
- 자주 사용하는 쿼리들의 집합을 Stroed Procedure로 저장하여 재사용
- 함수와 같이 parameter들을 전달하여 사용 가능

SQL의 사용자 지정 함수는 Stored Procedure와  비슷해보이지만 명확한 차이가 있다.

|                        | <font color="#8db3e2">Stored Function</font> | <font color="#8db3e2">Stored Procedure</font> |
| ---------------------- | :----------------------------------------------: | :-----------------------------------------------: |
| **Return Value**       |                return문으로 하나의 값 리턴                |                  여러개의 OUT파라미터 사용                  |
| **Create Syntax**      |               CREATE FUNCTION ....               |                CREATE PROCEDURE...                |
| **Call Syntax**        |                      SELECT                      |                       CALL                        |
| **Exception Handling** |                        -                         |                    TRY...CATCH                    |
| **호출 위치**              |      Function에서 Stored Procedure <br>호출 불가       |       Stored Procedure에서 Function <br>호출 가능       |
| **사용 가능 명령어**          |           INSERT, UPDATE, DELETE 사용 불가           |           INSERT, UPDATE, DELETE 사용 가능            |

> **Stored Procedure는** DB값을 조작 및 조회 쿼리문들을 묶어 사용하는 하나의 **`API`**,<br>
> **Stored Function**은 DB를 조회하고 이를 연산하는 등 간단한 작업을 위한 SQL 묶음으로 볼 수 있다.

<br>

**Stored Procedure(SP)**는 부족한 최적화로 인해 처리 속도가 느리고 코드 재사용성의 비효율적이지만, 서버에서 한번의 요청으로 여러 SQL문을 실행 할 수 있기 때문에 네트워크 부하를 줄일 수 있고 DB 유저 대신 프로시져에 권한을 부여할 수 있어 보안성 향상을 위해 사용하기도 한다. <br><br>
SP가 무엇인지 글로 알아봤으니 이제부터는 MySQL을 직접 조작하며 알아보자. 다른 DB를 사용해도 문법의 차이만 존재할 뿐 아래 실습 내용들의 결과는 크게 다르지 않다.

```
mysql> select @@version;
+-------------------------+
| @@version               |
+-------------------------+
| 8.0.36-0ubuntu0.22.04.1 |
+-------------------------+
```

```sql
CREATE DATABASE testDB;

CREATE TABLE employees(
	id int NOT NULL,
	name varchar(50),
	location varchar(50),
	age int,
	PRIMARY KEY (id)
);

INSERT INTO employees (id, name, location, age) 
VALUES
	(1, 'Alpha', 'Australia', 20),
	(2, 'Bravo', 'Brazil', 28),
	(3, 'Charlie', 'China', 19),
	(4, 'Delta', 'Denmark', 33);
```

***

### **0x3. Stored Procedure 활용**

#### Stored Procedure 생성, 삭제

```sql
USE testDB;
DROP procedure IF EXISTS genSP; -- SP 삭제. `IF EXISTS` 생략 가능

-- SP 생성
DELIMITER $$
USE testDB $$
CREATE PROCEDURE genSP(IN employeeName VARCHAR(255), OUT userCnt INT, INOUT plus INT)
BEGIN
	DECLARE sum INT DEFAULT 10;
		-- sum 지역변수를 선언하고 10으로 초기화
	SELECT * FROM testDB.employees WHERE name = employeeName;
		-- emplyees table에서 name 컬럼이 입력받은 employeeName인 행 출력
	SELECT count(User) INTO userCnt FROM mysql.user;
		-- MySQL 유저의 수를 userCnt에 해당하는 변수에 리턴
	SET plus = plus + sum;
		-- 초기화된 값 plus에 sum의 값을 더해 plus에 해당하는 변수에 리턴
END $$ 
DELIMITER ;
```

 - **DELIMITER :** 프로시저 앞,뒤의 위치하여 안에 있는 부분을 한번에 실행될 수 있게 하는 역할
 - **Stored Procedure 매개변수의 3가지 모드** 
	 - **IN :** 프로시저 내부에서 값이 변경 가능하지만 프로시저 반환 후 호출자는 변경 불가. (call-by-value 와 유사)
	 - **OUT :** 초기값은 내부에서 NULL. 프로시저 반활될 때 내부 값 리턴 (call-by-reference 와 유사)
	 - **INOUT :** 호출자에 의해 변수가 초기화되고 프로시저에 의해 수정된다. IN + OUT 의 기능
- **DECLARE :** 프로시저 내부에서 사용하는 지역변수 선언
- **BEGIN ~ END :** 프로시저의 실질적인 코드가 들어있는 영역. UPDATE, INSERT, DELETE 등의 쿼리 뿐 만 아니라 조건식, 반복문 등을 사용하여 사용자가 원하는 기능을 직접 코딩

#### Stored Procedure 호출

```sql
SET @val = 10;
CALL genSP('Bravo', @cnt, @var);
```

`@val`을 10으로 초기화하고 `@cnt`변수와 함깨 `genSP`의 매개변수로 전달했다. 아래는 해당 SQL문의 실행 결과이다.

```
mysql> SET @val = 10;
mysql> CALL genSP('Bravo', @cnt, @var);
+----+-------+----------+------+
| id | name  | location | age  |
+----+-------+----------+------+
|  2 | Bravo | Brazil   |   28 |
+----+-------+----------+------+

mysql> SELECT @cnt, @var;
+------+------+
| @cnt | @var |
+------+------+
|    5 |   20 |
+------+------+
```

프로시저 내부의 쿼리문들이 성공적으로 실행된 것을 볼 수 있다.
- `SELECT * FROM testDB.employees WHERE name = 'Bravo';`의 결과값이 출력됐다.
- `SELECT count(User) INTO userCnt FROM mysql.user;`쿼리가 실행돼 `@cnt`변수에 리턴값이 저장됐다.
- `SET plus = plus + sum;`쿼리가 실행돼 10으로 초기화된 `@var`에  지역변수`sum`의 값 10이 더해져 20의 값이 리턴됐다.

---

### **0x5. Stored Procedure SQL Injection**

그렇다면 Stored Procedure에서 SQL Injection 취약점이 발생할 수 있는지 확인해보기 위해 아래와 같은 테이블과 프로시저를 만들어 테스트해보았다. 

```sql
CREATE TABLE user(
	seq int NOT NULL AUTO_INCREMENT,
	uid varchar(50),
	upw varchar(50),
	PRIMARY KEY (seq)
);

INSERT INTO user (uid, upw) 
VALUES
	('guest', 'guest'),
	('admin', 'ADM1NP4SSW0RD');
```

```sql
USE testDB;
DROP procedure IF EXISTS testSQLI;
DELIMITER $$
USE testDB $$
CREATE PROCEDURE testSQLI(IN id VARCHAR(255))
BEGIN
	SELECT id;  -- 입력값 디버깅을 위한 출력
	SELECT * FROM user WHERE uid=id;
END $$ 
DELIMITER ;
```

```
mysql> call testSQLI('\' or 1=1 -- a');
+---------------+
| id            |
+---------------+
| ' or 1=1 -- a |
+---------------+
```

`call testSQLI('\' or 1=1 -- a');`쿼리문으로 SQLI 를 시도해보면 SQL Injection이 실패한 것을 볼 수 있다.<br><br>

#### SQL Injection의 원인

SQL Injection은  **DB가 공격자의 입력과 쿼리문을 구분하지 못하기 때문에 발생한다**.<br>
**`SELECT * FROM user WHERE username='admin' OR password='foo' OR 1=1`**
위 쿼리에서  공격자가는 `foo' OR 1=1`이라는 payload를 입력해 SQL Injection 공격을 실행했다. DB는 해당 payload가 기존의 쿼리문인지 아니면 공격자의 입력인지 알 수가 없다.<br><br>

#### SQL Injection 대응방법

- **Prepared Statement**
SQL Injection으로부터 DB를 보호하기 위해서는 DB가 쿼리와 입력 데이터를 구분할 수 있어야 한다. 이를 위해 프로시저는 **Prepared Statement**를 사용한다. <br>
일반적으로 SQL문은 **`구문분석(parsing) -> 최적화 -> 실행 가능 코드로 포맷팅 -> 실행 -> 인출`** 이렇게 5단계를 거쳐 실행된다. <br>
**Prepared Statement**란 미리 쿼리에 대한 컴파일을 실행하고 사용자의 입력값을 다중에 입력하는 방식이다. 일반적인 방식과 달리 불필요한 동작들을 매번 수행하지 않고 실행과 인출 이전의 단계는 한번의 컴파일로 미리 만들어 놓고 나중에는 미리 컴파일된 Prepared Statement를 가져다 쓰는 것이다. 이렇게 되면 DB는 쿼리문과 공격자의 악의적 입력을 정확하게 구분할 수 있게 된다. 하지만 그렇다고 해서 Stored Procedure가 SQL Injection에 무적인건 아니다.<br><br>

#### SQL Injection 우회

- **Dynamic SQL**

Dynamic SQL은 SQL문을 문자열 변수에 담아 실행하는 SQL문이다.

```sql
USE testDB;
DROP procedure IF EXISTS testSQLI;
DELIMITER $$
USE testDB $$
CREATE PROCEDURE testSQLI(IN id VARCHAR(50))
BEGIN
	-- Dynamic SQL
	SET @sql = CONCAT('SELECT * FROM user WHERE uid=\'', id, '\''); 
	PREPARE s1 FROM @sql;
	EXECUTE s1;
	DEALLOCATE PREPARE s1;
END $$ 
DELIMITER ;
```
이전의 예시와 결과는 같지만 위 프로시저는 `@sql`변수에 쿼리를 담은 후 실행하는 Dynamic SQL을 사용한다. 이를 프로시저 안에 사용하면 SQL Injection의 근본적 방어기법인 Prepared Statement를 사용하지 않아 결국 SQL Injection 공격에 취약해진다.

```

mysql> call testSQLI('evil\' or 1#');
+-----+-------+---------------+
| seq | uid   | upw           |
+-----+-------+---------------+
|   1 | guest | guest         |
|   2 | admin | ADM1NP4SSW0RD |
+-----+-------+---------------+

mysql> call testSQLI('evil\' or 1 UNION SELECT 1,2,3#');
+-----+-------+---------------+
| seq | uid   | upw           |
+-----+-------+---------------+
|   1 | guest | guest         |
|   2 | admin | ADM1NP4SSW0RD |
|   1 | 2     | 3             |
+-----+-------+---------------+
```

위의 출력 결과를 보면 알 수 있듯이 SQL Injection공격이 실행되는 것을 알 수 있다.

***

### **0x4. Stored Procedure와 보안**

개발 과정에서 **Stored Procedure**는 보안을 위해 좋은 선택이 될 수 있다.

#### **DB유저 권한과 프로시저의 권한 분리**

```
mysql> SELECT * FROM employees;
ERROR 1142 (42000): SELECT command denied to user 'aestera'@'localhost' for table 'employees'

mysql> SHOW GRANTS FOR 'aestera'@'localhost';
+----------------------------------------------------------------------+
| Grants for aestera@localhost                                         |
+----------------------------------------------------------------------+
| GRANT USAGE ON *.* TO `aestera`@`localhost`                          |
| GRANT EXECUTE ON PROCEDURE `testDB`.`gensp` TO `aestera`@`localhost` |
+----------------------------------------------------------------------+
```

계정 `aestera`는 genSP프로시저의 실행 권한은 있지만 employees 테이블의 접근 권한이 없는 것을 볼 수 있다. 하지만 `aestera`는 genSP프로시저를 통해 employees테이블에 제한적으로 접근할 수 있다.

```
mysql> SET @var = 10;

mysql> CALL genSP('Bravo', @cnt, @var);
+----+-------+----------+------+
| id | name  | location | age  |
+----+-------+----------+------+
|  2 | Bravo | Brazil   |   28 |
+----+-------+----------+------+

mysql> SELECT @cnt, @var;
+------+------+
| @cnt | @var |
+------+------+
|    6 |   20 |
+------+------+
```

이러한 권한의 분리가 보안적으로는 이점이 된다. 예를 들어 아래 조건을 가진 DB가 있다고 가정하자.

- 유저 `admin`은  `test`테이블과 중요한 정보가 담긴 `secret`테이블의 권한을 가지고 있다. 
- `test`테이블을 사용하는 로직이 SQL Injection에 취약하다.

이때 `test`테이블에서 SQL Injection이 발생했다면, 중요 정보가 담긴 `secret`테이블도  같이 취약해진다. 그러나 아래처럼 권한을 분리했을때를 생각해보자.

- `secret`테이블을 다루는 로직을 **Stored Procedure**로 처리
- `admin`의 `secret`테이블에 대한 권한을 제거

이렇게 되면 `test`테이블에서 SQL Injection이 발생해도 `secret`테이블이 안전함과 동시에 `admin`은 제한적이지만 여전히 `secret`테이블을 사용할 수 있다.


---

### **0x5. Reference**
```
- https://www.w3schools.com/sql/sql_stored_procedures.asp
- https://www.c-sharpcorner.com/UploadFile/996353/difference-between-stored-procedure-and-user-defined-functio/
- https://www.shekhali.com/difference-between-stored-procedure-and-function-in-sql-server/
- https://dev.mysql.com/doc/refman/8.0/en/faqs-stored-procs.html#faq-mysql-where-procedures-functions-docs
- https://dev.mysql.com/doc/refman/8.0/en/information-schema-routines-table.html
- https://security.stackexchange.com/questions/68701/how-does-stored-procedure-prevents-sql-injection
- https://www.slideshare.net/topcredu/11-sql
- https://nive.tistory.com/148
```