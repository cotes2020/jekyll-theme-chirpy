# SQL

[TOC]

---

# give specific output

---

## GLOB 子句：
```sql
SQL> SELECT column1, column2....columnN
     FROM   table_name
     WHERE  column_name GLOB { PATTERN };

SQL> SELECT *
     FROM   table_name
     WHERE  column_name GLOB { PATTERN };

     SELECT DISTINCT column_name,column_name
     FROM table_name;
```

```sql
// example
SELECT DISTINCT country FROM Websites;

// SELECT studentNO FROM student WHERE 0 // false, return null
// SELECT studentNO FROM student WHERE 1 // true, reture all
// SELECT * FROM Websites WHERE country='CN';
// SELECT * FROM Websites WHERE id=1;
// Select * from emp where comm is null;
// Select * from emp where sal > 2000 and sal < 3000;
// Select * from emp where sal > 2000 or comm > 500;
SELECT * FROM Websites
WHERE alexa > 15
AND (country='CN' OR country='USA');
// select * from emp where not sal > 1500;
// Select * from emp where sal between 1500 and 3000;
// SELECT * FROM emp where sal NOT BETWEEN 1 AND 20;
// Select * from emp where sal in (5000,3000,1500);
// SELECT * FROM Websites WHERE (alexa BETWEEN 1 AND 20) AND country NOT IN ('USA', 'IND');
// SELECT * FROM Websites WHERE name BETWEEN 'A' AND 'H';
// SELECT * FROM Websites WHERE name NOT BETWEEN 'A' AND 'H';
// Select * from emp where ename like 'M%';
```

---

## ORDER BY

```sql
SQL> SELECT column_name,column_name
     FROM table_name
     ORDER BY column_name,column_name ASC|DESC 降序;
```

```sql
// example:
SELECT * FROM Websites
ORDER BY alexa;

SELECT * FROM Websites
ORDER BY country, alexa;

order by A,B        // 都是默认按升序排列
order by A desc,B   // A 降序，B 升序排列
order by A ,B desc  // A 升序，B 降序排列
```

---

## GROUP BY 子句：
```sql
SQL> SELECT SUM(column_name)
     FROM   table_name
     WHERE  CONDITION
     GROUP BY column_name;
```

* AVG(): finds the average value of *numeric attribute*
* MIN(): finds the minimum value of *string/numeric attribute*
* MAX(): finds the maximum value of *string/numeric attribute*
* SUM(): finds the sum total of a *numeric attribute*
* COUNT(): counts the number of rows in a set.

---

## HAVING 子句：
```sql
SQL> SELECT SUM(column_name)
     FROM   table_name
     WHERE  CONDITION
     GROUP BY column_name
     HAVING (arithematic function condition);
```
The `HAVING` clause can do the same thing as `WHERE` clause

* SELECT FID, Name FROM Faculty
* HAVING Rank = 'Professor';

* SELECT FID, Name FROM Faculty
* WHERE Rank = 'Professor';
/generate the same output,
/but the WHERE clause provides a better performance

`HAVING` clause ually used with GROUP BY, can include aggregate functions (previous page)

---

## SELECT TOP, LIMIT, ROWNUM 子句
// 用于规定要返回的记录的数目。
// 对于拥有数千条记录的大型表来说，非常有用。

注意:并非所有的数据库系统都支持 SELECT TOP 语句。
- MySQL 支持 LIMIT 语句来选取指定的条数数据，
- Oracle 可以使用 ROWNUM 来选取。

```sql
SQL Server / MS Access 语法
SELECT TOP number|percent column_name(s)
FROM table_name;

MySQL 语法
SELECT column_name(s)
FROM table_name
LIMIT number;

Oracle 语法
SELECT column_name(s)
FROM table_name
WHERE ROWNUM <= number;
```

```sql
// example
SELECT * FROM Persons LIMIT 5;
SELECT * FROM Persons WHERE ROWNUM <=5;
SELECT TOP 50 PERCENT * FROM Websites;
SELECT TOP 5 * from table
SELECT TOP 5 * from table order by id desc
// desc 表示降序排列 asc 表示升序
```

---

## Like 子句：
```sql
SQL> SELECT column1, column2....columnN
     FROM table_name
     WHERE column_name LIKE PATTERN%;
```

```sql
//example
SELECT * FROM Websites WHERE name LIKE 'G%';
```

*case sensetive*

* `LIKE` 'Toyota`%`';      /*start with Toyota*
* `LIKE` '`%`0';           /*end with 0*
* `LIKE` '`%`RX4`%`'       /*contain RX$*
* `NOT LIKE` '`%`RX4`%`'   /*do NOT match the pattern*
* `LIKE` '[CK]ars[eo]n' 将搜索下列字符串：Carsen、Karsen、Carson 和 Karson（如 Carson）。
* `LIKE` '[M-Z]inger' 将搜索以字符串 inger 结尾、以从 M 到 Z 的任何单个字母开头的所有名称（如 Ringer）。
* `LIKE` 'M[^c]%' 将搜索以字母 M 开头，并且第二个字母不是 c 的所有名称（如MacFeather）。

### SQL 通配符

通配符	| 描述
---|---
%	| 替代 0 个或多个字符
_	| 替代一个字符
[charlist]	| 字符列中的任何单一字符
^[charlist] | start with
[^charlist] [!charlist]	| no exist


```sql
SELECT * FROM Websites WHERE name LIKE 'G_o_le';

// MySQL 用 REGEXP / NOT REGEXP (RLIKE / NOT RLIKE) 来操作正则表达式。
SELECT * FROM Websites WHERE name REGEXP '^[GFs]';
SELECT * FROM Websites WHERE name REGEXP '^[A-H]'; // 以 A-H 字母开头
SELECT * FROM Websites WHERE name REGEXP '^[^A-H]'; //  不以 A-H 字母开头


```

---

# change the table data

## INSERT INTO 语句

```SQL
SQL> INSERT INTO table_name   // 没有指定插入数据的列名的形式需要列出插入行的每一列数据
     VALUES (value1,value2,value3,...);

     INSERT INTO table_name (column1,column2,column3,...)
     VALUES (value1,value2,value3,...);
```

```sql
//Examples
+----+--------------+---------------------------+-------+---------+
| id | name         | url                       | alexa | country |
+----+--------------+---------------------------+-------+---------+
| 1  | Google       | https://www.google.cm/    | 1     | USA     |

INSERT INTO Websites (name, url, alexa, country)
VALUES ('百度','https://www.baidu.com/','4','CN');

INSERT INTO Websites (name, url, country)   // other will be 0
VALUES ('stackoverflow', 'http://stackoverflow.com/', 'IND');

```

---

## UPDATE 语法

```sql
SQL> UPDATE table_name
     SET column1=value1,column2=value2,...
     WHERE some_column=some_value;

// 如果省略 WHERE 子句，所有的记录都将被更新！
```


```sql
//Examples
+----+--------------+---------------------------+-------+---------+
| id | name         | url                       | alexa | country |
+----+--------------+---------------------------+-------+---------+
| 1  | Google       | https://www.google.cm/    | 1     | USA     |

UPDATE Websites
SET alexa='5000', country='USA'
WHERE name='菜鸟教程';

```

---

##  DELETE 语法
```sql
SQL> DELETE FROM table_name
     WHERE some_column=some_value;

//如果省略了 WHERE 子句，所有的记录都将被删除！


// 删除所有数据
// 在不删除表的情况下，删除表中所有的行。
// 表结构、属性、索引将保持不变：

SQL> DELETE FROM table_name;
SQL> DELETE * FROM table_name;
```


```sql
//Examples
+----+--------------+---------------------------+-------+---------+
| id | name         | url                       | alexa | country |
+----+--------------+---------------------------+-------+---------+
| 1  | Google       | https://www.google.cm/    | 1     | USA     |

DELETE FROM Websites
WHERE name='Facebook' AND country='USA';

```

---

# return boolean


## IN 子句：
```sql
SQL> SELECT column1, column2....columnN
     FROM table_name
     WHERE column_name IN (val-1, val-2,...val-N);
```

```sql
SELECT column_name(s) FROM table_name
WHERE column_name IN (value1,value2,...);
=
SELECT column_name(s) FROM table_name
WHERE column_name=value1;

```

---

## NOT IN 子句：

![sql-join](https://i.imgur.com/ngKCsn7.png)

```sql
SQL> SELECT column1, column2....columnN
     FROM   table_name
     WHERE  column_name NOT IN (val-1, val-2,...val-N);
```

---

## JOIN 子句
combine rows from tables based on common field.

```sql

1. both side has the same value

SQL> SELECT columnA1, columnA2, columnB1, columnB2...
     FROM TableA
     JOIN TableB (= INNER JOIN TableB)
     ON tableA.column_name=tableB.column_name;


2. all data in table1, match data/Null in table 2

SQL> SELECT column_name(s)
     FROM table1
     LEFT JOIN table2 (= LEFT OUTER JOIN table2)
     ON table1.column_name=table2.column_name;


3. all data in table2, match data/Null in table 1

SQL> SELECT column_name(s)
     FROM table1
     RIGHT JOIN table2 (= RIGHT OUTER JOIN table2)
     ON table1.column_name=table2.column_name;

4. all data in tables, match each other with data/Null
SQL> SELECT column_name(s)
     FROM table1
     FULL JOIN table2 (= FULL OUTER JOIN table2)
     ON table1.column_name=table2.column_name;
```

1. `INNER JOIN`: 如果表中有至少一个匹配，则返回行
2. `LEFT JOIN`: Return all rows from the left table, and the matched rows from the right table. 即使右表中没有匹配，也从左表返回所有的行
3. `RIGHT JOIN`: Return all rows from the right table, and the matched rows from the left table.即使左表中没有匹配，也从右表返回所有的行
4. `FULL JOIN`: Return all rows when there is a match in ONE of the tables.只要其中一个表中存在匹配，则返回行. 结合了 LEFT JOIN 和 RIGHT JOIN 的结果。

---

## MINUS
返回存在于A表中，但不存在于B表中的数据。
```sql
SQL> SELECT  COL1,COL2
     FROM TABLE_A  [ WHERE conditions ]
     MINUS
     SELECT COL1 , COL2
     FROM TABLE_B [ WHERE conditions]。
```
Oracle 数据库支持 MINUS 用法，SQL Server, PostgreSQL, and SQLite 可以使用Except代替

---

## UNION 操作符
合并两个或多个 SELECT 语句的结果集。
- 每个 SELECT 语句必须拥有相同数量的列。
- 列也必须拥有相似的数据类型。
- 每个 SELECT 语句中的列的顺序必须相同。

```sql
SELECT column_name(s) FROM table1
UNION
SELECT column_name(s) FROM table2;
// 默认 UNION 操作符选取不同的值。


// 如果允许重复的值 UNION ALL
SELECT column_name(s) FROM table1
UNION ALL
SELECT column_name(s) FROM table2;
```

```sql
//Examples
SELECT country FROM Websites
UNION
SELECT country FROM apps
ORDER BY country;
```

---

# comparison

## NULL
* To check whether a value is NULL or not in MySQL,
* we can use `IS NULL` or `IS NOT NULL`
```
SELECT * FROM Section
WHERE Room IS NULL;
```


## Relational Algebra - Examples
* A X B X C, 要标注 where 条件 and key一一对应

1. example

    `Π`course.MCode, Course.Cno, Schedule, Room, Credit
(`σ`SID = "625018" (Enrollment `X` Section `X` Course))

```sql
SQL> SELECT C.MCode, C.CNo, Credit, Schedule, Room
     FROM Enrollment E, Section S, Course C
     WHERE E.SID='20000006'
     AND E.CallNo=S.CallNo AND S.MCode=C.MCode AND S.CNo=C.CNo;
```

2. SID

    SID --- `Π`SID(Student) - `Π`SID(Transcript)
`Π`Student.SID,Name (SID   Student))

```sql
SQL> SELECT SID, Name FROM Student
     WHERE SID IN ( )
     SELECT SID FROM Student
     MINUS SELECT SID FROM Transcript;
```

3. group

    SC -- `SID`G`sum(Credit)(Transcript `X` Course)`Π`SC.SID,Name `σ`sum(Credit) >= 6 (SC `X` Student))

```sql
SQL> SELECT S.SID, S.Name, SUM(Credit)
     FROM Student S, Transcript T, Course C
     WHERE S.SID=T.SID AND T.MCode=C.Mcode AND T.CNo= C.Cno
     GROUP BY S.SID
     HAVING SUM(Credit)>=6;
```


---

# table

## SQL 别名
为表名称或列名称指定别名。
- 在查询中涉及超过一个表
- 在查询中使用了函数
- 列名称很长或者可读性差
- 需要把两个列或者多个列结合在一起

```sql
// 列别名
SELECT column_name AS alias_name FROM table_name;
// 表别名
SELECT column_name(s) FROM table_name AS alias_name;
```


```sql
+----+--------------+---------------------------+-------+---------+
| id | name         | url                       | alexa | country |
+----+--------------+---------------------------+-------+---------+
| 1  | Google       | https://www.google.cm/    | 1     | USA     |

SELECT name AS n, country AS c FROM Websites;

+--------------+---------+
| n            | c       |
+--------------+---------+
| Google       | USA     |


SELECT name, CONCAT(url, ', ', alexa, ', ', country) AS site_info FROM Websites;


SELECT w.name, w.url, a.count, a.date
FROM Websites AS w, access_log AS a
WHERE a.site_id=w.id and w.name="菜鸟教程";
=
SELECT Websites.name, Websites.url, access_log.count, access_log.date
FROM Websites, access_log
WHERE Websites.id=access_log.site_id and Websites.name="菜鸟教程";


join more tables:

SELECT SID, C.MCode, C.Cno, C.Title
FROM Enrollment E, Section S, Course C
WHERE E.CallNo = S.CallNo AND S.Mcode = C.MCode AND S.CNo = C.CNo
ORDER BY SID

```


---

## oracle修改Table的主键的方法

```sql
第一步：增加列key_no
SQL> ALERT TABLE table_name ADD key_no int;

第二部：给key_no更新值
SQL> UPDATE table_name SET key_no =rownum;
     commit;

第三步：将key_no置为非空
SQL> ALERT TABLE table_name MODIFY key_no   int   not null;


第四步：查找主键
SQL> select    constraint_name from    user_constraints where constraint_type='P' and   owner=user    and    table_name='TB_ZHAOZHENLONG'

第五步：删除主键
      ALTER TABLE TB_ZHAOZHENLONG DROP CONSTRAINT PK_TB_ZHAOZHENLONG;

第五步：删除主键
      ALTER TABLE TB_ZHAOZHENLONG DROP CONSTRAINT PK_TB_ZHAOZHENLONG;

第六步：增加主键
      ALTER TABLE TB_ZHAOZHENLONG ADD (CONSTRAINT PK_TB_ZHAOZHENLONG PRIMARY KEY(c_1,c_2,c_3);
```


```
SQL> ALERT TABLE table_name ADD CONSTRAIN T1_C1 (PRIMARY KEY(column1, column2...));

SQL> ALERT TABLE table_name MODIFY( column1 PRIMARY KEY);
```
