# SQL

[TOC]

# SQLite 语法
SQLite 是遵循一套独特的称为语法的规则和准则。
**大小写敏感性**：有SQLite 不区分大小写的，但也有一些命令是大小写敏感的，比如 GLOB 和 glob 在 SQLite 的语句中有不同的含义。
**注释**： SQLite 注释是附加的注释，可以在 SQLite 代码中添加注释以增加其可读性，他们可以出现在任何空白处，包括在表达式内和其他 SQL 语句的中间，但它们不能嵌套。

* 以两个连续的 `-` 字符（ASCII 0x2d）开始，并扩展至下一个换行符（ASCII 0x0a）或直到输入结束，以先到者为准。
* 也可以使用 C 风格的注释，以 `/*注释*/` 字符对或直到输入结束，以先到者为准。SQLite的注释可以跨越多行。

```
sqlite>.help -- 这是一个简单的注释
```

# SQLite 语句

所有的 SQLite 语句可以以任何关键字开始，如 SELECT, INSERT, UPDATE, DELETE, ALTER, DROP 等，所有的语句以分号 `;` 结束。


## ANALYZE 语句：(SQLite)

```sql
SQL> ANALYZE;
SQL> ANALYZE database_name;
SQL> ANALYZE database_name.table_name;
```

## AND/OR 子句：(SQLite, SQL)

```sql
SQL> SELECT column1, column2....columnN
     FROM table_name
     WHERE CONDITION-1 AND/OR CONDITION-2;
```

## Alter Table 语句：(SQLite)

```sql
SQL> ALTER TABLE table_name ADD COLUMN column_def;     /加新列
SQL> ALTER TABLE table_name RENAME TO new_table_name;  /改表名
SQL> ALTER TABLE table_name RENAME column 旧的字段名 to 新的字段名;

```



## ATTACH DATABASE 语句：(SQLite)
```sql
SQL> ATTACH DATABASE 'DatabaseName' As 'Alias-Name';
```

## BEGIN TRANSACTION 语句：(SQLite)

```sql
SQL> BEGIN;
SQL> BEGIN EXCLUSIVE TRANSACTION;
```

## BETWEEN 子句：(SQLite)
```sql
SQL> SELECT column1, column2....columnN
     FROM   table_name
     WHERE  column_name BETWEEN val-1 AND val-2;
```

## Column 修改length size：(Oracle)
```sql
SQL> column/col column_name format a30
SQL> set linesize 300   /这个好，自动调整
```

*a30 - alphanumeric30*


## COMMIT 语句：(SQLite)
```sql
SQL> COMMIT;
```

## CREATE INDEX 语句：(SQLite)
```sql
SQL> CREATE INDEX index_name
     ON table_name ( column_name COLLATE NOCASE );
```

### CREATE UNIQUE INDEX 语句：(SQLite)
```sql
CREATE UNIQUE INDEX index_name
ON table_name ( column1, column2,...columnN);
```

### CREATE TABLE 语句 创建表(SQLite)
```sql
SQL> CREATE TABLE table_name(
     column1 datatype  PRIMARY KEY(one or more columns),
     column2 datatype,
     column3 datatype,
     .....
     columnN datatype  PRIMARY KEY( one or more columns )
     );
```

`CREATE TABLE` 语句:

* 用于在任何给定的数据库创建一个新表。
* 创建基本表，涉及到命名表、定义列及每一列的数据类型。
* CREATE TABLE 是告诉数据库系统创建一个新表的关键字。
* CREATE TABLE 语句后跟着表的唯一的名称或标识。
* 您也可以选择指定带有 `table_name` 的 `database_name`。


```sql
sqlite> CREATE TABLE COMPANY(
        ID INT PRIMARY KEY     NOT NULL,
        NAME           TEXT    NOT NULL,
        AGE            INT     NOT NULL,
        ADDRESS        CHAR(50),
        SALARY         REAL
        );
//创建一个 COMPANY 表, ID 作为主键，
//NOT NULL 的约束表示在表中创建纪录时这些字段不能为 NULL

sqlite> CREATE TABLE DEPARTMENT(
        ID INT PRIMARY KEY      NOT NULL,
        DEPT           CHAR(50) NOT NULL,
        EMP_ID         INT      NOT NULL
        );
//再创建一个表，我们将在随后章节的练习中使用


sqlite>.tables
COMPANY     DEPARTMENT
//这里可以看到我们刚创建的两张表 COMPANY、 DEPARTMENT。


sqlite>.schema COMPANY
        CREATE TABLE COMPANY(
        ID INT PRIMARY KEY     NOT NULL,
        NAME           TEXT    NOT NULL,
        AGE            INT     NOT NULL,
        ADDRESS        CHAR(50),
        SALARY         REAL
        );
//使用 SQLite .schema 命令得到表的完整信息
```



### CREATE TRIGGER 语句：(SQLite)
```sqlite3
sqlite> CREATE TRIGGER database_name.trigger_name
        BEFORE INSERT ON table_name FOR EACH ROW
        BEGIN
        stmt1;
        stmt2;
        ...
        END;
```


### CREATE VIEW 语句：(SQLite)
```sqlite3
sqlite> CREATE VIEW database_name.view_name  AS
        SELECT statement....;
```


### CREATE VIRTUAL TABLE 语句：(SQLite)
```sqlite3
sqlite> CREATE VIRTUAL TABLE database_name.table_name USING weblog( access.log );

sqlite> CREATE VIRTUAL TABLE database_name.table_name USING fts3( );
```



## COMMIT TRANSACTION 语句：(SQLite)
```sql
SQL> COMMIT;
```



## COUNT 子句：(SQLite)
```sqlite3
sqlite> SELECT COUNT(column_name)
        FROM   table_name
        WHERE  CONDITION;
```



## DELETE 语句：(SQLite)
```sqlite3
sqlite> DELETE FROM table_name
        WHERE  {CONDITION};
```


## DETACH DATABASE 语句：(SQLite)
```sqlite3
sqlite> DETACH DATABASE 'Alias-Name';
```


## DISTINCT 子句：(SQLite)
```sqlite3
sqlite> SELECT DISTINCT column1, column2....columnN
        FROM   table_name;
```





## DROP INDEX 语句：(SQLite)
```sqlite3
sqlite> DROP INDEX database_name.index_name;
```


### DROP TABLE 语句：(SQLite) 删除表
`DROP TABLE` 语句:
删除表定义及其所有相关数据、索引、触发器、约束和该表的权限规范。
一旦一个表被删除，表中所有信息也将永远丢失。

```sqlite3
sqlite> DROP TABLE database_name.table_name;
```


### DROP VIEW 语句：(SQLite)
```sqlite3
sqlite> DROP VIEW view_name;
```

### DROP TRIGGER 语句：(SQLite)
```sqlite3
sqlite> DROP TRIGGER trigger_name
```


## Group By Having 分组

```sql
SELECT B.bookid, B.title, SUM(quantity)
FROM book B, orderitem O
WHERE B.bookid=O.bookid
GROUP BY B.bookid, B.title     /结果若有他，一定要每个指明合并
HAVING SUM(quantity)>=6;


SELECT C.custid, C.name
FROM customer C, orders O, orderitem I, book B
WHERE C.custid=O.custid AND O.orderid=I.orderid AND I.bookid=B.bookid                /多个表用AND
AND B.ISBN= '0087621658';


```
/现在select列表中的字段，如果没有在组函数中，那么必须出现在group by 子句中。（select中的字段不可以单独出现，必须出现在group语句中或者在组函数中。）


## Update 修改表内容（oracle）
```sql
update 表名 set 字段名1='修改后的值', 字段名2='修改后的值'  where id=1
```



## EXISTS 子句：(SQLite)
```sqlite3
sqlite> SELECT column1, column2....columnN
        FROM   table_name
        WHERE  column_name EXISTS (SELECT * FROM   table_name );
```

## EXPLAIN 语句：(SQLite)
```sqlite3
sqlite> EXPLAIN INSERT statement...;
sqlite> EXPLAIN QUERY PLAN SELECT statement...;
```

## INSERT INTO 语句：(SQLite, SQL)
`INSERT INTO` 语句:
用于向数据库的某个表中添加新的数据行。
如果要为表中的所有列添加值，可以不需要在 SQLite 查询中指定列名称。但要确保值的顺序与列在表中的顺序一致。

```sqlite3
sqlite> INSERT INTO TABLE_NAME [(column1, column2, column3,...columnN)]
sqlite> INSERT INTO TABLE_NAME VALUES (value1, value2, value3,...valueN);
/在这里，column1, column2,...columnN 是要插入数据的表中的列的名称。
```

```sqlite3
sqlite> CREATE TABLE COMPANY(
        ID   INT PRIMARY KEY     NOT NULL,
        NAME           TEXT    NOT NULL,
        AGE            INT     NOT NULL,
        ADDRESS        CHAR(50),
        SALARY         REAL
        );

sqlite> INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY)
VALUES (1, 'Paul', 32, 'California', 20000.00 );

or

sqlite> INSERT INTO COMPANY VALUES (7, 'James', 24, 'Houston', 10000.00 );
```

```
//使用一个表来填充另一个表
//可以通过在一个有一组字段的表上使用 select 语句，填充数据到另一个表中。

INSERT INTO first_table_name [(column1, column2, ... columnN)]
   SELECT column1, column2, ...columnN
   FROM second_table_name
   [WHERE condition];
```



## PRAGMA 语句：(SQLite)
**PRAGMA pragma_name;**
For example:

```sqlite3
sqlite> PRAGMA page_size;
sqlite> PRAGMA cache_size = 1024;
sqlite> PRAGMA table_info(table_name);
```




## RELEASE SAVEPOINT 语句：(SQLite)
```sqlite3
sqlite> RELEASE savepoint_name;
```



## REINDEX 语句：(SQLite)
```sqlite3
sqlite> REINDEX collation_name;
sqlite> REINDEX database_name.index_name;
sqlite> REINDEX database_name.table_name;
```




## ROLLBACK 语句：(SQLite)
```sqlite3
sqlite> ROLLBACK;
sqlite> ROLLBACK TO SAVEPOINT savepoint_name;
```

## SAVEPOINT 语句：(SQLite)
```sqlite3
sqlite> SAVEPOINT savepoint_name;
```

## SELECT 语句：(SQLite, SQL)
`SELECT` 语句:
用于从 SQLite 数据库表中获取数据，以结果表的形式返回数据。
这些结果表也被称为结果集。

1. choose info
```sql
sqlite>.header on
sqlite>.mode column
/前三个命令被用来设置正确格式化的输出。

SQL> SELECT column1, column2... FROM table_name;

SQL> SELECT * FROM table_name; /获取所有可用的字段


SQL> SELECT
     MM.DEPT_ID
     FROM MES_MACHINE MM, MT_OVERHAUL_RECORD MR
     WHERE
     MR.MACHINE_ID = MM.MACHINE_ID；

```

2. choose specific info
```sql
SQL> SELECT column1,2 FROM table_name; /只获取指定的字段
```

## .width num, num.... 设置输出列的宽度 (SQLite)
使用 .width num, num.... 命令设置显示列的宽度，如下所示：

```sqlite3
sqlite>.width 10, 20, 10
/第一列的宽度为 10，第二列的宽度为 20，第三列的宽度为 10

sqlite> SELECT * FROM COMPANY;

ID          NAME                  AGE         ADDRESS     SALARY
----------  --------------------  ----------  ----------  ----------
1           Paul                  32          California  20000.0
2           Allen                 25          Texas       15000.0
3           Teddy                 23          Norway      20000.0
4           Mark                  25          Rich-Mond   65000.0
5           David                 27          Texas       85000.0
6           Kim                   22          South-Hall  45000.0
7           James                 24          Houston     10000.0
```

## Schema 信息(SQLite)
因为所有的点命令只在 SQLite 提示符中可用，所以当您进行带有 SQLite 的编程时，您要使用下面的带有 sqlite_master 表的 SELECT 语句来列出所有在数据库中创建的表：

sqlite> `SELECT` **tbl_name** `FROM` **sqlite_master** `WHERE type =` **'table'**;


```sqlite3
//使用下面的带有 sqlite_master 表的 SELECT 语句来列出所有在数据库中创建的表：
sqlite> SELECT tbl_name FROM sqlite_master WHERE type = 'table';

//假设在 testDB.db 中已经存在唯一的 COMPANY 表
tbl_name
----------
COMPANY


sqlite> SELECT sql FROM sqlite_master
        WHERE type = 'table' AND tbl_name = 'COMPANY';
/列出关于 COMPANY 表的完整信息

//假设在 testDB.db 中已经存在唯一的 COMPANY 表，则将产生以下结果：
CREATE TABLE COMPANY(
   ID INT PRIMARY KEY     NOT NULL,
   NAME           TEXT    NOT NULL,
   AGE            INT     NOT NULL,
   ADDRESS        CHAR(50),
   SALARY         REAL
)
```

#### SQL UNION 和 UNION ALL 操作符(SQLite)
