# SQL

[toc]

## SQL

SQL statements usually are divided into two categories:

* Data Definition Language (DDL)
    * define relation/table structures including the schema for each relation, the domain of values associated with each attribute, and integrity constraints.
    * Example, CREATE DATABASE, ALTER DATABASE, DROP DATABASE, CREATE TABLE, ALTER TABLE, DROP TABLE, TRUNCATE TABLE, and so on.
    * DDL statements do `COMMIT` automatically
* Data Manipulation Language (DML)
    * used to *retrieve, insert, update, and delete* data in database.
    * Example, SELECT, INSERT, UPDATE, DELETE, and MERGE
    * DML may not do a COMMIT automatically in some RDBMS, like Oracle.
    * have to explicitly issue the `COMMIT` statement



## database security
https://www.aseatw.com/html/Present.aspx?id=DatabaseFundamentals&num=26

the first line of database for a database:

* **change the default user password** immediately
* **lock** unused user account.
* **enforce** stronger passwords.
* **remove** public accounts, or all access from all accounts.
* **choose** *domain authentication* or *database authentication* for your database users, and stick with it.
* **Examine** roles and groups closely.
* **Protect** administrative functions from users.
* **divide** database admin duties.

# mysql
table - database - server
## for database
### choose database:

mysql> `show databases`;
+--------------------+
| Database           |
+--------------------+
| information_schema |
| first              |
| mysql              |
| performance_schema |
| sys                |
+--------------------+
5 rows in set (0.02 sec)

mysql> `use` test1;
database changed

mysql> `show tables`;
+-----------------+
| Tables_in_first |
+-----------------+
| parrots         |
| student         |
+-----------------+
2 rows in set (0.00 sec)

### create database
mysql> `create database` test0 `charset utf8`;
Query OK, 1 row affected (0.01 sec)

mysql> `show tables`;
Empty set (0.00 sec)

### drop database
mysql> `drop database` test0;
Query OK, 0 rows affected (0.03 sec)

### change name?
mysql数据库不能改名
只能呢该表，列名字

## for table
### create table
建表其实就是声明表头列的过程

mysql> `create table` stu(
id `int`,
name `varchar(10)`)
`engine myisam charset utf8`;
Query OK, 0 rows affected (0.04 sec)

mysql> `show tables`;
+-----------------+
| Tables_in_test1 |
+-----------------+
| stu             |
+-----------------+
1 row in set (0.00 sec)

#### 三大列类型

* 数值类型
    * 整型: 字节越多，存的范围越大。
        * int默认是有符号的。
            * *unsigned*：无符号, 无正负[0,255]
            * *M*: 必须跟zerofill才有意义，单独使用无意义。表示补0的宽度
            * *zerofill*: 补0,默认unsigned。
        * **tinyint**：1字节，正负[-128,+127] 或 无正负[0,255]
            * 1 byte = 8 bits, 00000000 - 111111111
            * 计算机为了表示正负数，最高位最左侧的0/1当成符号。
            * 用补码规则
            * 0 0000000 = 0
            * 0 1111111 = 127
            * 1 0000000 = -1
            * 1 1111111 = 128
            * [-2^7,2^7-1]
        * **smallint**：2字节,16bits，3万
            * [-2^15,2^15-1]
        * **mediumint**：3字节，800+万
        * **int**：4字节，40+亿
        * **bigint**：8字节
            * XX `int` not null default 0;
            * XX `int` `(5)` `zerofill` not null default 0;

    * 浮点数:
        * **float (M,D)**:
        * **decimal (M,D)**:定点
        * M:精度，总位数, D: 标度，小数点后面
        * 正负的9999.99
            * `XX decimal(6,2)` 总共6位数，小数点后1位，正负都可以。
        * float 能存10^38，10^-38.
        * M<=24 4bytes, xor 8 bytes
        * 定点是把整数部分和小数部分分开存的，比float精确。float取出时有可能不一样！！像账户银行敏感的，建议用decimal。

mysql> `insert into` account values
    -> (1, 1234567.23,1234567.23);
Query OK, 1 row affected (0.00 sec)

mysql> `select` * `from` account;
+----+------------+------------+
| id | acc1       | acc2       |
+----+------------+------------+
|  1 | 1234567.25 | 1234567.23 |
+----+------------+------------+
2 rows in set (0.00 sec)

* 字符串型
    * *M 限制的是字符数不是字节，6个utf8或其他任何都是6个*。
    * **char(M)** 定长字符串 M,[0,255]
        * 存储定长，容易计算文件指针的移动量，*速度更快*
        * 不论够不够长，实际都占据N个长度
        * char(N),如果不够N个长度，用空格在末尾补齐长度
        * 取出时再把右侧空格去掉（*字符串本身右侧有空格将会丢失*）
        * 宽度M，可存字符M，实存字符i(i<=M),
        * 实占空间：M
        * 定长的利用率：M<=可能达到100%
        * 会有浪费
    * **varchar(M)** 变长字符串 M,[0,65535]
        * 不用空格补气，但是数据前面有1或2个字节来记录开头
        * 实占空间：i+(1或2个字节)
        * 变长的利用率：i+(1或2个字节)<100%, 不可能100%
        * 和text差不多，但是比他慢一点
    * **text**：
        * 不用加默认值，存较大的文本段，搜索速度慢。
        * 一万以内可以用varchar
    * **mediumblob**
    * **mediumtext**：一千多万
    * **longblob**
    * **longtext**
    * **blob**:
        * 是二进制类型，用来储存图像音频等二进制信息，0-255都有可能出现。
        * 意义在于防止因为字符集的问题，导致信息丢失
        * 比如一张图片中有0xFF字节，这个在ascii字符集中人文非法，在入库是被过滤了。如果是二进制，就是原原本本存进去，拿出来，隐形防范字符集的问题导致数据流失

```
//char varchar 区别
mysql> create table test(
    -> char(6) not null default'',
    -> varchar(6) not null default'')
    -> engine myisam charset utf8;

mysql> insert into test2 values ('aa ','aa ');
mysql> select concat(ca,'!'),concat(vca,'!') from test2;
+----------------+-----------------+
| concat(ca,'!') | concat(vca,'!') |
+----------------+-----------------+
| hello!         | hello!          |
| aa!            | aa !            |
+----------------+-----------------+
2 rows in set (0.01 sec)
```

```
//text 不需要默认值
mysql> create table test3(
    -> article **text** not null default''
    -> )engine myisam charset utf8;
ERROR 1101 (42000): BLOB, TEXT, GEOMETRY or JSON column 'article' can't have a default value

mysql> create table test3(
    -> article text);
Query OK, 0 rows affected (0.05 sec)

mysql> alter table test3 add img blob;
Query OK, 0 rows affected (0.04 sec)
Records: 0  Duplicates: 0  Warnings: 0
```

```
//blob
mysql> desc test3;
+--------+------+------+-----+---------+-------+
| Field  | Type | Null | Key | Default | Extra |
+--------+------+------+-----+---------+-------+
| article | text | YES  |     | NULL    |       |
| img    | blob | YES  |     | NULL    |       |
+--------+------+------+-----+---------+-------+

mysql> insert into test3
    -> values('qingqiongmaima','zhangfeiganlu');

mysql> select * from test3;
+----------------+---------------+
| article         | img           |
+----------------+---------------+
| qingqiongmaima | zhangfeiganlu |
+----------------+---------------+

```

* 时间类型
    * 比起用char来使用各省时间空间。
    * **date**：3个字节
        * 1934-04-12
        * 范围：1000-01-01到9999-12-31
    * **datetime**:  8个字节
        * YYYY-mm-dd HH:ii:ss
    * **time**: 3个字节
        * 20:20:20
    * **timestamp**：4个字节
        * 可以取当前的时间
    * **year**: 1个字节
        * [0000, 1901,2155]
        * 可以简化成两位数 year(2)

```
mysql> create table test4(
    -> sname varchar(20) not null default'',
    -> logintime datetime not null,
    -> ts timestamp default current_timestamp
    -> )engine myisam charset utf8;
```

`primery key`
`auto_increment`
`not null`
`default '' `
`engine myisam/innodb/bdb charset utf8/gbk/latin1...`

```
create table test5(
id int unsigned primary key not null default,
username char(10) not null default 'admimn',
gender char(1) not null,
weight tinyint unsigned not null,
birth date not null,
salary decimal(8,2) unsigned not null,
lastlogin datetime not null,
intro char(1500)not null

//除username和intro之外都是定长
//都是定长的话 搜索会快很多
//*优化：就是空间换时间*
//username varchar(10) 可以有优化 char(10)
//intro varchar(1500) 变 char(1500)就浪费太多了
//*优化：把常用到的信息，优先考虑效率，把不常用比较占空间的信息，放到附表*
//把intro单独拿出来，改变次数也很少

create table intro(
id int unsigned primary key not null default,
username char(10) not null default 'admimn',
lastlogin datetime not null,
intro char(1500)not null

create table member(
id int unsigned auto_increment primary key,
username char(20) not null default '',
gender char(1) not null default '',
weight tinyint unsigned not null default 0,
birth date not null,
salary decimal(8,2) not null default 0.00,
lastlogin int unsigned not null default 0)
engine myisam charset utf8;
```

### 删除表 `drop table table_A`
mysql> `drop table` stu;
//表就不在了

### 改名 `rename table table_A to table_B`
mysql> `rename table` stu `to` newstu;


### 修改表
#### 添加列 `alter table table_A add Z (after/first) X`
//加在最后
mysql> `alter table` class1 `add` score2 tinyint unsigned not null default 0;

//加在指定位置
mysql> `alter table` class1 `add` score1 tinyint unsigned not null default 0 `after` id;

//加在第一位
mysql> `alter table` class1 `add` score1 tinyint unsigned not null default 0 `first`;

#### 删除列 `alter table table_A drop X`
mysql> `alter table` class1 `drop` score2；

#### 修改列参数 `alter table table_A modify X .../ Change X TO Y...) `
//不能改列名
mysql> `alter table` class1 `modify` score2 int unsigned not null default 100;

//可以修改列名
mysql> `alter table` class1 `change` score2 `to` score234 int unsigned not null default 100;

//如果列类型改变了，导致数据保存不下来
//一般会往大了该
//1. 丢数据
//2. 严格模式下，不能改


### 查找 `desc table_A`
mysql> `desc` table_name;
+---------+--------------+------+-----+---------+----------------+
| Field   | Type         | Null | Key | Default | Extra          |
+---------+--------------+------+-----+---------+----------------+
| id      | int(11)      | NO   | PRI | NULL    | auto_increment |
| sname   | varchar(10)  | NO   |     |         |                |
| gender  | varchar(1)   | NO   |     |         |                |
| company | varchar(20)  | NO   |     |         |                |
| salary  | decimal(6,2) | NO   |     | 0.00    |                |
| fanbu   | smallint(6)  | NO   |     | 0       |                |
+---------+--------------+------+-----+---------+----------------+
6 rows in set (0.00 sec)


### add date `insert into table_A (X,Y,Z) values (X,1), (Y,2), (Z,3)`
mysql> `insert into` newstu (X,Y,Z) `values`(
    -> (1,'a'),
    -> (2,'b'),
    -> (3,'c'));

### 修改data
mysql> `update` table_name
    -> `set` X = 100;
//X栏全部都改了

mysql> `update` * `from` table_name
    -> `set` X = X+2;
    -> `where` Y = 6;

### 删除data
删除就是整行
一个data属于修改
mysql> `delete from` stu `where` id=2;
mysql> `delete` * `from` stu `where` id=2;
//都是删除整行 不需要 *


### 清空表数据
mysql> `truncate` newstu;
Query OK, 0 rows affected (0.01 sec)
//删除表，扔了重写，（全删的情况下更快）

mysql> `delete` `from` newstu;
//delete把数据删除重写

### data 没并行
set name utf8;

`\c` 退出继续打
