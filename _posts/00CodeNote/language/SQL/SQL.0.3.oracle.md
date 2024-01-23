# SQL

[TOC]

# oracle

# oracle win

# 1. Oracle SQL Plus
* SQL Plus is a command-line program to submit SQL and PL/SQL statements to an Oracle database.
* You can submit statements interactively or as SQL`*`Plus scripts.
* SQL`*`Plus is installed with the database and is located in your ORACLE_HOME/bin directory.
* You can start SQL`*`Plus from the command line.

1. To start SQLPlus,
    * C:> `sqlplus /nolog`
2. To connect to the Oracle 12c RDBMS as a DBA,
    * SQL> `conn / as sysdba;`
3. To exit SQL*Plus,
    * SQL> `exit`


## explore some system information
1. To retrieve the current OS host name
    * SQL> `SELECT SYS_CONTEXT('USERENV','SERVER_HOST') FROM DUAL;`
        * SYS_CONTEXT('USERENV','SERVER_HOST')--------**desktop-53v65bb**
2. To retrieve the current OS user name
    * SQL> `SELECT SYS_CONTEXT('USERENV','OS_USER') FROM DUAL;`
        * SYS_CONTEXT('USERENV','OS_USER')------**DESKTOP-53V65BB\chris**
3. To retrieve the current Oracle user name
    * SQL> `SELECT USER FROM DUAL;`
        * USER------**SYS**
5. To retrieve the current Oracle instance name or SID
    * SQL> `SELECT SYS_CONTEXT('USERENV','INSTANCE_NAME') FROM DUAL;`
        * SYS_CONTEXT('USERENV','INSTANCE_NAME')---**orcl**
    * SQL> `SELECT instance FROM V$THREAD;`
        * instance---**orcl**
5. To retrieve the current Oracle database name
    * SQL> `SELECT SYS_CONTEXT('USERENV','DB_NAME') FROM DUAL;`
        * SYS_CONTEXT('USERENV','DB_NAME')---**orcl**
    * SQL> `SELECT NAME FROM V$DATABASE;`
        * NAME-------**orcl**
    * SQL> `SELECT * FROM GLOBAL_NAME;`
        * Global_name-------**orcl**
6. To retrieve the current Oracle version
    * SQL> `COL PRODUCT FORMAT A40`
    * SQL> `COL VERSION FORMAT A15`
    * SQL> `COL STATUS FORMAT A15`
    * SQL> `SELECT * FROM PRODUCT_COMPONENT_VERSION;`
        * PRODUCT        VERSION         STATUS
        * ---
        * 表格
        *


# 1.first step
## 1.1. connect to oracle

```sql
C:> sqlplus /nolog         /start SQL
SQL> CONN / as sysdba;    /connect to the Oracle 12c RDBMS as a DBA

SQL> CONN shawmoo;         /connec as user.

SQL> EXIT                  /exit SQL*Plus

.
```

## 1.2 list table info

**list the table’s structure**

```sql
SQL> DESC DBA_TABLESPACES;

 Name                                      Null?    Type
 ----------------------------------------- -------- ----------------------------
 TABLESPACE_NAME                           NOT NULL VARCHAR2(30)
 BLOCK_SIZE                                NOT NULL NUMBER
 INITIAL_EXTENT                                     NUMBER
 NEXT_EXTENT                                        NUMBER
 MIN_EXTENTS                               NOT NULL NUMBER
 MAX_EXTENTS                                        NUMBER
 MAX_SIZE                                           NUMBER
 PCT_INCREASE                                       NUMBER
 MIN_EXTLEN                                         NUMBER
 STATUS                                             VARCHAR2(9)
 CONTENTS                                           VARCHAR2(21)
 LOGGING                                            VARCHAR2(9)
 FORCE_LOGGING                                      VARCHAR2(3)
 EXTENT_MANAGEMENT                                  VARCHAR2(10)
 ALLOCATION_TYPE                                    VARCHAR2(9)
 PLUGGED_IN                                         VARCHAR2(3)
 SEGMENT_SPACE_MANAGEMENT                           VARCHAR2(6)
 DEF_TAB_COMPRESSION                                VARCHAR2(8)
 RETENTION                                          VARCHAR2(11)
 BIGFILE                                            VARCHAR2(3)
 PREDICATE_EVALUATION                               VARCHAR2(7)
 ENCRYPTED                                          VARCHAR2(3)
 COMPRESS_FOR                                       VARCHAR2(30)
 DEF_INMEMORY                                       VARCHAR2(8)
 DEF_INMEMORY_PRIORITY                              VARCHAR2(8)
 DEF_INMEMORY_DISTRIBUTE                            VARCHAR2(15)
 DEF_INMEMORY_COMPRESSION                           VARCHAR2(17)
 DEF_INMEMORY_DUPLICATE                             VARCHAR2(13)
 SHARED                                             VARCHAR2(13)
 DEF_INDEX_COMPRESSION                              VARCHAR2(8)
 INDEX_COMPRESS_FOR                                 VARCHAR2(13)
 DEF_CELLMEMORY                                     VARCHAR2(14)
 DEF_INMEMORY_SERVICE                               VARCHAR2(12)
 DEF_INMEMORY_SERVICE_NAME                          VARCHAR2(1000)
 LOST_WRITE_PROTECT                                 VARCHAR2(7)
 CHUNK_TABLESPACE                                   VARCHAR2(1)
```


**list all tablespaces and their status**
```sql
SQL> SELECT Tablespace_Name,Status FROM DBA_TABLESPACES;

TABLESPACE_NAME                STATUS
------------------------------ ---------
SYSTEM                         ONLINE
SYSAUX                         ONLINE
UNDOTBS1                       ONLINE
TEMP                           ONLINE
USERS                          ONLINE
SHAWMOO                        ONLINE

```

**list these tables’ structures**
```sql
SQL> DESC DBA_DATA_FILES;
SQL> DESC DBA_TEMP_FILES;
```

**check free spaces in Tablespaces**
```sql
SQL> DESC DBA_FREE_SPACE;
SQL> DESC DBA_TEMP_FREE_SPACE
```

**list all tablespaces and their data files**
```sql
SQL> SELECT File_ID,File_Name,Tablespace_Name,Bytes FROM DBA_DATA_FILES;
SQL> SELECT File_ID,File_Name,Tablespace_Name,Bytes FROM DBA_TEMP_FILES;
```

## 1.3. create new tablespace

```sql
SQL> CREATE TABLESPACE shawmoo DATAFILE 'c:\app\*chris*\virtual\oradata\orcl\shawmoo.dbf' SIZE 2G EXTENT MANAGEMENT LOCAL AUTOALLOCATE;

Tablespace created.
```

## 1.4. create a new user

```sql
SQL> CREATE USER chris IDENTIFIED by wang DEFAULT TABLESPACE shawmoo TEMPORARY TABLESPACE TEMP;

SQL> CREATE USER shawmoo IDENTIFIED by wang DEFAULT TABLESPACE shawmoo TEMPORARY TABLESPACE TEMP;
```

* If you do not specify default tablespace and temporary tablespace, Oracle will use the current system.
* default *tablespace* and *temporary tablespace* for holding the new account.

## 1.5. To check the current system default tablespace,
```sql
SQL> `SELECT` * `FROM` DATABASE_PROPERTIES `WHERE` PROPERTY_NAME `LIKE` 'DEFAULT%TABLESPACE';
```

## 1.6. To check what users the system has, access the DBA_USERS table.

```sql
SQL> `COL` Username `FORMAT` A20
SQL> `COL` Account_Status `FORMAT` A20
SQL> `SELECT` Username, Account_Status `FROM` DBA_USERS
     `ORDER `BY` Username;
```

## 1.7. Grant Privileges (Permissions) to user to login to Oracle RDBMS
```sql
SQL> `GRANT CREATE SESSION to` xwang;

Grant succeeded.
```

/login with xwang/xw123, but you cannot do anything with the account. You must grant other privileges to allow the account to be able to do something.

```sql
SQL> `conn` shawmoo
Enter password: wang;

SQL> `conn` as sysdba;

SQL> GRANT `UNLIMITED TABLESPACE` to shawmoo;
SQL> GRANT `CREATE ANY TABLE` to shawmoo;
SQL> GRANT `ALTER ANY TABLE`, `DROP ANY TABLE` to shawmoo;
SQL> GRANT `INSERT ANY TABLE`, `UPDATE ANY TABLE`, `DELETE ANY TABLE`, `SELECT ANY TABLE` to shawmoo;
```

# 2. Creating Tables

```sql
SQL> CREATE TABLE Department(
     DCode NUMBER (8) PRIMARY KEY,
     Name VARCHAR (50) NOT NULL,
     Phone VARCHAR (16),
     Chair NUMBER (8)，
     PRIMARY KEY (CallNo),
     FOREIGN KEY (AAA) REFERENCES tableA(AAA),
     FOREIGN KEY (BBB) REFERENCES tableB(BBB)
     );
```

# 3. insert info
```sql
SQL> INSERT INTO Department VALUES
     ('MATH','Mathematics','703-111-0003', '10003');

1 row created.

SQL> COMMIT;          /commit complete.
```


# 4. SQL Script file
## 4.1. Create a SQL Script file
```sql
SQL Script file *uis.sql*

SQL> conn xwang/xw123;
SQL> @uis.sql;
```
Now your UIS database with testing data has been created.


## 4.2. Save commands and outputs in SQL file

```sql
SQL> SPOOL D:\filename.txt;
SQL> SELECT * FROM table_name;
SQL> SPOOL OFF
```

# 5. look your table
## 5.1. USER_TABLES
```sql
SQL> SELECT table_name, tablespace_name FROM user_tabels;
/user_tabels: 本user下的所有文件

TABLE_NAME    TABLESPACE_NAME
-----------------------------
FACULTY       SHAWMOO
DEPARTMENT    SHAWMOO

```

## 5.2 ALL_TABLES
```sql
SQL> SELECT table_name, tablespace_name FROM all_tabels;
/user_tabels: 本user下的所有文件

TABLE_NAME    TABLESPACE_NAME
-----------------------------
FACULTY       SHAWMOO
DEPARTMENT    SHAWMOO

```


## 5.2 TABLE_A
```sql
SQL> SELECT * FROM department;

DCODE    NAME                 PHONE       CHAIR
-------------------------------------------------
CS     Computer Science    703-333-3333   10005

```
