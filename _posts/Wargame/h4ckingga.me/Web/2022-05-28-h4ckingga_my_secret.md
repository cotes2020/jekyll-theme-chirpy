---
title: my_secret
date: 2022-05-28 21:42 +0900
categories: [Wargame,h4ckingga.me]
tags: [h4ckingga.me web, unserialize 취약점, magic hash]
---

## my_secret
<hr style="border-top: 1px solid;"><br>

```
can you see my secret?

http://web.h4ckingga.me:10010/

made by D0RI
```

<br>

```php
<?php
    class obj{
        var $pass = "I_WANT_FLAG";
        var $key = "TOP SECRET";
    }

    $obj  = new obj;
    $sol = $_GET["sol"];
    $obj = unserialize($sol);

    if(isset($obj->pass)){
        if($obj->key == I_am_robot){
            echo "oh! you are access my secret<br>";
            echo "<h1>".$flag."</h1>";
        }
    }
?>
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Solution
<hr style="border-top: 1px solid;"><br>

코드 상에서 내 입력 값을 대놓고 unserialize를 하므로 obj 클래스를 serialize하여 문제에서 원하는 값을 보내면 된다.

<br>

serialize
: <a href="https://www.php.net/manual/en/function.serialize.php" target="_blank">php.net/manual/en/function.serialize.php</a>
: ```O:strlen(object name):object name:object size:{s:strlen(property name):property name:property definition;(repeated per property)}```

<br>

repl에서 obj 클래스를 serialize하면 값이 나오므로 그 값에서 문제 상에서 원하는 값으로 변경 후 보내주면 된다.

근데 문제에서 I_am_robot이 문자열이 아니라 변수이다.

이 값은 어디서 확인하나면 ```/robots.txt```에서 확인할 수 있다.
: 0e로 시작한다. 따라서 0을 넣어주면 통과한다.

<br>

그러므로 serialize 한 값에서 key 값으로 0을 넣어주면 플래그가 나온다.

<br><br>
<hr style="border: 2px solid;">
<br><br>
