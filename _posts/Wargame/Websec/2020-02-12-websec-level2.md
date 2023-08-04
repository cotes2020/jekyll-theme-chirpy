---
title : "Websec - Level 2"
categories : [Wargame, Websec]
tags: [SQLi]
---

## Level 2
<hr style="border-top: 1px solid;"><br>

``` php
ini_set('display_errors', 'on');

class LevelTwo {
    public function doQuery($injection) {
        $pdo = new SQLite3('leveltwo.db', SQLITE3_OPEN_READONLY);

        $searchWords = implode (['union', 'order', 'select', 'from', 'group', 'by'], '|');  #단어들 사이에 | 을 해준다는 것
        $injection = preg_replace ('/' . $searchWords . '/i', '', $injection);

        $query = 'SELECT id,username FROM users WHERE id=' . $injection . ' LIMIT 1';
        $getUsers = $pdo->query ($query);
        $users = $getUsers->fetchArray (SQLITE3_ASSOC);

        if ($users) {
            return $users;
        }

        return false;
    }
}

if (isset ($_POST['submit']) && isset ($_POST['user_id'])) {
    $lt = new LevelTwo ();
    $userDetails = $lt->doQuery ($_POST['user_id']);
}
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Solution
<hr style="border-top: 1px solid;"><br>

level1과 다른점은 필터링을 한다는 점이다.

근데 필터링하는 문자들을 공백으로 바꿔준다는 점에서 취약점이 발생한다. 

```uniunionon``` 이런식으로 써준다면 필터링을 우회할 수 있다.

<br><br>
<hr style="border: 2px solid;">
<br><br>
