---
title : "Wechall - Register Globals"
categories : [Wargame, Wechall]
tags : ["PHP register_globals"]
---

## Register Globals
<hr style="border-top: 1px solid;"><br>

```
This challenge is a relict of old PHP times, 
where register globals has been enabled by default, 
which often lead to security issues.

Again, your job is to login as admin

I have also setup a test account: test:test

Enjoy!
```

<br>

```php
<?php
# EMULATE REGISTER GLOBALS = ON
foreach ($_GET as $k => $v) { $$k = $v; }
 
 
# Send request?
if (isset($_POST['password']) && isset($_POST['username']) && is_string($_POST['password']) && is_string($_POST['username']) )
{
        $uname = GDO::escape($_POST['username']);
        $pass = md5($_POST['password']);
        $query = "SELECT level FROM ".GWF_TABLE_PREFIX."wc_chall_reg_glob WHERE username='$uname' AND password='$pass'";
        $db = gdo_db();
        if (false === ($row = $db->queryFirst($query))) {
                echo GWF_HTML::error('Register Globals', $chall->lang('err_failed'));
        } else {
                # Login success
                $login = array($_POST['username'], (int)$row['level']);
        }
}
 
if (isset($login))
{
        echo GWF_HTML::message('Register Globals', $chall->lang('msg_welcome_back', array(htmlspecialchars($login[0]), htmlspecialchars($login[1]))));
        if (strtolower($login[0]) === 'admin') {
                $chall->onChallengeSolved(GWF_Session::getUserID());
        }
}
else 
{
?>
<form action="globals.php" method="post">
<table>
<tr>
        <td><?php echo $chall->lang('th_username'); ?>:</td>
        <td><input type="text" name="username" value="" /></td>
</tr>
<tr>
        <td><?php echo $chall->lang('th_password'); ?>:</td>
        <td><input type="password" name="password" value="" /></td>
</tr>
<tr>
        <td></td>
        <td><input type="submit" name="send" value="<?php echo $chall->lang('btn_send'); ?>" /></td>
</tr>
</table>
</form>
<?php
}
 
# EMULATE REGISTER GLOBALS = OFF
foreach ($_GET as $k => $v) { unset($$k); }
 
require_once 'challenge/html_foot.php';
?>
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Solution
<hr style="border-top: 1px solid;"><br>

php 5.4 이전 버전에서 발생한 취약점으로 php.ini에 register_globals 라는 옵션이 있음.

이 옵션을 On 하게 되면 GET 또는 POST 방식 등으로 전될 된 모든 변수가 자동으로 php의 변수로 변환이 됨.

<br>

```php
<?php 
	if(isset($get)){
		echo "success";
	}
	else{
		echo "fail";
	}
?>
```

<br>

위에 코드를 예시로 하면 ```$get```에 아무 값도 없으므로 fail이 출력되지만 ```?get=1```을 하면 register_globals 옵션에 의해 get 변수로 변환되어 success가 출력됨.

따라서 문제 상에서 우리가 봐야 할 코드는 다음과 같음.

<br>

```php
if (strtolower($login[0]) === 'admin') {
       $chall->onChallengeSolved(GWF_Session::getUserID());
}
```

<br>

```$login[0] == 'admin'``` 이어야 하므로 ```?login[0]=admin -> 성공```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 참고
<hr style="border-top: 1px solid;"><br>

Link 
: <a href="https://lactea.kr/entry/php-registerglobals-on-취약점" target="_blank">lactea.kr/entry/php-registerglobals-on-취약점</a>

<br><br>
<hr style="border: 2px solid;">
<br><br>
