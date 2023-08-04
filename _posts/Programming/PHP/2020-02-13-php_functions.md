---
title : PHP Functions
categories: [Programming, PHP]
tags: [PHP Functions, scandir, file_get_contents, print_r, var_dump, var_export]
---

## scandir()
<hr style="border-top: 1px solid;"><br>

```php
scandir ( string $directory ) : array

## ex) websec level 8
print_r(scandir('/'));

Array
(
    [0] => .
    [1] => ..
    [2] => flag.txt
    [3] => index.php
    [4] => php-fpm.sock
    [5] => source.php
    [6] => uploads
)
```

<br>

지정된 디렉토리의 파일 및 디렉토리의 배열을 반환.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## file_get_contents()
<hr style="border-top: 1px solid;"><br>

+ 전체파일을 문자열로 읽어들이는 PHP 함수

+ 로컬파일, 원격파일 모두 가능

<br>

```php
file_get_contents( string $filename ) : string

# ex) 
$flag = file_get_contents('flag.php');
echo $flag;

$str = file_get_contents('http://zetawiki.com/ex/txt/utf8hello.txt');
echo $str;
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## printr_r()
<hr style="border-top: 1px solid;"><br>

+ 변수에 대해 사람이 읽을 수있는 정보를 출력

  + PHP에서 변수는 배열(Array)과 객체(Object)도 포함

<br>

```php
print_r ( mixed $expression [, bool $return = FALSE ] ) : mixed

#ex) 
$a = array ('a' => '1', 'b' => '2', 'c' => array ('3', '4', 'hi'));
print_r ($a);
# $results = print_r($b, true); 
Array
(
    [a] => 1
    [b] => 2
    [c] => Array
        (
            [0] => 3
            [1] => 4
            [2] => hi
        )
)
```

<br>

+ 변수에 관한 정보를 사람이 읽기 편하게 출력, 두번째 인자로 true를 써주면 변수에 결과를 저장할 수 있음.

<br><br>
<hr style="border: 2px solid;">
<br><br>


## var_dump()
<hr style="border-top: 1px solid;"><br>

+ 변수의 정보를 출력
  + print_r()의 결과값에서 데이터형을 추가로 더 보여준다고 보면 됨.

+ 변수의 대한 정보를 덤프하고 리턴값은 없음.

<br>

```php
var_dump ( mixed $expression [, mixed $... ] ) : void

$a = array ('a' => '1', 'b' => '2', 'c' => array ('3', '4', 'hi'));
Array
(
    [a] => 
    int(1)
    [b] => 
    int(2)
    [c] => 
    array(3)
        (
            [0] => 
            int(3)
            [1] => 
            int(4)
            [2] => 
            string(2) "hi"
        )
)
```

<br><br>
<hr style="border: 2px solid;">
<br><br>


## var_export()
<hr style="border-top: 1px solid;"><br>

+ 변수를 처리가능한 문자열 표현으로 출력하거나 반환함. 

+ var_dump()와의 차이는 var_export()는 유효한 PHP 코드를 반환한다는 점. 

+ $return을 TRUE로 하면 변수에 저장가능.

<br>

```php
var_export ( mixed $expression [, bool $return = FALSE ] ) : mixed

#ex)
$a = array ('a' => '1', 'b' => '2', 'c' => array ('3', '4', 'hi'));
array (
  0 => 1,
  1 => 2,
  2 => 
  array (
    0 => 3,
    1 => 4,
    2 => 'hi',
  ),
)
```

<br>

참고
: <a href="http://chongmoa.com/php/5130" target="_blank">chongmoa.com/php/5130</a>

<br><br>
<hr style="border: 2px solid;">
<br><br>
