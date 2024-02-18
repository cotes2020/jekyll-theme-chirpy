---
title : "Wechall - PHP LFI"
categories : [Wargame, Wechall]
tags: [LFI]
---

## LFI
<hr style="border-top: 1px solid;"><br>

```php
$filename = 'pages/'.(isset($_GET["file"])?$_GET["file"]:"welcome").'.html';
include $filename;
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Solution
<hr style="border-top: 1px solid;"><br>


solution.php에 들어가야 함. solution.php의 경로는 ```/challenge/training/php/lfi/solution.php```

현재 경로는 ```/challenge/training/php/lfi/up/index.php```

우선 include 뒤에 확장자로 .html이 붙는데 이 부분은 null byte를 입력함으로써 무시할 수 있음.
: null byte는 url encode 값으로 ```%00```임.

입력을 ```../%00```을 해주면 다음과 같은 에러가 뜸.

<br>

```php
include(/home/wechall/www/wc5/www/challenge/training/php/lfi/up)
: failed to open stream
```

<br>

입력을 ```../../%00``` 을 해주면 에러로 뜬 경로가 ```/challenge/training/php/lfi/``` 로 바뀜.

따라서 ```../../solution.php%00``` 을 입력하면 성공.

<br><br>
<hr style="border: 2px solid;">
<br><br>
