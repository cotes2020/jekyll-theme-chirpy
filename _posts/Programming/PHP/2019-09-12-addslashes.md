---
title : PHP - addslashes()
categories: [Programming, PHP]
tags : [addslashes]
---

## addslashes()
<hr style="border-top: 1px solid;"><br>

따옴표, 쌍따옴표, 백슬래쉬, NULL 문자에 대해 escape 해주는 함수

<br>

```php
addslashes ( string $string ) : string

$str = "Is your name O'Reilly?";

echo addslashes($str) // Outputs: Is your name O\'Reilly?
```

<br><br>
<hr style="border: 2px solid;">
<br><br>
