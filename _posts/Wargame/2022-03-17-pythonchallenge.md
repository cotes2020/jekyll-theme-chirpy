---
title : PythonChallenge Writeup
categories: [Wargame, PythonChallenge]
date: 2022-03-17 12:29 +0900
---

## Level 1
<hr style="border-top: 1px solid;"><br>

문제를 보면 ```K -> M, O -> Q, E -> G```이다. 그 밑에에도 2번 생각한다고 되어있다.

즉, 2번 ROT 한 것이다. 코드를 짜면 아래와 같다.

<br>

```python
import string

cipher="g fmnc wms bgblr rpylqjyrc gr zw fylb. rfyrq ufyr amknsrcpq ypc dmp. bmgle gr gl zw fylb gq glcddgagclr ylb rfyr'q ufw rfgq rcvr gq qm jmle. sqgle qrpgle.kyicrpylq() gq pcamkkclbcb. lmu ynnjw ml rfc spj.".split()

plain=[]

for i in cipher:
    text=''
    for j in i :
        if ord(j) < 97 or ord(j) > 122 : 
            text+=j
            continue
        chk=ord(j)+2
        if chk > 122 : chk=(chk-122)+96
        text+=chr(chk)
    plain.append(text)

print(' '.join(plain))
```

<br>

해독하면 다음과 같다.
: ```i hope you didnt translate it by hand. thats what computers are for. doing it in by hand is inefficient and that's why this text is so long. using string.maketrans() is recommended. now apply on the url.```

<br>

url에 적용하라고 하므로 map에 rot2를 해주면 ocr이 된다. 

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Level 2
<hr style="border-top: 1px solid;"><br>

소스를 보면 rare character를 찾으라면서 매우 긴 문자열을 주었다.

rare character가 희귀한 문자를 찾으라는 뜻이므로 빈도수를 찾아야 한다.

이전에 알게 된 Python STL에 있는 collections.Counter를 이용하였다.

<br>

```python
from collections import Counter

text=Counter("""{TEXT}""") # mess text 

print(text.most_common())
```

<br>

```[(')', 6186), ('@', 6157), ('(', 6154), (']', 6152), ('#', 6115), ('_', 6112), ('[', 6108), ('}', 6105), ('%', 6104), ('!', 6079), ('+', 6066), ('$', 6046), ('{', 6046), ('&', 6043), ('*', 6034), ('^', 6030), ('\n', 1219), ('e', 1), ('q', 1), ('u', 1), ('a', 1), ('l', 1), ('i', 1), ('t', 1), ('y', 1)]```

<br>

빈도수가 1인 ```equality```를 url에 입력해주면 된다.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Level 3
<hr style="border-top: 1px solid;"><br>

정규식 문제였는데 3개의 대문자 1개의 소문자 3개의 대문자로 감싸진 소문자를 찾아야 한다.

정규식은 처음에 ```[A-Z]{3}([a-z])[A-Z]{3}```로 했는데 너무 길게 나왔다.

다른 풀이를 보니 이렇게 정규식을 설정하면 정확하게 나오지 않는다고 한다. 

앞, 뒤로 소문자도 같이 검색해주면 정확히 앞 뒤로 3개의 대문자와 가운데의 소문자를 가져올 수 있다.

최종 정규식은 ```[a-z][A-Z]{3}([a-z])[A-Z]{3}[a-z]```이며 값은 linkedlist가 나온다.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Level 4
<hr style="border-top: 1px solid;"><br>

사진을 누르면 nothing 값으로 계속 입력하게 하는데 400번 이상을 해야 한다고 써있다.

re, requests를 이용하였다.

<br>

```python
import requests
import re

url='http://www.pythonchallenge.com/pc/def/linkedlist.php'
headers={'Content-Type':'application/x-www-form-urlencoded'}

payload={'nothing':'63579'}
res=requests.get(url, headers=headers, params=payload)
num=re.findall("\d+", res.text)

for i in range(500) :
    payload={'nothing':num}
    res=requests.get(url, headers=headers, params=payload)
    print(res.text)
    num=re.findall("\d+", res.text)

```

<br>

중간에 2로 나누라거나 어떤 문장이 나오는 경우가 있었는데 noting 값이 66831 일 때, peak.html이라고 뜬다.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Level 5 (풀이 참조)
<hr style="border-top: 1px solid;"><br>

peakhell을 발음하면 pickle과 유사하다. 웹에 p확장자라 검색해도 pickle이라 나온다. 

따라서 banner.p에 있는 내용을 pickle.load()로 읽었다.

나오고 나니 2차원 리스트에 리스트 형태로 공백과 숫자, #과 숫자로 된 튜플 요소들이 들어있었다. 

<br>

해석이 안돼서 풀이를 찾아보니.. ASCII ART 형태라고 한다. 

즉, (' ', 95)는 공백을 95만큼 주는 것이고 ('#', 5)는 #을 5번 출력하라는 의미다.

따라서 코드를 짜면 아래와 같이 되며 출력 값으로 CHANNEL이라고 출력된다.

<br>

```python
import pickle

with open('banner.p', 'rb') as file:
	data = pickle.load(file)

for i in data:
    for j in i :
        print(j)
```

<br>

![image](https://user-images.githubusercontent.com/52172169/158942657-59f690e4-8a69-40e1-bc70-553aa6f41b2f.png)

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Level 6
<hr style="border-top: 1px solid;"><br>

소스코드를 보면 zip이라 되있다. zip.html로 가보니 zip download하라고 해서 channel.zip으로 가면 zip파일을 다운받는다.

zip 파일에 readme 파일이 있다. 

<br>

```
welcome to my zipped list.

hint1: start from 90052
hint2: answer is inside the zip
```

<br>

90052부터 읽으라고 하며 답은 zip 안에 있다고 한다. 우선 코드를 짜면 아래와 같다.

<br>

```python
import re

n="90052.txt"

while True:
    file = "channel"+'\\'+n
    
    with open(file, "r") as f :
        data=f.read()
        n=re.findall("\d+",data)[0]+".txt"
        print(data)
```

<br>

쭉 읽다가 46145.txt에서 막히는데 확인해보니 ```Collect the comments```라고 뜬다.

zip 파일 안에 답이 있다고 했으니.. 다 읽어보았지만 별 다른 게 없었다. 

확인해보니.. 파이썬 zipfile 모듈에 getinfo를 통해 comments가 있다고 한다.
: <a href="https://yganalyst.github.io/data_handling/memo_2/" target="_blank">yganalyst.github.io/data_handling/memo_2/</a>
: <a href="https://docs.python.org/ko/3/library/zipfile.html" target="_blank">docs.python.org/ko/3/library/zipfile.html</a>
: <a href="https://stackoverflow.com/questions/51792923/python-reading-a-zip-file-comment" target="_blank">stackoverflow.com/questions/51792923/python-reading-a-zip-file-comment</a> ---> 경로에 r 붙여야 함.

<br>

```python
import zipfile
import re

comment=''

with zipfile.ZipFile(r"[path]",'r') as files : # 앞에 r 붙여줘야 에러 안남.
    n = "90052.txt"
    while n != "46145.txt" :
        read = files.getinfo(n)
        comment+=read.comment.decode('utf-8')
        data = files.read(n).decode('utf-8')
        n = re.findall("\d+",data)[0] + ".txt"
        
print("Comments : ", comment)
```

<br>

![image](https://user-images.githubusercontent.com/52172169/159106181-8e5a40c1-ffbf-4f89-be9c-03f673ea9536.png)

<br>

hockey라고 떴다. hockey.html에 가보면 문자를 보라고 한다. 첫째 줄을 보면 아래와 같이 OXYGEN이라고 써있다.
: ```OO    OO    XX      YYYY    GG    GG  EEEEEE NN      NN```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Level 7 (풀이 참조)
<hr style="border-top: 1px solid;"><br>

사진 가운데 부분이 픽셀로 표현된 거라고 한다..  

PIL을 이용해 사진의 가운데 부분 픽셀에 접근해서 출력해보면 rgb 값이 동일한 픽셀들이 출력된다.

이 rgb 값은 아스키 값으로 문자로 변환해주면 된다.

<br>

```python
from PIL import Image

img = Image.open('oxygen.png')

pix = img.load()

txt=''

for i in range(4,img.width-22,7) :
    # print(pix[i,44])
    txt+=chr(pix[i,44][0])

print(txt)

ans = [105, 110, 116, 101, 103, 114, 105, 116, 121]
res=''

for i in ans :
    res+=chr(i)

print(res)
```

<br>

출력 결과
: ```smart guy, you made it. the next level is [105, 110, 116, 101, 103, 114, 105, 116, 121]```
: integrity

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Level 8
<hr style="border-top: 1px solid;"><br>

사진을 누르면 로그인 창이 뜬다.

소스를 보니 un, pw 값이 있는데 검색해보니 파이썬 bz2 압축 데이터라고 한다.
: <a href="https://wikidocs.net/119859" target="_blank">wikidocs.net/119859</a>

<br>

un
: ```'BZh91AY&SYA\xaf\x82\r\x00\x00\x01\x01\x80\x02\xc0\x02\x00 \x00!\x9ah3M\x07<]\xc9\x14\xe1BA\x06\xbe\x084'```

pw
: ```'BZh91AY&SY\x94$|\x0e\x00\x00\x00\x81\x00\x03$ \x00!\x9ah3M\x13<]\xc9\x14\xe1BBP\x91\xf08'```

<br>

```python
import bz2

un=b'BZh91AY&SYA\xaf\x82\r\x00\x00\x01\x01\x80\x02\xc0\x02\x00 \x00!\x9ah3M\x07<]\xc9\x14\xe1BA\x06\xbe\x084'
pw=b'BZh91AY&SY\x94$|\x0e\x00\x00\x00\x81\x00\x03$ \x00!\x9ah3M\x13<]\xc9\x14\xe1BBP\x91\xf08'

username= bz2.decompress(un).decode('utf-8')
password= bz2.decompress(pw).decode('utf-8')


print(username) # huge
print(password) # file
```

<br>

로그인하면 Level9로 넘어간다.

<br><br>
<hr style="border: 2px solid;">
<br><br>
