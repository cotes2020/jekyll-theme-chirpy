---
title : "Wechall - Substitution II"
categories : [Wargame, Wechall]
---

## Crypto - Substitution II
<hr style="border-top: 1px solid;"><br>

```
I have created an advanced version of the simple substitution cipher.
It can now use chars in range from 0-255, but that should not stop you.

The ciphertext is in the language of this text, 
and uses correct punctuation and case-sensitivity.
```

<br>

```
0A 2A 68 48 03 F7 5D 07 19 F7 5D 3C 2A 68 7E DE
91 2E 30 3C 7E 91 2A 68 C4 91 AA F7 7E 91 30 F7
03 FC C4 03 B4 91 5A 07 5D 91 3B 2A 07 91 48 2A
5D 91 3C 5D DE 91 43 C4 03 3B 91 AA C4 19 19 91
FC 2A 68 C4 91 00 C4 19 19 2A AA 91 30 F7 C8 1C
C4 03 DE 91 2E 30 C4 91 6B 03 2A 5A 19 C4 92 91
AA 3C 5D 30 91 5D 30 3C 7E 91 C8 3C 6B 30 C4 03
91 3C 7E 91 5D 30 F7 5D 91 5D 30 C4 91 1C C4 3B
91 3C 7E 91 6B 03 C4 5D 5D 3B 91 19 2A 68 48 DE
91 15 91 AA 3C 19 19 91 C8 2A 92 C4 91 07 6B 91
AA 3C 5D 30 91 F7 91 5A C4 5D 5D C4 03 91 C4 68
C8 03 3B 6B 5D 3C 2A 68 91 7E 30 C4 92 C4 91 F7
68 3B 91 7E 2A 2A 68 DE 91 08 2A 07 03 91 7E 2A
19 07 5D 3C 2A 68 91 3C 7E FF 91 68 C4 30 03 68
3C F7 C8 C4 00 5A 3C DE 
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Solution
<hr style="border-top: 1px solid;"><br>

빈도수를 분석해 보았음. 문제에서 구두점 사용 및 대소문자를 구분하였다고 함.

확실한 건 문장의 젤 마지막인 DE는 점이 된다는 것. 따라서 DE 뒤에 오는 값인 91이 공백이 됨.

대문자가 되는 값들은 0A, 2E, 43, 15, 08

<br>

빈도수별로 넣어주면서 같은 빈도수가 나오면 그 전 결과에서 추측해서 넣는 방식으로 substitution 1과 동일한 방식으로 풀어 나갔음..

<br>

```python
cipher="""0A 2A 68 48 03 F7 5D 07 19 F7 5D 3C 2A 68 7E DE
91 2E 30 3C 7E 91 2A 68 C4 91 AA F7 7E 91 30 F7
03 FC C4 03 B4 91 5A 07 5D 91 3B 2A 07 91 48 2A
5D 91 3C 5D DE 91 43 C4 03 3B 91 AA C4 19 19 91
FC 2A 68 C4 91 00 C4 19 19 2A AA 91 30 F7 C8 1C
C4 03 DE 91 2E 30 C4 91 6B 03 2A 5A 19 C4 92 91
AA 3C 5D 30 91 5D 30 3C 7E 91 C8 3C 6B 30 C4 03
91 3C 7E 91 5D 30 F7 5D 91 5D 30 C4 91 1C C4 3B
91 3C 7E 91 6B 03 C4 5D 5D 3B 91 19 2A 68 48 DE
91 15 91 AA 3C 19 19 91 C8 2A 92 C4 91 07 6B 91
AA 3C 5D 30 91 F7 91 5A C4 5D 5D C4 03 91 C4 68
C8 03 3B 6B 5D 3C 2A 68 91 7E 30 C4 92 C4 91 F7
68 3B 91 7E 2A 2A 68 DE 91 08 2A 07 03 91 7E 2A
19 07 5D 3C 2A 68 91 3C 7E FF 91 68 C4 30 03 68
3C F7 C8 C4 00 5A 3C DE""".split()
# 0A, 2E, 43, 15, 08 -> 

plain=''

for i in cipher :
    if i == 'DE' :
        plain+='.'
    elif i == '91' :
        plain+=' '
    elif i == 'C4' :
        plain+='e'
    elif i == '5D' :
        plain+='t'
    elif i == '2A' :
        plain+='o'
    elif i == '3C' :
        plain+='i'
    elif i == '68':
        plain+='n'
    elif i == '19' :
        plain+='l'
    elif i == '7E' :
        plain+='s'
    elif i == '07' :
        plain+='u'
    elif i == 'OA' :
        plain+='C'
    elif i == '48' :
        plain+='g'
    elif i == '03' :
        plain+='r'
    elif i == 'F7' :
        plain+='a'
    elif i == '30' :
        plain+='h'
    elif i == '0A' :
        plain+='C'
    elif i == '2E' :
        plain+='T'
    elif i == '08' :
        plain+='Y'
    elif i == '15' :
        plain+='I'
    elif i == 'AA' :
        plain+='w'
    elif i == 'C8' :
        plain+='c'
    elif i == '3B' :
        plain+='y'
    elif i == '6B' :
        plain+='p'  
    elif i == '5A' :
        plain+='b'
    elif i == '43' :
        plain+='V'
    elif i == '92' :
        plain+='m'
    elif i == '1C' :
        plain+='k'
    else : 
        plain+='_'
    
for i in plain :
    print(i, end='')
    if i == '.' :
        print('')
```

<br>

```
Congratulations.
This one was har_er_ but you got it.
Very well _one _ellow hacker.
The problem with this cipher is that the key is pretty long.
I will come up with a better encryption sheme any soon.
Your solution is_ nehrniace_bi.
```

<br>

정답 부분에 한자리 밖에 안남았고, 저 부분이 00인데 ```_ellow``` 부분도 00이어서 적절한 값인 f를 넣었더니 통과.


<br><br>
<hr style="border: 2px solid;">
<br><br>
