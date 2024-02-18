---
title : "Wechall - Caesar II"
categories : [Wargame, Wechall]
---

## Crypto - Caesar II
<hr style="border-top: 1px solid;"><br>

```
I guess you are done with Caesar I, aren't you?
The big problem with caesar is that it does not allow digits or other characters.
I have fixed this, and now I can use any ascii character in the plaintext.
The keyspace has increased from 26 to 128 too. \o/

Enjoy!
```

<br>

```
59 01 01 76 20 7C 01 74 3E 20 0B 01 07 20 05 01
7E 08 77 76 20 01 00 77 20 7F 01 04 77 20 75 7A
73 7E 7E 77 00 79 77 20 7B 00 20 0B 01 07 04 20
7C 01 07 04 00 77 0B 40 20 66 7A 7B 05 20 01 00
77 20 09 73 05 20 78 73 7B 04 7E 0B 20 77 73 05
0B 20 06 01 20 75 04 73 75 7D 40 20 69 73 05 00
39 06 20 7B 06 51 20 43 44 4A 20 7D 77 0B 05 20
7B 05 20 73 20 03 07 7B 06 77 20 05 7F 73 7E 7E
20 7D 77 0B 05 02 73 75 77 3E 20 05 01 20 7B 06
20 05 7A 01 07 7E 76 00 39 06 20 7A 73 08 77 20
06 73 7D 77 00 20 0B 01 07 20 06 01 01 20 7E 01
00 79 20 06 01 20 76 77 75 04 0B 02 06 20 06 7A
7B 05 20 7F 77 05 05 73 79 77 40 20 69 77 7E 7E
20 76 01 00 77 3E 20 0B 01 07 04 20 05 01 7E 07
06 7B 01 00 20 7B 05 20 7B 05 74 01 7A 7F 7A 76
04 78 76 77 40 
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Solution
<hr style="border-top: 1px solid;"><br>

1번과의 차이점은 1번에서는 알파벳 이외의 값은 암호화되지 않았는데 이번 문제에선 숫자 공백 등 모든 값을 암호화 했다는 점.

너무 길어서 첫 번째 암호문만 디코딩 해보니 key=110임을 확인함. 그런데 띄워쓰기가 안되어있어서 공백은 그냥 추가하도록 함.

<br>

```python
cipher="""59 01 01 76 20 7C 01 74 3E 20 0B 01 07 20 05 01
7E 08 77 76 20 01 00 77 20 7F 01 04 77 20 75 7A
73 7E 7E 77 00 79 77 20 7B 00 20 0B 01 07 04 20
7C 01 07 04 00 77 0B 40 20 66 7A 7B 05 20 01 00
77 20 09 73 05 20 78 73 7B 04 7E 0B 20 77 73 05
0B 20 06 01 20 75 04 73 75 7D 40 20 69 73 05 00
39 06 20 7B 06 51 20 43 44 4A 20 7D 77 0B 05 20
7B 05 20 73 20 03 07 7B 06 77 20 05 7F 73 7E 7E
20 7D 77 0B 05 02 73 75 77 3E 20 05 01 20 7B 06
20 05 7A 01 07 7E 76 00 39 06 20 7A 73 08 77 20
06 73 7D 77 00 20 0B 01 07 20 06 01 01 20 7E 01
00 79 20 06 01 20 76 77 75 04 0B 02 06 20 06 7A
7B 05 20 7F 77 05 05 73 79 77 40 20 69 77 7E 7E
20 76 01 00 77 3E 20 0B 01 07 04 20 05 01 7E 07
06 7B 01 00 20 7B 05 20 7B 05 74 01 7A 7F 7A 76
04 78 76 77 40""".split()

plain=''
for i in cipher :
    if i == '20' :
        plain+=chr(int(i,16))
    else :
        plain+=chr((int(i,16)+110)%128)
print(plain)
```

<br>

```
Good job, you solved one more challenge in your journey. 
This one was fairly easy to crack. 
Wasn't it? 128 keys is a quite small keyspace, 
so it shouldn't have taken you too long to decrypt this message. 
Well done, your solution is isbohmhdrfde.
```

<br><br>
<hr style="border: 2px solid;">
<br><br>
