---
title: "[corCTF 2022] whack-a-frog"
date: 2022-08-16 16:59:00 +09:00
author: aestera
categories: [CTF, Writeup]
tags: [Writeup]
---

# whack-a-frog


![Untitled](/assets/img/post_images/whack-a-frog/main.png)

나름 재미있었던 forensic 문제였다.
<br><br>

![Untitled](/assets/img/post_images/whack-a-frog/frog.png)
문제 페이지를 보면 무수한 우물안 개구리들이 다를 반겨준다. 클릭한 상태고 드래그하면 그 경로에 있는 개구리들이 우물 속으로 숨는다. 이것만 봐서는 무슨 문제인지 모르겠다.
<br><br>

![Untitled](/assets/img/post_images/whack-a-frog/wireshark.png)
.pcap 확장자라 wireshark로 파일을 열어 HTTP필터를 걸었더니 x, y, event 인자가 넘어간다.
<br>
x, y 는 마우스의 좌표, event는 마우스 클릭 여부를 알려주는 것 같다. 클릭한 상태로 이동한 길이 FLAG를 얻을 수 있을 것 같다. 
<br><br>

![Untitled](/assets/img/post_images/whack-a-frog/text.png)
이를 알아내기 위해 패킷을 plain text를 추출해서 x, y, event를 파싱해서 사용하며 될 것 같다.<br>
파이썬 처음 배울 때 정말 왜 배우는지 몰랐었던 Python의 turtle 모듈로 그림을 그려봤다.
<br><br>

****

# Exploit

```python
import turtle as t

with open("./frog.txt", "r", encoding='UTF8') as f:
    data = f.readlines()
    f.close()

def parse_text(data):  #parse data
    x = []
    y = []
    event = []
    for string in data:
        if "event" in string:
            question_split = string.split("?")[1]
            ampersand_split = question_split.split("&")
            x.append(ampersand_split[0].split("=")[1])
            y.append(ampersand_split[1].split("=")[1])
            event.append(ampersand_split[2].split(" ")[0].split("=")[1])    

    print_flag(x, y, event)
    return 0

def print_flag(X, Y, EVENT):  #draw FLAG with turtle
    t.penup()
    for i, j, eve in zip(X, Y, EVENT):
        if eve == "mousedown":
            t.pendown()
        if eve == "mousemove":
            t.goto(int(i), -int(j))
        if eve == "mouseup":
            t.penup()

parse_text(data)
```
위 스크립트를 사용해 그림을 그려보면
<br><br>

![Untitled](/assets/img/post_images/whack-a-frog/turtle.png)
이렇게 LILYXO 라는 문자열이 나타난다. 이 문자열이 FLAG였다. 
<br><br>

****
<br><br>
Turtle 모듈을 사용하지 않더라도 별찍기로도 풀이가 가능하다.

```python
with open("./frog.txt", "r") as f:
    data = f.readlines()
    f.close()

def parse_text(data):
    x = []
    y = []
    event = []
    for string in data:
        if "event" in string:
            question_split = string.split("?")[1]
            ampersand_split = question_split.split("&")
            x.append(ampersand_split[0].split("=")[1])
            y.append(ampersand_split[1].split("=")[1])
            event.append(ampersand_split[2].split(" ")[0].split("=")[1])    

    print_flag(x, y, event)
    return 0

def print_flag(X, Y, EVENT):
    height = 100
    width = 700
    table = [ [ ' ' for _ in range(width) ]  for _ in range(height) ]
    mousedown = 1
    for i, j, eve in zip(X, Y, EVENT):
        if eve == "mousedown":
            mousedown = 0
        elif eve == "mousemove" and mousedown == 0:
            table[int(j)][int(i)] = "*"
        elif eve == "mouseup":
            mousedown = 1
    for j in table:
        for i in j:
           print(i, end = "")
        print("")



parse_text(data)
```
![Untitled](/assets/img/post_images/whack-a-frog/star.png)
개인적으론 Turtle graphics를 사용하는게 좀 더 편했던 것 같다. 
<br><br>
그림을 그릴 때 배열은 왼쪽 위를 (0, 0) 으로 처리하는데,<br>
Turtle graphics를 이용할 때에는 ```t.goto(int(i), -int(j)) ``` 처럼 y좌표에 -를 붙여 음수로 처리해 주어야 정확한 그림이 그려진다.
<br><br>

**FLAG : corctf{LILYXOX}**