---
title : "Wechall - Programming 1"
categories : [Wargame, Wechall]
---

## Programming 1
<hr style="border-top: 1px solid;"><br>

```
When you visit this link you receive a message.
Submit the same message back to 
https://www.wechall.net/challenge/training/programming1/index.php?answer=the_message
Your timelimit is 1.337 seconds
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Solution
<hr style="border-top: 1px solid;"><br>

link에 걸려있는 사이트로 가면 메시지가 있음. 이 메시지를 제출하라는 것. 단, 1.337초 안에

주의사항은 처음 로그인 시 Restrict sessions to IP를 체크해제 후 로그인해야 함. 체크하고 로그인 시 아래 코드 실행 시 인증이 안됨.

<br>

```python
import requests

url='https://www.wechall.net/challenge/training/programming1/index.php'
headers={'Content-Type':'application/x-www-form-urlencoded'}
cookies={'WC':"[redacted]"}

res1=requests.get(url,params={'action':'request'},headers=headers,cookies=cookies)
ans=res1.text
res2=requests.get(url,params={'answer':ans},headers=headers,cookies=cookies)
ans=res2.text
print(ans)
```

<br><br>
<hr style="border: 2px solid;">
<br><br>
