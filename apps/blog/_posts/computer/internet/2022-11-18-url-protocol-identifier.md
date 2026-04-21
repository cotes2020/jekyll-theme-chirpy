---
title: "URL의 프로토콜 식별자 (Protocol Identifier)"
# description: ""
categories: [컴퓨터, 인터넷]
tags: [Web]
image: "/assets/img/background/kururu-lab.jpg"

date: 2022-11-18. 10:42
# last_modified_at: 2022-11-18. 10:42
---

## 프로토콜 식별자 (Protocol Identifier)

---

일반적인 Web URL을 보면  
:// 앞에 https 가 있는 것을 볼 수 있다.  

:// 앞은 프로토콜 식별자 (Protocol Identifier) 로 사용되는 공간으로,  
여기에 적힌 프로토콜로 통신을 하겠다는 의미이다.  

그래서 여기에 https 가 적혀있으면 https 프로토콜로 통신을 하겠다는 의미,  
다른 프로토콜이 적혀 있으면 그 프로토콜로 통신을 하겠다는 의미다  

## 예시 - TopazChat

---

VRChat 월드 영상 스트리밍 서비스인 [TopazChat](https://github.com/TopazChat/TopazChat)에서 사용하는  
영상 스트리밍 송신/수신 링크를 살펴보면,  

송신 링크 - **rtmp://topaz.chat/live**  
수신 링크 - **rtspt://topaz.chat/live/[StreamKey]**  

이런식으로 프로토콜 식별자에 rtmp와 rtspt가 적힌 걸 볼 수 있는데,  

이를 통해 TopazChat이  
rtmp - RealTime Messaging Protocol,  
rtspt - RealTime Streaming Protocol TCP,  
두 프로토콜을 이용해 영상 스트리밍 통신을 하고 있다는 것을 알 수 있다.  
