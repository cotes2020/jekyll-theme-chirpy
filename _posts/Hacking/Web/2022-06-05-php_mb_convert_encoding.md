---
title : PHP mb_convert_encoding 취약점
categories: [Hacking,Web]
tags : [mb_convert_encoding 취약점]
---

## mb_convert_encoding 취약점
<hr style="border-top: 1px solid;"><br>

멀티바이트 언어에서 유니코드 언어로 변경 시 발생할 수 있는 취약점 
: <a href="https://dydgh499.tistory.com/32" target="_blank">dydgh499.tistory.com/32</a>

<br>

+ ```싱글바이트(SBCS)``` 코드

  + 모든 캐릭터가 정확히 한바이트 차지한다. 
  
  + 쉽게 설명하자면 프로그래밍 언어에서 ```char```를 생각하면 됨.
  
  + ```ASCII```도 ```SBCS```이다. 문자열의 끝마다 끝을 알리는 ```NULL,\x00```이 반드시 존재한다.

<br>

+ ```멀티바이트(MBCS) 코드```
  
  + ```MBCS```는 2가지 방식이 있는데, ```SBCS```와 ```DBCS(Double-Byte-CharacherS)``` 이렇게 두가지 인코딩 방식이 있다. 
  
  + 영어같이 **1바이트를 요구로 하는 문자셋**은 ```SBCS```를 사용하고 **한글, 중국어**와 같은 **2바이트를 요구로 하는 문자셋**은 ```DBCS```를 사용한다.
  
  + ```DBCS```로 3바이트를 표현할 수도 있지만, 아직은 지구상에 3바이트를 표현하는 문자셋이 없기 때문에 표현을 하지 못한다.

<br>

+ ```유니코드(Unicode)```
  
  + 유니코드는 모든 캐릭터 문자셋을 2바이트로 나타내는 표준 인코딩방식이다.

<br>

멀티바이트는 1, 2바이트 표현, 유니코드는 모든 문자 2바이트 표현

멀티에서 유니로 인코딩 시, 멀티바이트 환경에서 백슬래시 앞에 ```%a1 ~ %fe```의 값이 들어오면 인코딩이 깨지면서 백슬래시를 덮어씌어버려서 **2바이트의 멀티바이트를 하나의 문자(1바이트)처럼 표현이 되는 취약점**이 있음.

<br>

출처
: <a href="https://dydgh499.tistory.com/32" target="_blank">dydgh499.tistory.com/32 [Creating my everything]</a>

<br><br>
<hr style="border: 2px solid;">
<br><br>
