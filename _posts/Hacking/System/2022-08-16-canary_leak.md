---
title: Bypass Canary (Canary Leak)
date: 2022-08-16 14:36 +0900
categories: [Hacking,System]
tags: [Canary Bypass, Canary Leak]
---

## Canary Leak
<hr style="border-top: 1px solid;"><br>

Canary
: <a href="https://ind2x.github.io/posts/stack_memory_protection/#sspstack-smashing-protector" target="_blank">ind2x.github.io/posts/stack_memory_protection/#sspstack-smashing-protector</a>

<br>

x86은 4바이트, x64는 8바이트의 카나리가 생성되는데, 카나리에는 첫 바이트에 null바이트가 포함되어 있어 실제로는 ```null+7바이트```, ```null+3바이트``` 이다.

따라서 브루트포싱하는 것은 연산량이 많을 뿐만 아니라, 실제 서버에서는 불가능하므로 바람직하지 않다.

<br>

카나리는 TLS에 전역변수로 저장되며 매 함수마다 참조해서 사용한다.

TLS의 주소는 매 실행마다 변경되지만 실행 중에 TLS의 주소를 알 수 있고, 임의 주소에 대한 읽기와 쓰기가 가능하다면, TLS의 값을 읽거나 조작할 수 있다.

<br>

+ Canary Leak
  + 함수의 프롤로그에서 카나리 값을 저장하므로, 이 값을 읽을 수 있다면 우회할 수 있다.
  + 카나리는 ```rbp-0x8```에 위치하므로 이 값만 읽을 수 있게 된다면 쉘 코드를 넣어주고 ```rbp-0x8``` 부분에는 알아낸 카나리 값을 통해 우회를 해 줄 수 있다.

<br><br>
<hr style="border: 2px solid;">
<br><br>
