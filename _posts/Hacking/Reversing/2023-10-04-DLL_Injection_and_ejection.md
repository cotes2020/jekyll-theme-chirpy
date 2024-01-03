---
title : DLL Injection and Ejection
categories : [Hacking, Reversing]
tags : [Reversecore, DLL Injection]
---

## DLL Injection
<hr style="border-top: 1px solid;"><br>

Link
: <a href="https://maple19out.tistory.com/37" target="_blank">maple19out.tistory.com/37</a>

<br>

DLL Injection은 실행 중인 다른 프로세스에 특정 DLL 파일을 강제로 삽입하는 것이다.

다른 프로세스에게 ```LoadLibrary()``` API를 스스로 호출하도록 명령하여 사용자가 원하는 DLL을 로딩하는 것이다.

프로세스에 DLL이 로딩되면 자동으로 해당 DLL의 ```DllMain()``` 함수가 실행된다. 따라서 ```DllMain()```에 사용자가 원하는 코드를 추가하면 DLL이 로딩될 때 자연스럽게 해당 코드가 실행된다.

삽입된 DLL은 해당 프로세스의 메모리에 대한 접근 권한을 갖기 때문에 사용자가 원하는 다양한 일(버그 패치, 기능 추가, 기타)을 수행할 수 있다.

<br>

DLL 인젝션 방법은 다음과 같다.

+ 원격 스레드 생성 (```CreateRemoteThread() API```)
+ 레지스트리 이용 (```레지스트리 AppInit_DLLs 값```)
+ 메시지 후킹 (```SetWindowsHookEx() API```)

<br>

실습 부분은 위의 블로그에서 자세히 확인

<br><br>
<hr style="border: 2px solid;">
<br><br>

## DLL Ejection
<hr style="border-top: 1px solid;"><br>

Link
: <a href="https://maple19out.tistory.com/38" target="_blank">maple19out.tistory.com/38</a>

<br>

이젝션은 프로세스에 강제로 삽입한 DLL을 빼내는 기법이다.

기본 동작 원리는 ```CreateRemoteThread API```를 이용한 DLL 인젝션의 동작 원리와 같다.

이젝션은 대상 프로세스가 ```FreeLibrary() API```를 호출하도록 만들어주면 된다.

<br><br>
<hr style="border: 2px solid;">
<br><br>
