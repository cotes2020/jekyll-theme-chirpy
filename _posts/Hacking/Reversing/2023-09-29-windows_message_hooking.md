---
title : Windows 메시지 후킹
categories : [Hacking, Reversing]
tags : [Reversecore, 윈도우 메시지 후킹]
---

## 후킹
<hr style="border-top: 1px solid;"><br>

Link
: <a href="https://maple19out.tistory.com/35" target="_blank">maple19out.tistory.com/35</a>
: <a href="https://chive0402.tistory.com/10" target="_blank">chive0402.tistory.com/10</a>

<br>

hook은 중간에서 오고가는 정보를 엿보거나 가로채는 행위를 뜻한다. 이를 hooking이라 한다.

그래서 OS - 애플리케이션 - 사용자 사이에 오고가는 정보도 마찬가지로 후킹이 가능하다.

여러 방식이 있으며 가장 기본적인 것이 메시지 후킹이라고 한다.

<br>

윈도우는 GUI 환경이므로 Event를 통해 동작하는데 간단히 보면 이벤트가 발생하면 OS 메시지 큐에 저장되었다가 OS에서 메시지를 확인하고 해당 애플리케이션 메시지 큐로 보낸다. 그럼 해당 애플리케이션은 메시지를 확인하고 이벤트를 처리한다.

만약 여기에 훅이 설치되었다면 오고가는 메시지를 먼저 확인할 수 있다.

메시지 훅 기능은 윈도우 운영체제에서 제공하는 **기본 기능**이며, 대표적인 프로그램으로 Visual Studio에서 제공되는 SPY++가 있다고 한다.

메시지 훅은 ```SetWindowsHookEx()``` API를 사용해서 간단히 구현할 수 있다고 한다.

멤버 변수 중에 ```HOOKPROC lpfn``` 변수는 hook procedure로, 운영체제가 호출해주는 콜백 함수라고 한다.

메시지 훅을 걸 때 hook procedure는 DLL 내부에 존재해야 하며, 그 DLL의 인스턴스 핸들이 바로 hMod라고 한다.

<br>

그래서 이 API를 이용해서 훅을 설치해 놓으면, 어떤 프로세스에서 해당 메시지가 발생했을 때 운영체제가 해당 DLL 파일을 해당 프로세스에 강제로 인젝션하고 등록된 hook procedure를 호출한다.

실습과 더 자세한 내용들은 맨 위의 블로그들에 있다.

<br><br>
<hr style="border: 2px solid;">
<br><br>