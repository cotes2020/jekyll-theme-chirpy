---
layout: post
title: PeachTri 네번째 세미나 - 가비지 컬렉션
date: 2022-06-14 12:02:00 +0900
description: PeachTri 네번째 세미나 # Add post description (optional)
published: true
img : Seminar_banner_small.png
tags: [세미나, PeachTri]
---
# 가비지 컬렉션  :wastebasket:
## 가비지 컬렉션이란? ❓
 흔히 Managed 언어와 Unmanaged언어로 언어의 특징을 나눈다.
 각각의 언어는 Memory allocation이 언어 자체적으로 지원하는지, 아니면 개발자가 직접 만져줘야 하는지에서 나오는 차이가 있다. 각각의 언어의 특징은 다음과 같다.
 ### Unmanaged 💻
 - 언어가 따로 메모리 관리를 해주지 않고, 직접 할당과 해제를 해줘야 한다.
 - pros
    - Managed언어에 비해 빠르고, 코딩 자유도가 높다.
 - cons
    - 매번 할당하는게 번거롭고, 메모리 누수 문제가 발생할 수 있다.

### Managed 👨‍🏭
 - 언어에서 따로 메모리 관리를 해주는 언어를 말한다.
 - pros
    - 자체적으로 관리하기 때문에, 개발하는데 더 편안하다. 이미 해제한 메모리를 또 해제한다던지, 유효하지 않은 포인터에 접근한다던지 하는 문제가 생기지 않는다.
 - cons
    - UnManaged언어에 비해 느리고, **잠재적인 문제에 대비하여서 어느정도 메모리 관리에 대한 지식이 요구된다.** 
### 그렇다면 가비지 컬렉션은 무엇일까?
가비지 컬렉션은 Managed언어에서 더 이상 사용하지 않는 변수가 메모리에 저장되어 용량을 차지하는 것을 막기 위해서 사용하지 않는 리소스를 찾아 메모리에서 free()시키는 기능이다.  
예를 들어, Java의 가비지 컬렉션의 경우, 영역을 두가지로 나눠서 가비지 컬렉션을 한다.
1. Young 영역
   - 새로이 생성된 객체가 할당되는 영역
   - 대부분의 객체가 금방 Unreachable한 상태가 되기 때문에 금방금방 생성되고 사라진다.
2. Old 영역
   - Young영역에서 Reachable한 상태를 유지하여 살아남은 객체가 복사되는 영역
   - Young에 비해 영역이 크게 할당되고, 영역의 크기가 커서 가비지도 적게 발생한다.  
![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fva8qQ%2FbtqUSpSocbS%2FkxTvtnmrdhf4bnVPXth0UK%2Fimg.png)  

이런식으로 자바는 두가지로 나눠서 영역에 메모리를 할당한다.
그리고 가비지 컬렉션은 **Stop The World** 어플리케이션의 실행을 멈추고, **Mark and Sweep** 메모리를 식별해 사용하지 않는 객체들을 메모리에서 제거한다. 

## Flutter의 구조
Flutter는 디버그와 배포단에서 다른 컴포넌트를 들고 간다. Debug모드에서는 Runtime과 JIT/Interpreter, 그리고 개발자 서비스 등이 포함되어 있고, 배포단에서는 런타임 컴포넌트만 가져가게 된다.

> 가비지 컬렉션을 두려워 하지 말아줘!

### The Dart Garbage Collector
다트의 가지비 컬렉터는 두가지 페이즈를 가지게 된다.
- Scheduling
- Young Space Scavenger

**Scheduling**  
이 전의 Stop The World에서 알다시피, GC를 진행하기 위해서는 어플리케이션의 중지가 필요하다. GC에서는 플러터 엔진에게 Hook을 제공하는데 이 Hook을 통해서 플러터의 어플리케이션이 구동되다가 유저와의 Interaction이 없는 순간을 알려준다. 퍼포먼스에 크게 영향을 끼치지 않으면서 사용하지 않는 메모리 공간을 확보하기 위해서 존재하는 페이즈이다.  
  
**Young Space Scavenger**  
플러터 또한 자바의 그것과 비슷한 방식으로 동작한다. stateless widget과 같이 짧은 생명 주기의 객체들을 위해서 고안된 단계이며, Mark and Sweep 보다 빠른 방식으로 동작한다. 플러터가 이러한 방식을 추구하는 이유는 아무래도 어플리케이션의 퍼포먼스 떄문이다.

### bump pointer allocation
![](https://miro.medium.com/max/1400/1*pNaeZ0l8oMCP-f1UUs-V1g.png)  
다트는 bump pointer allocation이라는 방식을 통해서 메모리를 할당하는데. 먼저 메모리 공간을 두 가지로 나눈다. 하나는 Active Space고 하나는 Inactive space인데, 먼저 오브젝트들을 active 공간에 집어넣고, active space가 전부 차게 되면 그 안에서 오브젝트들이 Dead인지 Live한지 참조하는 다른 오브젝트들을 확인하여서, live하다면 Inactive Space에 옮기고 Dead라면 놔둔다. 그리고 두 공간의 용도를 바꿔주면, live한 오브젝트만 존재하는 메모리는 Active Space가 되고, Dead한 오브젝트들은 Inactive Space에 존재하는게 된다. 이러한 방식으로 메모리 공간과 활성화 비활성화 오브젝트에 대해서 계속해서 관리해나간다.  

### Parallel Marking and Concurrent Sweeping
객체가 특정 생명주기에 다다르게 되면(보통은 어느정도 장기화 되었다고 생각되면), 두번째 단계의 GC의 관리하에 들어가게 된다. Mark and Sweep방식인데, 이것은 위의 범프 포인터 방식보다는 좀 느리지만 확실한 방식으로 동작한다. 먼저 오브젝트들을 순회하면서 사용중인지 아닌지, 사용중이라면 표시하고(Mark), 한번 더 메모리 전체를 순회하면서 더 이상 사용하지 않는 오브젝트 공간들을 비운다. 이 때 표시한 마크들도 전부 회수한다.(Sweep)
 마킹 단계에서 UI 스레드가 차단되고, 메모리 변화도 차단하기 때문에, Young Scavenger보다는 성능에 영향을 끼치게 된다. 따라서 Mark and Sweep 과정에 있어서 항상, **약한 세대 가설(weak generational hypothesis)**를 준수하도록 생각해야한다. 만약 이러한 가설을 생각해놓지 않는다면, 모든 오브젝트들이 장기화를 가정으로 생성이 되고, 그렇다면 GC과정의 대부분이 Mark and Sweep 과정으로 일어나게 되고, 결국 UI스레드나 어플리케이션의 성능에 영향을 끼치게 된다.

 > 약한 세대 가설은 생성된 오브젝트의 대부분이 금방 쓰레기가 되어서 메모리에 공간만 차지하게 된다는 가설이다.

### Isolates
Usolates는 다트의 코드가 실행되는 공간으로, 싱글 스레드로 동작한다. 다만 기존의 언어들과는 차이점이 존재하는데,  
![image](https://user-images.githubusercontent.com/74250270/174203282-feacc2b2-7bc1-4880-8c32-fa17cfef2bdc.png)  
자바는 다음과 같이 프로세스내의 스레드간에는 메모리가 공유가 되는 형태이다.  
  

![image](https://user-images.githubusercontent.com/74250270/174203334-b0a313ff-3fa6-4a6a-9171-2a4e096aca18.png)  
하지만, Dart의 isolate는 스레드 자체적으로 메모리를 가지고 있으며, 메모리가 공유되지 않는다는 특징을 가지고 있다. isolate간의 메시지를 주고 받아야 하는 방식을 사용해야하는 불편함은 존재하지만 멀티 스레드를 사용함에 있어서 공유자원에 대한 컨트롤에 신경쓰지 않아도 된다는 장점이 존재한다.

> 이부분은 차후에 Dart언어에서 비동기를 처리하는 부분에서도 사용이 된다.

isolate들은 각자 전용의 heap메모리를 가지고 있고 이러한 구조는 자연스럽게 내부적으로 레이어들이 분리되는 효과를 가지게 된다.

### 참고 문헌
---
https://medium.com/flutter/flutter-dont-fear-the-garbage-collector-d69b3ff1ca30
플러터 공식 미디엄에 올라온 글
