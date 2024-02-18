---
title : PE 재배치
categories : [Hacking, Reversing]
tags : [Reversecore, PE Relocation, Relocation Table]
---

## PE 재배치
<hr style="border-top: 1px solid;"><br>

PE 파일의 재배치 과정에서 Base Relocation Table을 사용하는데 이것의 구조와 동작 원리를 살펴본다.

<br>

PE파일이 메모리에 로딩될 때, PE헤더의 ImageBase 주소에 로딩된다.

DLL 파일의 경우 해당 위치에 이미 다른 DLL 파일이 로딩되어 있으면 다른 비어 있는 주소 공간에 로딩되는데 이를 재배치라 한다.

즉, PE파일이 ImageBase 주소에 로딩되지 못하고 다른 주소에 로딩될 때 수행되는 일련의 작업들을 의미한다.

<br>

단, MS에서는 각 OS의 주요 시스템 DLL들에 대해 버전별로 각자 고유한 ImageBase를 부여했다고 한다.

그래서 한 시스템에서 kernel32.dll, user32.dll 등은 각자만의 고유 ImageBase에 로딩되기 떄문에 실제로 시스템 DLL들끼리는 재배치가 발생할 일이 없다고 한다.

<br>

그래서 DLL/SYS 파일은 위에처럼 이미 해당 ImageBase 주소에 다른 파일이 로딩되있다면 빈 공간에 로딩이 된다는 것.

EXE 파일의 경우 원래는 가장 먼저 메모리에 로딩되어서 재배치를 고려할 필요가 없었으나 ASLR 기능이 추가되어서 파일이 실행될 때마다 랜덤 위치에 로딩된다.

DLL/SYS 파일 또한 ASLR이 적용되어서 고유한 ImageBase를 가지고 있지만 로딩 주소는 **매 부팅 시마다** 달라진다. 

즉, 시스템이 살아있는 동안에는 프로세스마다 같은 주소에 매핑되어 있다고 보면 된다.

윈도우는 DLL이 최초로 메모리에 올라갈 때 로딩이라 하고 그 후 사용될 때는 기존에 로딩되어 있는 DLL의 코드와 리소스를 매핑하는 방식을 사용하여 메모리를 효율적으로 사용한다.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## PE 재배치 발생시 수행되는 작업
<hr style="border-top: 1px solid;"><br>

Link
: <a href="https://chive0402.tistory.com/5" target="_blank">chive0402.tistory.com/5</a>

<br>

그니까 ASLR 기능으로 인해 프로그램이 실행 시 로딩되는 주소가 랜덤으로 바뀐다.

이에 맞춰서 프로그램에 하드코딩 되어 있던 메모리 주소들도 로딩되는 주소들에 맞춰서 변경된다. 이것이 PE 재배치다.

즉, ImageBase가 ```0x0100 0000```이고 프로그램이 로딩된 주소가 ```0x0028 0000```이라고 치자. ()

프로그램이 하드코딩된 주소들을 Hxd(파일 형태)로 확인했을 때는 각각 ```0x0100 10FC, 0x0100 1100, 0x0100 C0A4```이다.

메모리에 올라갔을 때의 주소로 확인해보면 각각 ```0x0028 10FC, 0x0028 1100, 0x0028 C0A4```로 오프셋에 맞춰서 변경된 것을 볼 수 있다.

<br>

하드코딩된 주소들은 ImageBase 주소를 기준으로 되어있는데 왜냐하면 프로세스가 실행될 때 실제 어느 주소로 로딩될 지 모르기 때문이다.

하지만 실행되는 순간 PE 재배치 과정을 거치면서 이 주소들은 로딩된 랜덤 주소에 맞춰서 변경이 된다. (물론 각각의 오프셋만큼만 변경됨)

그래서 에러없이 정상적으로 실행되는 것이다.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## PE 재배치 동작 원리
<hr style="border-top: 1px solid;"><br>

재배치 기본 동작 원리는 위에도 써놨지만 절차를 쓰면 다음과 같다.

<br>

+ 프로그램에서 하드코딩된 주소 위치를 찾는다.
+ 값을 읽은 후 ImageBase만큼 뺀다. (VA -> RVA)
+ 실제 로딩 주소를 더한다. (RVA -> VA)

<br>

핵심은 하드코딩된 주소 위치를 찾는 것인데, 이를 위해 PE 파일 내부에 Relocation Table이라고 하는 하드코딩 주소들의 오프셋(위치)을 모아 놓은 목록이 존재한다. 

Relocation Table은 PE 파일 빌드 과정에서 제공되며, 여기를 찾아가는 방법은 PE 헤더의 Base Relocation Table 항목을 따라가는 것이다.

그에 대한 설명은 여기가 더 나은 것 같다.
: <a href="https://maple19out.tistory.com/25" target="_blank">maple19out.tistory.com/25</a>

<br><br>
<hr style="border: 2px solid;">
<br><br>