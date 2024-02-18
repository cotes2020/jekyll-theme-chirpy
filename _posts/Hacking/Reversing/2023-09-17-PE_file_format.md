---
title : Reversing 내용 정리
categories : [Hacking, Reversing]
tags : [Reversecore, PE File Format]
---

## PE File Format
<hr style="border-top: 1px solid;"><br>

패치: 프로그램의 파일 혹은 실행 중인 프로세스 메모리의 내용을 변경하는 작업

크랙: 패치와 의미가 동일하지만 악의적인 의도인 작업

<br>

Link
: <a href="https://majg.tistory.com/16" target="_blank">majg.tistory.com/16</a>

<br>

위의 링크에도 설명하지만 NT Header 부분에서 32비트에서는 HEADER32를 쓰고 64비트에서는 HEADER64를 쓴다.

그에 대한 보충 설명은 아래에서 확인, Machine 값에 대해서도 설명해줌.
: <a href="https://ddatg.tistory.com/m/61" target="_blank">ddatg.tistory.com/m/61</a>

<br><br>
<hr style="border: 2px solid;">
<br><br>

## RVA to RAW
<hr style="border-top: 1px solid;"><br>

Link: <a href="https://rninche01.tistory.com/entry/RVA-to-RAW" target="_blank">rninche01.tistory.com/entry/RVA-to-RAW</a>

<br>

PE 파일이 메모리에 로딩되었을 때 각 섹션에서 메모리의 주소(RVA, 상대주소)와 파일 옵셋(RAW)을 잘 매핑할 수 있어야 한다.

이 매핑을 RVA to RAW라고 하며, 방법은 아래와 같다.


1. RVA가 속해 있는 섹션을 찾는다.

2. 간단한 비례식을 사용해서 파일 옵셋(RAW)을 계산한다.
    + ```VA = RVA + ImageBase```
    + ```RAW = RVA - VirtualAddress(VA) + PointerToRawData```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## IAT (Import Address Table)
<hr style="border-top: 1px solid;"><br>

IAT에는 윈도우 운영체제 핵심 개념인 process, memory, DLL 구조 등에 대한 내용이 함축되어 있다.

<br>

Link: <a href="https://rninche01.tistory.com/entry/IATImport-Address-Table" target="_blank">rninche01.tistory.com/entry/IATImport-Address-Table</a>

<br>

DLL이란 ```Dynamic Linked Library```로 ```동적 연결 라이브러리```라 하는데, linux의 ```.so``` 파일과 같은 용도

DLL 로딩 방식은 2가지이다.
+ 프로그램에서 사용되는 순간에 로딩하고 사용이 끝나면 메모리에서 해제되는 방법(Explicit Linking)
+ 프로그램 시작할 때 같이 로딩되어 프로그램 종료할 때 메모리에서 해제되는 방법(Implicit Linking)

<br>

DLL의 ImageBase는 ```0x10000000```이지만 DLL Relocation으로 인해 PE헤더에 명시된 ImageBase에 로딩된다고 보장할 수 없다.

단, DLL 시스템 파일들(kernel32, user32, gdi32 등)은 자신만의 고유한 ImageBase가 있어서 DLL Relocation이 발생하지 않는다고 한다.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## EAT (Export Address Table)
<hr style="border-top: 1px solid;"><br>

Link
: <a href="https://rninche01.tistory.com/entry/EATExport-Address-Table-1" target="_blank">rninche01.tistory.com/entry/EATExport-Address-Table-1</a>

<br><br>
<hr style="border: 2px solid;">
<br><br>