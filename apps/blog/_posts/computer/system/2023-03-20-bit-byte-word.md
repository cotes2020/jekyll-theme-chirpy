---
title: "Bit Byte Word"
# description: ""
categories: [컴퓨터, 시스템]
tags: []
image: "/assets/img/background/kururu-lab.jpg"

date: 2023-03-20. 16:53
# last_modified_at: 2023-04-06. 11:16
# last_modified_at: 2023-11-08. 15:03
# last_modified_at: 2023-11-26. 01:52
# last_modified_at: 2024-08-29. 22:10
last_modified_at: 2024-11-14. 00:32 # -1K=1024, -BinaryDigit
---

글 계승.
`2020-10-12 03:33:00: 32-Bit, 64-Bit 관련 글 스크랩 (네이버 블로그)`  

## Byte = 8 Bit ?

---

Byte는 중요하다.  
Byte는 CPU가 한번에 처리하는 데이터 크기 = Word(TDU)의 기준이 되고,  
이는 컴퓨터가 데이터를 다루는 기본 단위, 메모리 주소의 크기의 기준이 된다.  

Byte의 크기 역시 중요하다.  
현대 컴퓨터 아키텍쳐에서, Byte = 8 Bit 가 표준.  

본래 Byte는 컴퓨터에서 문자 Character 하나를 표현하기 위한 Bit 수였는데,  
이가 확장되어, '디지털 정보의 가장 작은 단위'가 되었다.  

과거 Byte의 크기는 HW에 종속되었고, 명확한 표준이 없어, 곳에 따라 1 ~ 48 Bit 등의 다양한 크기로 사용되었다고 한다.  

그 중에서도, 6-Bit를 사용하는 문자 표현 방식이 주로 사용되었고, 1960년대에는 6-Bit, 9-Bit 를 사용하는 컴퓨터가 일반적이었다고 함.  
이런 컴퓨터들은 2, 3, 4, 5, 6, 8, 10 6-Bit Byte에 상응하는, 12, 18, 24, 30, 36, 48, 60 Bit의 Memory Word를 주로 사용했음.  
이 시대엔 이런 Bit Groupings를 Syllables, Slab 등으로 불렀음. (Byte가 일반화되기 전까지)  

ASCII Code는 7-Bit만으로도 필요로 하는 문자를 모두 표현 할 수 있었는데,  
2-진수 Binary를 사용하는 컴퓨터 아키텍쳐 특성상, 편리하게 2-배수로 만들기 위해,  
7-Bit에 1-Bit를 더하여 8-Bit로 만들어 사용  

IBM의 System/360 컴퓨터가 이런 8-Bit Byte의 시초.  
ISO/IEC 2382-1:1933 에 문서화 되었으며,  
8-Bit Byte 마이크로프로세서가 득세한 70년대부터 표준으로 굳어지기 시작함 (사실상 표준 De Facto Standard)  

이런 8-Biy Byte에 기반하여 8의 배수인, 8-Bit, 16-Bit, 32-Bit, 64-Bit Words를 사용하게 됨.  

ISO International Organization for Standardization  
IEC International Electrotechnical Organization  

[참고-0](https://softwareengineering.stackexchange.com/questions/120126/what-is-the-history-of-why-bytes-are-eight-bits)  
[참고-1](https://en.wikipedia.org/wiki/Byte)  

## 32-Bit, 64-Bit ?

---

Program Counter  
32-Bit = 2^32 = 4,294,967,296  
64-Bit = 2^64 = 18,446,744,073,709,551,616  
32-Bit = 약 4-GB 메모리  
64-Bit = 약 256-TB 메모리 (48-Bit만 사용)  

왜 48-Bit만 사용하냐면, '일반적으로', 256-TB 이상의 주소 공간을 사용하지 않기 때문이다.  

운영체제도 32-Bit, 64-Bit 로 나뉜다.  
32-Bit CPU 에는 64-Bit 운영체제가 동작하지 않는다.  
64-Bit CPU 에는 32-Bit 운영체제가 동작하기는 하지만, 하위 호환 Backward Compatibility 된다.  

앱 역시 32-Bit, 64-Bit 로 나뉜다.  
32-Bit 운영체제에는 64-Bit 앱 (Programs File)이 동작하지 않는다.  
64-Bit 운영체제에는 32-Bit 앱 (Programs File (x86))이 동작하기는 하지만, 하위 호환 Backward Compatibility 된다.  

[참고-0](https://blog.naver.com/sharpsoul/221777128846)  
[참고-1](https://eine.tistory.com/entry/64%EB%B9%84%ED%8A%B8-32%EB%B9%84%ED%8A%B8-CPU%EC%99%80-%EC%9A%B4%EC%98%81%EC%B2%B4%EC%A0%9C-%EC%97%90-%EB%8C%80%ED%95%98%EC%97%AC)  

## x86, x64 (x86-64) ?

---

x85 = 8-bit  
x86 = 32-Bit (일반적으로)  
x64 = 64-Bit (x86-64)  

x86 (80x86) =  
1978년 인텔이 개발한 인텔 8086에 적용된 아키텍쳐,  
그 호환 프로세서와 후속작 (8086의 명령어 세트를 기반하여 확장한, 386, 486, ...  )  

IA-16, IA-32, IA-64 를 모두 포함하는 단어이지만,  
일반적으로 x86이라 하면 IA-32을 지칭  

IA = Intel Architecture  

[참고-0](https://ko.wikipedia.org/wiki/X86)  

## 워드 WORD ?

---

기계어 명령어나 연산을 통해 저장된 장치로부터 레지스터에 옮겨 놓을 수 있는 데이터 단위  
= CPU가 처리할 수 있는, 버스에 한 번에 지나갈 수 있는 크기의 단위  
= DTU Data Transport Unit (DTU를 사용하는 용어가 많았기에 WORD를 사용하기 시작)  

컴퓨터 아키텍쳐에서,  
32-Bit = WORD: 32-Bit  
64-Bit = WORD: 64-Bit  

반먄 프로그래밍에서,  
Win32 API의 WORD는 16-Bit다.  

왜 Why  

IA = Intel Architecture  
IA-16의 기본 처리 단위 DTU = WORD = 16-Bit  
추후 32-Bit, 64-Bit 등의 프로세서 (IA-32, IA-64 등) 등장  

호환성의 문제으로 인해, 기존 단위 크기를 바꿀 수는 없고,  
때문에 기존 16-Bit Word를 기반으로 한 새로운 단위를 만들어 썼다.  

DWORD = Double Word = 32-Bit  
QWORD = Quad/Quotable Word = 64-Bit  

[참고-0](https://bebesoft.tistory.com/12?category=887595)  
