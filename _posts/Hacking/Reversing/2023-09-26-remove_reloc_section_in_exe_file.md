---
title : 실행 파일에서 reloc 섹션 제거하기 
categories : [Hacking, Reversing]
tags : [Reversecore, reloc]
---

## reloc 섹션
<hr style="border-top: 1px solid;"><br>

EXE 형식의 PE 파일에서 Base Relocation Table 항목은 실행에 큰 영향을 미치지 않는다고 한다.

제거 후 실행했을 때 정상적으로 실행된다. (메모리에 먼저 올라가서 그런가?)

이유는 다음과 같다.

<br>

출처
: <a href="https://geun-yeong.tistory.com/39" target="_blank">geun-yeong.tistory.com/39</a>
: EXE 파일의 경우 자신만의 공간을 할당받기 때문에 ASLR을 사용하도록 빌드한 경우가 아니면 .reloc 섹션은 만들어지지 않는다. 
  DLL의 경우 다른 프로세스도 사용할 수 있기 때문에 공용 공간에 DLL이 로드되므로 자신의 ImageBase가 아닌 다른 ImageBase를 할당받을 수도 있다. 이처럼 ImageBase 값이 변할 경우 이를 대처하기 위한 섹션이 .reloc 섹션이다.

<br>

reloc 섹션은 보통 마지막에 위치하므로 제거하기 어렵지 않다.

과정은 다음과 같다.

+ ```.reloc``` 섹션 헤더 정리
+ ```.reloc``` 섹션 제거
+ ```IMAGE_FILE_HEADER``` 수정
+ ```IMAGE_OPTIONAL_HEADER``` 수정

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 
<hr style="border-top: 1px solid;"><br>



<br><br>
<hr style="border: 2px solid;">
<br><br>