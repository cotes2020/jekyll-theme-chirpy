---
title : UPack PE 헤더 분석
categories : [Hacking, Reversing]
tags : [Reversecore, UPack PE Header 분석]
---

## UPACK PE 헤더 분석
<hr style="border-top: 1px solid;"><br>

Link
: <a href="https://maple19out.tistory.com/28" target="_blank">maple19out.tistory.com/28</a>
: <a href="https://chive0402.tistory.com/7" target="_blank">chive0402.tistory.com/7</a>

<br><br>
<hr style="border: 2px solid;">
<br><br>

### SizeOfOptionalHeader
<hr style="border-top: 1px solid;"><br>

대부분 설명은 위에 써있지만 책에 나와있는 내용을 조금 더 추가하였다.

```IMAGE_FILE_HEADER```의 ```SizeOfOptionalHeader``` 부분을 수정하는 부분에서 "```IMAGE_OPTIONAL_HEADER```는 구조체이므로 수정이 불가능하지 않냐" 생각될 수 있는데 왜 PE 파일 설계자들은 구조체의 크기를 따로 입력하게 하였는가?

원래 의도는 PE 파일의 형태에 따라서 각각 다른 ```IMAGE_OPTIONAL_HEADER``` 형태의 구조체를 바꿔 낄 수 있도록 설계한 것이다.

그러니까 ```IMAGE_OPTIONAL_HEADER``` 구조체의 종류가 여러 개이므로 구조체의 크기를 따로 입력할 필요가 있는 것이다.

예를 들면 64비트용에서는 이 구조체의 크기는 F0이지만 32비트에서는 E0이다.

<br>

또한 ```IMAGE_FILE_HEADER```의 ```SizeOfOptionalHeader```의 또 다른 의미는 섹션 헤더(```IMAGE_SECTION_HEADER```)의 시작 오프셋을 결정하는 것이다.

PE 헤더를 그냥 보면 ```IMAGE_OPTIONAL_HEADER``` 다음에 ```IMAGE_SECTION_HEADER```가 나타나는 것으로 보이지만 정확히는 ```IMAGE_OPTIONAL_HEADER 시작 오프셋 + SizeOfOptionalHeader``` 값을 더한 위치부터 ```IMAGE_SECTION_HEADER```가 시작하게 된다.

<br>

UPack은 왜 이 값을 기존 값보다 더 큰(E0, F0) 148로 바꿨는가? (책, 위 블로그에 있음) 

위의 내용처럼 이 값을 늘리면 두 구조체 사이의 공간은 늘어나게 된다.

**UPack은 바로 이 늘어난 공간에 Decoding Code를 추가한다.**

<br><br>
<hr style="border: 2px solid;">
<br><br>

### NumberOfRvaAndSizes
<hr style="border-top: 1px solid;"><br>

이 값은 ```IMAGE_OPTIONAL_HEADER``` 구조체의 멤버이다.

이 값을 변경하는 이유 또한 헤더에 자신의 코드를 삽입하기 위한 목적이다.

<br>

이 값의 의미는 바로 뒤에 이어지는 IMAGE_DATA_DIRECTORY 구조체 배열의 원소 개수를 의미하는데, 이 값을 기존의 0x10보다 작은 값인 0xA로 변경한다.

즉, 이 값을 줄였다는 것은 나머지 6개의 원소들은 무시된다.

UPack은 무시된 영역에 자신의 코드를 덮어써서 사용한다. 

<br>

근데 의문점은 원소들 중에 잘못 변경하면 실행에러가 발생하는 원소들이 있다는데 왜 문제가 없는지? 

이러한 의문점들을 2번째 링크 블로그에서 잘 정리해주셨다.

<br><br>
<hr style="border: 2px solid;">
<br><br>

### RVA to RAW
<hr style="border-top: 1px solid;"><br>

정상적인 RVA -> RAW 방법을 이용하면 에러가 발생한다. 이거때문에 각종 유틸리티가 에러가 났었던 것이다.

책을 보면 RVA 1018를 예시로 드는데 이 RVA 1018은 첫 번째 섹션에 속하고 첫 번째 섹션의 PointerToRawData 값은 10이라고 한다.

그래서 정상적인 변환 방법을 이용해서 RAW를 계산했을 때의 영역을 살펴보면 이상한 곳을 보게 된다.

<br>

왜냐하면 UPack은 FileAlignment 값을 200으로 바꾼다고 한다.

따라서 PointerToRawData 값도 200의 배수가 되어야하므로 0, 200, 400, 600 등의 값이 되어야 한다.

두 번째 블로그에 있는 내용인데, PointerToRawData를 FileAlignment에 맞출 땐 반드시 내려야 한다고 한다.

<br><br>
<hr style="border: 2px solid;">
<br><br>