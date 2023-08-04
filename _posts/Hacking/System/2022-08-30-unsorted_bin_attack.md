---
title: Unsorted Bin Attack (Incomplete)
date: 2022-08-30 23:17 +0900
categories: [Hacking, System]
tags: [ptmalloc2, tcache, unsorted bin attack, malloc bin chunk, dreamhack uaf overwrite, Incomplete]
---

## 내용 출처
<hr style="border-top: 1px solid;"><br>

uaf_overwrite 풀이에서 unsorted bin이 나오길래 정리.

<br>

내용 출처
: <a href="https://www.lazenca.net/pages/viewpage.action?pageId=1148135" target="_blank">lazenca.net/pages/viewpage.action?pageId=1148135</a>
: <a href="https://umbum.dev/386" target="_blank">umbum.dev/386</a>
: <a href="https://cyber0946.tistory.com/101" target="_blank">cyber0946.tistory.com/101</a>
: <a href="https://wyv3rn.tistory.com/73" target="_blank">wyv3rn.tistory.com/73</a>  ---> GOOD

<br><br>
<hr style="border: 2px solid;">
<br><br>


## heap tcache & main_arena
<hr style="border-top: 1px solid;"><br>

tcache는 glibc 2.26 이상부터 적용됬다고 한다.

heap 영역에 tcache와 bins가 존재하는데 32 bit 에서는 516 byte 이하, 64 bit 에서는 1032 byte 이하의 사이즈가 할당되었을때 tcache를 사용하고 그 외에 bins를 사용한다.

이 bins를 관리하는 것이 ```main_arena```이다.

밑에서도 설명하지만, arena 자체는 heap에 없고 libc의 data segment에 있어서 주소를 leak할 수 있게 된다면, ```main_arena```의 주소 또한 leak 할 수 있게 된다.

그래서 ```main_arena``` 주소를 이용해 libc의 베이스 주소, 오프셋을 구할 수 있다.

**중요한 사실로 ```main_arena```의 주소는 못찾을 수 있는데, ```__malloc_hook + 0x10```에 위치한다고 한다.**

<br><br>
<hr style="border: 2px solid;">
<br><br>


## Unsorted Bin
<hr style="border-top: 1px solid;"><br>

unsorted bin attack을 이해하기 위해서는 malloc 함수가 unsorted bin에 등록된 chunk를 재할당 할때 해당 chunk를 unsorted bin의 list에서 삭제하는 방식에 대해 이해가 필요하다고 한다.

애초에 Heap 영역이 요청에 따라 할당되는 chunk 형태로 나뉠 수 있는 인접한 연속된 메모리 영역이라고 한다.

여기서 chunk는 malloc으로 8바이트 (64bit에서는 16바이트) 형태로 할당되는 영역이라고 한다.

그리고 이러한 chunk는 하나의 Heap 내부에 있으며 당연히 하나의 arena에 속한다고 한다.

<br>

arena는 heap chunk들의 연결 리스트(이를 bins라고 한다)를 포함하고 있는 구조를 뜻한다.

이 arena는 (물론 main_arena도 포함) heap에 존재하는게 아니라 libc.so의 data segment에 존재한다.

top chunk는 아직 할당되지 않는 힙의 최상단에 위치한 chunk를 뜻하는데, top chunk가 아닐 경우 chunk가 free가 되면 바로 계산되어 small bin이나 large bin으로 들어가는 게 아니라, 일단 Unsorted bin으로 들어가서 재할당을 기다리게 된다고 한다.

top chunk는 재사용 가능한 chunk가 없을 때 할당 요청이 오면 Top chunk에서 분리해서 영역을 반환해준다.

**즉, 동일한 크기의 chunk가 free되어 있을 때, Top Chunk로부터 영역을 받는 것이 아닌 해제되어 있는 영역을 할당해준다.**

또한 Top chunk와 인접한 chunk가 free되면 병합하며, 일반적으로 Top chunk의 크기는 0x21000이고, Top Chunk의 크기보다 큰 할당은 불가능하다고 한다.

<br>

즉, 위에서 말했듯이 unsorted bin attack을 이해하기 위해서는 malloc 함수가 unsorted bin에 등록된 chunk를 재할당 할 때, 해당 chunk를 unsorted bin의 list에서 삭제하는 방식에 대해 이해가 필요하다.

이 bins(연결 리스트)에는 fast bin, small bin, large bin이 있는데, 메모리 할당을 malloc()에 요청하면, 할당자는 요청한 메모리의 크기가 fast bin, small bin, large bin에서 사용 가능한 chunk가 있는지 확인한다고 한다.

할당자는 해당 bins에서 사용가능한 chunk를 찾지 못하면, ```unsorted_chunks (av)-> bk```가 가지고 있는 값과 ```unsorted_chunks (av)```의 반환값이 다른 값인지 확인한다.

<br>

```c
// /release/2.25/master/malloc/malloc.c - _int_malloc()

for (;; )
  {
    int iters = 0;
    while ((victim = unsorted_chunks (av)->bk) != unsorted_chunks (av))
      {
        bck = victim->bk;
```

<br>

이 값들이 서로 다르다면 Unsorted bin에는 free chunk가 있다는 것을 나타내는데, Unsorted bin에 free chunk가 있다면 ```unsorted_chunks->bk```의 값을 ```victim```에 저장하고 ```victim->bk```가 가지고 있는 데이터는 ```bck```에 저장한다.

자세한 과정은 lazenca님께서 정리한 글을 참조하면 된다.

<br>

나머지 과정을 요약하면 다음과 같다...

할당자는 Unsorted list에 배치된 chunk를 재할당 한 후 chunk를 unsorted list에서 삭제할 때, ```bck```가 가지고 있는 데이터를 ```main_arena.bin[0]->bk```에 저장.

그리고 할당자는 ```unsorted_chunks()```이 반환한 값을 ```bck→fd```에 저장.

<br>

unsorted bin attack은 할당자가 Unsorted list에서 chunk를 삭제하기 전에, free chunk의 bk에 값을 덮어써서 원하는 영역에 main_arena의 주소```(&main_arena.bin[0] - 16)```를 저장하는 기술이라고 한다.



<br><br>
<hr style="border: 2px solid;">
<br><br>

## Unsorted Bin Attack
<hr style="border-top: 1px solid;"><br>



<br><br>
<hr style="border: 2px solid;">
<br><br>
