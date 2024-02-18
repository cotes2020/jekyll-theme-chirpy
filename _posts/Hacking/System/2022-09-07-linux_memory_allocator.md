---
title: ptmalloc2
date: 2022-09-07 19:48 +0900
categories: [Hacking, System]
tags: [ptmalloc2, unsorted bin]
---

## 내용 출처
<hr style="border-top: 1px solid;"><br>

Dreamhack
: <a href="https://dreamhack.io/lecture/courses/98" target="_blank">Background:ptmalloc2</a>

<br><br>
<hr style="border: 2px solid;">
<br><br>

## ptmalloc2
<hr style="border-top: 1px solid;"><br>

Memory Allocator로 **리눅스에서는 ptmalloc2 알고리즘**을 사용한다.

ptmalloc2는 어떤 메모리가 해제되면, **해제된 메모리의 특징을 기억**하고 있다가 **비슷한 메모리의 할당 요청이 발생하면 이를 빠르게 반환**해준다.

<br>

ptmalloc은 리눅스에서 사용하고 있으며, **GLibc에 구현**돼 있다.

ptmalloc의 구현 **목표는 메모리의 효율적인 관리**이다.

즉, 메모리 낭비 방지, 빠른 메모리 재사용, 메모리 단편화 방지를 추구한다.

<br><br>

### 메모리 낭비 방지
<hr style="border-top: 1px solid;"><br>

메모리 동적 할당 및 해제는 매우 빈번하게 발생하지만, **메모리는 한정**되있다. 

따라서 ptmalloc은 메모리 할당 요청이 발생하면, 먼저 **해제된 메모리 공간 중에서 재사용할 수 있는 공간이 있는지 탐색**한다.

해제된 메모리 공간 중에서 **요청된 크기와 같은 크기의 메모리 공간이 있다면 이를 그대로 재사용**한다.

작은 크기의 할당 요청이 발생했을 때, 해제된 메모리 공간 중 매우 큰 메모리 공간이 있으면 그 영역을 나누어 주기도 한다.

<br><br>

### 빠른 메모리 재사용
<hr style="border-top: 1px solid;"><br>

운영체제가 프로세스에게 제공해주는 가상 메모리 공간은 매우 넓다.

따라서 특정 메모리 공간을 해제한 이후에 이를 빠르게 재사용하려면 해제된 메모리 공간의 주소를 기억하고 있어야 한다.

이를 위해 ptmalloc은 메모리 공간을 해제할 때, tcache 또는 bin이라는 연결 리스트에 해제된 공간의 정보를 저장한다.

tcache와 bin은 여러 개가 정의되어 있으며, 각각은 서로 다른 크기의 메모리 공간들을 저장한다.

이렇게 하면 특정 크기의 할당 요청이 발생했을 때, 그 크기와 관련된 저장소만 탐색하면 되므로 더욱 효율적으로 공간을 재사용할 수 있다.

<br><br>

### 메모리 단편화 방지
<hr style="border-top: 1px solid;"><br>

내부 단편화는 할당한 메모리 공간의 크기에 비해 실제 데이터가 점유하는 공간이 적을 때 발생

외부 단편화는 할당한 메모리 공간들 사이에 공간이 많아서 발생하는 비효율을 의미
 
전체 메모리 공간이 부분적으로 점유되기 때문에, ptmalloc은 단편화를 줄이기 위해 **정렬(Alignment)**과 **병합(Coalescence)** 그리고 **분할(Split)**을 사용한다.

<br>

64비트 환경에서 ptmalloc은 **메모리 공간을 16바이트 단위로 할당(정렬)**한다.

사용자가 어떤 크기의 메모리 공간을 요청하면, 그보다 조금 크거나 같은 16바이트 단위의 메모리 공간을 제공한다.

예를 들면, 4바이트 요청하면 16바이트 공간을, 17바이트를 요청하면 32바이트를 제공한다.

비슷한 크기의 요청에 대해서는 모두 같은 크기의 공간을 반환해야 해제된 청크들의 재사용률을 높이고, 외부 단편화도 줄일 수 있다.

<br>

ptmalloc은 특정 조건을 만족하면 해제된 공간들을 **병합**하기도 한다.

병합으로 생성된 큰 공간은 그 공간과 같은 크기의 요청에 의해, 또는 그보다 작은 요청에 의해 분할되어 재사용된다.

잘게 나뉜 영역을 병합하고, 필요할 때 구역을 다시 설정함으로써 해제된 공간의 재사용률을 높이고, 외부 단편화를 줄일 수 있다.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## ptmalloc 
<hr style="border-top: 1px solid;"><br>

ptmalloc2는 청크(Chunk), bin, tcache, arena를 주요 객체로 사용한다.

<br><br>

### chunk
<hr style="border-top: 1px solid;"><br>

![image](https://user-images.githubusercontent.com/52172169/188871597-3b51e242-10cf-4499-a8a3-a6d4e28c5221.png)

<br>

위의 사진은 청크의 구조이다.

청크는 ptmalloc이 할당한 메모리 공간으로 헤더와 데이터로 구성된다.

헤더는 청크의 상태를 나타내므로 사용 중인 청크와 해제된 청크의 구조는 다르다.

사용 중인 청크는 fd와 bk를 사용하지 않고, 그 영역에 사용자가 입력한 데이터를 저장한다.

<br>

청크 헤더의 요소의 특징은 아래와 같다.

+ prev_size
  + 8바이트
  + 인접한 직전 청크의 크기. 
  + 청크를 병합할 때 직전 청크를 찾는 데 사용 

<br>

+ size
  + 8바이트
  + 현재 청크의 크기 (헤더의 크기도 포함한 값)
  + 64비트 환경에서, 사용 중인 청크 헤더의 크기는 16바이트이므로 사용자가 요청한 크기를 정렬하고, 그 값에 16바이트를 더한 값

<br>

+ flags
  + 3비트
  + 64비트 환경에서 청크는 16바이트 단위로 할당되므로, size의 하위 4비트는 의미 X
  + ptmalloc은 size의 하위 3비트를 청크 관리에 필요한 플래그 값으로 사용
    + prev-in-use 플래그는 직전 청크가 사용 중인지를 나타내므로, ptmalloc은 이 플래그를 참조하여 병합이 필요한지 판단

<br>

+ fd
  + 8바이트
  + 해제된 청크에 있고, 연결 리스트에서 다음 청크를 가리킴

<br>

+ bk
  + 8바이트
  + 해제된 청크에 있고, 연결 리스트에서 이전 청크를 가리킴 

<br><br>

### bin
<hr style="border-top: 1px solid;"><br>

![image](https://user-images.githubusercontent.com/52172169/188895150-913e81bc-c76a-4536-a10b-8661d0f3c04b.png)

<br>

bin은 문자 그대로, 사용이 끝난 청크들이 저장되는 객체이다.

메모리의 낭비를 막고, 해제된 청크를 빠르게 재사용할 수 있게 한다.

ptmalloc에는 총 128개의 bin이 정의되어 있고, 이 중 62개가 smallbin, 63개는 largebin, 1개는 unsortedbin으로 사용되고, 나머지 2개는 사용되지 않는다.

<br><br>

#### smallbin
<hr style="border-top: 1px solid;"><br>

**smallbin에는 32 바이트 이상 1024 바이트 미만의 크기를 갖는 청크들이 보관**된다.

하나의 smallbin에는 같은 크기의 청크들만 보관되며, index가 증가하면 저장되는 청크들의 크기는 16바이트씩 커진다.

즉, ```smallbin[0]```는 32바이트 크기의 청크를, ..., ```smallbin[61]```은 1008 바이트 크기의 청크를 보관한다.

<br>

smallbin은 원형 이중 연결 리스트이며, 먼저 해제된 청크가 먼저 재할당된다. (FIFO)

연결 리스트 특성 상 청크를 추가하거나 해제할 때, 연결 고리를 끊어야 하는데 이 과정을 unlink라고 한다.

smallbin의 청크들은 병합 대상으로, 메모리상에서 인접한 두 청크가 해제되어 있고, smallbin에 있다면 이 둘은 병합된다.

<br><br>

#### fastbin
<hr style="border-top: 1px solid;"><br>

일반적으로 크기가 작은 청크들이 큰 청크들보다 빈번하게 할당되고 해제된다.

fastbin은 **smallbin보다 더 작은, 어떤 정해진 크기보다 작은 청크들을 관리**한다.

특이한 점은 fastbin에서는 메모리 단편화보다 속도를 조금 더 우선순위에 둔다고 한다.

<br>

**fastbin에는 32 바이트 이상 176 바이트 이하 크기의 청크들이 보관**되며, 이에 따라 16바이트 단위로 총 10개의 fastbin이 있다.

리눅스는 이 중에서 작은 크기부터 7개의 fastbin만을 사용합니다. 

즉, **리눅스에서는 32바이트 이상, 128바이트 이하의 청크들을 fastbin에 저장**한다.

<br>

fastbin은 단일 연결 리스트로, unlink 과정이 필요 없으며 FILO 방법을 이용한다.

즉, 먼저 해제된 것이 나중에 재할당되고, 나중에 해제된 것이 먼저 할당된다.

<br>

또한 fastbin에 저장되는 청크들은 서로 병합되지 않는다.

<br><br>

#### largebin
<hr style="border-top: 1px solid;"><br>

**largebin은 1024 바이트 이상의 크기를 갖는 청크들이 보관**된다.

총 63개의 largebin이 있으며, **한 largebin에서 일정 범위 안의 크기를 갖는 청크들을 모두 보관**한다.

예를 들어, ```largebin[0]```는 1024 바이트 이상, 1088 바이트 미만의 청크를 보관하며, ```largebin[32]```는 3072 바이트 이상, 3584 바이트 미만의 청크를 보관한다.

이 범위는 largebin의 인덱스가 증가하면 로그적으로 증가한다.

<br>

largebin은 범위에 해당하는 모든 청크를 보관하기 때문에, 재할당 요청이 발생했을 때 ptmalloc은 그 안에서 크기가 가장 비슷한 청크(best-fit)를 꺼내 재할당한다.

이 과정을 빠르게 하려고 ptmalloc은 largebin 안의 청크를 크기를 기준으로 내림차순으로 정렬한다.

<br>

largebin은 이중 연결 리스트이므로 재할당 과정에서 unlink 과정이 있다.

또한, 연속된 largebin 청크들은 병합의 대상이 된다.

<br><br>

#### unsorted bin
<hr style="border-top: 1px solid;"><br>

분류되지 않은 청크들을 보관하는 bin이다.

unsortedbin은 하나만 존재하며, fastbin에 들어가지 않는 모든 청크들은 해제되었을 때 크기를 구분하지 않고 unsortedbin에 보관된다.

unsortedbin은 원형 이중 연결 리스트이며 내부적으로 정렬되지는 않는다.

<br>

**smallbin 크기에 해당하는 청크를 할당 요청하면, ptmalloc은 fastbin 또는 smallbin을 탐색한 뒤 unsorted bin을 탐색**한다.

**largebin의 크기에 해당하는 청크는 unsortedbin을 먼저 탐색**한다.

unsortedbin에서 적절한 청크가 발견되면 해당 청크를 꺼내어 사용하고, 이 과정에서 탐색된 청크들은 크기에 따라 적절한 bin으로 분류된다.

<br><br>

### arena
<hr style="border-top: 1px solid;"><br>

arena는 fastbin, smallbin, largebin 등의 정보를 모두 담고 있는 객체이다.

ptmalloc은 최대 64개의 arena를 생성할 수 있게 하고 있다.

<br><br>

### tcache
<hr style="border-top: 1px solid;"><br>

tcache(thread local cache)는 각 쓰레드에 독립적으로 할당되는 캐시 저장소이다.

tcache는 GLibc 버전 2.26에서 도입되었으며, 멀티 쓰레드 환경에 더욱 최적화된 메모리 관리 메커니즘을 제공한다.

<br>

각 쓰레드는 64개의 tcache를 가지고 있다.

tcache는 fastbin과 마찬가지로 LIFO 방식으로 사용되는 단일 연결리스트이며, 하나의 tcache는 같은 크기의 청크들만 보관한다.

리눅스는 각 tcache에 보관할 수 있는 청크의 갯수를 7개로 제한한다.

이는 쓰레드마다 정의되는 tcache의 특성상, 무제한으로 청크를 연결할 수 있으면 메모리가 낭비될 수 있기 때문이라고 한다.

<br>

tcache에 들어간 청크들은 병합되지 않는다.

<br>

**tcache에는 32 바이트 이상, 1040 바이트 이하의 크기를 갖는 청크들이 보관**된다.

**이 범위에 속하는 청크들은 할당 및 해제될 때 tcache를 가장 먼저 조회**한다.

**청크가 보관될 tcache가 가득찼을 경우에는 적절한 bin으로 분류**된다.

<br>

**arena의 bin에 접근하기 전에 tcache를 먼저 사용**하므로 arena에서 발생할 수 있는 병목 현상을 완화하는 효과가 있다.

tcache는 보안 검사가 많이 생략되어 있어서 공격자들에게 힙 익스플로잇의 좋은 도구로 활용되고 있다고 한다.

<br><br>
<hr style="border: 2px solid;">
<br><br>
