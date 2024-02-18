---
title: Double Free Bug And Bypass
date: 2022-09-08 00:29 +0900
categories: [Hacking, System]
tags: [Double Free Bug, tcache, tcache duplication, tcache poisoning]
---

## 출처
<hr style="border-top: 1px solid;"><br>

Dreamhack
: <a href="https://dreamhack.io/lecture/courses/116" target="_blank">Memory Corruption: Double Free Bug</a>
: <a href="https://dreamhack.io/lecture/courses/107" target="_blank">Exploit Tech: Tcache Poisoning</a>

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 서론
<hr style="border-top: 1px solid;"><br>

free 함수로 청크를 해제하면, ptmalloc2는 이를 tcache나 bins에 추가하여 관리한다.

그리고 이후에 malloc으로 비슷한 크기의 동적 할당이 발생하면, 이 연결리스트들을 탐색하여 청크를 재할당한다.

이 메커니즘에서, 해커들은 free로 해제한 청크를 free로 다시 해제했을 때 발생하는 현상에 주목했다고 한다.

<br>

tcache와 bins를 free list라고 통칭한다면, free list의 관점에서 free는 청크를 추가하는 함수, malloc은 청크를 꺼내는 함수이다.

그러므로, 임의의 청크에 대해 free를 두 번이상 적용할 수 있다는 것은, 청크를 free list에 여러 번 추가할 수 있음을 의미한다고 한다.

<br>

해커들은 duplicated free list를 이용하면 임의 주소에 청크를 할당할 수 있음을 밝혀냈다고 한다.

이렇게 할당한 청크의 값을 읽거나 조작함으로써 해커는 임의 주소 읽기 또는 쓰기를 할 수 있다고 한다.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Double Free Bug
<hr style="border-top: 1px solid;"><br>

Double Free Bug (DFB)는 같은 청크를 두 번 해제할 수 있는 버그이다.

ptmalloc2에서 발생하는 버그 중 하나이며, 공격자에게 임의 주소 쓰기, 임의 주소 읽기, 임의 코드 실행, 서비스 거부 등의 수단으로 활용될 수 있다고 한다.

<br>

ptmalloc2에서, free list의 각 청크들은 fd와 bk로 연결된다. 

fd는 자신보다 이후에 해제된 청크를, bk는 이전에 해제된 청크를 가리킨다.

해제된 청크에서 fd와 bk 값을 저장하는 공간은 할당된 청크에서 데이터를 저장하는 데 사용된다.
: uaf_overwrite 문제를 풀 때 확인 가능

<br>

그러므로 **만약 어떤 청크가 free list에 중복해서 포함된다면, 첫 번째 재할당에서 fd와 bk를 조작하여 free list에 임의 주소를 포함시킬 수 있다**고 한다.

드림핵에서 확인해볼 수 있는데, malloc 후 free가 두 번 되면 아래와 같이 fd와 bk 값에 같은 값이 들어가게 된다.

<br>

![image](https://user-images.githubusercontent.com/52172169/189047834-061ee43b-b52a-4c8a-8880-ee30efa2a344.png)

<br>

이 상태에서 재할당을 해주면 공간이 재사용 되므로 아래와 같이 된다.

즉, **중복으로 연결된 청크를 재할당하면, 그 청크는 할당된 청크이면서, 동시에 해제된 청크가 되는 것**이다.

<br>

![image](https://user-images.githubusercontent.com/52172169/189048019-f7425117-848b-427a-bcfe-3dddf51bac87.png)

<br>

이 때, 값을 쓰게 되면 fd와 bk가 있던 공간에 데이터가 저장되므로 free list에 있던 chunk에도 값이 씌여진다.

<br>

![image](https://user-images.githubusercontent.com/52172169/189064287-bb523813-8a3a-4bd4-9c10-02ad42156b02.png)

<br>

![image](https://user-images.githubusercontent.com/52172169/189048134-11978709-78b9-4dc7-9889-635714556c93.png)

<br>

즉, **공격자가 중첩 상태인 청크에 임의의 값을 쓸 수 있다면, 그 청크의 fd와 bk를 조작할 수 있으며, 이는 다시 말해 ptmalloc2의 free list에 임의 주소를 추가할 수 있음을 의미**한다.

이를 tcache poisoning 이라고 하는데, 아래에서 다시 정리.

최근에는 관련한 보호 기법이 glibc에 구현되면서, 이를 우회하지 않으면 같은 청크를 두 번 해제하는 즉시 프로세스가 종료된다.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Mitigation
### tcache_entry
<hr style="border-top: 1px solid;"><br>

tcache_entry 가 추가가 되었다.

<br>

```c
typedef struct tcache_entry {
  struct tcache_entry *next;
+ /* This field exists to detect double frees.  */
+ struct tcache_perthread_struct *key;
} tcache_entry;
```

<br>

tcache_entry는 해제된 tcache 청크들이 갖는 구조로, fd의 역할을 하는 next와 tcache는 LIFO로 bk가 필요가 없다.

<br><br>

### tcache_put
<hr style="border-top: 1px solid;"><br>

tcache_put은 해제한 청크를 tcache에 추가하는 함수다.

<br>

```c
tcache_put(mchunkptr chunk, size_t tc_idx) {
  tcache_entry *e = (tcache_entry *)chunk2mem(chunk);
  assert(tc_idx < TCACHE_MAX_BINS);
  
+ /* Mark this chunk as "in the tcache" so the test in _int_free will detect a
       double free.  */
+ e->key = tcache;
  e->next = tcache->entries[tc_idx];
  tcache->entries[tc_idx] = e;
  ++(tcache->counts[tc_idx]);
}
```

<br>

tcache_put 함수는 해제되는 청크의 key에 tcache라는 값을 대입하도록 변경되었다.

tcache는 ```tcache_perthread```라는 구조체 변수를 가리킨다.
: ```tcache_perthread_struct```는 tcache를 처음 사용하면 할당되는 구조체

<br><br>

### tcache_get
<hr style="border-top: 1px solid;"><br>

tcache_get은 tcache에 연결된 청크를 재사용할 때 사용하는 함수로, tcache_get함수는 재사용하는 청크의 key값에 NULL을 대입하도록 변경되었다.

<br>

```c
tcache_get (size_t tc_idx)
   assert (tcache->entries[tc_idx] > 0);
   tcache->entries[tc_idx] = e->next;
   --(tcache->counts[tc_idx]);
+  e->key = NULL;
   return (void *) e;
 }
```

<br><br>

### _int_free
<hr style="border-top: 1px solid;"><br>

```_int_free```은 청크를 해제할 때 호출되는 함수다.

재할당하려는 청크의 key값이 tcache이면 Double Free가 발생했다고 보고 프로그램을 abort시킨다.

<br>

```c
_int_free (mstate av, mchunkptr p, int have_lock)
 #if USE_TCACHE
   {
     size_t tc_idx = csize2tidx (size);
-
-    if (tcache
-       && tc_idx < mp_.tcache_bins
-       && tcache->counts[tc_idx] < mp_.tcache_count)
+    if (tcache != NULL && tc_idx < mp_.tcache_bins)
       {
-       tcache_put (p, tc_idx);
-       return;
+       /* Check to see if it's already in the tcache.  */
+       tcache_entry *e = (tcache_entry *) chunk2mem (p);
+
+       /* This test succeeds on double free.  However, we don't 100%
+          trust it (it also matches random payload data at a 1 in
+          2^<size_t> chance), so verify it's not an unlikely
+          coincidence before aborting.  */
+       if (__glibc_unlikely (e->key == tcache)) // bypass it
+         {
+           tcache_entry *tmp;
+           LIBC_PROBE (memory_tcache_double_free, 2, e, tc_idx);
+           for (tmp = tcache->entries[tc_idx];
+                tmp;
+                tmp = tmp->next)
+             if (tmp == e)
+               malloc_printerr ("free(): double free detected in tcache 2");
+           /* If we get here, it was a coincidence.  We've wasted a
+              few cycles, but don't abort.  */
+         }
+
+       if (tcache->counts[tc_idx] < mp_.tcache_count)
+         {
+           tcache_put (p, tc_idx);
+           return;
+         }
       }
   }
 #endif
```

<br>

```if (__glibc_unlikely (e->key == tcache))```만 우회하면 DFB를 할 수 있다고 한다.

즉, 해제된 청크의 key값을 1비트만이라도 바꿀 수 있으면, 이 보호 기법을 우회할 수 있다고 한다.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Bypass
<hr style="border-top: 1px solid;"><br>

DFB 보호기법을 우회하는 방법으로, tcache duplication이 있다고 한다.

코드는 드림핵에서 가져왔다.

<br>

```c
// Name: tcache_dup.c
// Compile: gcc -o tcache_dup tcache_dup.c
#include <stdio.h>
#include <stdlib.h>
int main() {
  void *chunk = malloc(0x20);
  
  printf("Chunk to be double-freed: %p\n", chunk);
  free(chunk);
  
  *(char *)(chunk + 8) = 0xff;  // manipulate chunk->key
  free(chunk);                  // free chunk in twice
  
  printf("First allocation: %p\n", malloc(0x20));
  printf("Second allocation: %p\n", malloc(0x20));
  
  return 0;
}
```

<br>

```shell
$ ./tcache_dup
Chunk to be double-freed: 0x55d4db927260
First allocation: 0x55d4db927260
Second allocation: 0x55d4db927260
```

<br>

chunk가 tcache에 중복 연결되어 연속으로 재할당되는 것을 확인할 수 있다.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## tcache poisoning
<hr style="border-top: 1px solid;"><br>

tcache poisoning은 tcache를 조작하여 임의 주소에 청크를 할당시키는 공격이다.

중복으로 연결된 청크를 재할당하면, 그 청크는 할당된 청크이면서, 동시에 해제된 청크가 되는 것을 확인할 수 있다.

<br>

![tcache poisoning](https://kr.object.ncloudstorage.com/dreamhack-content/page/ad87a12c40e687d150365ed2462a0d7bc40891d25bfb5a32baa49fdc4cf669cd.gif)

<br>

따라서 **공격자가 중첩 상태인 청크에 임의의 값을 쓸 수 있다면, 그 청크의 fd와 bk를 조작**할 수 있다.

즉, ptmalloc2의 free list(bin, tcache)에 임의 주소를 추가할 수 있음을 의미한다.

<br>

ptmalloc2는 동적 할당 요청에 대해 free list의 청크를 먼저 반환하므로, 이를 이용하면 **공격자는 임의 주소에 청크를 할당**할 수 있다.

따라서 만약, Tcache poisoing으로 할당한 청크에 대해 값을 출력하거나 조작할 수 있다면, **임의 주소 읽기, 임의 주소 쓰기가 가능**하다는 것이다.

<br><br>
<hr style="border: 2px solid;">
<br><br>
