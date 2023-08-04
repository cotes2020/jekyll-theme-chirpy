---
title: Use-After-Free (UAF)
date: 2022-08-30 17:19 +0900
categories: [Hacking,System]
tags: [Use After Free, UAF]
---

## UAF
<hr style="border-top: 1px solid;"><br>

드림핵 내용 정리.

Use-After-Free 취약점은 메모리 참조에 사용한 포인터를 메모리 해제 후에 적절히 초기화하지 않거나, 해제한 메모리를 초기화하지 않고 다음 청크에 재할당해주면서 발생하는 취약점이다. 

<br>

컴퓨터 과학에서, Dangling Pointer는 유효하지 않은 메모리 영역을 가리키는 포인터를 말한다.

메모리에 동적 할당을 해줄 땐 malloc 함수를 사용하는데, malloc 함수는 할당한 메모리의 주소를 반환한다.

일반적으로 메모리 할당 시, 포인터를 선언하고 그 포인터에 malloc 함수가 할당한 메모리 주소를 저장한다.

그리고 그 포인터를 참조하여 할당한 메모리에 접근한다.

<br>

메모리를 다 사용한 후 해제할 때는 free 함수를 사용한다.

하지만 free 함수는 청크를 ptmalloc에 반환만 할 뿐이지, 포인터 변수를 초기화시켜주진 않는다고 한다.

따라서 포인터 변수를 초기화하지 않는다면, 포인터 변수에는 허용되지 않는 메모리의 주소가 담겨져 있는 상태이다.

즉, Dangling Pointer가 된다.

<br>

그래서 UAF는 이 해제된 메모리에 접근할 수 있게 되었을 때 발생하는 취약점이다.

코드는 드림핵에서 가져왔다.

<br>

```c
// Name: uaf.c
// Compile: gcc -o uaf uaf.c -no-pie
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
struct NameTag {
  char team_name[16];
  char name[32];
  void (*func)();
};
struct Secret {
  char secret_name[16];
  char secret_info[32];
  long code;
};
int main() {
  int idx;
  
  struct NameTag *nametag;
  struct Secret *secret;
  secret = malloc(sizeof(struct Secret));
  strcpy(secret->secret_name, "ADMIN PASSWORD");
  strcpy(secret->secret_info, "P@ssw0rd!@#");
  secret->code = 0x1337;
  free(secret);
  
  secret = NULL;
  
  nametag = malloc(sizeof(struct NameTag));
  strcpy(nametag->team_name, "security team");
  memcpy(nametag->name, "S", 1);
  
  printf("Team Name: %s\n", nametag->team_name);
  printf("Name: %s\n", nametag->name);
  
  if (nametag->func) {
    printf("Nametag function: %p\n", nametag->func);
    nametag->func();
  }
}
```

<br>

```console
$ ./uaf
Team Name: security team
Name: S@ssw0rd!@#
Nametag function: 0x1337
Segmentation fault (core dumped)
```

<br>

위 코드에서 UAF가 발생한다. 이유는 아래와 같다.

ptmalloc2는 새로운 할당 요청이 들어왔을 때, 요청된 크기와 비슷한 청크가 bin이나 tcache에 있는지 확인한다.

그리고 만약 있다면, 해당 청크를 꺼내어 재사용한다고 한다.

예제 코드에서 Nametag와 Secret은 같은 크기의 구조체로, 앞서 할당한 secret을 해제하고 nametag를 할당하면 nametag는 secret과 같은 메모리 영역을 사용하게 된다고 한다.

이때 free는 해제한 메모리의 데이터를 초기화하지 않으므로, nametag에는 secret의 값이 일부 남아있게 된다고 한다!

<br>

동적 할당한 청크를 해제한 뒤에는 해제된 메모리 영역에 이전 객체의 데이터가 남아 있어서 이러한 특징을 이용해 초기화되지 않은 메모리의 값을 읽어내거나, 새로운 객체가 악의적인 값을 사용하게 유도하여 프로그램의 정상적인 실행을 방해할 수 있다고 한다.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 내용 출처
<hr style="border-top: 1px solid;"><br>

dreamhack
: <a href="https://dreamhack.io/lecture/courses/106" target="_blank">Memory Corruption: Use After Free</a>

<br><br>
<hr style="border: 2px solid;">
<br><br>
