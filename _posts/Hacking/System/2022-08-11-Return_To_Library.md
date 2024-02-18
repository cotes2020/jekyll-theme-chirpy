---
title: Return To Library (RTL)
date: 2022-08-11 22:20  +0900
categories: [Hacking, System]
tags: [Return To Library, RTL, movaps, rtl movaps issue]
---

## Library
<hr style="border-top: 1px solid;"><br>

라이브러리는 컴퓨터 시스템에서 프로그램들이 함수나 변수를 공유해서 사용할 수 있게 한다.

대개의 프로그램은 서로 공통으로 사용하는 함수들이 많다.  ```(printf, scanf, strlen, memcpy, malloc)```

<br>

C언어를 비롯하여 많은 컴파일 언어들은 자주 사용되는 함수들의 정의를 묶어서 하나의 라이브러리 파일로 만들고, 이를 여러 프로그램이 공유해서 사용할 수 있도록 지원하고 있다.

라이브러리를 사용하면 같은 함수를 반복적으로 정의해야 하는 수고를 덜 수 있어서 코드 개발의 효율이 높아진다는 장점이 있다.

C의 표준 라이브러리인 libc는 우분투에 기본으로 탑재된 라이브러리이며, 실습환경에서는 ```/lib/x86_64-linux-gnu/libc-2.27.so```에 있다.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## RTL(Return To Libc)
<hr style="border-top: 1px solid;"><br>

NX가 적용되어 있으면 스택과 데이터 메모리 영역에는 rw권한만 있고 실행권한이 없다.

따라서 보호기법이 적용되지 않았을 때 BOF하는 것처럼 쉘코드를 삽입하거나 설사 코드 내에 있더라도 실행이 불가능하다.

그래서 쉘코드 대신에 **실행 가능한 영역에 있는 코드들을 사용하여 익스플로잇**해야 한다.

<br>

libc는 ```printf()```와 ```exit()``` 같은 다양한 기본 함수를 갖고 있는 표준 C 라이브러리다.

이 함수들은 **공유**되므로 ```printf()``` 함수를 사용하는 어떤 프로그램도 libc의 적절한 실행 위치를 가리키게 된다.

RTL은 취약점이 있는 프로그램(함수)가 스택에서 어떤 것도 실행하지 않고 libc에 위치한 ```system()``` 함수로 리턴해 셸을 만들게 하는 것이다.

<br>

먼저 libc에서 ```system()``` 함수의 위치는 정해져 있다. 

시스템마다 위치가 다르기는 하나 **한 번 위치가 정해지면 libc가 재컴파일되기 전까지는 위치가 같다.** 

<br>

32비트는 함수 호출 규약이 cdecl이다. (인자는 오른쪽에서 왼쪽으로 스택에 push)

즉, 스택에 인자가 쌓이는데 스택 안에서 함수를 호출을 하게 되면은 어떻게 되는가?

만약 인자가 3개인 func 함수를 호출한다고 가정한다면 아래와 같이 된다.

<br>

![image](https://user-images.githubusercontent.com/52172169/184145546-6038080d-856c-41e8-bf5d-b8bcfaca1bf1.png)

<br>

함수를 호출하면 인자가 먼저 쌓인 다음 함수가 끝나고 리턴할 주소를 push하게 된다.

따라서 ```system("/bin/sh")```를 호출한다고 하면 ```system 함수가 끝나고 돌아갈 리턴 주소 + "/bin/sh"```가 된다.

<br>

그래서 리턴 주소에 쉘 코드 대신 ```system("/bin/sh")```가 실행되도록 넣어줘야 한다면, ```system 함수 주소 + 리턴 주소(아무 값 가능) + 인자 주소 ("/bin/sh" 주소)```가 된다.

print 명령어로 system 주소를 찾고, find 명령어로 libc에 존재하는 ```/bin/sh``` 주소를 찾을 수 있다.

system이나 popen 등의 셸 명령어 실행 함수들이 내부적으로 ```/bin/sh``` 문자열을 사용하기 때문에 라이브러리 메모리에서 ```/bin/sh``` 문자열을 찾을 수 있습니다.

gdb에서 ```info proc maps```를 통해 process를 확인하면 된다.

<br>

```
$ gdb -q ./example1_nx
Reading symbols from ./example1_nx...(no debugging symbols found)...done.
(gdb) b main
Breakpoint 1 at 0x8048479
(gdb) r aaaabbbb
Starting program: ~/example1_nx aaaabbbb
Breakpoint 1, 0x080484fb in main ()
(gdb) info proc map
process 110780
Mapped address spaces:
	Start Addr   End Addr       Size     Offset objfile
	 0x8048000  0x8049000     0x1000        0x0 ~/example1_nx
	 0x8049000  0x804a000     0x1000        0x0 ~/example1_nx
	 0x804a000  0x804b000     0x1000     0x1000 ~/example1_nx
	0xf7e02000 0xf7e03000     0x1000        0x0 
	0xf7e03000 0xf7fb3000   0x1b0000        0x0 /lib/i386-linux-gnu/libc-2.23.so
	0xf7fb3000 0xf7fb5000     0x2000   0x1af000 /lib/i386-linux-gnu/libc-2.23.so
	0xf7fb5000 0xf7fb6000     0x1000   0x1b1000 /lib/i386-linux-gnu/libc-2.23.so
	0xf7fb6000 0xf7fb9000     0x3000        0x0 
	0xf7fd3000 0xf7fd4000     0x1000        0x0 
	0xf7fd4000 0xf7fd7000     0x3000        0x0 [vvar]
	0xf7fd7000 0xf7fd9000     0x2000        0x0 [vdso]
	0xf7fd9000 0xf7ffc000    0x23000        0x0 /lib/i386-linux-gnu/ld-2.23.so
	0xf7ffc000 0xf7ffd000     0x1000    0x22000 /lib/i386-linux-gnu/ld-2.23.so
	0xf7ffd000 0xf7ffe000     0x1000    0x23000 /lib/i386-linux-gnu/ld-2.23.so
	0xfffdd000 0xffffe000    0x21000        0x0 [stack]
(gdb) p system
$1 = {<text variable, no debug info>} 0xf7e3dda0 <system>
(gdb) find 0xf7e03000, 0xf7fb3000, "/bin/sh"
0xf7f5ea0b
1 pattern found.
(gdb) x/s 0xf7f5ea0b
0xf7f5ea0b:	"/bin/sh"
(gdb) 
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## x64 RTL and stack alignment
<hr style="border-top: 1px solid;"><br>

위에서 진행한 방식은 32비트일 때로, 32비트는 스택에 인자를 push하기 때문에 가능한 것이다.

64비트에서는 가젯을 찾아서 진행해야 하는데, 예를 들어, 인자가 1개라면 ```pop rdi; ret``` 코드 가젯을 찾아서 진행해야 한다.

그 후 가젯을 통해 rdi에 ```/bin/sh``` 문자열을 넣어주고 system 함수의 주소를 찾아서 넣어주면 되는데..  주의사항이 있다.

system 함수에는 내부적으로 movaps 명령어를 사용하는데, 이 명령어는 스택 포인터가 16의 배수가 아니면 segmentation fault를 일으킨다.

<br><br>

출처는 <a href="https://hackyboiz.github.io/2020/12/06/fabu1ous/x64-stack-alignment/" target="_blank">hackyboiz.github.io/2020/12/06/fabu1ous/x64-stack-alignment/</a>

위의 블로그 내용을 정리하면 아래와 같다.

<br>

movaps는 stack alignment가 지켜져야 한다.

stack alignment는 항상 스택의 top이 16배수여야 한다는 것이다.

이것이 지켜지기 위해서, 프로그램의 흐름(control)이 함수의 entry로 옮겨지는 시점에선 스택 포인터(rsp)+8이 항상 16의 배수여야 한다.

<br>

stack align을 지키면서 함수를 호출하는 흐름을 요약하면 아래와 같다.

1. call 실행 직전 RSP는 16의 배수 ( stack align O )

2. 함수의 entry point에선 RSP+8이 16의 배수 ( stack align X )

3. 함수의 프롤로그 실행 후 RSP는 16의 배수 ( stack align O )

4. RBP는 항상 16의 배수 ( stack align O )

<br>

그리고 jmp 명령어와 call 명령어, ret 명령어에 대해서 자세히 알아야 한다.

+ jmp는 단순히 프로그램의 흐름을 옮기는 것이라, 스택에 변화는 없다.

+ call은 함수 종료 후 돌아올 ret 주소를 저장하기 때문에 rsp가 8만큼 감소하게 된다. 
  + 즉, rsp가 원래 16배수였는데, 감소해서 8배수가 된다.
  + 따라서 일시적으로 entry point에서 stack alignment가 깨지게 된다.

+ ret는 call이 저장한 ret 주소를 스택에서 pop을 하여 돌아가므로 rsp가 8만큼 증가하게 된다.
  + 따라서 함수 종료 시 leave나 pop으로 인해 stack alignment가 깨지지만, ret를 통해 다시 맞춰진다. 

<br>

그래서 bof나 rop, rtl 공격 시 ret로 system 등의 함수를 호출하게 된다.

그래서 stack alignment가 깨져버리므로 segmentation fault가 발생하여 exploit이 실패하는 것이라고 한다.

확인해보면 다음과 같다.

버퍼오버플로우로 main 함수가 끝나면 system 함수로 넘어가도록 페이로드를 익스플로잇했다고 가정한다.

그럼 main이 ret 후에 system으로 넘어가게 되는데.. 원래는 call 명령어를 통해 함수를 호출해야 하지만, ret 명령어를 통해 함수로 넘어가게 된다.

이 시점에서 원래의 흐름을 생각해보면, call 직전 rsp는 16배수에서 8만큼 감소하여 함수의 entry point에서는 rsp가 8의 배수이고, rsp+8이 16배수이다.

<br>

하지만 call로 호출되어야 하는 함수가 ret로 호출이 되면?

ret 후에는 rsp가 16배수이므로 프롤로그가 실행이 되면 rsp가 8의 배수가 되어 stack alignment가 깨지게 되는 것이다.

<br>

따라서 해결하기 위해서는 스택 포인터는 +8 씩 증가하므로 에러가 발생했다면, 스택 포인터를 8만큼 증가 또는 감소시켜줘야 한다.

가장 좋은 방법으로 ret 가젯을 찾아서 넣어주는 것이라고 한다.

ret 가젯을 넣어줌으로써 rsp를 8 증가시켜서 stack alignment를 만족시켜서 해결할 수 있다고 한다.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 출처 및 참고
<hr style="border-top: 1px solid;"><br>

내용 출처
: <a href="https://dreamhack.io/lecture/courses/2" target="_blank">Linux Exploitation & Mitigation Part 1</a>
: <a href="https://c0wb3ll.tistory.com/entry/ret2libc-x64" target="_blank">c0wb3ll.tistory.com/entry/ret2libc-x64</a>

<br>

x64 system movaps issue (매우 중요)
: <a href="https://hackyboiz.github.io/2020/12/06/fabu1ous/x64-stack-alignment/" target="_blank">hackyboiz.github.io/2020/12/06/fabu1ous/x64-stack-alignment/</a>
: <a href="https://c0wb3ll.tistory.com/entry/ret2libc-x64" target="_blank">c0wb3ll.tistory.com/entry/ret2libc-x64</a>

<br>

보면 좋음 
: <a href="https://opentutorials.org/module/4290/27060" target="_blank">opentutorials.org/module/4290/27060</a>
: <a href="https://pwnkidh8n.tistory.com/178?category=883849" target="_blank">pwnkidh8n.tistory.com/178?category=883849</a> -> x86 RTL
: <a href="https://www.lazenca.net/display/TEC/02.RTL%28Return+to+Libc%29+-+x64" target="_blank">lazenca.net/display/TEC/02.RTL%28Return+to+Libc%29+-+x64</a> -> x64 RTL
: <a href="https://pwnkidh8n.tistory.com/179?category=883849" target="_blank">pwnkidh8n.tistory.com/179?category=883849</a> -> x64 RTL

<br><br>
<hr style="border: 2px solid;">
<br><br>
