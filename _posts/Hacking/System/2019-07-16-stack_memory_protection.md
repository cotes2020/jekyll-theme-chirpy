---
title : Linux Stack Memory Protection And Bypass
categories : [Hacking, System]
tags : [스택 메모리 보호기법, ASLR, NX, SSP, Stack Smashing Protector, Stack Canary, RELRO, PIE]
---

## 필독
<hr style="border-top: 1px solid;"><br>

Link
: <a href="https://www.lazenca.net/display/TEC/02.Protection+Tech" target="_blank">lazenca.net/display/TEC/02.Protection+Tech</a>
: <a href="https://bpsecblog.wordpress.com/memory_protect_linux/" target="_blank">bpsecblog.wordpress.com/memory_protect_linux/</a>
: <a href="https://sechack.tistory.com/63" target="_blank">sechack.tistory.com/63</a>  --> 보호기법 정리 잘되있음

<br>

드림핵
: NX bit -> <a href="https://dreamhack.io/lecture/courses/2" target="_blank">Linux Exploitation & Mitigation Part 1</a>
: 카나리 -> <a href="https://dreamhack.io/lecture/courses/112" target="_blank">Mitigation: Stack Canary</a>
: ASLR -> <a href="https://dreamhack.io/lecture/courses/3" target="_blank">Linux Exploitation & Mitigation Part 2</a>
: 그 외 -> <a href="https://dreamhack.io/lecture/courses/4" target="_blank">Linux Exploitation & Mitigation Part 3</a>
: PIE 추가 -> <a href="https://dreamhack.io/lecture/courses/113" target="_blank">Background: PIE</a> 

<br><br>
<hr style="border: 2px solid;">
<br><br>

## NX
<hr style="border-top: 1px solid;"><br>

```NX : No -eXecute```

<br>

실행할 코드영역을 제외한 메모리 영역에 실행권한 제거. 

**즉, 스택을 실행 불가능하게 만드는 것이다.**

NX가 안걸려 있다면 아래처럼 코드 영역 이외에도 실행 권한이 부여되어 있다.

<br>

![image](https://user-images.githubusercontent.com/52172169/185734339-0a0ce540-30a5-4d52-b7a1-6de05be170ef.png)

<br>

NX를 설정하면 코드 영역을 제외한 메모리 영역에 실행 권한이 없어진다.

<br>

![image](https://user-images.githubusercontent.com/52172169/185734375-1f3e513a-2ba4-48af-b171-64c94e4f7082.png)

<br>

+ 해제: (컴파일시) ```gcc ~~ -z execstack```

<br>

+ bypass : 실행권한이 있는 메모리를 이용한 기법 (RTL,ROP,etc ...)

<br>

확인방법 
: ```readelf -a [file] | grep STACK``` -> RW가 있으면 적용되어 있는 것

<br><br>
<hr style="border: 2px solid;">
<br><br>

## ASLR
<hr style="border-top: 1px solid;"><br>

```Address Space Layout Randomization (공격자가 쉽게 주소값을 알지 못하게 해주는 보호기법)```

라이브러리, 힙, 스택 영역 등의 주소를 바이너리가 실행될 때마다 랜덤하게 바꿔 RTL과 같이 정해진 주소를 이용한 공격을 막기 위한 보호 기법

<br>

+ ```/proc/sys/kernel/randomize_va_space```  파일의 값을 확인하면 알 수 있음.
  + 0 : 적용 x
  + 1 : 스택, 라이브러리 메모리 랜덤화
  + 2 : 스택, 힙, 라이브러리 메모리 랜덤화

<br>

그러나 우회할 수 있는 방법은 있다.

<br>

+ 스택을 포함한 메모리의 주소를 계속해서 랜덤하게 할당하지만, 코드와 데이터 영역은 변경되지 않음.

+ 리눅스는 ASLR이 적용됐을 때, 파일을 페이지(page)1 단위로 임의 주소에 매핑하므로, 페이지의 크기인 12비트 이하로는 주소가 변경되지 않는다.

+ 완전한 랜덤은 아니고 순서와 패턴을 어느정도 유지한 채로 랜덤하게 함.
  + 전체 메모리 주소를 랜덤화 하는 것이 아닌, 각 메모리 영역의 base 주소만 랜덤화 하는 것.
  + 따라서 주소가 매번 바뀌어도 취약한 함수로부터 리턴 주소까지의 오프셋이 일정하므로 둘 사이의 차이(오프셋 값)을 알아내면 됨.
  + 즉, ASLR이 적용되면 라이브러리를 매핑하는데, 그대로 가져오는 것이므로 매핑된 주소로부터 다른 심볼들 사이의 오프셋은 동일함.

+ 해제: ```(sudo) sysctl -w kernel.randomize_va_space=0 (1,2 는 on, 0은 off)```

+ bypass : ROP

<br><br>
<hr style="border: 2px solid;">
<br><br>

## SSP(Stack Smashing Protector)
<hr style="border-top: 1px solid;"><br>

메모리 커럽션 취약점 중 스택 버퍼 오버플로우 취약점(Return Address Overwrite)을 막기 위해 개발된 보호 기법.

스택 버퍼와 스택 프레임 포인터 사이에 랜덤 값을 삽입하여 함수 종료 시점에서 랜덤 값 변조 여부 검사

<br>

SSP 보호 기법이 적용되어 있으면 함수에서 스택 사용 시 **카나리** 생성

마스터 카나리는 main함수 호출 전 랜덤으로 생성된 카나리를 스레드 전역 변수로 사용되는 TLS(Thread Local Storage)에 저장한다.

<br>

TLS 영역의 ```header.stack_guard```에 카나리 값이 삽입되고, 이 값을 gs, fs 세그먼트 레지스터에 저장한다.

gs, fs 세그먼트 레지스터는 목적이 정해지지 않아 운영체제가 임의로 사용할 수 있는 레지스터로, ```TLS(Thread Local Storage)```를 가리키는 포인터로 사용된다.

<br>

TLS에는 프로세스에 필요한 여러 데이터(카나리 등)가 저장되어 있다.

<br><br>

### Stack Canary
<hr style="border-top: 1px solid;"><br>

함수의 프롤로그에서 스택 버퍼와 반환 주소 사이에 임의의 값(카나리 값)을 삽입, 함수 종료 시(함수 에필로그) 카나리 값이 변조되었으면 스택 오염으로 판단

카나리가 변조되면 프로세스 강제 종료

<br>

- 리눅스의 경우 ```gs:0x14```, ```fs:0x28```에 카나리 저장

- 해제 : (컴파일시) ```gcc ~~ -fno-stack-protector```

- bypass : leak? got overwrite?

<br><br>

### TLS 생성 과정과 카나리 값 설정
<hr style="border-top: 1px solid;"><br>

Canary는 프로세스가 시작될 때, TLS에 전역 변수로 저장되고 각 함수마다 프롤로그와 에필로그에서 이 값을 참조한다.

fs는 TLS를 가리킨다고 위에서 말했듯이, fs의 값을 알면 TLS의 주소를 알 수 있다.

단, 리눅스에서는 fs의 값은 특정 시스템 콜을 사용해야만 알 수 있다.

따라서, fs의 값을 설정할 때 호출되는 ```arch_prctl(int code, unsigned long addr)``` 시스템 콜을 확인해야 한다.

시스템 콜을 ```arch_prctl(ARCH_SET_FS, addr)``` 형태로 호출하면 addr에 fs의 값이 담긴다.

gdb에는 특정 이벤트 발생 시 프로세스를 중지하는 catch 명령어가 있다.

따라서 rsi 값을 확인하면 될 것이다.

<br>

```
$ gdb -q ./canary
pwndbg> catch syscall arch_prctl
Catchpoint 1 (syscall 'arch_prctl' [158])
pwndbg> run

Catchpoint 1 (call to syscall arch_prctl), 0x00007ffff7dd6024 in init_tls () at rtld.c:740
740	rtld.c: No such file or directory.
 ► 0x7ffff7dd4024 <init_tls+276>    test   eax, eax
   0x7ffff7dd4026 <init_tls+278>    je     init_tls+321 <init_tls+321>
   0x7ffff7dd4028 <init_tls+280>    lea    rbx, qword ptr [rip + 0x22721]
pwndbg> info register $rdi
rdi            0x1002   4098          // ARCH_SET_FS = 0x1002
pwndbg> info register $rsi
rsi            0x7ffff7fdb4c0   140737354032320 
pwndbg> x/gx 0x7ffff7fdb4c0+0x28
0x7ffff7fdb4e8:	0x0000000000000000
```

<br>

rsi의 값이 ```0x7ffff7fdb4c0```이므로, 이 프로세스는 TLS를 ```0x7ffff7fdb4c0```에 저장할 것이며, fs는 이를 가리키게 될 것이다.

카나리의 값은 ```fs:0x28```이므로, TLS가 있는 ```0x7ffff7fdb4c0```에 ```0x28```을 더한 곳에 있다.

확인해보면 아직은 값이 설정되어 있지 않다.

<br>

```TLS+0x28```에 값을 쓸 때를 확인하기 위해 gdb 명령어인 watch를 사용하였다.

watch는 특정 주소에 저장된 값이 변경되면 프로세스를 중단한다.

<br>

```
pwndbg> watch *(0x7ffff7fdb4c0+0x28)
Hardware watchpoint 4: *(0x7ffff7fdb4c0+0x28)
pwndbg> continue
Continuing.
Hardware watchpoint 4: *(0x7ffff7fdb4c0+0x28)
Old value = 0
New value = -1942582016
security_init () at rtld.c:807
807	in rtld.c
pwndbg> x/gx 0x7ffff7fdb4c0+0x28
0x7ffff7fdb4e8:	0x2f35207b8c368d00
```

<br>

TLS+0x28 주소의 값을 확인해보면 카나리 값이 설정된 것을 알 수 있다.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## RELRO(Relocation Read-Only)
<hr style="border-top: 1px solid;"><br>

GOT와 같은 다이나믹 섹션들에 쓰기 권한을 없애고 읽기 권한만 갖게 하는 보호기법이다.

이렇게 되면 GOT에 라이브러리 함수의 주소를 덮어쓰기 위해서는 쓰기 권한이 필요한데, 읽기 권한만 있으므로 GOT Overwrite, ROP를 할 수 없게 된다.

<br>

RELRO는 바이너리 섹션에 Read Only가 적용된 정도에 따라 크게 No RELRO, Partial RELRO, Full RELRO 세 단계로 나뉠 수 있다.

<br>

+ No RELRO 는 바이너리에 RELRO 보호기법이 아예 적용되어 있지 않은 상태로, 코드 영역을 제외한 거의 모든 메모리 영역에 쓰기 권한이 있음.
  + 코드 영역은 원래 쓰기 권한이 없다

+ Partial RELRO 는 ```.init_arry```나 ```.fini_array``` 등 ```non-PLT GOT``` 에 대한 쓰기 권한을 제거한 상태

+ Full RELRO 는 GOT 섹션에 대한 쓰기 권한까지 제거해 ```.data```, ```.bss``` 영역을 제외한 모든 바이너리 섹션에서 쓰기 권한이 제거된 상태
  + GOT Overwrite 불가능

<br>

```readelf -a``` 명령어로 확인할 수 있고, ```BIND_NOW```가 있다면 Full RELRO, ```GNU_RELRO```가 있다면 Partial RELRO, 둘 다 없다면 적용되어 있지 않은 것이다.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## PIE(Position Independent Executable)
<hr style="border-top: 1px solid;"><br>

PIE는 Executable, 즉 **바이너리가 로딩될 때 랜덤한 주소에 매핑**되는 보호기법이다.

즉, binary의 base 주소를 랜덤화 하는 것으로, ASLR과 유사하며 ASLR이 binary에 적용됬다고 보면 된다.

<br>

컴파일러는 바이너리가 메모리 어디에 매핑되어도 실행에 지장이 없도록 바이너리를 위치 독립적으로 컴파일한다.

이는 결국 **코드 영역의 주소 랜덤화**를 가능하게 해준다. 

PIE가 설정되어 있으면 코드 영역의 주소가 실행될 때마다 변하기 때문에 ROP와 같은 코드 재사용 공격(가젯)을 막을 수 있다.

<br>

즉, PIE 가 설정되어 있으면 **코드, 힙, 라이브러리, 스택 등 모든 메모리 영역의 주소가 랜덤화**된다.

<br>

자세하게는 리눅스에서 ELF는 실행 파일(Executable)과 공유 오브젝트(Shared Object, SO)로 두 가지가 존재한다.

공유 오브젝트는 기본적으로 재배치(Relocation)이 가능하도록 설계가 됬는데, 재배치가 가능하다는 것은 메모리의 어느 주소에 적재되어도 코드의 의미가 훼손되지 않음을 의미하는데, 컴퓨터 과학에서는 이런 성질을 만족하는 코드를 Position-Independent Code(PIC)라고 부른다고 한다.

<br>

PIE는 무작위 주소에 매핑돼도 실행 가능한 실행 파일을 뜻한다.

ASLR이 도입되기 전에는 실행 파일을 무작위 주소에 매핑할 필요가 없어서, 실행 파일은 재배치를 고려하지 않고 설계되었다.

이후 ASLR이 도입되면서 실행 파일도 무작위 주소에 매핑될 수 있게 하고 싶었으나, 호환성 문제가 발생할 것임이 분명해서, 원래 재배치가 가능했던 공유 오브젝트를 실행 파일로 사용하기로 하였다.

<br>

PIE는 재배치가 가능하므로, ASLR이 적용된 시스템에서는 실행 파일도 무작위 주소에 적재된다.

PIE가 적용되면 코드 영역도 주소가 계속 바뀐다.

<br>

```shell
$ ./pie
buf_stack addr: 0x7ffc85ef37e0
buf_heap addr: 0x55617ffcb260
libc_base addr: 0x7f0989d06000
printf addr: 0x7f0989d6af00
main addr: 0x55617f1297ba

$ ./pie
buf_stack addr: 0x7ffe9088b1c0
buf_heap addr: 0x55e0a6116260
libc_base addr: 0x7f9172a7e000
printf addr: 0x7f9172ae2f00
main addr: 0x55e0a564a7ba

$ ./pie
buf_stack addr: 0x7ffec6da1fa0
buf_heap addr: 0x5590e4175260
libc_base addr: 0x7fdea61f2000
printf addr: 0x7fdea6256f00
main addr: 0x5590e1faf7ba
```

<br>

PIE 보호기법이 적용되어있는 ELF 바이너리는 실행될 때 메모리의 동적 주소에 로딩된다.

readelf를 통해 파일 헤더를 확인하면 된다.

위의 내용을 토대로 보면 단순 ELF 파일은 pie가 적용되지 않은 것이고, pie가 적용되면 공유 오브젝트 파일이 되는 것이다.

<br>

```linux
$ readelf -h ./no_pie | grep Type
  Type:                              EXEC (Executable file)
$ readelf -h ./pie | grep Type
  Type:                              DYN (Shared object file)
```
    
<br><br>
<hr style="border: 2px solid;">
<br><br>
