---
title: Introduction to Pwning - Part 1
author: P0ch1ta
date: 2023-04-07 12:33:00 +0530
categories: [Pwning]
tags: [Basics]
math: true
mermaid: true
---

This is the part one of simple pwning lecture series. The target of this series is to get started with pwning from the very basics to some advanced attack. We will be trying challenges from different CTFs as well to get familiar with the different exploitation vectors. In this lecture we will breifly try to understand how out code works and how can we get started with pwning.

# Introduction

When we write code we generally write it in higher level programming languages such as `C`/`C++` or `python`. But the computer cannot directly understand such higher level languages and requires preprossing before it can run the program. Here we will take a look at how a program is actually converted from higher level language to machine understandable binary instructions.  

Below is a high level `C` code.
```c
#include <stdio.h>

int main()
{
    printf("Hello World.\n");
    return 0;
}
```

After processing the code looks like this
```asm
	.file	"test.c"
	.def	___main;	.scl	2;	.type	32;	.endef
	.section .rdata,"dr"
LC0:
	.ascii "Hello World.\0"
	.text
	.globl	_main
	.def	_main;	.scl	2;	.type	32;	.endef
_main:
LFB10:
	.cfi_startproc
	pushl	%ebp
	.cfi_def_cfa_offset 8
	.cfi_offset 5, -8
	movl	%esp, %ebp
	.cfi_def_cfa_register 5
	andl	$-16, %esp
	subl	$16, %esp
	call	___main
	movl	$LC0, (%esp)
	call	_puts
	movl	$0, %eax
	leave
	.cfi_restore 5
	.cfi_def_cfa 4, 4
	ret
	.cfi_endproc
LFE10:
	.ident	"GCC: (MinGW.org GCC-6.3.0-1) 6.3.0"
	.def	_puts;	.scl	2;	.type	32;	.endef
```

<br>

### The basic process of how a code is processed and run can be understood by the following diagram
<br>

<!-- ![alt text](./images/language_processing_system.jpg "Code flow") -->
<img src="https://raw.githubusercontent.com/manasghandat/manasghandat.github.io/master/assets/img/Images/Intro_to_pwn_1/language_processing_system.jpg" alt="Code flow">


# Code processing

## Compiler

A compiler is a program which compiles the high level code into lower level `machine code` (C/C++ compiler) or into `bytecode`. Machine code is a set of instructions that the machine can directly execute without the help of any other software. It is considered as low level code. The `assembler` then converts this set of instructions into binary instructions that is executed by the CPU.

## Interpreter

The `bytecode` (eg. Java / Python) which acts as an intermediate code that can be executed by the interpreter. The execution is done line by line and thus is often not prefered as it is slower.

## Assembler

An assembler is a program that takes basic computer instructions and converts them into a pattern of bits that the computer's processor can use to perform its basic operations. It is an embedded system tool. 

# Code Execution

The running of the code is comprehensively divided into 4 categories.

## Stack (Program Stack / Call Stack)

It is the stack which contains all the return address of the programs that are being executed and some other important data that is required for the execution of the program. 

<!-- ![alt text](./images/stack.png "Stack") -->
<img src="https://raw.githubusercontent.com/manasghandat/manasghandat.github.io/master/assets/img/Images/Intro_to_pwn_1/stack.png" alt="stack">


## Heap

Heap is the large empty region reserved for dynamic memory allocation. Funcitons like `malloc` can be used to instantiate such memory.

<!-- ![alt text](./images/heap.png "Heap") -->
<img src="https://raw.githubusercontent.com/manasghandat/manasghandat.github.io/master/assets/img/Images/Intro_to_pwn_1/Heap.png" alt="heap">

## Code

It is the section that stores the instruction codes, this is, the actual code that is written by the user. It is represented by the `.text` section of the program.

<!-- ![alt text](./images/code.png "Code") -->
<img src="https://raw.githubusercontent.com/manasghandat/manasghandat.github.io/master/assets/img/Images/Intro_to_pwn_1/code.png" alt="code">

## Data

It is a portion contains initialized static variables, that is, global variables and static local variables. The size of this segment is determined by the size of the values in the program's source code, and does not change at run time. It is represented by the `.data` and `.bss` section.

The actual execution of the code is done by the program registers. A register is a small set of data storing cell. It may contain actual data value or the address of a memory location. There are several program register. They are as follows

Register  | Description
---- | ----
R8-R11 | General purpose registers
R12-R15 | Callee save register
RIP (Instruction Pointer) | Points ot the next instruction that should be executed
RSP (Stack Pointer) | Pointe to the top of the stack (bottom of memory)
RBP (Base Pointer) | Points to the base of current frame
RDI (Destination index) | Scratch register used to pass first argument
RSI (Source index) | Scratch register used to paas second argument
RAX | Contains the return value of the functions
RBX | Callee Save register
RCX | Scratch register used to pass fourth argument
RDX | Scratch register used to pass third argument

<!-- ![alt text](./images/registers.png "registers") -->
<img src="https://raw.githubusercontent.com/manasghandat/manasghandat.github.io/master/assets/img/Images/Intro_to_pwn_1/registers.png" alt="registers">

**Note: In case of x86 architecture the name of registers is changed and instead of `r` the prefix `e` is used. Eg. `RIP` becomes `EIP`**

# Getting started

## Finding protections

One of the first things that we must do when we get any challenge is to run `checksec` command. This command helps us understand various protections that are enabled in the binary given to us. The output of the command looks as follows

You can install the command using:
```bash
pip install checksec.py
```

You can run checksec using:
```bash
checksec --file=./(file_name)
```

<!-- <img src="/images/Checksec.png"> -->
<img src="https://raw.githubusercontent.com/manasghandat/manasghandat.github.io/master/assets/img/Images/Intro_to_pwn_1/Checksec.png" alt="checksec command">

Let us understand what the various protections are:
* Canary : It is random generated value stored in the `fs` register at an offset of `0x28`. It is used to prevent buffer-overflow.
* NX : It stands for no-execute. It makes the stack of a program unexecutable. Thus it prevents writing shellcode on the stack.
* PIE : Position Independent Executable loads the program at any random address everytime. As a result we cannot find the exact address of any function in the binary
* RelRO : Full relro makes the `GOT` (global offset table) read-only while partial relro makes the `GOT` to come before the `data` section
* Fortify : It is used to verify and protect various overflows. It includes various functions like `__gethostname_chk`, `__printf_chk`, etc.

## Debugging the challenge

In order to understand what exactly the program is doing, we need to use a debugger. The preferred choice for debugger is `gdb` though you can use any debugger like `lldb`, etc. Using plain `gdb` may be inefficient so we will install `gef` or `pwndbg` . You can find a link on how to install `pwndbg` and `gef` simultaneously on your system in the resources section

You need to first install plain `gdb`. It can be done using the following command:

```bash
sudo apt install gdb
```

We will learn how to effectively use `gdb` as we solve various challenges

# References

<a href="https://cs.brown.edu/courses/cs033/docs/guides/x64_cheatsheet.pdf">x64 Programming</a><br>
<a href="https://www.geeksforgeeks.org/memory-layout-of-c-program/">Memory Layout</a><br>
<a href="https://resources.infosecinstitute.com/topic/gentoo-hardening-part-3-using-checksec-2/">Checksec Command</a><br>
<a href="https://infosecwriteups.com/pwndbg-gef-peda-one-for-all-and-all-for-one-714d71bf36b8">Installing gdb plugings</a>
