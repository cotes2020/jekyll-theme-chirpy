---
layout: post
title: "10-AT&T汇编"
date: 2020-05-03 20:10:00.000000000 +09:00
categories: [逆向工程]
tags: [逆向工程, AT&T汇编, Assembly]
---

## AT&T汇编

+ 基于x86架构的处理器所使用的汇编指令一般有2种格式

  + **Intel汇编**
    + DOS(8086处理器)、Windows
    + Windows派系 -> VC编译器
  + **AT & T汇编**
    + Linux、Unix、Mac OS、iOS(模拟器)
    + Unix派系 -> GCC编译器
  + **AT & T**（American Telephone & Telephone）
  + 作为iOS工程师，最主要的汇编语言是
    + `AT&T` 汇编 -> iOS模拟器
    + `ARM` 汇编 -> iOS真机设备、

+ **AT & T汇编** VS **Intel汇编**

  | 项目         |                        AT&T                         |                      Intel                       |             说明              |
  | ------------ | :-------------------------------------------------: | :----------------------------------------------: | :---------------------------: |
  | 寄存器命名   |                        %eax                         |                       eax                        |          Intel不带%           |
  | 操作数顺序   |                   movl %eax, %edx                   |                   mov edx, eax                   |      将eax的值赋值给edx       |
  | 常数\立即数  |       movl $3, %eax         movl $0x10, %eax        |            mov eax, 3   mov eax, 0x10            | 将3赋值给eax, 将0x10赋值给eax |
  | jmp指令      |      jmp *%edx   jmp *0x4001002   jmp *(%eax)       |        jmp edx  jmp 0x4001002   jmp[eax]         | 在AT&T的jmp地址签名要加星*号  |
  | 操作数的长度 | movl %eax, %eax  movb $0x10, %al  leaw 0x(%dx), %ax | mov edx, eax   mov al, 0x10  lea ax, [ex + 0x10] |                               |

+ **AT & T汇编** VS **Intel汇编** 寻址方式

  | AT&T                       |               Intel               |                                      说明 |
  | -------------------------- | :-------------------------------: | ----------------------------------------: |
  | imm(base,index,indexscale) | [base + index * indexscale + imm] | 两种结构的实际须知都是 imm + base + index |
  | -4(%ebp)                   |             [ebp - 4]             |                                           |
  | 0x40014(, %eax, 3)         |        [0x40014 + eax * 3]        |                                           |
  | 0x40014(%ebx, %eax,2)      |     [ebx + eax * 2 + 0x40014]     |                                           |
  | movw $6, %ds:(%eax)        |      mov word ptr ds:[eax],6      |                                           |

+ 64位AT&T汇编语言的寄存器

  + 有16个常用64位寄存器
    + `%rax`、`%rbx`、`%rcx`、`%rdx`、`%rsi`、`%rdi`、`%rbp`、`%rsp`
    + `%r8`、`%r9`、`%r10`、`%r11`、`%r12`、`%r13`、`%r14`、`%r15`

  + 寄存器的具体用途
    + `%rx`作为函数返回值使用
    + `%rsp`指向栈顶
    + `%rdi`、`%rsi`、`%rdx`、`%rcx`、`%r8`、`%r9`等寄存器用于存放函数参数

+ 常见代码反汇编

  + `sizeof`
    + 计算大小
    + `sizeof`不是个函数，本质是编译器特性。
  + `a++  +  a++  +  a++`
    + 结果: 1 + 2 + 3 = 6
    + 可以通过汇编代码查看原理
  + `++a + ++a + ++a`
    + 结果: 2 + 3 + 4 = 9
  + `++a + a++ + ++a`
    + 结果: 2 + 2 + 4 = 8
  + `if-else`
  + `for`
  + `switch` 和 `if` 效率
    + 可以通过反汇编可以看出**switch**的执行效率比**if**高，switch在判断前已经计算好了判断，所以在jmp指令后不用一个个判断，直接跳到符合的结果。

+ `lldb`常用指令

  + 读取寄存器的值
    + `register read/格式`
    + `register read/x`
  + 修改内存中的值
    + `memory write 内存地址 数值`
    + `memory write 0x0000010 10`
  + `expression` 表达式
    + 可以简写: `expr 表达式`
    + `expression $rax`
    + `expression $rax = 1`
  + 修改寄存器的值
    + `register write 寄存器名称 数值`
    + `register write %rax 0`
  + 格式
    + `x是16进制`、`f是浮点`、`d是十进制`
  + `po 表达式`
  + `print 表达式`
  + 读取内存中的值
    + `x/数量 - 格式 - 字节大小 内存地址`
    + `x/3xw 0x0000010`
  + 字节大小
    + `b - byte` 1字节
    + `h - half word` 2字节
    + `w - word` 4字节
    + `g - giant word` 8字节
  + `po/x $rax`
  + `po (int)$rax `

