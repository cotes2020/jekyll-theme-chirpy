---
title: High vs Low Programming Languages
# author: Grace JyL
date: 2021-10-05 11:11:11 -0400
description:
excerpt_separator:
categories: [00CodeNote]
tags: []
math: true
# pin: true
toc: true
# image: /assets/img/note/tls-ssl-handshake.png
---

- [programming languages](#programming-languages)
  - [low-level programming languages](#low-level-programming-languages)
    - [machine code](#machine-code)
    - [Object code](#object-code)
    - [Assembly code](#assembly-code)
  - [High-level programming languages](#high-level-programming-languages)
    - [FORTUNE语言](#fortune语言)
    - [FORTRAN](#fortran)


---

# programming languages


Building a complete program involves:
- writing source code for the program in either `assembly` or `higher level language like C++`.

- The `source code` is **assembled (for assembly code) or compiled (for higher level languages)** to object code, and individual modules are linked together to become the machine code for the final program.
- In the case of very simple programs the linking step may not be needed.
  - In other cases, such as with an IDE (integrated development environment) `the linker and compiler` may be invoked together.
  - In other cases, a complicated make script or solution file may be used to tell the environment how to build the final application.

```
The assembly code -> assembled -> object code,
The higher level languages code -> compiled -> object code,
```


![kKTfY](https://i.imgur.com/Vq0zKcD.png)


![SJO6y](https://i.imgur.com/5VftPC7.png)

```
Machine code 是低级语言，或者说是底层语言，

他是用二进制代码表示的，计算机能直接识别和执行的机器指令集合，用于直接控制CPU。他的优点有灵活，直接执行和速度快（越快越好）。不同种类的计算机其机器语言是不相容的。程序员可以直接写machine code，但是他就要计算每个bits的位置和数字，不科学。

所以我们有了高级语言
- 程序员用高级语言编程或者assembly code写代码source code，
- 然后用Assemblers或者compiler来把source code assemble/comply 成 machine language module (object file, 里面会有placeholders or offsets)，然后linker把这些不同的object library/file连接起来生成一个 executable file
- 或者有的Assembler也可以直接把source code assemble 成machine code。


然后高级语言也有分不同的：

Compiled languages (e.g. C, C++)
- Source code -> compiler -> object file ->
- runtime: linker -> machine executable file -> output (上述)

Interpreted programming languages (e.g. Python, Perl)
- it rely on the machine code of a special interpreter program.

- runtime: Source code -> interpreter （combines with runtime libraries）-> output

  - 程序不会有预编译，而是每次运行时都会进行一次转换过程
  - 这也是为什么compiled languages 一般比interpreted languages快
  - At the basic level, an interpreter parses the source code and immediately converts the commands to new machine code and executes them.
- Modern interpreters are now much more complicated
  - evaluating whole sections of source code at a time, caching and optimizing where possible, and handling complex memory management tasks.


Java又不一样了
- source code -> Java compiler (javac) -> runtime: bytecode.

- bytecode file (.class file) can be run on any operating system by using the Java interpreter (java) for that platform.
- The interpreter is referred to as a Virtual Machine.
- Thus, Java is an example of a Virtual Machine programming language.
```

the use of a `runtime-environment` or `virtual machine`.
- In this situation, a program is first pre-compiled to a lower-level intermediate language or byte code.
- The `byte code` is then loaded by the virtual machine, which just-in-time compiles it to native code.
- The advantage here is the virtual machine can take advantage of optimizations available at the time the program runs and for that specific environment.
- A compiler belongs to the developer, and must produce `relatively generic (less-optimized) machine code` that could run in many places.
- The runtime environment or virtual machine, however is located on the end user's computer and therefore can take advantage of all the features provided by that system.




----

## low-level programming languages


### machine code

machine code is any low-level programming language,
- machine code 是电脑的CPU可直接解的资料
- 是用二进制代码表示的、计算机能直接识别和执行的一种机器指令的集合。
- 它是计算机的设计者通过计算机的硬件结构赋予计算机的操作功能。
- 机器语言具有灵活、直接执行和速度快等特点。
- 不同种类的计算机其机器语言是不相容的，按某种计算机的机器指令编制的程式不能在另一种计算机上执行。
- 要用机器语言编写程序，编程人员需首先熟记所用计算机的全部指令代码和代码的涵义。
  - 手编程序时，程序员要自己处理每条指令和每一数据的存储分配和输入输出，还需记住编程过程中每步所使用的工作单元处在何种状态。这是一件十分繁琐的工作，编写程序花费的时间往往是实际运行时间的几十倍或几百倍。
  - 而且，这样编写出的程序完全是0与1的指令代码，可读性差且容易出错。
  - 在现今，除了计算机生产厂家的专业人员外，绝大多数程序员已经不再学习机器语言。



- binary (1's and 0's) code that can be executed directly by the CPU.

- used to control a computer's central processing unit (CPU).
  - Each instruction causes the CPU to perform a very specific task,
  - such as `a load, a store, a jump, or an arithmetic logic unit (ALU) operation`
  - on one or more units of data in the CPU's registers or memory.
  - 现今存在着超过100000种机器语言的指令。

- a strictly numerical language designed to run as fast as possible
  - and may be considered as the lowest-level representation of
    - a `compiled`
    - or `assembled computer program`
    - or as a `primitive and hardware-dependent programming language`

While it is possible to write programs directly in machine code, managing individual bits and calculating numerical addresses and constants manually is tedious and error-prone.

For this reason, programs are very rarely written directly in machine code in modern contexts, but may be done for low level debugging, program patching (especially when assembler source is not available) and assembly language disassembly.

The majority of practical programs today are written in `higher-level languages` or `assembly language`.
- The source code is then translated to
  - `executable machine code` by utilities such as `compilers`, `assemblers`, and `linkers`,
  - with the `important exception of interpreted programs`, which are not translated into machine code.
- However, the `interpreter` itself,
  - which may be seen as an executor or processor performing the instructions of the source code,
  - typically consists of directly executable machine code (generated from assembly or high-level language source code).

Machine code is by definition the lowest level of programming detail visible to the programmer
- but internally many `processors` use `microcode` or `optimise` and `transform machine code instructions into sequences of micro-ops`. This is not generally considered to be a machine code.


以下是一些範例：

```bash
# Machine code is pure binary
# it is an amount of stored electricity in a circuit

# Hex code is just a convenient representation of binary
5F 3A E3 F1

# 指令部份的範例
0000 代表 載入（LOAD）
0001 代表 儲存（STORE）
...
# 暫存器部份的範例
0000 代表暫存器 A
0001 代表暫存器 B
...
# 記憶體部份的範例
000000000000 代表位址為 0 的記憶體
000000000001 代表位址為 1 的記憶體
000000010000 代表位址為 16 的記憶體
100000000000 代表位址為 2^11 的記憶體
# 整合範例
0000,0000,000000010000 代表 LOAD A, 16
0000,0001,000000000001 代表 LOAD B, 1
0001,0001,000000010000 代表 STORE B, 16
0001,0001,000000000001 代表 STORE B, 1
```

---


### Object code

- a portion of machine code not yet linked into a complete program.
- It's the machine code for one particular library or module that will make up the completed product.
- It may also contain placeholders or offsets not found in the machine code of a completed program.
- The `linker` will use these `placeholders` and `offsets` to connect everything together.

- In computing, object code or object module is the product of a compiler.
- In a general sense object code is a sequence of statements or instructions in a computer language, usually a `machine code language (i.e., binary)` or an intermediate language such as `register transfer language (RTL)`.
- The term indicates that the code is the goal or result of the compiling process, with some early sources referring to source code as a "subject program".


- Object files can in turn be linked to form an `executable file` or `library file`.
- In order to be used, object code must either be placed in an `executable file, a library file, or an object file`.

Object code is a portion of machine code that has not yet been linked into a complete program. It is the machine code for one particular library or module that will make up the completed product. It may also contain placeholders or offsets, not found in the machine code of a completed program, that the linker will use to connect everything together.

Whereas machine code is binary code that can be executed directly by the CPU, object code has the jumps partially parametrized so that a linker can fill them in.

- An assembler is used to convert assembly code into machine code (object code).
- A linker links `several object (and library) files` to generate an executable.
- Assemblers can also assemble directly to machine code executable files without the object intermediary step.



```
lmylib.so containing 8B 5D 32 is object code
```

---

### Assembly code

- a low-level language for programming computers.
  - but intermediate language between high-level language and machine code.

- It implements a symbolic representation of the numeric machine codes and other constants needed to program a particular CPU architecture.


- plain-text and (somewhat) human read-able source code
- mostly has a direct 1:1 analog with machine instructions.


- This is accomplished using mnemonics for the actual instructions, registers, or other resources.
- Examples include `JMP` and `MULT` for the `CPU's jump and multiplication instructions`.

- Unlike machine code, the CPU does not understand assembly code.
  - You `convert assembly code to machine code` with the use of an `assembler` or a `compiler`,
  - though we usually think of `compilers` in association with high-level programming language that are abstracted further from the CPU instructions.



```bash
# Assembly code is a human readable representation of machine code:
mov eax, 77
jmp anywhere

rainbow_lp:
  lda ColorTbl,x
  sta WSYNC
  sta COLUBK
  dex
  bpl rainbow_lp
```


---

## High-level programming languages

In computer science, a high-level programming language is a `programming language with strong abstraction from the details of the computer`.
- In contrast to low-level programming languages, it may use natural language elements, be easier to use, or may automate (or even hide entirely) significant areas of computing systems (e.g. memory management), making the process of developing a program simpler and more understandable than when using a lower-level language.
- The amount of abstraction provided defines how "high-level" a programming language is.


In the 1960s, high-level programming languages using a compiler were commonly called autocodes.[2] Examples of autocodes are COBOL and Fortran.

- The first high-level programming language designed for computers was `Plankalkül`, created by Konrad Zuse.
  - However, it was not implemented in his time, and his original contributions were largely isolated from other developments due to World War II, aside from the language's influence on the "Superplan" language by Heinz Rutishauser and also to some degree Algol.
- The first significantly widespread high-level language was Fortran, a machine-independent development of IBM's earlier Autocode systems.
- The Algol family, with Algol 58 defined in 1958 and Algol 60 defined in 1960 by committees of European and American computer scientists, introduced recursion as well as nested functions under lexical scope.
  - Algol 60 was also the first language with a clear distinction between value and name-parameters and their corresponding semantics.
  - Algol also introduced several structured programming concepts, such as the while-do and if-then-else constructs and its syntax was the first to be described in formal notation – "Backus–Naur form" (BNF).
- During roughly the same period, Cobol introduced records (also called structs) and Lisp introduced a fully general lambda abstraction in a programming language for the first time.

---

### FORTUNE语言
- 比较老的计算机语言，比尔当年创造的第一代计算机语言.
- 医学模型分析都是一些数值分析程序，很多都是用Fortune等语言来编程的.
- 应用于比较专业的领域，一般接触不到的，也就是说你学了也基本上应用不上，而且学起来是非常困难的！

---


> vs FORTUNE, 这两种语言的共同点是都属于比较老的计算机语言。

### FORTRAN
- “FORmula TRANslator”的缩写,译为“公式翻译器”,
- 世界上最早出现的计算机高级程序设计语言,广泛应用于科学和工程计算领域。
- FORTRAN语言以其特有的功能在数值、科学和工程计算领域发挥着重要作用。

- Fortran语言是為了滿足数值计算的需求而發展出來的。
  - 1953年12月，IBM公司工程師約翰·巴科斯（J. Backus）因深深體會編寫程序很困難，而寫了一份備忘錄給董事長斯伯特·赫德（Cuthbert Hurd），建議為IBM704系統設計全新的電腦語言以提升開發效率。
  - 當時IBM公司的顾问冯·诺伊曼强烈反对，因為他認為不切實際而且根本不必要。
  - 但赫德批准了這項計劃。
  - 1957年，IBM公司开发出第一套FORTRAN语言，在IBM704電腦上運作。歷史上第一支FORTRAN程式在馬里蘭州的西屋貝地斯核電廠試驗。1957年4月20日星期五的下午，一位IBM軟體工程師決定在電廠內編譯第一支FORTRAN程式，當程式碼輸入後，經過編譯，印表機列出一行訊息：“原始程式錯誤……右側括號後面沒有逗號”，這讓現場人員都感到訝異，修正這個錯誤後，印表機輸出了正確結果。而西屋電氣公司因此意外地成為FORTRAN的第一個商業用戶。1958年推出FORTRAN Ⅱ，幾年後又推出FORTRAN Ⅲ，1962年推出FORTRAN Ⅳ後，開始廣泛被使用。
- Fortran语言的最大特性是接近数学公式的自然描述，在计算机里具有很高的执行效率。易学，语法严谨。
- 可以直接对矩阵和复数进行运算，這點Matlab有繼承。
- 自诞生以来广泛地应用于数值计算领域，积累了大量高效而可靠的源程序。
- 很多专用的大型数值运算计算机针对Fortran做了优化。
- 广泛地应用于并行计算和高性能計算领域。
- Fortran 90，Fortran 95，Fortran 2003的相继推出使Fortran语言具备了现代高级编程语言的一些特性。
- 其矩陣元素在記憶空間儲存順序是採用`列優先（Column major）`，Matlab也承襲這點，目前最多使用的C語言則採用`行優先（Row major）`。

```py
! 驚嘆號之後是註解
program main ! 這行可以省略，但是寫大程式的時候會發生混亂
    write (*,*) "hello, world!" ! 第一個* 表示輸出縮排使用內定值，第二個* 表示不指定輸出格式
    write (unit = *, fmt = * ) "hello, world!" ! 做和上一行一樣的事
    stop ! 這行代表程式結束，可以省略
end program main ! end之後的program main也可以省略，但寫上是比較嚴謹
```
