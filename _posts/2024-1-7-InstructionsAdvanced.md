---
title: RISC-V instructions - advanced
date: 2024-1-7 15:35:00 +0900
categories: [컴퓨터구조(Computer Architecture)]
tags: [instructions, RISC-V, computer architecture]
math: true
img_path: /assets/img/post5/
image: preview2.png
---

이 포스트는 RISC-V의 Instructions 중 Conditional Operations와 Procedure Call Instructions에 대해 설명한다.

## Conditional Operations
만약 조건에 맞으면 지정한 Instruction으로 이동하고 그렇지 않다면 그냥 다음 Instruction 실행.

beq rs1, rs2, L1: rs1 == rs2 이면 L1으로 이동

bne rs1, rs2, L1: rs1 != rs2이면 L1으로 이동

blt rs1, rs2, L1: rs1 < rs2이면 L1으로 이동

bge rs1, rs2, L1: rs1 >= rs2이면 L1으로 이동

ex1) 
```c
if (i==j) f = g+h;
else f = g-h;
```

```
assembly code
//f, g, h in x19, x20, x21
//i, j in x22, x23
    bne x22, x23, Else
    add x19, x20, x21
    beq x0, x0, Exit //unconditional
Else: sub x19, x20, x21
Exit: 
```
bne에서 i==j을 확인하고 같으면 그대로 진행, 다르면 Else로 이동. 그대로 진행 할 때는 Else 부분 코드 실행하면 안되니까 beq 무조건 참으로 만들어서 Exit으로 이동.

ex2)
```c
while (save[i]==k) i+=1;
```

```
assembly code
//i, k in x22, x24
//address of save in x25
//assume that array save has 8 byte big data type.(ex. long long)

Loop: slli x10, x22, 3
    add x10, x10, x25
    ld x9, 0(x10)
    bne x9, x24, Exit
    addi x22, x22, 1
    beq x0, x0, Loop
Exit:
```
한 칸에 8byte니까 i에 8을 곱해주고 그것을 base address에 더해야 save[i]의 주소가 나온다.

ex3)
```c
if (a > b) a+=1;
```

```
assembly code
//a, b in x22, x23
    bge x23, x22, Exit
    addi x22, x22, 1
Exit:
```


### Basic Blocks
branch 없고(마지막 빼고) branch target 없는(처음 빼고) sequence of instructions를 Basic Block이라 한다. 컴파일러는 프로그램을 Basic Blocks로 분해해서 최적화 과정을 진행한다.

### Signed vs Unsigned
앞서 배운 blt, bge은 signed comparison을 한다. 하지만 bltu, bgeu는 unsigned comparison을 한다.

x22 = 0xFFFFFFFF  
x23 = 0x00000001  
signed comparison 이면 x22 = -1, x23 = 1 이라서 x22 < x23이라고 한다.  
하지만 unsigned comparison이면 x22 = 4294967295, x23 = 1 이라서 x22 > x23이라고 한다.

## Procedure Call Instructions
c언어의 함수를 생각하면 된다.  
```c
int func()
{
    return 1; //procedure return
}

int main(void)
{
    int a = func(); //procedure call
    return 0;
}
```
### Procedure call: jump and link
```
jal x1, ProcedureLabel1
```
이 instruction의 다음 instruction의 주소를 x1 register에 저장해두고 ProcedureLabel1으로 jump한다.  

### Procedure return: jump and link register
```
jalr x0, 0(x1)
```
return 시키고 x1+0의 주소로 jump한다.

### Leaf Procedures
leaf(잎)은 여기서 끝이라는 의미다. 즉 다른 함수를 또 call 하지 않고 값을 return 하는 함수를 가리킨다.
```c
long long int leaf_example (
    long long int g, long long int h, long long int i, long long int j
)
{
    lont lont int f;
    f = (g + h) - (i + j);
    return f; 
}
```

```
assembly code
//g, h, i, j in x10, x11, x12, x13
//f in x20
//x5, x6 temporaries
//need to save original values of x5, x6, x20 on stack
leaf_example:
    addi sp, sp, -24 //sp stands for stack pointer, sp == x2
    sd x5, 16(sp) //push
    sd x6, 8(sp) //push
    sd x20, 0(sp) //push
    add x5, x10, x11 //x5 = g + h
    add x6, x12, x13 //x6 = i + j
    sub x20, x5, x6 //f = x5 - x6
    addi x10, x20, 0 //copy f to return register
    ld x20, 0(sp) //pop
    ld x6, 8(sp) //pop
    ld x5, 16(sp) //pop
    addi sp, sp, 24
    jalr x0, 0(x1) //return to caller
```
![stack](6.png)
register 하나에 8byte(64bit)라는 가정하에서 register 3개 저장해야하니 24를 내리는거다.

### Register Usage
register 마다 각각 쓰임이 있다. 이것은 대부분 지키는 관용적인 것이다.
- x5~x7, x28~x31: temporary registers
이것들은 stack에 굳이 넣어서 저장했다가 pop하지 않고 그냥 덮어써도 되는 register이다.
- x8, x9, x18~x27: saved registers
이것들은 stack에 저장해줘야한다. 

### non-leaf procedures
leaf가 아니면 다른 함수를 call 하게 된다. 이런 함수들은 다음 함수를 call 할 때 돌아올 주소(return address)와 call다음에 필요한 arguments 와 temporaries를 저장해놔야 한다.  
예시와 함께 살펴보자.

```c
long long int fact (long long int n)
{
    if (n < 1) return 1;
    else return n * fact(n-1);
}
```

```
assembly code
//argument n and result in x10
fact:addi sp, sp, -16 //save return address and n on stack
    sd x1, 8(sp)
    sd x10, 0(sp)
    addi x5, x10, -1 //x5=n-1
    bge x5, x0, L1 //if n-1>=0, goto L1
    addi x10, x0, 1 //else, set return value to 1
    addi sp, sp, 16 //pop stack, don't bother restoring values
    jalr x0, 0(x1) //return
L1:addi x10, x10, -1 //n=n-1
    jal x1, fact //call fact(n-1)
    addi x6, x10, 0 //move result of fact(n-1) to x6
    ld x10, 0(sp) //restore caller's n
    ld x1, 8(sp) //restore caller's return address
    addi sp, sp, 16 //pop stack
    mul x10, x10, x6 //return value = n*fact(n-1)
    jalr x0, 0(x1) //return
```

### Memory layout & stack
![memory](7.png)
64-bit architecture에서는 메모리 주소가 64bit까지 가능하니 $$2^{64}$$개의 메모리 주소가 존재한다. 그중 $$2^{38}$$개의 메모리 주소만이 user가 접근 가능한 메모리 공간이다.  
Text는 program code가 있는 곳이다.  
Dynamic data는 malloc in C, new in Java처럼 메모리 공간을 할당해주는 느낌이다.  
Stack은 아까 위에서 봤던 그 stack이다. sp가 아래로 내려가면서 stack 공간을 늘린다. argument또는 temporaries를 저장할 수 있다.

![stack](9.png)
stack공간에 precedure이 진행되는 동안 local data를 저장하는 부분을 procedure frame이라고 한다. 그림과 같은 것들이 저장된다. 이게 위의 memory layout에서 나온 아래로 자라는 stack부분이다. 
