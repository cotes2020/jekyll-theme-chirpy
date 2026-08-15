-------------

### ASSEMBLY

--------------

- Common verbs-:

```asm
add (some data together)
sub (subtract)
mul (multiply)
div divide some data
mov move some data into or out of storage
comp (compare two pieces of data together)
test some other properties of data
```

- Registers
 -  CPUs need to ne fast, CPUs need a place to access data
 -  Registers are fast, temporary places to store data
 -  You get several "general purpose" register
   ```asm
    x86: eax,ecx,edx,ebx,esp,ebp,esi,edi
arm: r0 -r14
```
-  The address of the next instruction is in a register-:
```asm
eio(x86), rip(amd64),r15(arm)
```
- Load registers into assembly with `mov`

```
mov rax, 0x539
mov ax, 1337
```

- 32-bit caveat ( if you write to a 32-bit partial e.g eax, the CPU will zero out the rest from the register.

```asm
mov rax to 0xfffffffffffffffffff
mov eax, 0x539
```

- You can move a general purpose register to another general purpose register.
- Consider

```asm
mov eax, -1
```
- If you want to operate on that -1 in 64-bit land

```asm
mov eax, -1
movsx rax, eax
```

- `movsx` does a sign-extending move by copying the top bit to the rest of the register.

---------------

### Register arrithmetic

-------------

- Don't mess with `rip`, it stores the memory address of next instruction to be executed.
- Also, don't forget about `rsp` as it contains the address of a region to store temporary data (sp = Stack Pointer)

<img width="658" height="318" alt="image" src="https://github.com/user-attachments/assets/c6bc338c-1aeb-4cf8-88f4-ca81cbf98b43" />

------------

### Memory

-------------

- Process memory is addressed linearly.

```
0x10000 ( for security reasons)
0x7ffffffffffffff ( for architecture/ OS purposes)
```

- 

---------------
