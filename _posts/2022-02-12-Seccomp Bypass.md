---
title: Bypassing Seccomp
author: P0ch1ta
date: 2022-02-13 12:33:00 +0530
categories: [Pwning]
tags: [Seccomp Bypass, Kernel Seccomp]
math: true
mermaid: true
---

# What is Seccomp?

A large number of system calls are exposed to every userland process with many of them going unused for the entire lifetime of the process. Seccomp filtering provides a means for a process to specify a filter for incoming system calls. The filter is expressed as a Berkeley Packet Filter (BPF) program, as with socket filters, except that the data operated on is related to the system call being made: system call number and the system call arguments.

Seccomp has three primary modes:

* `SECCOMP_MODE_STRICT` — Turn all security measures that Seccomp provides on
* `SECCOMP_MODE_FILTER` — Allows the developer/user to restrict certain actions via filters
* `SECCOMP_MODE_DISABLED` — Disable Seccomp on the machine

We can easily find which syscalls are blocked by the process by using <a href="https://github.com/david942j/seccomp-tools">seccomp tools</a>

# Bypassing Seccomp

Bypassing Seccomp is the userland is impossible and it can only be evaded due to improper implementation. For the purpose of demonstration I will be exploiting the *gissa 2* challenge in the **Midnight Sun CTF Quals 2019**. In this challenge the seccomp was not properly implementated and can be bypassed easily. You can download <a href="https://github.com/manasghandat/manasghandat.github.io/blob/master/assets/img/chal/gissa_igen" download>here</a>.

If we use `seccomp-tool` on this file we get the following result.

```
seccomp-tools dump ./gissa_igen
```

<img src="https://github.com/manasghandat/manasghandat.github.io/raw/master/assets/img/Images/Blog2/1.png" alt="Seccomps that are disabled">

As we can see we get that the syscalls like `open` , `execve` , etc are disabled but on a close look we find that this filter is different from what the general seccomp. 

<img src="https://github.com/manasghandat/manasghandat.github.io/raw/master/assets/img/Images/Blog2/2.png" alt="Seccomp when properly implemented">

As we can clearly see that on line 3 and 4 of the latter example it blocks syscalls that have value greater than `0x40000000`. IN the case of first example we can try to pass syscalls that are greater than `0x40000000`. We add `0x40000000` the offset to the original syscall value o get those syscalls. Thus seccomp is bypassed. 

You can find the writeup in detail of the challenge in the references section.

# Advanced Bypassing techniques

We saw that in the above examples seccomp filters are applied at the start of the process but it can also be applied at the end of the process. In that case if the process creates a child process then the seccomp will not be applied to the child. Thus we can read the memory of the child process.

One such challenge was in **Google CTF 2020** called *write only*. In that challenge all syscalls except `open` and `write` were blocked. Thus we had to read the flag from the memory of the child process. A writeup to that challenge can be found in the resources section.

# Disabling Seccomp

Seccomp is implemented using the Berkeley Packet Filter and is done by the kernel using the `__secure_computing` (more info <a href="https://elixir.bootlin.com/linux/latest/source/kernel/seccomp.c#L1325">here</a>). The information about the process seccomp is stored inside `seccomp` struct which is inside the `task_struct` (more info <a href="https://elixir.bootlin.com/linux/latest/source/include/linux/sched.h#L1124">here</a>). The kernel sets a `TIS_SECCOMP` bit to `1` to indicate that seccomp is enabled.

If we manage to change the `TIF_SECCOMP` bit to `0` then we would disable the seccomp in the process. Current `task_struct` of the process is stored in the `gs` register. The `seccomp` struct is at an offset of `0x15d00` from the base of the `gs` register. On referencing this address we find the variable `flags` whose 8th bit is the bit we want to reset.

```cpp
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/cred.h>

MODULE_LICENSE ("GPL");

void disable_seccomp(void){
    current->thread_info.flags &= ~(_TIF_SECCOMP);
}
```

The following module disables the seccomp in the current process that we are working.

# Writing the Shellcode

The above code works fine but when we objdump the following code in order to convert it to working shellcode we can run the following command to get the assembly.

```
Objdump -M intel -d (kernel module name)
```

<img src="https://github.com/manasghandat/manasghandat.github.io/raw/master/assets/img/Images/Blog2/3.png" alt="Disassembled view">

We can then use the shellcode that is generated ...... but it wont work. If we carefully see we find that the offset of the `gs` register is wrong. This is because `fs` and `gs` are exceptions that were added to address thread-specific data. Their real base addresses are stored in MSRs (model specific registers) instead of the descriptor table. The MSRs are only accessible in kernel mode. Thus we have to manually write assemble and then convert it to shellcode using <a href="https://defuse.ca/online-x86-assembler.htm#disassembly">defuse.ca</a>

I would not be providing the exact details on how to generate the shellcode since it is a challenge in <a href="https://dojo.pwn.college/">pwn college</a> kernel module. You can always ask for hints to solve this challenge on their official <a href="https://pwn.college/discord">discord</a> server. If you want to try this exact challenge then solve the `babykernel_level8.0` challenge in the kernel module of pwn college.

## References

Article 1 : https://ajxchapman.github.io/linux/2016/08/31/seccomp-and-seccomp-bpf.html <br>
Article 2 : http://blog.redrocket.club/2019/04/11/midnightsunctf-quals-2019-gissa2/ <br>
Article 3 : https://blog.bi0s.in/2020/08/24/Pwn/GCTF20-Writeonly/ <br>
Article 4 : https://reverseengineering.stackexchange.com/questions/21033/windbg-why-does-the-gs-register-resolve-to-offset-0x0
