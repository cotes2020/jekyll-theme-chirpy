---
layout: post
title: "Process loading in linux"
categories: explained
tags: [under the hood,linux]
---

## Overview

In this article we will explain how linux kernel loads a process into memory

## What is a process

Everyone who uses linux will eventually hear the word "process" but what is a really a process?? 

Every program on your computer when it starts running it runs as a process. Your browser,terminal,editor... are all considered processes.

A process is composed of:

### task_struct: 

So that Linux can manage the processes in the system, each process is represented by a  task_struct  data structure (task and process are terms which Linux uses interchangeably). The  task vector is an array of pointers to every task_struct data structure in the system.

This means that the maximum number of processes in the system is limited by the size of the  task vector. By default it has 512 entries. As processes are created, a new  task_struct  is allocated from system memory and added into the  task  vector. To make it easy to find, the current, running, process is pointed to by the  current  pointer.

Although the task_struct  data structure is quite large and complex, but its fields can be divided into a number of functional areas:

### State: 

As a process executes it changes state according to its circumstances. Linux processes have the following states

- Running:
The process is either running (it is the current process in the system) or it is ready to run (it is waiting to be assigned to one of the system's CPUs).

- Waiting:
The process is waiting for an event or for a resource. Linux differentiates between two types of waiting process: interruptible and uninterruptible. Interruptible waiting processes can be interrupted by signals whereas uninterruptible waiting processes are waiting directly on hardware conditions and cannot be interrupted under any circumstances.

- Stopped:
The process has been stopped, usually by receiving a signal. A process that is being debugged can be in a stopped state.

- Zombie:
This is a halted process which, for some reason, still has a task_struct data structure in the task vector. It is what it sounds like, a dead process.

### Scheduling Information

The scheduler needs this information in order to fairly decide which process in the system most deserves to run

### Identifiers

Every process in the system has a process identifier. The process identifier is not an index into the task vector, it is simply a number. Each process also has User and group identifiers, these are used to control this processes access to the files and devices in the system
it's a security measure so that a process can't abuse resources on the system

### Parent, children, sibling

every process is forked and can fork another process. use the command ```pstree``` it will give the relation between your running processes

### Shared resources

depends on the behaviour of the process it can use some file descriptors, pipes, sockets, libraries...

### Virtual memory space

every process has it's own virtual memory space so that it doesn't interfere with the memory of other processes running. think of it like a virtual machine given to each process.

## Where do process comes from ?

process in linux propagate by mitosis

the system call fork creates an exact copy of the calling process so we can have a parent and a child. later the child will use the syscall execve to replace itself into another process. check my blog post named *what happens when you type ls in your terminal* to better understand that.

## Loading a process

### Can we load ?

We already mentioned that the system call ```execve``` will try to load this new process. but before that the kernel will check if the program is executable.

### What and how to load

to find out what to load the kernel will read the beginning of the file and make decisions:

**1.** If the file starts with **#!**, the kernel extracts the interpreter from the rest of that line and executes this interpreter with the original file as an argument. so if you have a shebang line like ```#!/bin/bash``` the kernel will use ```/bin/bash``` to execute your program

write a small script and use ```#!/bin/echo``` as your shebang line then executes it ...

**2.** If the file matches a format in **/proc/sys/fs/binfmt_misc**, the kernel executes the interpreter specified for that format with the original file as an argument. **binfmt_misc** has a set of configuration for different files type. when using ```ls /proc/sys/fs/binfmt_misc``` i have and output of ```jar  llvm-10-runtime.binfmt  python2.7  python3.8  register  status``` I can distinguish that i have the python and java interpreters. using ```cat python2.7``` i get 

    ```
    enabled
    interpreter /usr/bin/python2.7
    flags: 
    offset 0
    magic 03f30d0a
    ```

This means that if the files has the magic bytes 03f30d0a at offset 0 the Kernel will use python2.7 to run it.

### Loading a dynamically linked ELF

**1.** the kernel reads the interpreter/loader defined in the ELF, loads the interpreter and the original file, and lets the interpreter take control. using the command ```readelf -a a.out | grep interpreter``` on a compiled C program we get ```Requesting program interpreter: /lib64/ld-linux-x86-64.so.2``` this means that the kernel will load **ld-linux-x86-64.so.2** which will handle the loading of our program

**2.** The interpreter locates the libraries
    
  -  **LD_PRELOAD** environment variable, and anything in /etc/ld.so.preload
   - **LD_LIBRARY_PATH** environment variable (can be set in the shell)
   -  **DT_RUNPATH** or **DT_RPATH** specified in the binary file (both can be modified with patchelf)
    - system-wide configuration (/etc/ld.so.conf)
    - /lib and /usr/lib

**3.** Where is all this getting loaded to?

- the binary
    
-   the libraries
    
-   the "heap" (for dynamically allocated memory)
    
-   the "stack" (for function local variables)
    
-   any memory specifically mapped by the program
    
-   some helper regions
    
-   kernel code in the "upper half" of memory (above 0x8000000000000000 on 64-bit architectures), inaccessible to the process

  here is a mapping of /bin/cat on my system when executed
  
![](/assets/img/process-loading/mapping.png)  

#### Sum it up.

The execution of a program starts inside the kernel, in the exec("/bin/program",...) system call takes a path to the executable file. The kernel reads the ELF header and the program header table [PHT](https://docs.oracle.com/cd/E19683-01/816-1386/chapter6-83432/index.html#:~:text=An%20executable%20or%20shared%20object,described%20in%20%22Segment%20Contents%22.), followed by lots of sanity checks.

The kernel then loads the parts specified in the LOAD directives in the PHT into memory. If an INTERP entry is present, the interpreter is loaded too. Statically linked binaries can do without an interpreter. dynamically linked programs always need /lib64/ld-linux-x86-64.so.2 as interpreter because it includes some startup code, loads shared libraries needed by the binary, and performs relocations.

Finally control can be transferred to the entry point of the program or to the interpreter, if linking is required.

In case of a statically linked binary that's pretty much it, however with dynamically linked binaries a lot more has to go on.

First the dynamic linker (contained within the interpreter) looks at the .dynamic section, whose address is stored in the PHT.

There it finds the NEEDED entries determining which libraries have to be loaded before the program can be run, the **REL** entries giving the address of the relocation tables, the **VER** entries which contain symbol versioning information, etc.

So the dynamic linker loads the needed libraries and performs relocations (either directly at program startup or later, as soon as the relocated symbol is needed, depending on the relocation type).

Finally control is transferred to the address given by the symbol `_start` in the binary. Normally some gcc/glibc startup code lives there, which in the end calls main()

This was a quick beginner friendly resume of how a process is loaded in linux. There is a lot more hidden magic performed by the kernel. I recommend this [post](https://0xax.gitbooks.io/linux-insides/content/SysCall/linux-syscall-4.html) to get a deeper look at code level

I hope you learned new concepts and triggered your curiosity to explore more