---
layout: post
title: "ptrace反调试"
date: 2020-05-10 22:51:00.000000000 +09:00
categories: [逆向工程]
tags: [逆向工程, ptrace, 反调试]
---

## iOS调试程序

`LLDB`调试是Xcode自带的调试工具，它既可以本地调试Mac应用程序，也可以远程调试iPhone应用程序。当使用Xcode调试iPhone应用程序时，Xcode会将`debugsever`文件复制到手机中，以便在手机上启动一个服务，等待Xcode进行远程调试，然后通过LLDB调试指令发给手机`debugserver`进行调试。

`debugserver`是通过`ptrace`函数调试应用程序的，`ptrace`是系统函数，此函数提供一个进程去监听和控制另一个进程，并且可以检测被控制进程的内存和寄存器里面的数据。ptrace可以用来实现断点调试和系统调用跟踪。

## 什么是ptrace

`ptrace()`系统调用函数提供了一个进程`the tracer`监察和控制另一个进程`the tracee`的方法。并且可以检查和改变“tracee”进程的内存和寄存器里的数据。它可以用来实现断点调试和系统调用跟踪。

+ MyPtrace.h

  ```c
  #ifndef    _SYS_PTRACE_H_
  #define    _SYS_PTRACE_H_
  
  #include <sys/appleapiopts.h>
  #include <sys/cdefs.h>
  
  enum {
      ePtAttachDeprecated __deprecated_enum_msg("PT_ATTACH is deprecated. See PT_ATTACHEXC") = 10
  };
  
  
  #define    PT_TRACE_ME    0    /* child declares it's being traced */
  #define    PT_READ_I    1    /* read word in child's I space */
  #define    PT_READ_D    2    /* read word in child's D space */
  #define    PT_READ_U    3    /* read word in child's user structure */
  #define    PT_WRITE_I    4    /* write word in child's I space */
  #define    PT_WRITE_D    5    /* write word in child's D space */
  #define    PT_WRITE_U    6    /* write word in child's user structure */
  #define    PT_CONTINUE    7    /* continue the child */
  #define    PT_KILL        8    /* kill the child process */
  #define    PT_STEP        9    /* single step the child */
  #define    PT_ATTACH    ePtAttachDeprecated    /* trace some running process */
  #define    PT_DETACH    11    /* stop tracing a process */
  #define    PT_SIGEXC    12    /* signals as exceptions for current_proc */
  #define PT_THUPDATE    13    /* signal for thread# */
  #define PT_ATTACHEXC    14    /* attach to running process with signal exception */
  
  #define    PT_FORCEQUOTA    30    /* Enforce quota for root */
  #define    PT_DENY_ATTACH    31
  
  #define    PT_FIRSTMACH    32    /* for machine-specific requests */
  
  __BEGIN_DECLS
  
  
  int    ptrace(int _request, pid_t _pid, caddr_t _addr, int _data);
  
  
  __END_DECLS
  
  #endif    /* !_SYS_PTRACE_H_ */
  ```

```c
int ptrace(int _request, pid_t _pid, caddr_t _addr, int _data);
```

- `_request`: 表示要执行的操作类型，我们反调试会用到`PT_DENY_ATTACH`，也就是去除进程依附
- `_pid`: 要操作的目的进程ID，因为我们是反调试，所以就传递0，表示对当前进程进行操作
- `_addr`: 要监控的内存地址，目前用不上所以就传0
- `_data`: 保存读取出或者要写入的数据，也用不上，所以就传0

合到一句简单的代码:

```c
ptrace(PT_DENY_ATTACH, 0, 0, 0)
```

在iOS工程中，我们没法儿直接引入`sys/ptrace.h`，这是因为苹果没有对iOS项目公开。不过，我们可以先新建一个macOS下的`command Line Tool`类型工程，在这个工程中进入到`sys/ptrace.h`文件里面，然后复制文件内的所有内容，放到iOS工程里我们随便新建的一个`.h`文件里面，比如`my_ptrace.h`。这样，我们就可以通过`import my_ptrace.h`，做到在iOS工程里面调用`ptrace`了。

## 编写ptrace

ptrace反调试实现方式有很多种，比如:

```
if (ptrace(PTRACE_TRACEME, 0) < 0) {
	printf("This process is being traced!");
	exit(-1);
}
```

#### 调用ptrace函数方式

```c
#import <UIKit/UIKit.h>
#import "AppDelegate.h"
#import "MyPtrace.h"
#import <dlfcn.h>

int main(int argc, char * argv[]) {
    
// 反调试
#ifndef PT_DENY_ATTACH
#define PT_DENY_ATTACH 31
#endif
    typedef int (*ptrace_ptr_t)(int _request, pid_t _pid, caddr_t _addr, int _data);
    ptrace(PT_DENY_ATTACH, 0, 0, 0);
    
  // 二次反调试
    void *handle = dlopen(0, RTLD_GLOBAL | RTLD_NOW);
    ptrace_ptr_t ptrace_ptr = (ptrace_ptr_t)dlsym(handle, "ptrace");
    ptrace_ptr(PT_DENY_ATTACH,0,0,0);
    
    @autoreleasepool {
        return UIApplicationMain(argc, argv, nil, NSStringFromClass([AppDelegate class]));
    }
}
```

这种调用`ptrace`可以做到反调试，但是会很容易通过fishhook攻破，我们可以换种稍微相对安全点的方案来做。我们知道`ptrace`的本质是一种linux的系统调用函数，所以可以通过直接调用系统函数的方式来调用`ptrace`。

系统调用的API:

```c
int syscall(int, ...);
```

系统调用类型很多，可以参照`sys/syscall.h`里面的定义，对于我们反调试来说，就是`SYS_ptrace`.

```c
syscall(SYS_ptrace, PT_DENY_ATTACH, 0, 0, 0)
```

#### 汇编方案

关于ARM64汇编知识可以看篇文章[ARM64](https://jovins.cn/posts/11ARM64Assembly/)

使用`syscall`的方案稍微相对安全一点，但是都是API调用，无法避免被fishhook攻破。最直接的方式是直接用内联汇编的方式来实现。

```
static __attribute__((always_inline)) void asm_ptrace() {
#ifdef __arm64__
    __asm__("mov X0, #31\n"
            "mov X1, #0\n"
            "mov X2, #0\n"
            "mov X3, #0\n"
            "mov X16, #26\n"
            "svc #0x80\n"
            );
#endif
}
```

这段汇编代码就是`ptrace`调用的汇编写法。其次，`X0`、`X1`、`X2`、`X3`寄存器，存贮着我们调用`ptrace`的传参。

```
static __attribute__((always_inline)) void asm_ptrace()
```

定义了一个C方法`asm_ptrace`，同时设置为内联函数inline，在编译阶段，就会把这段代码复制到各个调用位置，最终编译的结果里面，我们调用了几次，这段代码就会出现几次，揉杂在其他汇编里，分散在各处加大了攻破的难度。

> `SVC`：进入异常同步，即使CPU跳转到同步异常的入口地址
>
> int 0x80`对应的就是`SVC #0x80