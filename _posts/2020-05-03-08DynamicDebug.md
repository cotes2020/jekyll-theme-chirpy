---
layout: post
title: "08-动态调试"
date: 2020-05-02 13:22:00.000000000 +09:00
categories: [逆向工程]
tags: [逆向工程, 动态调试, debugserver, LLDB, ASLR]
---

## 什么叫动态调试

+ 将程序运行起来，通过下断点、打印等方式，查看参数、返回值、函数调用流程等。

## Xcode的动态调试原理

![1.xcodetiaoshi](/assets/images/reverse/1.xcodetiaoshi.png)

+ 关于`GCC`、`LLVM`、`GDB`、`LLDB`
  + Xcode的编译器发展历程: [GCC](https://www.gnu.org/software/gcc/) → [LLVM](https://llvm.org/)
  + Xcode的调试器发展历程: [GDB](https://www.gnu.org/software/gdb/) → [LLDB](https://lldb.llvm.org/)
+ `debugserver`一开始存放在Mac的Xcode里面
  + `/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/De viceSupport/9.1/DeveloperDiskImage.dmg/usr/bin/debugserver`
+ 当Xcode识别到手机设备时，Xcode会自动将`debugserver`安装到iPhone上
  + `/Developer/usr/bin/debugserver`
+ Xcode调试的局限性
  + 一般情况下，只能调试通过Xcode安装的App

## 动态调试任意App

![](/assets/images/reverse/2.dongtaitiaoshiapp.png)

+ #### debugserver的权限问题

  + 默认情况下，`/Developer/usr/bin/debugserver`权限一定是全新的，只能调试通过Xcode安装的App，无法调试其他App，比如来自App Store的App。

  + 如果希望调试其他App，需要对debugserver重签名，签上2个调试相关的权限

    > get-task-allow: YES
    >
    > task_for_pid-allow: YES

+ #### 如果给debugserver签上权限

  + iPhone上的`/Developer`目录是只读的，无法直接对`/Developer/usr/bin/debugserver`文件签名，需要把debugserver复制到Mac重签名。

  + 通过ldid命令导出文件以前的签名权限

    ```
    $ ldid -e debugserver > debugserver.entitlements
    ```

  + 给`debugserver.plist`文件加上`get-task-allow`和`task_for_pid-allow`权限

  ![](/assets/images/reverse/3.debugquanxian.png)

  + 通过ldid命令重新签名

  ````
  $ ldid -Sdebugserver.entitlements debugserver
  ````

  + 将已经签好权限的`debugserver`放到`/usr/bin`目录，便于找到`debugserver`指令

    > 有可能会出现: `-sh: /usr/bin/debugserver: Permission denied`
    >
    > // 执行
    >
    > $ `chmod +x /usr/bin/debugserver`

  + 关于权限的签名，也可以使用`codesign`

  ```
  # 查看权限信息
  $ codesign -d --entitlements - debugserver
  # 签名权限
  $ codesign -f -s - --entitlements debugserver.entitlements debugserver 
  # 或者简写为
  $ codesign -fs- --entitlements debugserver.entitlements debugserver
  ```

+ #### 让debugserver附加到某个App进程

  ```
  $ debugserver *:端口号 -a 进程
  jovinteki-iPhone:~ root# debugserver *:10011 -a WeChat
  debugserver-@(#)PROGRAM:LLDB  PROJECT:lldb-900.3.106
   for arm64.
  Attaching to process WeChat...
  Listening to port 10011 for a connection from *...
  Failed to get connection from a remote gdb process.
  Exiting.
  jovinteki-iPhone:~ root#
  ```

  + 问题1: 执行`debugserver`附加进程的时候报错`Failed to get connection from a remote gdb process. Exiting.`

    + 解决方式: 删除`debugserver` 的以下权限，重新签名

    ```
    com.apple.security.network.server
    com.apple.security.network.client
    seatbelt-profiles
    ```

  + 问题二: `lldd+debugserver`启动APP时`Attaching to process ting... Segmentation fault: 11`

    + 可能该App做了`ptrace反调试`。

  + ***:端口号**
    
    + 使用iPhone的某个端口启动`debugserver`服务(只要不是保留端口号就行)
  + **输入App的进程信息**(进程ID或者进程名称)

+ #### 在Mac上启动LLDB，远程连接iPhone上的debugserver服务

  + 启动LLDB

    ```
    $ lldb
    (lldb)
    ```

  + 连接`debugserver`服务

    ```
    (lldb) process connect connect://手机ip:debugserver服务端口
    // 因为手机的10011端口已经映射到电脑的10011端口
    // 所以可以localhost:10011这样写
    $ process connect connect://localhost:10011
    ```

    > 注意: 
    >
    > lldb窗口报：
    >
    > error: failed to get reply to handshake packet
    >
    > debugserver窗口报：
    >
    > error: rejecting incoming connection from ::ffff:127.0.0.1 (expecting ::1)
    > 解决办法: 
    >
    > 把debugserver启动试设置的监听 *:10011 改成 localhost:10011
    >
    > 也就是debugserver localhost:10011 -a WeChat

  + 连接成功，App是出于打断点、暂停状态。

    ```
    (lldb) process connect connect://localhost:10011
    Process 47921 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGSTOP
        frame #0: 0x0000000199e7c198 libsystem_kernel.dylib`mach_msg_trap + 8
    libsystem_kernel.dylib`mach_msg_trap:
    ->  0x199e7c198 <+8>: ret
    
    libsystem_kernel.dylib`mach_msg_overwrite_trap:
        0x199e7c19c <+0>: mov    x16, #-0x20
        0x199e7c1a0 <+4>: svc    #0x80
        0x199e7c1a4 <+8>: ret
    Target 0: (Carben) stopped.
    (lldb)
    ```

  + 使用`LLDB的c命令`让程序先继续运行

    ```
    (lldb) c    // continue
    ```

  + 接下来就可以使用LLDB命令调试App

+ #### 通过debugserver启动App

  ```
  $ debugserver -x auto *:端口号 APP的可执行文件路径
  ```

## 常用LLDB指令

+ 指令的格式

  ```
   <command> [<subcommand> [<subcommand>...]] <action> [-options [option-
  value]] [argument [argument...]]
  ```

  + `<command>` : 命令

  + `<subcommand>` : 子命令

  + `<action> `: 命令操作

  + `<options>`: 命令选项

  + `<arguments>` : 命令参数

  + 比如给test函数设置断点

    ```
    breakpoint set -n test
    ```

    ```
    $ lldb
    (lldb) process connect connect://localhost:10011
    ...
    (lldb) breakpoint set -n "[BaseMsgContentViewController scheduleInMsg:]"
    // 如果是已发布的App，也就是release版本的，不能通过函数名设置断点
    // 来调试App
    // 错误信息:
    // Breakpoint 1: no locations (pending).
    // WARNING: Unable to resolve breakpoint to any actual locations
    // 解决办法:
    // 可以通过方法的内存地址去设置断点
    (lldb) breakpoint set -a 0x102096354
    ```

    + `breakpoint`是<command>
    + `set`是<action>
    + `-n`是<options>
    + `test`是<arguments>

+ `help`

  + 查看指令的用法
  + 比如`help breakpoint`、`help breakpoint set`

+ `expression <cmd-options> -- <expr>`

  + 执行一个表达式

    +  `<cmd-options>` : 命令选项
    + -- : 命令选项结束符，表示所有的命令选项已经设置完毕，如果没有命令选项，--可以省略
    + <expr> : 需要执行的表达式

    ```
    (lldb) expression self.view.backgroundColor = [UIColor redColor]
    (lldb) expression self.view.backgroundColor = .red
    ```

  + `expression`、`expression --`和指令`print`、`p`、`call`的效果一样

  + `expression -O --`和指令`po`的效果一样

    ```
    expression -O -- self.view
    po self.view
    ```

+ `thread backtrace`

  + 打印现场的堆栈信息
  + 和指令`bt`的效果一样
  + 注意: frame由往上调用的

+ `thread return [<expr>]`

  + 让函数直接返回某个值，不会执行断点后面的代码

    ```swift
    	 private func test() {
            
    ->      let a = 10		// 设置断点
            let b = 20
            let total = a + b
            print("a + b = \(total)")
        }
    // 输入thread return,不会执行断点下面的代码
    ```

+ `frame variable [<variable-name>]`

  + 打印当前栈帧的变量

  ```
  (lldb) frame variable
  (Int) a = 10
  (Int) b = 20
  (Int) total = 30
  (lldb) frame variable a
  (Int) a = 10
  (lldb) frame variable total
  (Int) total = 30
  ```

+ `thread continue`、`continue`、`c` : 程序继续执行

+ `thread step-over`、`next`、`n` : 单步运行，把子函数当做整体一步执行

+ `thread step-in`、`step`、`s` : 单步执行，遇到子函数会进入子函数

+ `thread step-out`、`finish` : 直接执行完当前函数的所有代码，返回到上一个函数

+ `thread step-inst-over`、`nexti`、`ni`

+ `thread step-ints`、`stepi`、`si`

+ `si`、`ni`和`s`、`n`类似

  + `s`、`n`是源码级别
  + `si`、`ni`是汇编指令级别，汇编指令调试。

+ `breakpoint set`

  + 设置断点

  + `breakpoint set -a `函数地址

  + `breakpoint set -n `函数名

    + `breakpoint set -n test`
    + `breakpoint set -n touchesBegan:withEvent:`
    + `breakpoint set -n "-[ViewController touchesBegan:withEvent]"`

  + `breakpoint set -r`正则表达式

    ```
    // 意思是包含test的函数都打断点
    breakpoint set -r test 
    ```

  + `breakpoint set -s` 动态库 **-n** 函数名

+ `breakpoint list`

  + 列出所有的断点(每个断点都有自己的编号)

+ `breakpoint disbale 断点编号` : 禁用断点

+ `breakpoint enable 断点编号` : 启用断点

+ `breakpoint delete 断点编号` : 删除断点

+ `breakpoint command add 断点编号`

  +  给断点预先设置需要执行的命令，到触发断点时，就会按顺序执行

    ```
    (lldb) breakpoint command add 2
    Enter your debugger command(s).  Type 'DONE' to end.
    > po self.view.backgroundColor = .red
    > DONE
    (lldb) c
    Process 81310 resuming
    123
     po self.view.backgroundColor = .red
    0 elements
    ```

+ `breakpoint command list 断点编号`

  + 查看某个断点设置的命令

+ `breakpoint command delete 断点编号`

  + 删除某个断点设置的命令

+ 内存断点

  + 在内存数据发生改变的时候触发
  + `watchpoint set variable` 变量
    + `watchpoint set variable self->_age`
  + `watchpoint set expression 地址`
    + `watchpoint set expression &(self->_age)`
  + `watchpoint list`
  + `watchpoint disable 断点编号`
  + `watchpoint enable 断点编号`
  + `watchpoint command add 断点编号`
  + `watchpoint command list 断点编号`
  + `watchpoint command delete 断点编号`

+ `image lookup`

  + `image  lookup -t 类型`: 查找某个类型的信息

  + `image lookup -a 地址` : 根据内存地址查找在模块中的位置

    > image lookup -a 地址: 用在查找奔溃位置比较多

  + `image lookup -a 符号或者函数名`: 查找某个符号或者函数的位置

+ `image list`

  + 列出所加载的模块信息
  + `image list -o -f`
    + 打印出模块的偏移地址、全路径

+ 小技巧

  + 敲Enter，会自动执行上次的命令
  + 绝大部分执行都可以使用缩写

## ASLR

+ 什么是ASLR

  + `Address Space Layout Randomization`，地址空间布局随机化

  + 是一种针对缓冲区溢出的安全保护技术，通过对堆、栈、共享库映射等线性区布局的随机化，通过增加攻击者预测目的地址的难度，防止攻击者直接定位攻击代码位置，达到阻止溢出攻击的目的的一种技术

  + iOS4.3开始引入了ASLR技术

+ Mach-O的文件结构

  ![mach-o](/assets/images/reverse/machoimage.png)

+ 未使用`ASLR`

  + 函数代码存放在`_TEXT`段中, 全局变量存放在`_DATA`段中

  + 可执行文件的内存地址是0x0

  + 代码段`_TEXT`的内存地址，就是LC_SEGMENT(`_TEXT`)中的VM Address

  + arm64：0x100000000，非arm64：0x4000

  + 可以使用`size -l -m -x`来查看Mach-O的内存分布

  + 地址分布

    ![noalsr](/assets/images/reverse/noaslr.png)

    ```
    VM Address: Virtual Memory Address 虚拟内存地址
    VM Address: Virtual Memory Size 占多少内存，内存大小
    如:
    _PAGEZERO
    VM Address: 0x0
    VM Size: 0x100000000
    _TEXT
    VM Address: 0x100000000
    VM Size: 0x380C000
    _DATA
    VM Address: 0x10380C000
    VM Size: 0xD4C000
    _LINKEDIT
    VM Address: 0x104558000
    VM Size: 0x2C8000
    
    File Offset: 在Mach-O文件中的位置
    File Size: 在Mach-O文件中占据的大小
    Mach-O文件中一开始是没有_PAGEZERO的，File Size是0，当载入内存才有_PAGEZERO，分配内存是0x10000000
    ```

+ 使用了`ASLR`

  + LC_SEGMENT(`_TEXT`)的VM Address

    + `0x100000000`

  + ASLR随机产生的Offset(偏移)

    + 0x5000

  + 也就是可执行文件的内存地址

  + 地址分布

    ![aslradress](/assets/images/reverse/alsradress.png)

    ```
    breakpoint set -a 0x102095f90+0x10000
    ```

+ **函数的内存地址**

  + 函数的内存地址（VM Address） = File Offset(固定) + ASLR Offset(随机) + __PAGEZERO Size(固定)
  + Hopper、IDA64中的地址都是未使用`ASLR`的VM Address

+ **寄存器**

  + LLDB指令

    + 读取所有寄存器的值

      ```
      memory read
      ```

    + 给某个寄存器写入值

      ```
      memory write  寄存器  值
      ```

    + 打印方法调用者

      ```
      po $x0
      ```

    + 打印方法名

      ```
      x/s $x1
      ```

    + p打印参数（以此类推，x3、x4也可能是参数）

      ```
      po $x2
      ```

    + p如果是非arm64，寄存器就是r0、r1、r2