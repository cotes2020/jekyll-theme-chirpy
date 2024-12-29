---
layout:     post
title:      "Unix/Linux 扫盲笔记"
subtitle:   "不适合人类阅读，非常水的自我笔记"
date:       2015-04-14 
author:     "Hux"
header-img: "img/post-bg-unix-linux.jpg"
catalog: true
tags:
  - 笔记
---

> This document is not completed and will be updated anytime.


## Unix 


> Unix is a **family** of multitasking, multiuser computer OS.


Derive from the original **AT&T Unix**, Developed in the 1970s at **Bell Labs** (贝尔实验室), initially intended for use inside the **Bell System**.

- #### Bell Labs
Bell 和 AT&A 在那时已经是一家了，可以看到那时的通信公司真是一线 IT 公司呢。 
**C 语言也是 Bell Labs 的产物**，从一开始就是为了用于 Unix 而设计出来的。所以 Unix （在 73 年用 C 重写）在高校流行后，C 语言也获得了广泛支持。



AT&T licensed Unix to outside parties(第三方) from the late 1970s, leading to a variety of both **academic** (最有有名的 BSD ) and **commercial** (Microsoft Xenix, IBM AIX, SunOS Solaris)

- #### Xenix
微软 1979 年从 AT&A 授权来的 Unix OS，配合着 x86 成为当时最受欢迎的 Unix 发行版。后来 M$ 和 IBM 合作开发 OS/2 操作系统后放弃，后来最终转向 **Windows NT**。

- #### BSD
**Barkeley Software Distribution**, also called Berkeley Unix. Today the term "BSD" is used to refer to any of the BSD descendants(后代) which together form a branch of the family of Unix-like OS.(共同组成了一个分支)
	- **BSD 最大的贡献是在 BSD 中率先增加了虚拟存储器和 Internet 协议**，其 TCP/IP(IPv4 only) 代码仍然在现代 OS 上使用（ Microsoft Windows and most of the foundation of Apple's OS X and iOS ）
	- BSD 后来发展出了众多开源后代，包括 FreeBSD, OpenBSD, NetBSD 等等……很多闭源的 vendor Unix 也都从 BSD 衍生而来。

- #### FreeBSD & Apple
FreeBSD 不但是 Open Source BSD 中占有率最高的，还直接影响了 Apple Inc : NeXT Computer 的团队在 FreeBSD 上衍生出了 NeXTSTEP 操作系统，这货后来在 Apple 时期演化成了 **Darwin** ，这个“达尔文”居然还是个开源系统，而且是 the Core of **Mac OS X** and **iOS**.

- #### NeXTSTEP
An **object-oriented**, multitasking OS. Low-level C but High-level OC language and runtime the first time, combined with an **OO aplication layer** and including several "kits".    
大家都知道 NeXT 是 Steve Jobs 被 forced out of Apple 后和 a few of his coworkers 创办的，所以 **NeXTSTEP 绝对是证明 Jobs 实力的作品。** 

- #### Darwin
[Darwin](https://en.wikipedia.org/wiki/Darwin_(operating_system)), the core set of components upon which Mac OS X and iOS based, mostly POSIX compatible, but has never, by itself, been certified as being compatible with any version of **POSIX**. (OS X, since Leopard, has been certified as compatible with the Single UNIX Specification version 3)  
**所以说 Mac OS X 算是很正统 Unix 的了**

- #### POSIX
可移植操作系统接口, Portable Operating System Interface, is a family of standards specified by the IEEE from maintaining compatibility between OS, defines the API along with Command Line Shells and utility interfaces, for software comaptibility with variants of Unix and other OS.
	- Fully POSIX compliant:
		- OS X
		- QNX OS (BlackBerry)
	- Mostly complicant:
		- Linux
		- OpenBSD/FreeBSD
		- Darwin (Core of **iOS** & OS X)
		- **Android**
	- Complicant via compatibility feature （通过兼容功能实现兼容）
		- Windows NT Kernel
			- Windows Server 2000, 2003, 2008, 2008 R2, 2012
		- Symbian OS (with PIPS)
			- Symbian was a closed-source OS.


## Unix-like

> A Unix-like (sometimes referred to as UN*X or *nix) operating system is one that behaves in a manner similar to a Unix system, while not necessarily conforming to or being certified to any version of the **Single UNIX Specification**.

There is no standard for defining the term.  
其实 Unix-like 是个相对模糊的概念：

* 最狭义的 Unix 单指 Bell Labs's Unix 
* 稍广义的 Unix 指代所有 Licensed Unix, 即通过了 SUS 的 Unix-like ，比如 OS X
* 最广义的 Unix 即所有 Unix-like 系统，无论它是否通过过任何 SUS，包括 Linux，BSD Family 等

#### Single UNIX Specification
The Single UNIX Specification (SUS) is the collective name of a family of standards for computer OS, compliance with which is required to **qualify for the name "Unix"**, like **POSIX**.

#### Apple iOS
iOS is a **Unix-like OS based on Darwin(BSD)** and OS X, which share some frameworks including Core Foundation, Founadtion and the Darwin foundation with OS X, but, Unix-like shell access is not avaliable for users and restricted for apps, **making iOS not fully Unix-compatible either.**

The iOS kernal is **XNU**, the kernal of Darwin.

#### XNU Kernel
XNU, the acronym(首字母缩写) for ***X is Not Unix***, which is the **Computer OS Kernel** developed at Apple Inc since Dec 1996 for use in the Mac OS X and released as free open source software as part of Darwin.


## Linux


> Linux is a Unix-like and mostly POSIX-compliant computer OS.


![Unix_timeline](http://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Unix_timeline.en.svg/800px-Unix_timeline.en.svg.png)


#### Linux Kernel

严格来讲，术语 Linux 只表示 [Linux Kernel](http://en.wikipedia.org/wiki/Linux_kernel) 操作系统内核本身，比如说 Android is Based on Linux (Kernel). Linus 编写的也只是这一部分，一个免费的 Unix-like Kernel，并不属于 GNU Project 的一部分。

但通常把 Linux 作为 Linux Kernel 与大量配合使用的 GNU Project Software Kit (包括 Bash, Lib, Compiler, 以及后期的 GUI etc) 所组合成的 OS 的统称。（包括各类 Distribution 发行版）

这类操作系统也被称为 **GNU/Linux**


#### GNU Project

The GNU Project is a **free software, mass collaboration** project, which based on the following freedom rights:

* Users are free to run the software, share (copy, distribute), study and modify it.
* GNU software guarantees these freedom-rights legally (via its license).
* So it is not only FREE but, more important, FREEDOM.

In order to ensure that the *entire* software of a computer grants its users all freedom rights (use, share, study, modify), even the most fundamental and important part, **the operating system**, needed to be written. 

This OS is decided to called **GNU (a recursive acronym meaning "GNU is not Unix")**. By 1992, the GNU Project had completed all of the major OS components except for their kernel, *GNU Hurd*. 

With the release of the third-party **Linux Kernel**, started independently by *Linus Torvalds* in 1991 and released under the GPLv0.12 in 1992, for the first time it was possible to run an OS **composed completely of free software**.

Though the Linux kernel is not part of the GNU project, it was developed using GCC and other GNU programming tools and was released as free software under the GPL.

Anyway, there eventually comes to the **GNU/Linux**


* **GPL**: GNU General Public License
* **GCC**: GNU Compiler Collection

其他与 GPL 相关的自由/开源软件公共许可证：

* [Mozilla Public License](http://en.wikipedia.org/wiki/Mozilla_Public_License)
* [MIT License](http://en.wikipedia.org/wiki/MIT_License)
* [BSD Public License](http://en.wikipedia.org/wiki/BSD_licenses)
	* GPL 强制后续版本必须是自由软件，而 BSD 的后续可以选择继续开源或者封闭	
* [Apache License](http://en.wikipedia.org/wiki/Apache_License)


![Public License](/img/in-post/open-source-license.png)

#### Android

Android is a mobile OS based on **Linux Kernel**, so it's definitely **Unix-like**.  

**Linux is under GPL so Android has to be open source**. 
Android's source code is released by Google under open source licenses, although most Android devices ultimately ship with a combination of open source and proprietary software, including proprietary software developed and licensed by Google *(GMS are all proprietary)*  

#### Android Kernel
  
Android's kernel is based on one of the Linux kernel's long-term support (LTS) branches.   

**Android's variant of the Linux kernel** has further architectural changes that are implemented by Google outside the typical Linux kernel development cycle, and, certain features that Google contributed back to the Linux kernel. Google maintains a public code repo that contains their experimental work to re-base Android off the latest stable Linux versions.

Android Kernel 大概是 Linux Kernel 最得意的分支了，Android 也是 Linux 最流行的发行版。不过，也有一些 Google 工程师认为 Android is not Linux in the traditional Unix-like Linux distribution sense. 总之这类东西就算有各种协议也还是很难说清楚，在我理解里 Android Kernel 大概就是 fork Linux Kernel 之后改动和定制比较深的例子。


#### Android ROM

既然提到 Android 就不得不提提 Android ROM 

ROM 的本义实际上是只读内存：  

**Read-only memory** (ROM) is a class of storage medium used in computers and other electronic devices. Data stored in ROM can only be modified slowly, with difficulty, or not at all, so it is **mainly used to distribute firmware (固件)** (software that is very closely tied to specific hardware, and unlikely to need frequent updates).

ROM 在发展的过程中不断进化，从只读演变成了可编程可擦除，并最终演化成了 Flash  

* PROM (Programmable read-only memory)
* EPROM (Erasable programmable read-only memory)
* EEPROM (Electrically erasable programmable read-only memory)
	* Flash memory (闪存) 

Flash 的出现是历史性的，它不但可以作为 ROM 使用，又因其极高的读写速度和稳定性，先后发展成为U盘（USB flash drives）、移动设备主要内置存储，和虐机械硬盘几条街的固态硬盘（SSD），可以说这货基本统一了高端存储市场的技术规格。

所以我们平时习惯说的 ROM 其实还是来源于老单片机时代，那时的 ROM 真的是写了就很难（需要上电复位）、甚至无法修改，所以那时往 ROM 里烧下去的程序就被称作 firmware ，固件。久而久之，虽然技术发展了，固件仍然指代那些不常需要更新的软件，而 ROM 这个词也就这么沿用下来了。

所以在 wiki 里是没有 Android ROM 这个词条的，只有 [List of custom Android firmwares](http://en.wikipedia.org/wiki/List_of_custom_Android_firmwares)

> A custom firmware, also known as a custom ROM, ROM, or custom OS, is an aftermarket distribution of the Android operating system. They are based on the Android Open Source Project (AOSP), hence most are open-sourced releases, unlike proprietary modifications by device manufacturers.

各类 Android ROM 在 Android 词类下也都是属于 **Forks and distributions** 一类的。

所以我说，其实各类 Android ROM 也好，fork Android 之流的 YunOS、FireOS 也好，改了多少东西，碰到多深的 codebase ……**其实 ROM 和 Distribution OS 的界限是很模糊的**，为什么 Android 就不可以是移动时代的 Linux ，为什么 Devlik/ART 就不能是移动时代的 GCC 呢？

#### Chrome OS

Chrome OS is an operating system based on the **Linux kernel** and designed by Google to work with web applications and installed applications. 

虽然目前只是个 Web Thin Client OS ，但是 RoadMap 非常酷……

* **Chrome Packaged Application** (Support working offline and installed)
* **Android App Runtime** (run Android applications natively...fxxking awesome)

平复一下激动的心情，还是回到正题来：

#### Chromium OS

Chrome OS is based on Chromium OS, which is the open-source development version of Chrome OS, which is a **Linux distribution** designed by Google.

For Detail, Chromium OS based on [Gentoo Linux](http://en.wikipedia.org/wiki/Gentoo_Linux), emm...

