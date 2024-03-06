---
title: CommandList - Socket
date: 2020-07-16 11:11:11 -0400
categories: [30System, CommandTool]
tags: [CommandTool, Socket]
math: true
image:
---


# Linux Socket

[toc]

---

## 1. 网络中进程之间通信

我们深谙信息交流的价值，那网络中进程之间如何通信，如我们每天打开浏览器浏览网页时，浏览器的进程怎么与web服务器通信的？当你用QQ聊天时，QQ进程怎么与服务器或你好友所在的QQ进程通信？这些都得靠socket？那什么是socket？socket的类型有哪些？还有socket的基本函数

本地的`进程间通信（IPC）`有很多种方式，总结为下面4类：
- 消息传递（管道、FIFO、消息队列）
- 同步（互斥量、条件变量、读写锁、文件和写记录锁、信号量）
- 共享内存（匿名的和具名的）
- 远程过程调用（Solaris门和Sun RPC）

网络中进程之间通信？首要解决的问题是如何唯一标识一个进程，否则通信无从谈起！
- 在本地可以通过`进程PID`来唯一标识一个进程，但在网络中行不通。
- TCP/IP协议族 解决了这个问题
  - 网络层的“ip地址”可以唯一标识网络中的主机，
  - 而传输层的“协议+端口”可以唯一标识主机中的应用程序（进程）
  - 这样利用`三元组（ip地址，协议，端口）`就可以标识网络的进程了
  - 网络中的进程通信就可以利用这个标志与其它进程进行交互。

使用TCP/IP协议的应用程序通常采用应用编程接口：
- UNIX BSD的套接字（socket）
- 和UNIX System V的TLI（已经被淘汰），来实现网络进程之间的通信。
- 就目前而言，几乎所有的应用程序都是采用socket


## 2. Socket
socket起源于Unix，而Unix/Linux基本哲学之一就是“一切皆文件”，都可以用“打开open –> 读写write/read –> 关闭close”模式来操作。
- Socket就是该模式的一个实现，socket即是一种特殊的文件，一些socket函数就是对其进行的操作（读/写IO、打开、关闭）

> 在组网领域的首次使用是在1970年2月12日发布的文献IETF RFC33中发现的，撰写者为Stephen Carr、Steve Crocker和Vint Cerf。根据美国计算机历史博物馆的记载，Croker写道：“命名空间的元素都可称为套接字接口。一个套接字接口构成一个连接的一端，而一个连接可完全由一对套接字接口规定。”计算机历史博物馆补充道：“这比BSD的套接字接口定义早了大约12年。”

---

Socket programing is the key API for programming distributed applications on the Internet.

The basics
- `Program`: an executable file residing on a disk in a directory.
  - A program is read into memory and is executed by the kernel as a result of an `exec()` function.
  - The exec() has six variants, but we only consider the simplest one (exec()) in this course.
- `Process`: An executing instance of a program. Sometimes, `task` is the same meaning.
  - UNIX guarantees that every process has a unique identifier called the `process ID`.
  - The process ID is always a non-negative integer.
- `File descriptors`. File descriptors are normally small non-negative integers that the kernel uses to identify the files being accessed by a particular process.
  - Whenever it opens an existing file or creates a new file, the kernel returns a file descriptor that is used to read or write the file.
  - As we will see in this course, sockets are based on a very similar mechanism (socket descriptors).

---

# socket套接字：

`socket`就是抽象封装了传输层以下软硬件行为，为上层应用程序提供进程/线程间通信管道。让应用开发人员不用管信息传输的过程，直接用socket API就OK了。
Socket是应用层与TCP/IP协议族通信的中间软件抽象层，它是一组接口。在设计模式中，Socket其实就是一个门面模式，它把复杂的TCP/IP协议族隐藏在Socket接口后面，对用户来说，一组简单的接口就是全部，让Socket去组织数据，以符合指定的协议。

![3169895156-56de684038cf9_articlex](https://i.imgur.com/ZRWW5JY.png)

TCP/IP协议存在于OS中，网络服务通过OS提供，在OS中增加支持TCP/IP的系统调用——Berkeley套接字，如Socket，Connect，Send，Recv等

<!-- ![1334044049_2497](/assets/1334044049_2497.jpg) -->

TCP/IP协议族包括运输层、网络层、链路层，而socket所在位置如图，Socket是应用层与TCP/IP协议族通信的中间软件抽象层。

<!-- ![1334044170_5136](/assets/1334044170_5136.jpg) -->


---

## The client-server model
The client-server model is one of the most used communication paradigms in networked systems.
- Clients normally communicates with one server at a time.
- From a server’s perspective, usual communicating with multiple clients.
- Client need to know of the existence of and the address of the server, but the server does not need to know the address of (/existence of) the client prior to the connection being established
- Client and servers communicate by means of multiple layers of network protocols. In this course we will focus on the TCP/IP protocol suite.

---

1. The scenario of the client and the server on the same local network (LAN, Local Area Network)

![ethernet](https://i.imgur.com/symmro4.jpg)

2. The client and the server be in different LANs, with both LANs connected to a Wide Area Network (WAN) by routers.
    - The largest WAN is the Internet, but companies may have their own WANs.
    - The flow of information between the client and the server goes down the protocol stack on one side, then across the network and then up the protocol stack on the other side.

![wan](https://i.imgur.com/jhiR006.jpg)

---

## Transmission Control Protocol (TCP)
TCP provides a connection oriented service, based on connections between clients and servers.
- TCP provides reliability. When a TCP client send data to the server, it requires an acknowledgement in return. If an acknowledgement is not received, TCP automatically retransmit the data and waits for a longer period of time.
- TCP is instead a `byte-stream protocol`, without any boundaries at all.
- TCP is described in RFC 793, RFC 1323, RFC 2581 and RFC 3390.

Socket addresses
IPv4 socket address structure is named `sockaddr_in` and is defined by including the <netinet/in.h> header.
The POSIX definition is the following:

```py
struct in_addr{
in_addr_t s_addr;           # 32 bit IPv4 network byte ordered address
};

struct sockaddr_in {
   uint8_t sin_len;         # length of structure (16)
   sa_family_t sin_family;  # AF_INET
   in_port_t sin_port;      # 16 bit TCP or UDP port number
   struct in_addr sin_addr; # 32 bit IPv4 address
   char sin_zero[8];        # used but always set to zero
};
```
The uint8_t datatype is unsigned 8-bit integer.


---

# Socket通信过程和API全解析
udp和TCP socket通信过程基本上是一样的，只是调用api时传入的配置不一样，以TCP client/server模型为例子看一下整个过程。

![35710689-56de7a1a7c6b4_articlex](https://i.imgur.com/HBFIN8z.png)

既然socket是“open—write/read—close”模式的一种实现，那么socket就提供了这些操作对应的函数接口。

以TCP为例，几个基本的socket接口函数。

## socket API
socket: establish socket interface
gethostname: obtain hostname of system
gethostbyname: returns a structure of type hostent for the given host name
bind: bind a name to a socket
listen: listen for connections on a socket
accept: accept a connection on a socket
connect: initiate a connection on a socket
setsockopt: set a particular socket option for the specified socket.
close: close a file descriptor
shutdown: shut down part of a full-duplex connection

## `int socket(int domain, int type, int protocol);`
socket函数: 对应于普通文件的打开操作。
- 普通文件的打开操作返回一个文件描述字，而`socket()`用于创建一个`socket描述符（socket descriptor）`，它唯一标识一个socket。
- 这个socket描述字跟文件描述字一样，后续的操作都有用到它，把它作为参数，通过它来进行一些读写操作。
  - 正如可以给`fopen`的传入不同参数值，以打开不同的文件。
  - 创建socket的时候，也可以指定不同的参数创建不同的socket描述符

`int socket(int domain, int type, int protocol);`

- socket函数的三个参数分别为：
  - `domain`：即协议域/协议族（family）。
    - 设定socket双方通信协议域，是`本地/internet` / `ip4` / `ip6`
    - 常用协议族: AF_INET, AF_INET6, AF_LOCAL（AF_UNIX，Unix域socket）, AF_ROUTE..
    - domain决定socket的地址类型，在通信中必须采用对应的地址
      - `AF_UNIX/AF_LOCAL`：用在本機程序與程序間的傳輸，讓兩個程序共享一個檔案系統(file system)
        - `AF_UNIX` : 用一个绝对路径名作为地址。
      - `AF_INET, AF_INET6` ：讓兩台主機透過網路進行資料傳輸
        - `AF_INET` : IPv4協定, 用ipv4地址（32位的）与端口号（16位的）的组合
        - `AF_INET6` : IPv6協定。

            Name                Purpose                          Man page
            AF_UNIX, AF_LOCAL   Local communication              unix(7)
            AF_INET             IPv4 Internet protocols          ip(7)
            AF_INET6            IPv6 Internet protocols          ipv6(7)

- `type`：指定socket类型。
  - 常用:
    - SOCK_STREAM : 提供一個序列化的連接導向位元流，可做位元流傳輸。對應protocol TCP。一般对应TCP、sctp
    - SOCK_DGRAM : 提供的是一個一個的資料包(datagram)，對應的protocol為UDP
    - SOCK_RAW
    - SOCK_PACKET、SOCK_SEQPACKET..

- `protocol`：指定协议。一般來說都會設為0，讓kernel選擇type對應的默認協議。
  - 常用的协议:
    - IPPROTO_TCP : TCP传输协议
    - IPPTOTO_UDP : UDP传输协议
    - IPPROTO_SCTP : STCP传输协议
    - IPPROTO_TIPC : TIPC传输协议
    - ...

- protocol和type不是随意组合的。
  - 如SOCK_STREAM不可以跟IPPROTO_UDP组合。
  - 当protocol为0时，会自动选择type类型对应的默认协议。

Return Value
- 成功產生socket時，會返回該socket的檔案描述符(socket file descriptor)，我們可以透過它來操作socket。
- 若socket創建失敗則會回傳`-1(INVALID_SOCKET)`。

```py
Example:
#include<stdio.h>
#include<sys/socket.h>
int main(int argc , char *argv[])
{
    int sockfd = 0;
    sockfd = socket(AF_INET , SOCK_STREAM , 0);

    if (socket_fd == -1){
        printf("Fail to create a socket.");
    }

    return 0;
}
```

当我们调用socket创建一个socket时，返回的socket描述字它存在于协议族（address family，AF_XXX）空间中，但没有一个具体的地址。如果想要给它赋值一个地址，就必须调用`bind()`函数，否则就当调用`connect()`、`listen()`时系统会自动随机分配一个端口。

socket() API是在glibc中实现的，该函数又调用到了kernel的sys_socket()，调用链如下。

![35710689-56de7a1a7c6b4_articlex](https://i.imgur.com/CMG4WIX.png)

调用socket()会在内核空间中分配内存然后保存相关的配置。同时会把这块kernel的内存与文件系统关联，以后便可以通过filehandle来访问修改这块配置或者read/write socket。操作socket就像操作file一样，应了那句unix一切皆file。

提示系统的最大filehandle数是有限制的，/proc/sys/fs/file-max设置了最大可用filehandle数。当然这是个linux的配置，可以更改，方法参见Increasing the number of open file descriptors，有人做到过1.6 million connection。


## `bind()`函数
bind()函数:
- 把一个`domain`中的特定地址赋给socket。
- bind()设置socket通信的地址
- 如果为`INADDR_ANY`则表示server会监听本机上所有的interface
- 如果为`127.0.0.1`则表示监听本地的process通信（外面的process也接不进啊）。
- AF_INET、AF_INET6就是把一个ipv4/ipv6地址和端口号组合赋给socket。

`int bind(int sockfd, const struct sockaddr \*addr, socklen_t addrlen);`

- `sockfd`：之前socket()获得的file handle,
  - 即socket描述字，通过socket()函数创建，唯一标识一个socket。bind()函数就是将给这个描述字绑定一个名字。

- `addrlen`：地址长度
通常服务器在启动的时候都会绑定一个众所周知的地址（如ip地址+端口号），用于提供服务，客户就可以通过它来接连服务器；而客户端就不用指定，有系统自动分配一个端口号和自身的ip地址组合。这就是为什么通常服务器端在listen之前会调用bind()，而客户端就不会调用，而是在connect()时由系统随机生成一个。

网络字节序与主机字节序
主机字节序就是我们平常说的大端和小端模式：不同的CPU有不同的字节序类型，这些字节序是指整数在内存中保存的顺序，这个叫做主机序。引用标准的Big-Endian和Little-Endian的定义如下：

　　a) Little-Endian就是低位字节排放在内存的低地址端，高位字节排放在内存的高地址端。

　　b) Big-Endian就是高位字节排放在内存的低地址端，低位字节排放在内存的高地址端。

网络字节序：4个字节的32 bit值以下面的次序传输：首先是0～7bit，其次8～15bit，然后16～23bit，最后是24~31bit。这种传输次序称作大端字节序。由于TCP/IP首部中所有的二进制整数在网络中传输时都要求以这种次序，因此它又称作网络字节序。字节序，顾名思义字节的顺序，就是大于一个字节类型的数据在内存中的存放顺序，一个字节的数据没有顺序的问题了。

所以：在将一个地址绑定到socket的时候，请先将主机字节序转换成为网络字节序，而不要假定主机字节序跟网络字节序一样使用的是Big-Endian。由于这个问题曾引发过血案！公司项目代码中由于存在这个问题，导致了很多莫名其妙的问题，所以请谨记对主机字节序不要做任何假定，务必将其转化为网络字节序再赋给socket。

- `addr`：绑定地址，可能为本机IP地址或本地文件路径.
  - 一个`const struct sockaddr \*`指针，指向要绑定给sockfd的协议地址。这个地址结构根据地址创建socket时的地址协议族的不同而不同，如ipv4对应的是：

```py
struct sockaddr_in {
    sa_family_t    sin_family; /* address family: AF_INET */
    in_port_t      sin_port;   /* port in network byte order */
    struct in_addr sin_addr;   /* internet address */
};

/* Internet address. */
struct in_addr {
    uint32_t       s_addr;     /* address in network byte order */
};

ipv6对应的是：
struct sockaddr_in6 {
    sa_family_t     sin6_family;   /* AF_INET6 */
    in_port_t       sin6_port;     /* port number */
    uint32_t        sin6_flowinfo; /* IPv6 flow information */
    struct in6_addr sin6_addr;     /* IPv6 address */
    uint32_t        sin6_scope_id; /* Scope ID (new in 2.4) */
};

struct in6_addr {
    unsigned char   s6_addr[16];   /* IPv6 address */
};
Unix域对应的是：
#define UNIX_PATH_MAX    108

struct sockaddr_un {
    sa_family_t sun_family;               /* AF_UNIX */
    char        sun_path[UNIX_PATH_MAX];  /* pathname */
};
```


## listen()、connect()函数
如果作为一个服务器，在调用socket()、bind()之后就会调用`listen()`来监听这个socket，如果客户端这时调用`connect()`发出连接请求，服务器端就会接收到这个请求。

`int listen(int sockfd, int backlog);`

- `sockfd`：之前socket()获得的file handle, 要监听的socket描述字
- `backlog`：设置server可以同时接收的最大链接数
  - 相应socket可以排队的最大连接个数。
  - server端会有个处理connection的queue，listen设置这个queue的长度。
  - listen()只用于server端，设置接收`queue`的长度。如果queue满了，server端可以丢弃新到的connection或者回复客户端`ECONNREFUSED`。

socket()函数创建的socket默认是一个主动类型的，listen函数将socket变为被动类型的，等待客户的连接请求。


`int connect(int sockfd, const struct sockaddr *addr, socklen_t addrlen);`

- sockfd: socket的标示filehandle, 客户端的socket描述字
- addr：server端地址, 服务器的socket地址
- addrlen：地址长度,socket地址的长度

功能说明：
- `connect()` : 用于双方连接的建立。
  - 对于TCP连接，connect()实际发起了TCP三次握手，connect成功返回后TCP连接就建立了。 客户端通过调用connect函数来建立与TCP服务器的连接。
  - 由于UDP是无连接的，connect()可以用来指定要通信的对端地址，后续发数据`send()`就不需要填地址了。UDP也可以不使用connect(), socket()建立后，在`sendto()`中指定对端地址。


## accept()函数
- `TCP服务器端`依次调用`socket()`、`bind()`、`listen()`之后，就会监听指定的socket地址了。
- `TCP客户端`依次调用`socket()`、`connect()`之后就想TCP服务器发送了一个连接请求。
- `TCP服务器`监听到这个请求之后，就会调用`accept()`函数取接收请求，这样连接就建立好了。
- 之后就可以开始网络I/O操作了，即类同于普通文件的读写I/O操作。

`int accept(int sockfd, struct sockaddr *addr, socklen_t *addrlen);`

参数说明：
- `sockfd`: 服务器的socket描述字
- `addr`:对端地址, 参数为指向`struct sockaddr *`的指针，用于返回客户端的协议地址
- `addrlen`：协议地址的长度

功能说明：
- `accept()`从queue中拿出第一个pending的connection，新建一个socket并返回。
- 新建的socket我们叫`connected socket`，区别于前面的`listening socket`。
  - `connected socket`用来server跟client的后续数据交互
  - `listening socket`继续waiting for new connection。
- 当queue里没有connection时，如果socket通过`fcntl()`设置为 `O_NONBLOCK`，`accept()`不会block，否则一般会block。
- 如果accpet成功，那么其返回值是由内核自动生成的一个全新的描述字，代表与返回客户的TCP连接。

注意：accept的第一个参数为`服务器的socket描述字`，是服务器开始用`socket()`函数生成的，称为监听socket描述字；而accept函数返回的是`已连接的socket描述字`。
- 一个服务器通常通常仅仅只创建一个监听socket描述字，它在该服务器的生命周期内一直存在。
- 内核为每个由服务器进程接受的客户连接创建了一个已连接socket描述字，当服务器完成了对某个客户的服务，相应的已连接socket描述字就被关闭。


## `read()、write()`等函数
至此服务器与客户已经建立好连接了。可以调用网络I/O进行读写操作了，实现了网咯中不同进程之间的通信

网络I/O操作有下面几组：

    read()/write()
    recv()/send()
    readv()/writev()
    recvmsg()/sendmsg()
    recvfrom()/sendto()

`recvmsg()/sendmsg()`函数，最通用的I/O函数，实际上可以把上面的其它函数都替换成这两个函数。
它们的声明如下：

```
#include <unistd.h>

ssize_t read(int fd, void *buf, size_t count);
ssize_t write(int fd, const void *buf, size_t count);

#include <sys/types.h>
#include <sys/socket.h>

ssize_t send(int sockfd, const void *buf, size_t len, int flags);
ssize_t recv(int sockfd, void *buf, size_t len, int flags);

ssize_t sendto(int sockfd, const void *buf, size_t len, int flags,
              const struct sockaddr *dest_addr, socklen_t addrlen);
ssize_t recvfrom(int sockfd, void *buf, size_t len, int flags,
                struct sockaddr *src_addr, socklen_t *addrlen);

ssize_t sendmsg(int sockfd, const struct msghdr *msg, int flags);
ssize_t recvmsg(int sockfd, struct msghdr *msg, int flags);
```

read函数是负责从fd中读取内容.当读成功时，read返回实际所读的字节数，如果返回的值是0表示已经读到文件的结束了，小于0表示出现了错误。如果错误为EINTR说明读是由中断引起的，如果是ECONNREST表示网络连接出了问题。

write函数将buf中的nbytes字节内容写入文件描述符fd.成功时返回写的字节数。失败时返回-1，并设置errno变量。 在网络程序中，当我们向套接字文件描述符写时有俩种可能。1)write的返回值大于0，表示写了部分或者是全部的数据。2)返回的值小于0，此时出现了错误。我们要根据错误类型来处理。如果错误为EINTR表示在写的时候出现了中断错误。如果为EPIPE表示网络连接出现了问题(对方已经关闭了连接)。


`int recv(SOCKET socket, char FAR* buf, int len, int flags);`
- socket： 一个标识已连接套接口的描述字。
- buf： 用于接收数据的缓冲区。
- len： 缓冲区长度。
- flags： 指定调用方式。取值：MSG_PEEK 查看当前数据，数据将被复制到缓冲区中，但并不从输入队列中删除；MSG_OOB 处理带外数据。

若无错误发生，recv()返回读入的字节数。如果连接已中止，返回0。否则的话，返回SOCKET_ERROR错误，应用程序可通过WSAGetLastError()获取相应错误代码。


`ssize_t recvfrom(int sockfd, void buf, int len, unsigned int flags, struct socketaddr* from, socket_t* fromlen);`
参数说明
sockfd： 标识一个已连接套接口的描述字。
buf： 接收数据缓冲区。
len： 缓冲区长度。
flags： 调用操作方式。是以下一个或者多个标志的组合体，可通过or操作连在一起：
MSG_DONTWAIT：操作不会被阻塞；
MSG_ERRQUEUE： 指示应该从套接字的错误队列上接收错误值，依据不同的协议，错误值以某种辅佐性消息的方式传递进来，使用者应该提供足够大的缓冲区。导致错误的原封包通过msg_iovec作为一般的数据来传递。导致错误的数据报原目标地址作为msg_name被提供。错误以sock_extended_err结构形态被使用。
MSG_PEEK：指示数据接收后，在接收队列中保留原数据，不将其删除，随后的读操作还可以接收相同的数据。
MSG_TRUNC：返回封包的实际长度，即使它比所提供的缓冲区更长， 只对packet套接字有效。
MSG_WAITALL：要求阻塞操作，直到请求得到完整的满足。然而，如果捕捉到信号，错误或者连接断开发生，或者下次被接收的数据类型不同，仍会返回少于请求量的数据。
MSG_EOR：指示记录的结束，返回的数据完成一个记录。
MSG_TRUNC：指明数据报尾部数据已被丢弃，因为它比所提供的缓冲区需要更多的空间。
MSG_CTRUNC：指明由于缓冲区空间不足，一些控制数据已被丢弃。(MSG_TRUNC使用错误,4才是MSG_TRUNC的正确解释)
MSG_OOB：指示接收到out-of-band数据(即需要优先处理的数据)。
MSG_ERRQUEUE：指示除了来自套接字错误队列的错误外，没有接收到其它数据。
from：（可选）指针，指向装有源地址的缓冲区。
fromlen：（可选）指针，指向from缓冲区长度值。



`int sendto( SOCKET s, const char FAR* buf, int size, int flags, const struct sockaddr FAR* to, int token);`
参数说明
s： 套接字
buf： 待发送数据的缓冲区
size： 缓冲区长度
flags： 调用方式标志位, 一般为0, 改变Flags，将会改变Sendto发送的形式
addr： （可选）指针，指向目的套接字的地址
tolen： addr所指地址的长度
如果成功，则返回发送的字节数，失败则返回SOCKET_ERROR。



`int accept( int fd, struct socketaddr* addr, socklen_t* len);`
参数说明
fd： 套接字描述符。
addr： 返回连接着的地址
len： 接收返回地址的缓冲区长度
成功返回客户端的文件描述符，失败返回-1。



## close()函数
在服务器与客户端建立连接之后，会进行一些读写操作，完成了读写操作就要关闭相应的socket描述字，好比操作完打开的文件要调用fclose关闭打开的文件。

    #include <unistd.h>
    int close(int fd);

close一个TCP socket的缺省行为时把该socket标记为以关闭，然后立即返回到调用进程。该描述字不能再由调用进程使用，也就是说不能再作为read或write的第一个参数。

注意：close操作只是使相应socket描述字的引用计数-1，只有当引用计数为0的时候，才会触发TCP客户端向服务器发送终止连接请求。

# 4、socket: TCP的三次握手建立连接详解
tcp建立连接要进行“三次握手”，即交换三个分组
- 客户端向服务器发送一个SYN J
- 服务器向客户端响应一个SYN K，并对SYN J进行确认ACK J+1
- 客户端再想服务器发一个确认ACK K+1

socket中发送的TCP三次握手:
<!-- ![socket中发送的TCP三次握手](/assets/201012122157476286.png) -->

从图中可以看出
- 当客户端调用connect时，触发了连接请求，向服务器发送了SYN J包，这时connect进入`阻塞状态`；
- 服务器监听到连接请求(收到SYN J包)，调用`accept()`接收请求(向客户端发送SYN K ，ACK J+1)，这时accept进入`阻塞状态`；
- 客户端收到服务器的SYN K ，ACK J+1之后，这时connect返回，并对SYN K进行确认；
- 服务器收到ACK K+1时，accept返回，至此三次握手完毕，连接建立。

> 总结：
客户端的connect在三次握手的第二个次返回，
服务器端的accept在三次握手的第三次返回。

# 5、socket中TCP的四次握手释放连接详解

socket中的四次握手释放连接的过程

<!-- ![201012122157494693](/assets/201012122157494693.png) -->

某个应用进程首先调用close主动关闭连接，这时TCP发送一个FIN M；
另一端接收到FIN M之后，执行被动关闭，对这个FIN进行确认。它的接收也作为文件结束符传递给应用进程，因为FIN的接收意味着应用进程在相应的连接上再也接收不到额外数据；
一段时间之后，接收到文件结束符的应用进程调用close关闭它的socket。这导致它的TCP也发送一个FIN N；
接收到这个FIN的源发送端TCP对它进行确认。
这样每个方向上都有一个FIN和ACK。

6、一个例子（实践一下）
说了这么多了，动手实践一下。下面编写一个简单的服务器、客户端（使用TCP）——服务器端一直监听本机的6666号端口，如果收到连接请求，将接收请求并接收客户端发来的消息；客户端与服务器端建立连接并发送一条消息。

服务器端代码：
```
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<errno.h>
#include<sys/types.h>
#include<sys/socket.h>
#include<netinet/in.h>

#define MAXLINE 4096

int main(int argc, char** argv)
{
    int    listenfd, connfd;
    struct sockaddr_in     servaddr;
    char    buff[4096];
    int     n;

    if( (listenfd = socket(AF_INET, SOCK_STREAM, 0)) == -1 ){
    printf("create socket error: %s(errno: %d)\n",strerror(errno),errno);
    exit(0);
    }

    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = htonl(INADDR_ANY);
    servaddr.sin_port = htons(6666);

    if( bind(listenfd, (struct sockaddr*)&servaddr, sizeof(servaddr)) == -1){
    printf("bind socket error: %s(errno: %d)\n",strerror(errno),errno);
    exit(0);
    }

    if( listen(listenfd, 10) == -1){
    printf("listen socket error: %s(errno: %d)\n",strerror(errno),errno);
    exit(0);
    }

    printf("======waiting for client's request======\n");
    while(1){
    if( (connfd = accept(listenfd, (struct sockaddr*)NULL, NULL)) == -1){
        printf("accept socket error: %s(errno: %d)",strerror(errno),errno);
        continue;
    }
    n = recv(connfd, buff, MAXLINE, 0);
    buff[n] = '\0';
    printf("recv msg from client: %s\n", buff);
    close(connfd);
    }

    close(listenfd);
}
```
服务器端
客户端代码：
```
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<errno.h>
#include<sys/types.h>
#include<sys/socket.h>
#include<netinet/in.h>

#define MAXLINE 4096

int main(int argc, char** argv)
{
    int    sockfd, n;
    char    recvline[4096], sendline[4096];
    struct sockaddr_in    servaddr;

    if( argc != 2){
    printf("usage: ./client <ipaddress>\n");
    exit(0);
    }

    if( (sockfd = socket(AF_INET, SOCK_STREAM, 0)) < 0){
    printf("create socket error: %s(errno: %d)\n", strerror(errno),errno);
    exit(0);
    }

    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(6666);
    if( inet_pton(AF_INET, argv[1], &servaddr.sin_addr) <= 0){
    printf("inet_pton error for %s\n",argv[1]);
    exit(0);
    }

    if( connect(sockfd, (struct sockaddr*)&servaddr, sizeof(servaddr)) < 0){
    printf("connect error: %s(errno: %d)\n",strerror(errno),errno);
    exit(0);
    }

    printf("send msg to server: \n");
    fgets(sendline, 4096, stdin);
    if( send(sockfd, sendline, strlen(sendline), 0) < 0)
    {
    printf("send msg error: %s(errno: %d)\n", strerror(errno), errno);
    exit(0);
    }

    close(sockfd);
    exit(0);
}
```
客户端
当然上面的代码很简单，也有很多缺点，这就只是简单的演示socket的基本函数使用。其实不管有多复杂的网络程序，都使用的这些基本函数。上面的服务器使用的是迭代模式的，即只有处理完一个客户端请求才会去处理下一个客户端的请求，这样的服务器处理能力是很弱的，现实中的服务器都需要有并发处理能力！为了需要并发处理，服务器需要fork()一个新的进程或者线程去处理请求等。



# 代码示例
TCP server端
这是TCP server代码例子，server收到client的任何数据后再回返给client。主进程负责accept()新进的connection并创建子进程，子进程负责跟client通信。

```
#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <string.h>
#include <unistd.h>

#define MAXLINE 4096 /*max text line length*/
#define SERV_PORT 3000 /*port*/
#define LISTENQ 8 /*maximum number of client connections */

int main (int argc, char **argv) {
    int listenfd, connfd, n;
    socklen_t clilen;
    char buf[MAXLINE];
    struct sockaddr_in cliaddr, servaddr;

    //creation of the socket
    listenfd = socket (AF_INET, SOCK_STREAM, 0);

    //preparation of the socket address
    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = htonl(INADDR_ANY);
    servaddr.sin_port = htons(SERV_PORT);

    // bind address
    bind (listenfd, (struct sockaddr *) &servaddr, sizeof(servaddr));
    // connection queue size 8
    listen (listenfd, LISTENQ);
    printf("%s\n","Server running...waiting for connections.");

    while(1) {
        clilen = sizeof(cliaddr);
        connfd = accept (listenfd, (struct sockaddr *) &cliaddr, &clilen);
        printf("%s\n","Received request...");

        if (!fork()) { // this is the child process
            close(listenfd); // child doesn't need the listener
            while ( (n = recv(connfd, buf, MAXLINE,0)) > 0)  {
                printf("%s","String received from and resent to the client:");
                puts(buf);
                send(connfd, buf, n, 0);
                if (n < 0) {
                   perror("Read error");
                   exit(1);
                }
            }
            close(connfd);
            exit(0);
        }
    }
    //close listening socket
    close (listenfd);
}

TCP client端
TCP端代码，单进程。client与server建立链接后，从标准输入得到数据发给server并等待server的回传数据并打印输出，然后等待标准输入...

#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <string.h>
#include <arpa/inet.h>

#define MAXLINE 4096 /*max text line length*/
#define SERV_PORT 3000 /*port*/

int main(int argc, char **argv)
{
    int sockfd;
    struct sockaddr_in servaddr;
    char sendline[MAXLINE], recvline[MAXLINE];
    //basic check of the arguments
    if (argc !=2) {
        perror("Usage: TCPClient <IP address of the server");
        exit(1);
    }

    //Create a socket for the client
    //If sockfd<0 there was an error in the creation of the socket
    if ((sockfd = socket (AF_INET, SOCK_STREAM, 0)) <0) {
        perror("Problem in creating the socket");
        exit(2);
    }

    //Creation of the socket
    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr= inet_addr(argv[1]);
    servaddr.sin_port =  htons(SERV_PORT); //convert to big-endian order

    //Connection of the client to the socket
    if (connect(sockfd, (struct sockaddr *) &servaddr, sizeof(servaddr))<0) {
        perror("Problem in connecting to the server");
        exit(3);
    }

    while (fgets(sendline, MAXLINE, stdin) != NULL) {
        send(sockfd, sendline, strlen(sendline), 0);
        if (recv(sockfd, recvline, MAXLINE,0) == 0){
            //error: server terminated prematurely
            perror("The server terminated prematurely");
            exit(4);
        }
        printf("%s", "String received from the server: ");
        fputs(recvline, stdout);
   }
   exit(0);
}
高并发socket -- select vs epoll
上面举的server的例子是用多进程来实现并发，当然还有其他比较高效的做法，比如IO复用。select和epoll是IO复用常用的系统调用，详细分析一下。

select API
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
int select(int nfds, fd_set *readfds, fd_set *writefds,fd_set *exceptfds, struct timeval *timeout);

//fd_set类型示意
typedef struct
{
   unsigned long fds_bits[1024 / 64]; // 8bytes*16=128bytes
} fd_set;

参数说明：
readfds: 要监控可读的sockets集合，看是否可读
writefds：要监控可写的sockets集合，看是否可写
exceptfds：要监控发生exception的sockets集合，看是否有exception
nfds:上面三个sockets集合中最大的filehandle+1
timeout：阻塞的时间，0表示不阻塞，null表示无限阻塞

功能说明：
调用select()实践上是往kernel注册3组sockets监控集合，任何一个或多个sockets ready（状态跳变，不可读变可读 or 不可写变可写 or exception发生），
函数就会返回，否则一直block直到超时。
返回值>0表示ready的sockets个数，0表示超时，-1表示error。
epoll API
epoll由3个函数协调完成，把整个过程分成了创建，配置，监控三步。

step1 创建epoll实体

  #include <sys/epoll.h>
  int epoll_create(int size);

  参数说明：
  size：随便给个>0的数值，现在系统不care了。

  功能说明：
  epoll_create()在kernel内部分配了一块内存并关联到文件系统，函数调用成功会返回一个file handle来标识这块内存。

  #include <sys/epoll.h>
  int epoll_ctl(int epfd, int op, int fd, struct epoll_event *event);
Step2 配置监控的socket集合

  #include <sys/epoll.h>
  int epoll_ctl(int epfd, int op, int fd, struct epoll_event *event);

  typedef union epoll_data {
      void        *ptr;
      int          fd;
      uint32_t     u32;
      uint64_t     u64;
  } epoll_data_t;
  struct epoll_event {
      uint32_t     events;      /* Epoll events */
      epoll_data_t data;        /* User data variable */
  };
  参数说明：
  epfd：前面epoll_create()创建实体的标识
  op:操作符，EPOLL_CTL_ADD/EPOLL_CTL_MOD/EPOLL_CTL_DEL
  fd:要监控的socket对应的file handle
  event：要监控的事件链表

  功能说明：
  epoll_ctl()配置要对哪个socket做什么样的事件监控。

step3 监控sockets

  #include <sys/epoll.h>
  int epoll_wait(int epfd, struct epoll_event *events, int maxevents, int timeout);

  参数说明：
  epfd：epoll实体filehandle标识
  events：指示发生的事情。application分配一块内存用event指针来指向，epoll_wait()调用时kernel将发生的事件存入event这块内存。
  maxevents:最大可接收多少event
  timeout：超时时间，0表示立即返回，函数不block，-1表示无限block。

  功能说明：
  epoll_wait()真正开始监控之前设置好的sockets集合。如果有事件发生，通过事件链表的方式返回给application。

对比select和epoll
有了上面的API，我们可以比较直观的比较select和epoll的特点

select的memory copy比epoll多。

select每次调用都要有用户空间到kernel空间的内存copy，把所有要监控配置copy到内核。

epoll只需要epoll_ctl配置的时候copy，而且是增量copy，epoll_wait没有用户空间到内核的copy

select函数调用返回后的处理比epoll低效

select()返回给application有几件事情发生了，但是没说是谁有事情，application还得挨个遍历过去，看看谁有啥事

epoll_wait()返回给application更多的信息，谁发生了什么事都通知给application了，application直接处理这些事件就行了，不需要遍历

select相比epoll有处理socket数量的限制

select内核限定了1024最大的filehandle数，如果要修改需要编译内核

epoll没有固定的限制，可以达到系统最大filehandle数

小结一下两者的对比，通常可以看到epoll的效率更高，尤其是在大量socket并发的时候。有人说在少量sockets，比如10多个以内，select要有优势，我没有验证过。不过这么少的并发用哪个都行，不会差别太大。
```


# 從Client連向Server
客戶端要連向伺服端，需要先知道並儲存伺服端的IP及port
- `netinet/in.h`已經為我們定義好了一個`struct sockaddr_in`來儲存這些資訊：

```py
# IPv4 AF_INET sockets:
# IPv6參見 sockaddr_in6
struct sockaddr_in {
    short            sin_family;   # AF_INET,因為這是IPv4;
    unsigned short   sin_port;     # 儲存port No
    struct in_addr   sin_addr;     # 參見struct in_addr
    char             sin_zero[8];  # Not used, must be zero */
};

struct in_addr {
    unsigned long s_addr;          # load with inet_pton()
};
```



ref
- https://segmentfault.com/a/1190000004570985
- https://www.cs.dartmouth.edu/~campbell/cs50/socketprogramming.html
- https://zake7749.github.io/2015/03/17/SocketProgramming/
- https://www.tenouk.com/Module39a.html
- https://www.jb51.net/article/135558.htm
- https://hit-alibaba.github.io/interview/basic/network/Socket-Programming-Basic.html
- https://cighao.com/2016/07/12/c-linux-socket/
- https://blog.csdn.net/hguisu/article/details/7445768
- https://www.jianshu.com/p/6a5d273f3223
- https://blog.csdn.net/hguisu/article/details/7445768

















.
