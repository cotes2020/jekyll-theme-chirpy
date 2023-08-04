---
title : Leviathan
categories : [Wargame, Overthewire]
tags : [Linux commands, ltrace, symbolic link, 심볼릭 링크]
---

## Leviathan
<a href="https://overthewire.org/wargames/leviathan/" target="_blank">https://overthewire.org/wargames/leviathan/</a>

```
access : leviathan.labs.overthewire.org through SSH on port 2223

To login to the first level use:

Username: leviathan0
Password: leviathan0
password :  /etc/leviathan_pass
```
```
This machine has a 64bit processor and many security-features enabled
  by default, although ASLR has been switched off.  The following
  compiler flags might be interesting:

    -m32                    compile for 32bit
    -fno-stack-protector    disable ProPolice
    -Wl,-z,norelro          disable relro
```

## Level 0
```
ssh leviathan0.labs.overthewire.org -p 2223
```
```
leviathan0@leviathan:~/.backup$ grep "password" bookmarks.html
<DT><A HREF="http://leviathan.labs.overthewire.org/passwordus.html | This will be fixed later, the password for leviathan1 is rioGegei8m" ADD_DATE="1155384634" LAST_CHARSET="ISO-8859-1" ID="rdf:#$2wIU71">password to leviathan1</A>
```
```
password : rioGegei8m
```

## Level 1
```
ssh leviathan1.labs.overthewire.org -p 2223
```
```
   0x080485a6 <+107>:   lea    eax,[ebp-0x10]
   0x080485a9 <+110>:   push   eax
   0x080485aa <+111>:   lea    eax,[ebp-0xc]
   0x080485ad <+114>:   push   eax
   0x080485ae <+115>:   call   0x80483b0 <strcmp@plt>
   0x080485b3 <+120>:   add    esp,0x10
   0x080485b6 <+123>:   test   eax,eax
   0x080485b8 <+125>:   jne    0x80485e5 <main+170>
   0x080485ba <+127>:   call   0x80483e0 <geteuid@plt>
```
check를 까보면 ebp-0x10, ebp-0xc를 인자로 줌.  
123에 break를 걸고 확인.
```
(gdb) r
Starting program: /home/leviathan1/check 
password: 1234

Breakpoint 1, 0x080485ae in main ()
(gdb) x/wx $eax
0xffffd69c:     0x00333231
(gdb) x/wx $ebp-0x10
0xffffd698:     0x00786573
```
내 입력값은 ebp-0xc이고 ebp-0x10에는 이미 값이 들어가 있음.(위에 어셈블리 보면 됨.)  
따라서 입력값을 0x786573을 string으로 바꿔보면 ```sex```임;
```
$ cat /etc/leviathan_pass/leviathan2
ougahZi8Ta

password : ougahZi8Ta
```

## Level 2 (풀이 봄)
```
ssh leviathan2.labs.overthewire.org -p 2223
```
```
./printfile /etc/leviathan_pass/leviathan3 -> 실패함.

그래서 좀 해보다가 심볼릭 링크가 생각나서 해봤는데 안됨.

mkdir /tmp/lv2
ln -s /etc/leviathan_pass/leviathan3 /tmp/lv2/fk
./printfile /tmp/lv2/fk -> 실패..
```
무엇이 문제인가 풀이를 찾아봄.  
우선 그냥 파일을 생성해서 읽기 가능하면 ```access```는 0을 리턴.  
근데 심볼릭 링크는 읽기 권한이 없어서 불가능함.
```sh
leviathan2@leviathan:~$ ltrace ./printfile /tmp/lv2/fk
__libc_start_main(0x804852b, 2, 0xffffd774, 0x8048610 <unfinished ...>
access("/tmp/lv2/fk", 4)                                               = -1
puts("You cant have that file..."You cant have that file...
)                                                                 = 27
+++ exited (status 1) +++
```
```
그럼 access 함수를 어떻게 넘어갈 것인가? 
위에도 있지만 내가 생성한 파일을 읽게 해줘야 함.
인자는 한 개 뿐이라서 두 개의 파일은 읽히지 않음.

그러나 cat 명령어는 여러 인자를 읽을 수 있음.(중요)
```
```sh
leviathan2@leviathan:~$ mkdir /tmp/test
leviathan2@leviathan:~$ touch /tmp/test/abc
leviathan2@leviathan:~$ touch /tmp/test/def
leviathan2@leviathan:~$ cat > /tmp/test/abc
abc
^C
leviathan2@leviathan:~$ cat > /tmp/test/def
def
^C
leviathan2@leviathan:~$ ./printfile /tmp/test/abc /tmp/test/def
abc
leviathan2@leviathan:~$ cat /tmp/test/abc /tmp/test/def
abc
def
```
```
\을 통해 공백을 읽히게 할꺼임. 

즉, 첫 인자를 access 함수를 통과할 수 있는 파일을 주면은
공백 뒤에 있는 심볼릭 링크 파일은 검사를 하지 않고 넘어가게 됨.
```
```sh
leviathan2@leviathan:~$ touch /tmp/test/abc\ def
leviathan2@leviathan:~$ ltrace ./printfile /tmp/test/abc\ def
__libc_start_main(0x804852b, 2, 0xffffd774, 0x8048610 <unfinished ...>
access("/tmp/test/abc def", 4)                                                                     = 0
snprintf("/bin/cat /tmp/test/abc def", 511, "/bin/cat %s", "/tmp/test/abc def")                    = 26
geteuid()                                                                                          = 12002
geteuid()                                                                                          = 12002
setreuid(12002, 12002)                                                                             = 0
system("/bin/cat /tmp/test/abc def"abc
/bin/cat: def: No such file or directory
 <no return ...>
--- SIGCHLD (Child exited) ---
<... system resumed> )                                                                             = 256
+++ exited (status 0) +++
leviathan2@leviathan:~$  ./printfile /tmp/test/abc\ def
abc
/bin/cat: def: No such file or directory
```
```system("/bin/cat /tmp/test/abc def"abc``` 가 되므로 /tmp/test/abc와 def 파일을 출력하게 되는 것.
```sh
leviathan2@leviathan:~$ mkdir /tmp/test
leviathan2@leviathan:~$ ln -s /etc/leviathan_pass/leviathan3 /tmp/test/ok
leviathan2@leviathan:~$ ls -al /tmp/test/ok
lrwxrwxrwx 1 leviathan2 root 30 Jan 18 12:27 /tmp/test/ok -> /etc/leviathan_pass/leviathan3
leviathan2@leviathan:~$ touch /tmp/test/ok\ bye
leviathan2@leviathan:~$ ./printfile /tmp/test/ok\ bye
Ahdiemoo1j
/bin/cat: bye: No such file or directory
```

또 다른 풀이는 쉘을 따는 것이 있었음!
```
touch '/tmp/dir1/file1;sh'로 파일을 생성하고 이 파일을 읽으면

./prinfile '/tmp/dir1/file1;sh' -> /bin/cah /tmp/dir1/file1;sh

이렇게 되므로 세미콜론에 의해 leviathan3의 uid로 sh를 실행함.
```
```
password : Ahdiemoo1j 
```

## Level 3
```
ssh leviathan3.labs.overthewire.org -p 2223
```
```
leviathan3@leviathan:~$ ltrace ./level3
__libc_start_main(0x8048618, 1, 0xffffd784, 0x80486d0 <unfinished ...>
strcmp("h0no33", "kakaka")                                    = -1
printf("Enter the password> ")                                = 20
fgets(Enter the password> asb
"asb\n", 256, 0xf7fc55a0)                                     = 0xffffd590
strcmp("asb\n", "snlprintf\n")                                = -1
puts("bzzzzzzzzap. WRONG"bzzzzzzzzap. WRONG
)                                                             = 19
+++ exited (status 0) +++
```
```ltrace```로 확인해보니 ```password``` 입력을 받고 그 입력값과 ```snlprintf```를 비교함.  
따라서 입력을 ```snlprintf```라 해주면 성공.  
```
leviathan3@leviathan:~$ ltrace ./level3
__libc_start_main(0x8048618, 1, 0xffffd784, 0x80486d0 <unfinished ...>
strcmp("h0no33", "kakaka")                                    = -1
printf("Enter the password> ")                                = 20
fgets(Enter the password> snlprintf
"snlprintf\n", 256, 0xf7fc55a0)                               = 0xffffd590
strcmp("snlprintf\n", "snlprintf\n")                          = 0
puts("[You've got shell]!"[You've got shell]!
)                                                             = 20
geteuid()                                                     = 12003
geteuid()                                                     = 12003
setreuid(12003, 12003)                                        = 0
system("/bin/sh"$ ls
level3
$ id
uid=12003(leviathan3) gid=12003(leviathan3) groups=12003(leviathan3)
$ exit
 <no return ...>
--- SIGCHLD (Child exited) ---
<... system resumed> )                                        = 0
+++ exited (status 0) +++
```
```
leviathan3@leviathan:~$ ./level3
Enter the password> snlprintf
[You've got shell]!
$ id
uid=12004(leviathan4) gid=12003(leviathan3) groups=12003(leviathan3)
$ cat /etc/leviathan_pass/leviathan4
vuH0coox6m
```
```
password : vuH0coox6m
```

## Level 4
```
ssh leviathan4.labs.overthewire.org -p 2223
```
```
leviathan4@leviathan:~/.trash$ ltrace ./bin
__libc_start_main(0x80484bb, 1, 0xffffd774, 0x80485b0 <unfinished ...>
fopen("/etc/leviathan_pass/leviathan5", "r")                              = 0
+++ exited (status 255) +++
leviathan4@leviathan:~/.trash$ ./bin
01010100 01101001 01110100 01101000 00110100 01100011 01101111 01101011 01100101 01101001 00001010
```
저 2진수 값은 비밀번호이므로 변환시켜주면 됨.
```
password : Tith4cokei
```

## Level 5
```
ssh leviathan5.labs.overthewire.org -p 2223
```
```
leviathan5@leviathan:~$ ltrace ./leviathan5
__libc_start_main(0x80485db, 1, 0xffffd784, 0x80486a0 <unfinished ...>
fopen("/tmp/file.log", "r")                                               = 0
puts("Cannot find /tmp/file.log"Cannot find /tmp/file.log
)                                         = 26
exit(-1 <no return ...>
+++ exited (status 255) +++

leviathan5@leviathan:~$ ls -al /tmp/file.log
ls: cannot access '/tmp/file.log': No such file or directory
```
```/tmp/file.log```에 심볼릭 링크를 걸어주면 됨.
```
leviathan5@leviathan:~$ ln -s /etc/leviathan_pass/leviathan6 /tmp/file.log
leviathan5@leviathan:~$ ltrace ./leviathan5
leviathan5@leviathan:~$ ./leviathan5
UgaoFee4li
```
```
password : UgaoFee4li
```

## Level 6
```
ssh leviathan6.labs.overthewire.org -p 2223
```
```
leviathan6@leviathan:~$ ltrace ./leviathan6
__libc_start_main(0x804853b, 1, 0xffffd784, 0x80485e0 <unfinished ...>
printf("usage: %s <4 digit code>\n", "./leviathan6"usage: ./leviathan6 <4 digit code>
)                      = 35
exit(-1 <no return ...>
+++ exited (status 255) +++
```
```
leviathan6@leviathan:~$ ltrace ./leviathan6 1234
__libc_start_main(0x804853b, 2, 0xffffd784, 0x80485e0 <unfinished ...>
atoi(0xffffd8ad, 0, 0xf7e40890, 0x804862b)                                = 1234
puts("Wrong"Wrong
)                                                             = 6
+++ exited (status 0) +++
```
gdb로 까보면 입력값과 ```ebp-0xc```에 있는 값을 비교를 하여 같아야 함.  
```ebp-0xc```에는 ```0x1bd3```이 있는데 이는 정수로 ```7123```임.  
따라서 입력을 7123해주면 됨.
```
leviathan6@leviathan:~$ ./leviathan6 7123
$ cat /etc/leviathan_pass/leviathan7
ahy7MaeBo9
```
```
password : ahy7MaeBo9
```

## Level 7
```
ssh leviathan7.labs.overthewire.org -p 2223
```
```
leviathan7@leviathan:~$ cat CONGRATULATIONS 
Well Done, you seem to have used a *nix system before, now try something more serious.
(Please don't post writeups, solutions or spoilers about the games on the web. Thank you!)
```
