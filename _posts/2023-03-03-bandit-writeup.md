---
title: OverTheWire - bandit Writeup
author: dyl4n
date: 2023-03-03 09:55:23 +0700
categories: [CTF Writeup, Overthewire Bandit]
tags: [overthewire]
render_with_liquid: false
comments: true
---

## Bandit Level 0 &rarr; 1

The password for the next level is stored in a file called readme located in the home directory. Use this password to log into bandit1 using SSH. Whenever you find a password for a level, use SSH (on port 2220) to log into that level and continue the game

### Solution

The level 0 requires us to login wargame via ssh.

```
ssh username@host [-p port]
```

Back to wargame, I will use `https://bandit.labs.overthewire.org/` is the host, on port `2220`, with both username and password are `bandit0`. Open SSH, and login!

```
ssh bandit0@bandit.labs.overthewire.org -p 2220
```

![bandit0 login](/posts/bandit-writeup/1.png)

All I need to do is read it!

```
bandit0@bandit:~$ cat readme
NH2SXQwcBdpmTEzi3bvBHMM9H66vVXjL
```

Flag: _NH2SXQwcBdpmTEzi3bvBHMM9H66vVXjL_

---

## Bandit Level 1 &rarr; 2

The password for the next level is stored in a file called - located in the home directory

### Solution

```
ssh bandit1@bandit.labs.overthewire.org -p 2220
```

![dashed filename](/posts/bandit-writeup/2.png)

There are several ways to read dashed filename. Simply I just use `cat ./-`.
Additionally, you can use `cat < -`

```
bandit1@bandit:~$ cat ./-
rRGizSaX8Mk1RTb1CNQoXTcYZWU6lgzi
```

Flag: _rRGizSaX8Mk1RTb1CNQoXTcYZWU6lgzi_

---

## Bandit Level 2 &rarr; 3

The password for the next level is stored in a file called spaces in this filename located in the home directory

### Solution

![image](/posts/bandit-writeup/3.png)

There's the password file, just read it and move on. But... Let's take a closer look at file name. If you use `cat spaces in this filename`, cat thinks we want to read 4 files. So, in this case, I can put filename to `''` or put `\` before spaces character

```
cat 'spaces in this filename'
or
cat spaces\ in\ this\ filename
```

Result:

```
bandit2@bandit:~$ cat spaces\ in\ this\ filename
aBZ0W5EmUfAf7kHTQeOwd8bauFJ2lAiG
```

Flag: _aBZ0W5EmUfAf7kHTQeOwd8bauFJ2lAiG_

---

## Bandit Level 3 &rarr; 4

The password for the next level is stored in a hidden file in the inhere directory.

### Solution

There's a `inhere` directory in bandit3's home dir.

![hidden file](/posts/bandit-writeup/4.png)

Use `ls -a` to list all content in a directory, consist of hidden files.

Let's cat it and move on!

```
bandit3@bandit:~/inhere$ cat .hidden
2EW7BBsr6aMMoJ2HjW067dm8EgX26xNe
```

Flag: _2EW7BBsr6aMMoJ2HjW067dm8EgX26xNe_

---

## Bandit Level 4 &rarr; 5

The password for the next level is stored in the only human-readable file in the inhere directory.

Tip: if your terminal is messed up, try the “reset” command.

### Solution

There's `inhere` directory, let's see the content of this.

![content of inhere/](/posts/bandit-writeup/5.png)

Here you can see multiple of dash filename that already learnt in level 1. According to the challenge, the password's stored in the only `HUMAN-READABLE` file (ascii text format).

```
find . -exec file {} + | grep ASCII
```

![flag bandit4](/posts/bandit-writeup/6.png)

Flag: _lrIWWI6bB37kxfiCQZqUdOIYfr6eEeqR_

---

## Bandit Level 5 &rarr; 6

The password for the next level is stored in a file somewhere under the inhere directory and has all of the following properties:

- human-readable
- 1033 bytes in size
- not executable

### Solution

I make use of `find` command. Learn more `find` command [here](http://https://man7.org/linux/man-pages/man1/find.1.html "here")

```
bandit5@bandit:~/inhere$ find . ! -executable -size 1033c -exec file {} + | grep ASCII
./maybehere07/.file2: ASCII text, with very long lines (1000)
```

This means I find in current directory, a non-executable file with -size is 1033c (`c` for bytes), read more at `man find`.

```
bandit5@bandit:~/inhere$ cat ./maybehere07/.file2
P4L4vucdmLnm8I7Vl7jG1ApGSfjYKqJU
```

Flag: _P4L4vucdmLnm8I7Vl7jG1ApGSfjYKqJU_

---

## Bandit Level 6 &rarr; 7

The password for the next level is stored somewhere on the server and has all of the following properties:

- owned by user bandit7
- owned by group bandit6
- 33 bytes in size

### Solution

Seem likes it's similar to the previous level. Keyword is "somewhere on the **server**" So I'll look on entire filesystem instead of finding in current directory.

```
bandit6@bandit:~$ find / -user bandit7 -group bandit6 -size 33c 2>/dev/null
/var/lib/dpkg/info/bandit7.password
```

I used `2> /dev/null` what for? `2` refers to `stderr` (standard error). This means all of the error message in this output will be redirected to `/dev/null`, which is a null device file. This will discard anything written to it, and will return EOF on reading.

```
bandit6@bandit:~$ cat /var/lib/dpkg/info/bandit7.password
z7WtoNQU2XfjmMtWA8u5rN4vzqu4v99S
```

Flag: _z7WtoNQU2XfjmMtWA8u5rN4vzqu4v99S_

---

## Bandit Level 7 &rarr; 8

The password for the next level is stored in the file data.txt next to the word millionth

### Solution

There's a data.txt file in home directory, just grep `millionth` in it to find the password.

```
bandit7@bandit:~$ grep 'millionth' data.txt
millionth       TESKZC0XvTetK0S9xNwm25STk5iWrBvP
```

Flag: _TESKZC0XvTetK0S9xNwm25STk5iWrBvP_

---

## Bandit Level 8 &rarr; 9

The password for the next level is stored in the file data.txt and is the only line of text that occurs only once

### Solution

This challenge require us to know how to filter one line different from the others using some necessary command like `sort`, `uniq`

```
bandit8@bandit:~$ sort data.txt | uniq -u
EN632PlfYiZbn3PhVK3XOGSlNInNE00t
```

Where,
`sort` then sorts the output alphabetically

`uniq -u` show the unique lines , but it only works on ordered lists, so I've to sort first.

Flag: _EN632PlfYiZbn3PhVK3XOGSlNInNE00t_

---

## Bandit Level 9 &rarr; 10

The password for the next level is stored in the file data.txt in one of the few human-readable strings, preceded by several ‘=’ characters.

### Solution

Just `strings` file **data.txt** to meet with the human-readable characteristic and `grep` several '=' character to find the password

![bandit9 password](/posts/bandit-writeup/7.png)

Flag: _G7w8LIi6J3kTb8A7j9LgrywtEUlyyp6s_

---

## Bandit Level 10 &rarr; 11

The password for the next level is stored in the file data.txt, which contains base64 encoded data

### Solution

Read file `data.txt`, then I get a base64 encoded data. I solved it by using `base64` from **coreutils** package to decode it. The syntax is:

```
base64 [option]… [file]
```

Read more about base64 in [RFC4648](https://www.rfc-editor.org/rfc/rfc4648 "rfc4648")

```
bandit10@bandit:~$ base64 --decode data.txt
The password is 6zPeziLdR2RKNdNYFNb6nVCKzphlXHBM
```

Flag: _6zPeziLdR2RKNdNYFNb6nVCKzphlXHBM_

---

## Bandit Level 11 &rarr; 12

The password for the next level is stored in the file data.txt, where all lowercase (a-z) and uppercase (A-Z) letters have been rotated by 13 positions

### Solution

See the content of **data.txt**

```
bandit11@bandit:~$ cat data.txt
Gur cnffjbeq vf WIAOOSFzMjXXBC0KoSKBbJ8puQm5lIEi
```

Combined with challenge's description, the data have been rotated by 13 position => rot13

For that, I use `tr` command, completed syntax is:

```
tr '[a-zA-Z]' '[n-za-mN-ZA-M]'
```

**Explanation**: Every “A” is replaced by an “N”, every “B” by an “O”, so on.

```
bandit11@bandit:~$ cat data.txt | tr '[a-zA-Z]' '[n-za-mN-ZA-M]'
The password is JVNBBFSmZwKKOP0XbFXOoW8chDz5yVRv
```

Flag: _JVNBBFSmZwKKOP0XbFXOoW8chDz5yVRv_

---

## Bandit Level 12 → 13

The password for the next level is stored in the file data.txt, which is a hexdump of a file that has been repeatedly compressed. For this level it may be useful to create a directory under /tmp in which you can work using mkdir. For example: mkdir /tmp/myname123. Then copy the datafile using cp, and rename it using mv (read the manpages!)

### Solution

First, I'll open `data.txt` file in home dir and check it!

```shell
bandit12@bandit:~$ cat data.txt
00000000: 1f8b 0808 8c3f f563 0203 6461 7461 322e  .....?.c..data2.
00000010: 6269 6e00 0134 02cb fd42 5a68 3931 4159  bin..4...BZh91AY
00000020: 2653 5953 6696 8100 001b 7fff fbdb effb  &SYSf...........
00000030: b41f 6efa a7cb ebee fff3 b7ad 897d f77f  ..n..........}..
00000040: 67bf beff bb6b aaff ff3b ff7b b001 3b5b  g....k...;.{..;[
00000050: 4100 00d0 3101 881a 0d34 01a0 000d 0006  A...1....4......
00000060: 10c4 d006 41b5 1a0d 0064 0340 64c8 3468  ....A....d.@d.4h
00000070: 1934 1a0d 1a68 da26 26d3 50e4 d0d3 40d0  .4...h.&&.P...@.
00000080: d001 a341 b500 0032 320d 0323 47a9 a683  ...A...22..#G...
00000090: 4346 9a00 3d40 36a0 0308 184d 0640 0068  CF..=@6....M.@.h
000000a0: 0c43 466a 0d34 6832 9a68 6430 40d3 4d34  .CFj.4h2.hd0@.M4
000000b0: d0d0 7a80 d0c2 69a3 268d 1a06 81a0 00d0  ..z...i.&.......
000000c0: c83f 5232 3400 c406 8da8 0680 3400 6800  .?R24.......4.h.
000000d0: 001a 0020 2823 e282 2299 1ae9 cfa4 8ea0  ... (#..".......
000000e0: 716d 6e03 9844 dd8b 7260 8c1e e05c d068  qmn..D..r`...\.h
000000f0: 9a86 f4d8 b355 8786 1723 3041 695d f96a  .....U...#0Ai].j
00000100: f8c0 503b 8df1 eac8 138b 82ed 21cb 9611  ..P;........!...
00000110: 6d6a e5c3 c7ca 637c 26d9 ed7e 107a 14a2  mj....c|&..~.z..
00000120: 6c54 8868 511f 481a 6412 bb95 a771 0401  lT.hQ.H.d....q..
00000130: 3ca4 96cf 7e08 0e31 d967 e4c4 4fee 206b  <...~..1.g..O. k
00000140: 8793 ec23 4da7 44ba 3ded 12e2 b947 9288  ...#M.D.=....G..
00000150: 7809 0ca2 6b04 5f0d e0b2 6717 7e87 0628  x...k._...g.~..(
00000160: 11a3 d282 9d61 f0a4 340c af19 d501 4ddd  .....a..4.....M.
00000170: 1a8c c27b 154c 531f 345c b6a2 7298 a20c  ...{.LS.4\..r...
00000180: e02d bb16 9127 5b42 30d6 634c b7cd 54ae  .-...'[B0.cL..T.
00000190: bb26 9494 2a19 33bc b233 0d8c a75a ccf8  .&..*.3..3...Z..
000001a0: 401c d5f4 bd06 7c43 cd73 32d3 84d0 c440  @.....|C.s2....@
000001b0: 004e b2e9 de84 8251 e080 1a1e f506 e546  .N.....Q.......F
000001c0: cf30 31af 361e b04c 8f5a f636 f1e7 4c24  .01.6..L.Z.6..L$
000001d0: e14b 456b 109e 1421 99e5 ead9 3840 038f  .KEk...!....8@..
000001e0: c1d8 c71a 9b5d 5435 afa0 5eca 34ca a83c  .....]T5..^.4..<
000001f0: 309e 6b5d 532f a0af 20e0 bc3f bb03 a680  0.k]S/.. ..?....
00000200: 6616 4b13 9d09 bf8b 3a93 6f16 b48a e6cf  f.K.....:.o.....
00000210: ccb9 084c 8a35 12a7 447d 8224 4491 e534  ...L.5..D}.$D..4
00000220: 0c71 2f36 fda1 8b54 0808 a144 9894 966f  .q/6...T...D...o
00000230: be74 2140 952c 0294 a1d6 841e 1658 756f  .t!@.,.......Xuo
00000240: 0d7f c5dc 914e 1424 14d9 a5a0 4043 a8c0  .....N.$....@C..
00000250: f434 0200 00                             .4...
```

Here as you can see, it's a hexdump which is made by `xxd` command and reversed also by it

![man xxd](/posts/bandit-writeup/8.png)

Referring to challenge's description, Let's create a directory in tmp file and make edits on `data.txt` file

```shell
bandit12@bandit:~$ mktemp -d
/tmp/tmp.UXHp76B0fO
bandit12@bandit:~$ cp data.txt /tmp/tmp.UXHp76B0fO
bandit12@bandit:~$ cd /tmp/tmp.UXHp76B0fO
bandit12@bandit:/tmp/tmp.UXHp76B0fO$ ls
data.txt
```

Now, we have to reverse hexdump using `xxd`

```shell
bandit12@bandit:/tmp/tmp.UXHp76B0fO$ xxd -r  data.txt > password
```

`-r` option for running the reverse hexdump, and `>` redirect to **password** file.
Then, we should check what type of **password** file:

```
bandit12@bandit:/tmp/tmp.UXHp76B0fO$ file password
password: gzip compressed data, was "data2.bin", last modified: Tue Feb 21 22:02:52 2023, max compression, from Unix, original size modulo 2^32 564

```

From here on, the flow as below steps:

1. Using `file` command to determine the type of file
1. Using `mv` command to change the file type to specific file extension.
1. Using the appropriate unzip tools to decompress files (gzip/gunzip, bzip2, tar,...)
1. Repeat and repeat to get the password!

And, the summarized result is here:

```
bandit12@bandit:/tmp/tmp.UXHp76B0fO$ xxd -r data.txt > password

bandit12@bandit:/tmp/tmp.UXHp76B0fO$ file password
password: gzip compressed data, was "data2.bin", last modified: Tue Feb 21 22:02:52 2023, max compression, from Unix, original size modulo 2^32 564
bandit12@bandit:/tmp/tmp.UXHp76B0fO$ mv password password.gz
bandit12@bandit:/tmp/tmp.UXHp76B0fO$ gunzip password.gz

bandit12@bandit:/tmp/tmp.UXHp76B0fO$ file password
password: bzip2 compressed data, block size = 900k
bandit12@bandit:/tmp/tmp.UXHp76B0fO$ mv password password.bz2
bandit12@bandit:/tmp/tmp.UXHp76B0fO$ bzip2 -d password.bz2

bandit12@bandit:/tmp/tmp.UXHp76B0fO$ file password
password: gzip compressed data, was "data4.bin", last modified: Tue Feb 21 22:02:52 2023, max compression, from Unix, original size modulo 2^32 20480
bandit12@bandit:/tmp/tmp.UXHp76B0fO$ mv password password.gz
bandit12@bandit:/tmp/tmp.UXHp76B0fO$ gunzip password.gz

bandit12@bandit:/tmp/tmp.UXHp76B0fO$ file password
password: POSIX tar archive (GNU)
bandit12@bandit:/tmp/tmp.UXHp76B0fO$ mv password password.tar
bandit12@bandit:/tmp/tmp.UXHp76B0fO$ tar -xvf password.tar
data5.bin

bandit12@bandit:/tmp/tmp.UXHp76B0fO$ file data5.bin
data5.bin: POSIX tar archive (GNU)
bandit12@bandit:/tmp/tmp.UXHp76B0fO$ mv data5.bin data5.tar
bandit12@bandit:/tmp/tmp.UXHp76B0fO$ tar -xvf data5.tar
data6.bin

bandit12@bandit:/tmp/tmp.UXHp76B0fO$ file data6.bin
data6.bin: bzip2 compressed data, block size = 900k
bandit12@bandit:/tmp/tmp.UXHp76B0fO$ mv data6.bin data6.bz2
bandit12@bandit:/tmp/tmp.UXHp76B0fO$ bzip2 -d data6.bz2

bandit12@bandit:/tmp/tmp.UXHp76B0fO$ file data6
data6: POSIX tar archive (GNU)
bandit12@bandit:/tmp/tmp.UXHp76B0fO$ mv data6 data6.tar
bandit12@bandit:/tmp/tmp.UXHp76B0fO$ tar -xvf data6.tar
data8.bin

bandit12@bandit:/tmp/tmp.UXHp76B0fO$ file data8.bin
data8.bin: gzip compressed data, was "data9.bin", last modified: Tue Feb 21 22:02:52 2023, max compression, from Unix, original size modulo 2^32 49
bandit12@bandit:/tmp/tmp.UXHp76B0fO$ mv data8.bin data8.gz
bandit12@bandit:/tmp/tmp.UXHp76B0fO$ gunzip data8.gz

bandit12@bandit:/tmp/tmp.UXHp76B0fO$ file data8
data8: ASCII text

bandit12@bandit:/tmp/tmp.UXHp76B0fO$ cat data8
The password is wbWdlBxEir4CaE8LaPhauuOo6pwRmrDw
```

Flag: _wbWdlBxEir4CaE8LaPhauuOo6pwRmrDw_

---

## Bandit Level 13 → 14

The password for the next level is stored in /etc/bandit_pass/bandit14 and can only be read by user bandit14. For this level, you don’t get the next password, but you get a private SSH key that can be used to log into the next level. Note: localhost is a hostname that refers to the machine you are working on

### Solution

This challenge require us to know about **SSH** and **SSH Key**. If you haven't known about it yet, you can read it [here](https://help.ubuntu.com/community/SSH/OpenSSH/Keys "here")

![bandit13 home](/posts/bandit-writeup/9.png)

Try login the **bandit14**'s box using SSH Key in home directory's bandit13 with the provided information.

```
bandit13@bandit:~$ ssh -i "sshkey.private" bandit14@localhost -p 2220
```

Boom! I've just entered the bandit14's box! Let's get the password in the `/etc/bandit_pass/bandit14`:

```
bandit14@bandit:~$ cat /etc/bandit_pass/bandit14
fGrHPx402xGC7U7rXKDaxiWFTOiF0ENq
```

Flag: _fGrHPx402xGC7U7rXKDaxiWFTOiF0ENq_

---

## Bandit Level 14 → 15

The password for the next level can be retrieved by submitting the password of the current level to port 30000 on localhost.

### Solution

The solution here we can use either `nc` or `telnet`. This means, we submit the current password's bandit14 (you can retrieve in the `/etc/bandit_pass/bandit14`) to **localhost port 30000**.

```
bandit14@bandit:~$ nc localhost 30000
fGrHPx402xGC7U7rXKDaxiWFTOiF0ENq
Correct!
jN2kgmIXJ6fShzhT2avhotn4Zcka6tnt
```

Additionally, we can use `echo fGrHPx402xGC7U7rXKDaxiWFTOiF0ENq | nc localhost 30000` . When we run this command, we take the **STDOUT** of echo and pipe it as the **STDIN** of netcat. It returns the same result.

Or we can use `telnet` instead.

```
bandit14@bandit:~$ telnet localhost 30000
Trying 127.0.0.1...
Connected to localhost.
Escape character is '^]'.
fGrHPx402xGC7U7rXKDaxiWFTOiF0ENq
Correct!
jN2kgmIXJ6fShzhT2avhotn4Zcka6tnt

Connection closed by foreign host.
```

Flag: _jN2kgmIXJ6fShzhT2avhotn4Zcka6tnt_

---

## Bandit Level 15 → 16

The password for the next level can be retrieved by submitting the password of the current level to port 30001 on localhost using SSL encryption.

Helpful note: Getting “HEARTBEATING” and “Read R BLOCK”? Use -ign_eof and read the “CONNECTED COMMANDS” section in the manpage. Next to ‘R’ and ‘Q’, the ‘B’ command also works in this version of that command…

Helpful Reading Material

- [Secure Socket Layer/Transport Layer Security on Wikipedia](https://en.wikipedia.org/wiki/Transport_Layer_Security#SSL_1.0,_2.0,_and_3.0/)
- [OpenSSL Cookbook - Testing with OpenSSL](https://www.feistyduck.com/library/openssl-cookbook/online/)

### Solution

Read the hint, as we can see, the level goal mentions `ssl`, there's a tool called `openssl`. Let's look there:

![bandit15 image-1](/posts/bandit-writeup/10.png)

```
s_client
This implements a generic SSL/TLS client which can establish a
transparent connection to a remote server speaking SSL/TLS. It's
intended for testing purposes only and provides only rudimentary
interface functionality but internally uses mostly all
functionality of the OpenSSL ssl library.

-connect
Tests connectivity to an HTTPS service.
```

Let's connect to the given port via **openssl** as above options.

```
bandit15@bandit:~$ openssl s_client -connect localhost:30001
CONNECTED(00000003)
Can't use SSL_get_servername
depth=0 CN = localhost
verify error:num=18:self-signed certificate
verify return:1
depth=0 CN = localhost
verify error:num=10:certificate has expired
notAfter=Mar  4 13:56:53 2023 GMT
verify return:1
depth=0 CN = localhost
notAfter=Mar  4 13:56:53 2023 GMT
verify return:1
---
[...]
---
read R BLOCK
jN2kgmIXJ6fShzhT2avhotn4Zcka6tnt
Correct!
JQttfApK4SeyHwDlI9SXGR50qclOAil1

closed
```

Flag: _JQttfApK4SeyHwDlI9SXGR50qclOAil1_

---

## Bandit Level 16 → 17

The credentials for the next level can be retrieved by submitting the password of the current level to a port on localhost in the range 31000 to 32000. First find out which of these ports have a server listening on them. Then find out which of those speak SSL and which don’t. There is only 1 server that will give the next credentials, the others will simply send back to you whatever you send to it.

### Solution

Proceed to scan port on localhost in range 31000-32000 first. of course, using the `nmap`

![nmap scan](https://user-images.githubusercontent.com/98354414/232568606-44191a0b-ec0c-4a9b-a1da-f7d614a40be7.png)

We can see the `31790` port running ssl service. Try connecting this port and see:

![private key](https://user-images.githubusercontent.com/98354414/232569256-55fe8ac7-4d0e-45d9-8b6f-19402b450303.png)

Great, that's all we need. Copy it to clipboard and move on next Challenge!

---

## Bandit Level 17 → 18

There are 2 files in the homedirectory: passwords.old and passwords.new. The password for the next level is in passwords.new and is the only line that has been changed between passwords.old and passwords.new

NOTE: if you have solved this level and see ‘Byebye!’ when trying to log into bandit18, this is related to the next level, bandit19

### Solution

We use the private SSH key file to login in **bandit17**'s box

![bandit17 login](/posts/bandit-writeup/11.png)

"_It is required that your private key files are NOT accessible by others._" So I'll modify the permission of it for only user to read it.

```
chmod 400 bandit17.private
```

Then try reconnect and it's successful.

![bandit17 login successfully](/posts/bandit-writeup/12.png)

There are 2 files as mentioned in the description and we have to findout the only line that has been changed between them.

With the given hints, we just need to know about `diff` command usage.

![diff command](/posts/bandit-writeup/13.png)

Let's proceed to `diff` two password files!

```
bandit17@bandit:~$ diff passwords.old passwords.new
42c42
< f9wS9ZUDvZoo3PooHgYuuWdawDFvGld2
---
> hga5tuuCLF6fFzUpnagiMN8ssu9LFrdg

```

Flag: _hga5tuuCLF6fFzUpnagiMN8ssu9LFrdg_

---

## Bandit Level 18 → 19

The password for the next level is stored in a file readme in the homedirectory. Unfortunately, someone has modified .bashrc to log you out when you log in with SSH.

### Solution

Try connecting the **bandit18**'s box then immediately logged out. due to the modified **.bashrc** file.

```
Byebye !
Connection to bandit.labs.overthewire.org closed.

```

So I try using `dash` instead of `bash`

```shell
┌──(dyl4n㉿kali)-[~]
└─$ ssh -t bandit18@bandit.labs.overthewire.org -p 2220 /bin/dash
                         _                     _ _ _
                        | |__   __ _ _ __   __| (_) |_
                        | '_ \ / _` | '_ \ / _` | | __|
                        | |_) | (_| | | | | (_| | | |_
                        |_.__/ \__,_|_| |_|\__,_|_|\__|


                      This is an OverTheWire game server.
            More information on http://www.overthewire.org/wargames

bandit18@bandit.labs.overthewire.org's password:
$ ls
readme
```

Yep, it works. Let's get the password in `readme` file and move on!

```
$ cat readme
awhqfNnAbc1naukrpqDYcF95h7HoMTrC
```

Flag: _awhqfNnAbc1naukrpqDYcF95h7HoMTrC_

---

## Bandit Level 19 → 20

To gain access to the next level, you should use the setuid binary in the homedirectory. Execute it without arguments to find out how to use it. The password for this level can be found in the usual place (/etc/bandit_pass), after you have used the setuid binary.

### Solution

When logged in, we can see the highlighted file named "bandit20-do". It seems like a binary file.
![bandit20-do](/posts/bandit-writeup/14.png)

I proceed to execute it and then it displayed something like that

```shell
bandit19@bandit:~$ ./bandit20-do
Run a command as another user.
  Example: ./bandit20-do id
```

Maybe it's executable file and help me to be able to execute command under **bandit20**'s permission.
No-doubt I run the file along with `cat /etc/bandit_pass/bandit20` to get the flag!

```
bandit19@bandit:~$ ./bandit20-do cat /etc/bandit_pass/bandit20
VxCazJaVykI6W36BkBU0mJTCM8rR95XT
```

Flag: _VxCazJaVykI6W36BkBU0mJTCM8rR95XT_

---

## Bandit Level 20 → 21

There is a setuid binary in the homedirectory that does the following: it makes a connection to localhost on the port you specify as a commandline argument. It then reads a line of text from the connection and compares it to the password in the previous level (bandit20). If the password is correct, it will transmit the password for the next level (bandit21).

NOTE: Try connecting to your own network daemon to see if it works as you think

### Solution

My idea is to create a listener and simultaneously send the previous level password (I use `nc` with pipe line). Then using `suconnect` file to connect that port and it'll return my expect output.

```shell
bandit20@bandit:~$ echo -n "VxCazJaVykI6W36BkBU0mJTCM8rR95XT" | nc -lp 8888 &
[1] 2301973
```

The `-n` flag is to prevent newline characters in the input.

`-lp`: listener port

Then, we try using file `./suconnect` to connect to that port and get the password!

```shell
bandit20@bandit:~$ ./suconnect 8888
Read: VxCazJaVykI6W36BkBU0mJTCM8rR95XT
Password matches, sending next password
NvEJF7oVjkddltPSrdKEFOllh9V1IBcq
```

Flag: _NvEJF7oVjkddltPSrdKEFOllh9V1IBcq_

## Bandit Level 21 → 22

A program is running automatically at regular intervals from cron, the time-based job scheduler. Look in /etc/cron.d/ for the configuration and see what command is being executed.

### Solution

Try seeing the content of `cronjob_bandit22` at `/etc/cron.d`

```
bandit21@bandit:/etc/cron.d$ cat cronjob_bandit22
@reboot bandit22 /usr/bin/cronjob_bandit22.sh &> /dev/null
* * * * * bandit22 /usr/bin/cronjob_bandit22.sh &> /dev/null
```

As we can see, there's a bash script at `/usr/bin/cronjob_bandit22.sh` and it execute something.

```
bandit21@bandit:/etc/cron.d$ cat /usr/bin/cronjob_bandit22.sh
#!/bin/bash
chmod 644 /tmp/t7O6lds9S0RqQh9aMcz6ShpAoZKF7fgv
cat /etc/bandit_pass/bandit22 > /tmp/t7O6lds9S0RqQh9aMcz6ShpAoZKF7fgv
```

It seems like output of bandit22's password file is redirected to `/tmp/t7O6lds9S0RqQh9aMcz6ShpAoZKF7fgv` under the bandit22 user's privilege. Let's get password there!

```
bandit21@bandit:/etc/cron.d$ cat /tmp/t7O6lds9S0RqQh9aMcz6ShpAoZKF7fgv
WdDozAdTM2z9DiFEQ2mGlwngMfj4EZff
```

Flag: _WdDozAdTM2z9DiFEQ2mGlwngMfj4EZff_

## Bandit Level 22 → 23

A program is running automatically at regular intervals from cron, the time-based job scheduler. Look in /etc/cron.d/ for the configuration and see what command is being executed.

NOTE: Looking at shell scripts written by other people is a very useful skill. The script for this level is intentionally made easy to read. If you are having problems understanding what it does, try executing it to see the debug information it prints.

### Solution

```
bandit22@bandit:~$ cat /etc/cron.d/cronjob_bandit23
@reboot bandit23 /usr/bin/cronjob_bandit23.sh  &> /dev/null
* * * * * bandit23 /usr/bin/cronjob_bandit23.sh  &> /dev/null
```

Here's the content of `/usr/bin/cronjob_bandit23.sh`. First, we need to understand this bash script.

```
bandit22@bandit:~$ cat /usr/bin/cronjob_bandit23.sh
#!/bin/bash

myname=$(whoami)
mytarget=$(echo I am user $myname | md5sum | cut -d ' ' -f 1)

echo "Copying passwordfile /etc/bandit_pass/$myname to /tmp/$mytarget"

cat /etc/bandit_pass/$myname > /tmp/$mytarget
```

This script will do something like that: `myname` variable is the bandit's user, `mytarget` is the md5hash and echo the password of bandit23 to `/tmp/$mytarget` as I mentioned

Here's the flag!!

```
bandit22@bandit:~$ echo I am user bandit23 | md5sum | cut -d ' ' -f 1
8ca319486bfbbc3663ea0fbe81326349
bandit22@bandit:~$ cat /tmp/8ca319486bfbbc3663ea0fbe81326349
QYw0Y2aiA672PsMmh9puTQuhoz8SyR2G
```

Flag: _QYw0Y2aiA672PsMmh9puTQuhoz8SyR2G_

## Bandit Level 23 → 24

A program is running automatically at regular intervals from cron, the time-based job scheduler. Look in /etc/cron.d/ for the configuration and see what command is being executed.

NOTE: This level requires you to create your own first shell-script. This is a very big step and you should be proud of yourself when you beat this level!

NOTE 2: Keep in mind that your shell script is removed once executed, so you may want to keep a copy around…

### Solution

Let's see the bandit24 cronjob:
![cronjob_bandit24](https://user-images.githubusercontent.com/98354414/232648889-c15bc003-183e-4a25-9feb-487c232daead.png)

It's a piece of code. Let's see how it works

```
#!/bin/bash

myname=$(whoami)

cd /var/spool/$myname/foo
echo "Executing and deleting all scripts in /var/spool/$myname/foo:"
for i in * .*;
do
    if [ "$i" != "." -a "$i" != ".." ];
    then
        echo "Handling $i"
        owner="$(stat --format "%U" ./$i)"
        if [ "${owner}" = "bandit23" ]; then
            timeout -s 9 60 ./$i
        fi
        rm -f ./$i
    fi
done
```

To put it simply, I will explain the overview.

First of all, this script will "Executing and deleting all scripts in /var/spool/$myname/foo:" with `$myname` is the current user. Instead of bandit23 user, I replace it with **bandit24** in order to the script run Executing and deleting all scripts that I put in /var/spool/bandit24/foo/ . Based on that, I easily read the bandit24's password.

Thence, I myself write a tiny script to handle above idea.

Create a temp folder and make it full permission.
![mktemp](https://user-images.githubusercontent.com/98354414/232651158-24dae5de-748f-467f-9834-ece0fabc910c.png)

Write a small script to get the bandit24's password.
![dyl4n script](https://user-images.githubusercontent.com/98354414/232652669-1adae9ea-0b98-4066-89e9-2ed2c8549101.png)

Before moving the scripts, Besure that the script's executable
![dyl4n script handling](https://user-images.githubusercontent.com/98354414/232652318-fe0b6255-612d-445d-93d6-5ac310d8a5f2.png)

Wait a minute,

![flag](https://user-images.githubusercontent.com/98354414/232653530-398f2de6-eb63-4d33-8433-e5c101bad8d6.png)

Flag: _VAfGXJ1PBSsPSnvsjI8p759leLZ9GGar_

## Bandit Level 24 → 25

A daemon is listening on port 30002 and will give you the password for bandit25 if given the password for bandit24 and a secret numeric 4-digit pincode. There is no way to retrieve the pincode except by going through all of the 10000 combinations, called brute-forcing.

You do not need to create new connections each time

### Solution

![connect to listening port](https://user-images.githubusercontent.com/98354414/232739574-808082f4-10e7-4747-94f1-dff41f6846e7.png)

To pass the challenge, we need enter the password for user bandit24 and the secret 4-digit pincode on a single line, separated by a space.

There're 10000 combinations of 4-digit pincode, shell-script'll help us in this case.

```
#!/bin/bash

pass=VAfGXJ1PBSsPSnvsjI8p759leLZ9GGar

for i in {0000..9999};
do
        echo $pass' '$i;
done | nc localhost 30002
```

Go to /tmp and create a yourself directory, we work under it to run the shell script.

![bandit24 password](https://user-images.githubusercontent.com/98354414/232744859-9abdbbbe-089c-4016-8742-678717ccbefe.png)

Flag: _p7TaowMYrmu23Ol8hiZh9UvD0O9hpx8d_

## Bandit Level 25 → 26

Logging in to bandit26 from bandit25 should be fairly easy… The shell for user bandit26 is not /bin/bash, but something else. Find out what it is, how it works and how to break out of it.

### Solution

There's a private key file in **bandit25**'s home directory.

Lets try login as user **bandit26** using this key with command:

```
ssh -i bandit26.sshkey bandit26@localhost -p 2220
```

We connect to localhost because bandit26 also exists the same server device as bandit25.
![login bandit26](https://user-images.githubusercontent.com/98354414/232754674-db9a2b7f-e2f3-4526-8591-9264740d9503.png)

Looking the output looks like we logged in bandit26 but logout immediately.

According to the level goal, "**The shell for user bandit26 is not /bin/bash**". Lets findout what shell bandit26 is being used, shell's information is in the '/etc/passwd':

![bandit26's shell](https://user-images.githubusercontent.com/98354414/232758113-01ecc442-d8ba-44cc-aba1-5e628d250b0e.png)

Let's have a look at content of '**/usr/bin/showtext**'

![/usr/bin/showtext](https://user-images.githubusercontent.com/98354414/232758590-7bef6875-ff43-46e9-9a33-71016afde0f0.png)

The shell script using the '**more**' command and the output is the contents of **text.txt**, after that exit immeadiately.

After researching '**more**' for a while, I've gathered some cool stuff:

- **More** is a filter for paging through text one screenful at a time.
- If the amount of content to be displayed is larger than the terminal size, the **more** enable interactive mode.
- v' Start up an editor at current line. The editor is taken from the environment variable VISUAL if defined, or EDITOR if VISUAL is not defined, or defaults to "vi" if neither VISUAL nor EDITOR is defined.

I try resizing the terminal window size smaller till the interactive mode is enabled

![login bandit26](https://user-images.githubusercontent.com/98354414/232765675-b5585f64-9e86-4c9b-98ca-547ee406f8c9.png)

Yeah, now we can resize terminal windows bigger to handle easily

As I mentioned, we just hit `v` to convert to `vi` editor.

![vi mode](https://user-images.githubusercontent.com/98354414/232766619-d5f3a0bc-293f-4ff1-8fa8-ecd3241595b9.png)

Vim has a mode called **Command-Line Mode** where you can enter your own commands in Vim. Googling to know how to use it :)

![set shell](https://user-images.githubusercontent.com/98354414/232767273-8e5d1ce7-a26c-423e-be96-a3e6c288ea92.png)

![enter shell](https://user-images.githubusercontent.com/98354414/232767735-5d0be903-0731-4bac-99fe-fdf8a87698ee.png)

Flag: _c7GvcKlw9mC7aUQaPx7nwFstuAIBw1o1_

## Bandit Level 26 → 27

Good job getting a shell! Now hurry and grab the password for bandit27!

### Solution

Do the same steps as the previous level to login bandit26.

![bandit27-do](https://user-images.githubusercontent.com/98354414/232804750-32e4a450-8d08-47e1-a803-9e39cc5cffa3.png)

`bandit27-do` has suid bit, Try running it and see how it goes.

```
bandit26@bandit:~$ ./bandit27-do
Run a command as another user.
  Example: ./bandit27-do id
```

Let's get the password thanks to **bandit27-do** run command:

```
bandit26@bandit:~$ ./bandit27-do cat /etc/bandit_pass/bandit27
YnQpBuifNMas1hcUFk70ZmqkhUU2EuaS
```

Flag: _YnQpBuifNMas1hcUFk70ZmqkhUU2EuaS_

## Bandit Level 27 → 28

There is a git repository at ssh://bandit27-git@localhost/home/bandit27-git/repo. The password for the user bandit27-git is the same as for the user bandit27.

Clone the repository and find the password for the next level.

### Solution

Run the following command to clone the git repo (remember to create a folder in /tmp to work on):

```
git clone ssh://bandit27-git@localhost:2220/home/bandit27-git/repo
```

![git connect](https://user-images.githubusercontent.com/98354414/232808863-6291388b-ea84-4eb4-9c4e-e8dd119c4235.png)

Enter password as the bandit27's password that we can clone this repo.

After clone it, go to /repo and read `README` file to get password.

![bandit27 password](https://user-images.githubusercontent.com/98354414/232809609-d08d5887-d7ee-42d3-ba98-302bfde9ee77.png)

Flag: _AVanL161y9rsbcJIsFHuw35rjaOM19nR_

## Bandit Level 28 → 29

There is a git repository at ssh://bandit28-git@localhost/home/bandit28-git/repo. The password for the user bandit28-git is the same as for the user bandit28.

Clone the repository and find the password for the next level.

### Solution

Same as the previous challenge, let's connect and clone repo to your own temp directory.

```
ssh://bandit28-git@localhost:2220/home/bandit28-git/repo
```

There's a README.md file in the repo.

![README.md](https://user-images.githubusercontent.com/98354414/232812369-7bf91996-59f6-4e83-9dd2-ec084a417e46.png)

No password here but it mentions 'bandit29'. Try reviewing some recent commit by using `git log`.

![git log](https://user-images.githubusercontent.com/98354414/232825903-9b90a349-2e69-43d5-a2c9-8307eb327637.png)

`add missing data` sounds interesting. Let's see the changes of file in this commit using `git show`

```
 it show 6c3c5e485cc531e5d52c321587ce1103833ab7c3
```

![bandit29 password](https://user-images.githubusercontent.com/98354414/232826551-1a765057-226e-4fde-a7a3-fd8ed53ec5cb.png)

Flag: _tQKvmcwNYcFS6vmPHIUSI3ShmsrQZK8S_

## Bandit Level 29 → 30

There is a git repository at ssh://bandit29-git@localhost/home/bandit29-git/repo. The password for the user bandit29-git is the same as for the user bandit29.

Clone the repository and find the password for the next level.

### Solution

clone repo to your own temp directory and check README.md file.

```
git clone ssh://bandit29-git@localhost:2220/home/bandit29-git/repo
```

![clone repo](https://user-images.githubusercontent.com/98354414/232828744-6ee88f03-3d3f-4b98-9821-6ec9fb227496.png)

Maybe there is a development branch exisiting in the repository. Check existing branches:

![git branches](https://user-images.githubusercontent.com/98354414/232838713-5089b1e8-e87d-4ad6-8456-5a42c94760fd.png)

Let's checkout `remotes/origin/dev` and re-check the README.md file

```
bandit29@bandit:/tmp/tmp.G0EVhgUlAE/repo$ cat README.md
# Bandit Notes
Some notes for bandit30 of bandit.

## credentials

- username: bandit30
- password: xbhV3HpNGlTIdnjUrdAlPzc2L6y9EOnS
```

Flag: _xbhV3HpNGlTIdnjUrdAlPzc2L6y9EOnS_
