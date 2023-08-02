---
title: File Types in Linux
author: dyl4n
date: "2023-03-10 00:30:23 +0700"
categories:
  - Operating System
tags:
  - linux
render_with_liquid: false
comments: true
image:
  path: "/thumbnails/Linux-File-Types.png"
published: false
---

In Linux/UNIX, Files are mainly categorized into 3 parts:

- Regular Files
- Directory Files
- Special Files

The easiest way to find out file type in any operating system is by looking at its extension such as .txt, .sh, .py, etc. If the file doesn’t have an extension then in Linux we can use file utility to determine it.

There are all 7 file types in linux as shown in the following table:

| File Symbol |     Meaning      |      Located in      |
| :---------: | :--------------: | :------------------: |
|      -      |   Regular File   | Any directory/Folder |
|      d      |    Directory     |  It is a directory   |
|      l      | Symbol link File |         /dev         |
|      c      |  Character File  |         /dev         |
|      s      |   Socket File    |         /dev         |
|      p      |    Pipe File     |         /dev         |
|      b      |    Block FIle    |         /dev         |

Let's learn more about each file types!

## Regular File

- Regular files are ordinary files on a system that contains programs, texts, or data. It is used to store information such as text, or images.
- These files are located in a directory/folder.
- Regular files contain all readable files such as text files, Docx files, programming files, etc, Binary files, image files such as JPG, PNG, SVG, etc, compressed files such as ZIP, RAR, etc.

![regular file](https://user-images.githubusercontent.com/98354414/222961140-ce3035b5-af88-481b-9fca-0f4e4b21c6a5.png)

Using the "**file\***" to find out the file type
![file](https://user-images.githubusercontent.com/98354414/222961266-1443d423-b6e1-4eb3-87eb-302cf431a723.png)

## Directory Files

- Directory files store the other regular files, directory files, and special files and their related information.
- A directory file contains an entry for every file and sub-directory that it houses.
- We can navigate between directories using the `cd` command

We can filter the directory files using following command:

```bash
ls -l | grep ^d
```

![directory files](https://user-images.githubusercontent.com/98354414/222961506-5520415b-442a-4d27-b495-0eb01107f644.png)

## Special Files

### 1. Block Files

- Block files act as a direct interface to block devices hence they are also called block devices.
- These files are hardware files and most of them are present in /dev.

We can find out the block files in `/dev` by the following command:
![block files filter](https://user-images.githubusercontent.com/98354414/222961816-e39258f0-71bf-46e8-b59d-51dd561d10d5.png)

### 2. Character device files

- A character file is a hardware file that reads/writes data in character by character in a file.
- These files provide a serial stream of input or output and provide direct access to hardware devices. The terminal, serial ports, etc are examples of this type of file.

We can find out and determine character device file as the following command:
![image](https://user-images.githubusercontent.com/98354414/222962030-97ab81cc-bc50-43a7-9dd5-8851b740ed6b.png)

### 3. Pipe Files

- The other name of pipe is a “named” pipe, which is sometimes called a FIFO. FIFO stands for “First In, First Out” and refers to the property that the order of bytes going in is the same coming out.
- The “name” of a named pipe is actually a file name within the file system.
- This file sends data from one process to another so that the receiving process reads the data first-in-first-out manner.

We can find out pipe file by using the following command:
![pipe files 1](https://user-images.githubusercontent.com/98354414/222962407-a31dd016-e32b-4ca8-88d6-2e98eac85ebf.png)
Pipe File can be created by `mkfifo`

### 4. Symbol link files

- A symbol link file is a type of file in Linux which points to another file or a folder on your device.
- Symbol link files are also called **Symlink** and are similar to **shortcuts in Windows**.

![symbol link files 1](https://user-images.githubusercontent.com/98354414/222962608-8b9851e9-eacd-4269-86fd-3fc6aa7465e8.png)

### 5. Socket Files:

- A socket is a special file that is used to pass information between applications and enables the communication between two processes.
- We can create a socket file using the socket() system call.
- Socket files are the special files that use a file name as their address instead of an IP address and port number. Socket files use the sendmsg() and recvmsg() system calls to enable inter-process communication between local applications.
- A socket file is located in **/dev** of the root folder or you can use the `find / -type s` command to find socket files.
  ![socket files](https://user-images.githubusercontent.com/98354414/222962753-ff62256f-3d56-4831-bfc7-ff7a99722c55.png)

We can find out Symbol link file by using the following command:

```bash
ls -l | grep ^s
```
