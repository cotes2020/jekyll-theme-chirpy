---
title: Bring old pc to live
date: 2020-07-16 11:11:11 -0400
categories: [30System]
tags: [Lab]
math: true
image:
---

# old pc

1. check the version

```
dxdiag
```

---

2. make the usability

```py

- erase the sd Card
  - ExFAT
  - 2nd 主引导记录

- Download Zorin OS.
  - The 64-bit version is recommended for most computers.
  - Older PCs with less than 2GB of RAM: 32-bit version.

- make boot sd card

  $ diskutil list
  /dev/disk0
  /dev/disk1
  /dev/disk2
  disk2s1上面的/dev/disk2即是U盘挂载点。

  diskutil unmountDisk /dev/disk2
  这样就有了一个已经插入但是unmount的U盘了，在Finder下看不到这个U盘了，但是用diskutil list命令还可以看到。

  将iso文件写到这个U盘里
  dd if=~/Downloads/debian-6.0.6-i386-CD-1.iso of=/dev/disk2 bs=1m
  648+0 records in
  648+0 records out
  679477248 bytes transferred in 447.883182 secs (1517086 bytes/sec)


- Create an Install Drive with balenaEtcher.
  - https://www.balena.io/etcher/
  - Open balenaEtcher and press "Select image" to choose your downloaded Zorin OS ".iso" file.
  - select your USB flash drive
  - Press "Flash!" to begin writing Zorin OS to the USB drive

```

3. Boot from the Install Drive.

```py

- plugged in the written USB Install Drive.

- Boot: Boot Device Menu
    - win vaio E serie: 64bit
      - f2: setup
        - boot:
        - External device boot -> Enable.
        - 1st boot priority -> external device
        - exit: save change -> exit setup.
      - press F11 when poweron.
    - win vaio SZ serie: 32bit
      - esc: chose boot drive
      - f2: setup

```

balenaEtcher: system not found
yumi kali fat32



.
