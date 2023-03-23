---
title: Mac - Terminal Note
date: 2020-07-16 11:11:11 -0400
categories: [30System, MacOS]
tags: [MacOS]
math: true
image:
---


# Mac - Terminal Note

- [Mac - Terminal Note](#mac---terminal-note)
  - [文件目录：](#文件目录)
  - [基本命令:](#基本命令)
  - [compile C++ in MAC](#compile-c-in-mac)
- [compile C++/c in Win](#compile-cc-in-win)
- [terminal](#terminal)
  - [将普通成员改为管理员？](#将普通成员改为管理员)
  - [virtualbox clone](#virtualbox-clone)
  - [`brew`](#brew)
    - [`brew info [formula]`](#brew-info-formula)
    - [`brew cleanup`](#brew-cleanup)
    - [`brew doctor`](#brew-doctor)
    - [`brew update`](#brew-update)
    - [`brew upgrade`](#brew-upgrade)
    - [add line to .bash\_profile](#add-line-to-bash_profile)


---

## 文件目录：
" / "  ：根目录
" ~ " ：用户主目录的缩写。例如当前用户为hello，那么" ~ "展开来就是：/Users/hello
" . "  ：当前目录
".."   ：父目录

## 基本命令:

1. **clear** 清空当前输入

2. history 查看输入历史记录
- 上下方向键
- history查看输入的完整历史

```
  600  ls
  601  cd Public/
  602  ls
  603  cd /
  604  ls
  605  cd ..
  606  ls
  607  cd /
  608  ls
  609  history
  610  history
```

## compile C++ in MAC

`g++` is the C++ compiler frontend to GCC.
`gcc` is the C compiler frontend to GCC.

`g++ hw.cpp ./a.out`

# compile C++/c in Win
 
# terminal
`ifconfig` in mac
`ipconfig` in win


## 将普通成员改为管理员？
- 按住command+s，再按开机键。
- 需先登录进系统，出现命令行终端的时候按照以下顺序输入命令：

```
/sbin/mount -uaw
rm var/db/.applesetupdone
reboot
```

- reboot完成后，创建一个新的用户，按照提示操作。
- 新用户创建后，打开系统偏好设置-用户与群组点击原来的普通用户，右侧有个“允许用户管理这台电脑”，打勾然后重启。
- 这样就可以以管理员的身份登录到你原来的系统，再把刚新创建的管理员账户删除就可以了。


## virtualbox clone

```
common-lisp
J:~ luo$ cd /Users/luo/VirtualBox\ VMs/win0.0
J:win0.0 luo$ ls
Logs			win0.0.vbox-prev
share			win0.0.vdi
win0.0.vbox
J:win0.0 luo$ VBoxManage clonehd win0.0.vdi win0.vdi
0%...10%...20%...30%...40%...50%...60%...70%...80%...90%...100%
Clone medium created in format 'VDI'. UUID: 984e9128-1326-40d6-a192-c65e674f7da8
```


---



## `brew`

### `brew info [formula]`

get details about options on any Homebrew formula

`brew info [formula]`

### `brew cleanup`

### `brew doctor`

### `brew update`

### `brew upgrade`

### add line to .bash_profile

- adding the following line at the bottom of your `~/.bashrc` file.
- `export PATH=/usr/local/bin:$PATH`

1. to append the line to your .bash_profile:
`echo 'export PATH=/usr/local/bin:$PATH' >>~/.bash_profile`


2. command

```bash
#Start up Terminal
$ cd ~/                 # go to your home folder
$ touch .bash_profile   # create your new file
$ open -e .bash_profile # open it in TextEdit
$ . .bash_profile       # to reload&update .bash_profile
```

1. Open the TextEdit app
Navigate to File → Open.... In the center drop down, be sure to select Home. Or, select the Home directory in the left pane. Then, use "COMMAND+SHIFT+." to show hidden files:

```
.bash_profile is a script that is executed each time you start a new shell. On Linux, it's called under different circumstances than .bashrc, but on OS X, they work exactly the same way. Any command you add to the file will be run whenever you open a new terminal window (thus starting a new interactive shell).

$PATH is a variable that tells the shell where to look for executable files - so when you type a command, the system will search each directory specified in that variable until it finds an executable program with that command's name.

The command `export PATH=/usr/local/bin:$PATH` prepends the directory `/usr/local/bin` to the current PATH, so it becomes the first directory searched by the shell.

.bash_profile just a normal plain text file - you can edit it with any text editor, including vi or nano, or even a graphical editor like TextEdit. It's up to you - just remember to save it as a plain-text file.
```
