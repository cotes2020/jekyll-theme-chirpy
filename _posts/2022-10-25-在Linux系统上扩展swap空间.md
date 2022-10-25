---
title: 在Linux系统上扩展swap空间
author: mumu
date: 2020-10-25 16:48:00 +0800
categories: [Linux,内存,swap]
tags: [Linux,swap,内存]
pin: true
---

# 在Linux系统上扩展swap空间

## 一、what is swap？

SWAP的作用类似Windows系统下的“虚拟内存”。当物理内存不足时，拿出部分硬盘空间当SWAP分区（虚拟成内存）使用，从而解决内存容量不足的情况。

当某进程向OS请求内存发现不足时，OS会把内存中暂时不用的数据交换出去，放在SWAP分区中，这个过程称为SWAP OUT。

当某进程又需要这些数据且OS发现还有空闲物理内存时，又会把SWAP分区中的数据交换回物理内存中，这个过程称为SWAP IN。

swap使用完，操作系统会触发OOM-Killer机制，把消耗内存最多的进程kill掉以释放内存。

## 二、涉及命令

```shell
free：查看内存状态命令，可以显示memory、swap、buffer/cache等的大小及使用状况;
dd：读取，转换并输出数据命令；
mkswap：设置交换区
swapon：启用交换区，相当于mount
swapoff：关闭交换区，相当于umount
```

## 三、操作

### 3.1.查看当前swap情况

```shell
root@VM-4-6-ubuntu:~# free -h
              total        used        free      shared  buff/cache   available
Mem:           1.8G        955M        111M        4.9M        765M        697M
Swap:          4.0G        3.5M        4.0G
```

### 3.2.关闭所有swap

```shell
swapoff -a
```

### 3.3.创建swap分区的文件

```shell
dd if=/dev/zero of=swapfile bs=1M count=8192
```

`bs`是每块的大小，`count`是块的数量；`bs*count`，就是swap文件的大小

`swapfile`文件路径

### 3.4.格式化swapfile并启用swap分区文件

```shell
mkswap swapfile
swapon swapfile
```

### 3.5.添加开机启动

修改`/etc/fstab`这个文件，添加或者修改这一行：

```shell
/swapfile swap swap defaults 0 0
```

<font color=red>swapfile为全路径</font>

此项不是必须的。如果不修改开机启动配置，重启之后的swap空间会重置，之前配置丢失。

### 3.6. 设置swap占用

1. 查看系统的 **swappiness**

   ```sh
   $ cat /proc/sys/vm/swappiness
   ## 一般默认 60  （内存达40 开始使用swap）
   ```
2. 修改swappiness值为10
   ```sh
   $ sudo sysctl vm.swappiness=10
   ## 临时性的修改，在你重启系统后会恢复默认
   ```
3. 永久修改
   ```sh
   $ sudo vim /etc/sysctl.conf
   ## 添加 vm.swappiness=10 保存，重启，OK
   ```
### 3.7.[扩展方式] 通过新建分区来扩展原有swap空间

这个是Linode的默认做法，这里也介绍一下。此方法与swap文件类似，只是使用了一个独立分区，而不是文件。

1. 使用fdisk创建交换分区（假设是 `/dev/sdb2`）

2. 使用mkswap设置交换分区：

   ```shell
   mkswap /dev/sdb2
   ```

3. 启用交换分区

   ```shell
   swapon /dev/sdb2
   ```

4. 修改`/etc/fstab`添加到开启启动项：

   ```shell
   /dev/sdb2 swap swap defaults 0 0
   ```

# `参考资料`

- [Linux Swap是干嘛的？](https://www.cnblogs.com/pipci/p/11399250.html)
- [linux增加swap空间的方法小结](https://www.cnblogs.com/tocy/p/linux-swap-cmd-summary.html)



