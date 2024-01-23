---
title: Linux - Boot Process Concepts
date: 2020-07-16 11:11:11 -0400
categories: [00CodeNote, LinuxNote]
tags: [Linux, Boot]
math: true
image:
---

# CompTIA Linux+ XK0-004

[toc]

---

# Linux - Boot Process Concepts

## ✔️ | **Legacy GRUB**


<img alt="pic" src="https://i.imgur.com/pmLrHrH.png" width=600>

> BIOS >
> boot.img (load core file) (first 512 bytes) >
> core.img (address of boot disk, the actual boot partition) >
> /boot/grub >
> Linus get loaded by GRUB

```c
findmnt /boot
TARGET SOURCE    FSTYPE OPTIONS
/boot  /dev/sda1 ext4   rw,relatime,seclabel

// install grub to the device
grub-install device_name
grub-install /dev/sda1

// grub shell
find /grub/stage1

quit
```


<img alt="pic" src="https://i.imgur.com/ZcM1ybR.png" width=600>

MBR(Master Boot Record) change to GPT(GUID Partition Table)
- more partitions, and bigger partitions size
- GPT need UEFI(Unified extensible firmware interface) to boot:
    - UEFI replace BIOS,
    - avoid unauthorized OS from boot (boot with USB)

> UEFI BIOS >
> boot.img (load core file) (first 512 bytes) >
> GPT header > partition entry array >
> core.img (empty sectors address of boot disk, the actual boot partition) >
> /boot/efi (vfat or FAT32 ESP) >
> /boot/grub2
> Linus get loaded by GRUB

edit:
open > copy new file >


```c
grub2-editenv list

saved_entry=25f2bab51ef24f2aad905f9999f19bf3-4.18.0-147.8.1.el8_1.x86_64
kernelopts=root=/dev/mapper/cl-root ro crashkernel=auto resume=/dev/mapper/cl-swap rd.lvm.lv=cl/root rd.lvm.lv=cl/swap rhgb quiet
boot_success=1
boot_indeterminate=0


grub2-mkconfig
update-grub


# ls /etc/grub.d/
00_header          01_users      20_ppc_terminfo   40_custom
00_tuned           10_linux      30_os-prober      41_custom
01_menu_auto_hide  20_linux_xen  30_uefi-firmware  README
```

## Boot Loader

GRUB Legacy

```c
key:

grub:

A : append options to the kernel boot line

C : open up the GRUB commandline

grub > setup (hd0)

ESC : quiet

Arrow : highlight the option in the GRUB meny


grub2:

E : commandline

grub> ls
grub> ls (hd0,1)/
grub> linux /boot/vmlinuz-generic root=/dev/vda1
grub> initrd / boot/initrd.ing-generic
grub> boot
```

---

## initramfs

initial RAM disk

```c
ls /boot/initrad-default


lsinitrd | less
Image: /boot/initramfs-4.18.0-147.8.1.el8_1.x86_64.img: 27M
========================================================================
Early CPIO image
========================================================================
drwxr-xr-x   3 root     root            0 Jan  3 13:12 .
-rw-r--r--   1 root     root            2 Jan  3 13:12 early_cpio
drwxr-xr-x   3 root     root            0 Jan  3 13:12 kernel
drwxr-xr-x   3 root     root            0 Jan  3 13:12 kernel/x86
drwxr-xr-x   2 root     root            0 Jan  3 13:12 kernel/x86/microcode
-rw-r--r--   1 root     root        25600 Jan  3 13:12 kernel/x86/microcode/Genui
neIntel.bin
========================================================================

dracut modules:
bash
systemd
systemd-initrd
nss-softokn
rngd
i18n
network-legacy
network
ifcfg
drm
plymouth
prefixdevname
dm
kernel-modules
kernel-modules-extra
kernel-network-modules
lvm
resume
```

## dracut:
- create a new initramfs for kernel.
- add or remove modules and drivers from initramfs builds


```c
dracut -o "fcoe fcoe-uefi" -M -F

/etc/dracut.conf.d
// modify for new kernel install
omit_dracutmodules+="fcoe fcoe-uefi"

// run dracut config file
dracut -f
```


## other boot

boot from Network

![Screen Shot 2020-06-05 at 23.10.34](https://i.imgur.com/iLpqYZc.png)

PXE boot system.
after linux kernel and initrd are loaded from TFTP server.
the rest of the system file can be loaded from HTTP/NFS server.


---


# kernel


## kernel:

```c
uname
Linux

uname -m
x86_64

uname -rm
4.18.0-147.8.1.el8_1.x86_64 x86_64

uname -a
Linux server0.demo.local 4.18.0-147.8.1.el8_1.x86_64 #1 SMP Thu Apr 9 13:49:54 UTC 2020 x86_64 x86_64 x86_64 GNU/Linux
```

## module:

```c
lsmod
Module                  Size  Used by
fuse                  126976  3
xt_CHECKSUM            16384  1
ipt_MASQUERADE         16384  1
xt_conntrack           16384  1
nft_chain_nat_ipv6     16384  6
nf_conntrack_ipv6      20480  33
nf_defrag_ipv6         20480  1 nf_conntrack_ipv6
nf_nat_ipv6            16384  2 nft_chain_nat_ipv6,nft_masq_ipv6


modinfo module_name


modprobe module_name     // inload
modprobe -r module_name  // delete


ls /lib/modules/

ls /usr/lib/modules/$(uname -r)/

ls /lib/modules/$(uname -r)/
bls.conf       modules.alias.bin    modules.drm          source
build          modules.block        modules.modesetting  symvers.gz
config         modules.builtin      modules.networking   System.map
extra          modules.builtin.bin  modules.order        updates
kernel         modules.dep          modules.softdep      vdso
misc           modules.dep.bin      modules.symbols      vmlinuz
modules.alias  modules.devname      modules.symbols.bin  weak-updates


ls /lib/modules/$(uname -r)/kernel/drivers/block
brd.ko.xz   nbd.ko.xz       pktcdvd.ko.xz  virtio_blk.ko.xz    zram
loop.ko.xz  null_blk.ko.xz  rbd.ko.xz      xen-blkfront.ko.xz


rmmod loop.ko.xz
insmod /lib/modules/$(uname -r)/kernel/drivers/block/loop.ko.xz
// require full path


stop from running:

ls /etc/modprobe.d/
firewalld-sysctls.conf  lockd.conf  nvdimm-security.conf  tuned.conf
kvm.conf                mlx4.conf   truescale.conf        vhost.conf

vim /etc/modprobe.d/xxx-blacklist.conf
blacklist loop


depmod
```


## kernel panic

![Screen Shot 2020-06-05 at 23.37.38](https://i.imgur.com/vMAQztK.png)

**auto reboot**:
- contains the number of seconds that a system will wait before rebooting to recover from a kernel panic.
- default 0: will not reboot
- but volatile

```c
/proc/sys/kernel/panic
0

echo 5 > /proc/sys/kernel/panic
```

**keep change**:

```c
/etc/sysctl/con.f
kernel.panic=15

sysctl -p // reload
```






























.
