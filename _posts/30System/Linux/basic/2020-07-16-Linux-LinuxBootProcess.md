---
title: Linux - Linux Boot Process
date: 2020-07-16 11:11:11 -0400
categories: [30System, Basic]
tags: [Linux, Boot]
math: true
image:
---

# linux boot process

A Linux system goes through several stages when booting.

the <font color=bluepi> firmware </font> stage
-  the computer runs code in `BIOS` or `UEFI` during power on self test or post.
- Older computers have a `BIOS` and newer computers have `UEFI`.

the <font color=bluepi> boot loader </font> stage
- After the firmware stage, the BIOS or UEFI executes the boot loader stage.
- the boot loader is `grub2`, or grand unified boot loader two.
- `Grub`'s job is to <font color=red> read configuration file and boot the Linux kernel </font>
  - For BIOS machine, grub reads in `/boot/grub2/grub.cfg`.
  - For UEFI systems, it loads `/boot/efi/EFI/redhat/grub.efi`.
- The boot loader then executes the kernel.

the <font color=bluepi> kernel </font> stage
- the kernel loads a `ramdisk` into memory.
- This `ramdisk` serves as a temporary root file system, includes `kernel` `modules`, `drivers`, and possibly even `kickstart` files.
- Later, the kernel unmounts the `ramdisk` and mounts the `root file system on the hard drive`.
- And then, starts the initialization stage by executing the first process.

the <font color=bluepi> initialization </font> stage
- In the initialization stage, the grandfather process runs.
- In older versions of Red Hat this was the `Init` process. Init was replaced by `Upstart`, which has now been replaced by `systemd`.
- `Systemd` then starts all system services, a login shell or a graphical interface.
- When it's finished, the OS is ready to be used.
- `Systemd` has the concept of Targets that similar to the old `Init` run levels. You can think of a target as a system configuration.
  - default, boots up into the graphical.target.
- A system can be booted into different Targets for different purposes, such as rescuing the system after a crash.
