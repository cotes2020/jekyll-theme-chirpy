---
title: Meow's Testing Tools - hashcat
# author: Grace JyL
date: 2019-09-17 11:11:11 -0400
description:
excerpt_separator:
categories: [10CyberAttack, CyberAttackTools]
tags: [CyberAttack, CyberAttackTools]
math: true
# pin: true
toc: true
# image: /assets/img/sample/devices-mockup.png
---

[toc]

---

# hashcat

---

## install

```bash
# official web
https://hashcat.net/hashcat/

cd /hashcat-6.1.1
sudo make

# execute hash cat
./hashcat --help
```


## auto install script
```bash

#!/bin/bash
git clone https://github.com/hashcat/hashcat.git
mkdir -p hashcat/deps
git clone https://github.com/KhronosGroup/OpenCL-Headers.git hashcat/deps/OpenCL
cd hashcat/ && make
./hashcat --version
./hashcat -b -D 1,2
./example0.sh
```


# Sample To Use







.
