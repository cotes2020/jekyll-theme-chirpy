---
title: Terminal Aliases
author: Nick_Post
date: 2023-03-04 14:10:00 +0800
categories: [shell, bash, alias, linux]
tags: [notes, bash, linux, shell, alias, fish]
render_with_liquid: false
pin: false
---

# Aliases - Makes your terminal better

When using the $BASH shell you'll need to first edit the '.BashRC' file.
```bash
    sudo nano ~/.bashrc
```
If you're using $FISH you'll need to create a config fiile without SUDO
```bash
    nano ~/.config/fish/config.fish
```


> You may need to comment out already exsisting lines like LS.
{: .prompt-tip }


> Identify your network interface name with 'ip -a' 
{: .prompt-warning }

## New and updated Aliases

```bash
#Fixes the LS command
   alias ls="ls -lhF --time-style=long-iso --color=auto --ignore=lost+found"

#Fixes the df command
   alias df="df -h -x squashfs -x tmpfs -x devtempfs"

#Makes Debian installing faster
   alias install="sudo apt install"
   alias upgrade="sudo apt update && sudo apt dist-upgrade"
   alias aptclean="sudo apt autoclean && sudo apt autoremove"

   #Makes Fedora/RHEL installing faster
    #alias install="sudo dnf install"
    #alias upgrade="sudo dnf update -y"


#Tells you your IP address NEED CURL
   alias extip="curl icanhazip.com"
#AWK has issues inside of an alias
   alias intip="ip addr | awk -F' |/' '$5=="inet" && $8!="scope" {print $6}'"
#This is for VM's on proxmox - enp6s18 may be different on your system
   alias ethip="ip -4 addr show enp6s18 | grep -oP '(?<=inet\s)\d+(\.\d+){3}'"

#Cleans up the MOUNT command
   alias lsmount="mount | column -t"
   
#This needs python3 (whitch ppython3)
   alias speedtest="curl -s https://raw.githubusercontent.com/sivel/speedtest-cli/master/speedtest.py | python3 -"
```
When you're done, exit nano (How do I exit nano?) you may need to exit your bash session and open a new one. 

Thanks to [LearnLinuxTV](https://www.learnlinux.tv/linux-essentials-bash-aliases/) for this info

{% include embed/youtube.html id='Ok_kD_sgNcs' %}


I'll update this as i come up with new alias examples.