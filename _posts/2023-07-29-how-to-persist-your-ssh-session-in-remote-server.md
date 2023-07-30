---
title: How to Persist Your SSH Session in Remote Server
author: Hulua
date: 2023-07-29 20:55:00 +0800
categories: [Linux]
tags: [linux, development environment]
---

One of the major advatanage of developing server side applications in your local enviornment is that the terminal session never gets disconnected. The session is always available.
```bash
➜  yywe.github.io git:(master) ✗ ls
Gemfile                     _javascript                 assets
Gemfile.lock                _layouts                    gulpfile.js
LICENSE                     _plugins                    index.html
README.md                   _posts                      jekyll-theme-chirpy.gemspec
_config.yml                 _sass                       note.txt
_data                       _site                       package.json
_includes                   _tabs                       tools
```
However, in many companies, it is daily work to SSH into a server and do the development work, and it is annoying that you often see that your session session get disconnected.

```console
:client_loop: send disconnect: Broken pipe
```

This may happen while you are having launch or due to network issue or other whatever reason. The consequence is that your onging work/process will be terminated due to this closed session. 

To solve the problem, I have known the "screen" command for a long time, which can decouple the server side processes with the console window. However, it is still normal that you SSH session may be disconnected and you need to re-connect and attach the your screen session. 

Is there any way to make your SSH session in a remote server never gets lost and work like your local terminal session. Probably yes! In this post, I will introduce how to achieve this goal. The basic idea is to combine "autossh" and "screen".

If you are not familar with autossh and screen, you might want to learn some basics before continue reading. Usually screen is already installed in the remote server, and you need to install autossh in your local machine.

Let's make your remote SSH session like your local terminal session and never gets lost!

## 1. Prepare your screen config.

This is simple, vim ~/.screenrc and put below line:

```console
termcapinfo xterm* ti@:te@
```

in the file. If you do not do this, when you scroll your mouse wheel you cannot view your historical commands and output.

## 2. Write the below shell script to a file like myssh.sh

```bash

# get your local terminal session id (macos, other platform not verified)
session_id=$(echo $TERM_SESSION_ID | cut -d':' -f2)
# set your username
user_id=<username>
# the host you will be connecting, will be passed as a parameter
host=$1
if [ -z $host ]; then
    echo "please specify the host"
    exit
fi
echo "autossh connect to $host using sessionid=$session_id"

# key command, explanation will be followed. 
autossh -M 0  -o TCPKeepAlive=yes -o ServerAliveCountMax=20 -o ServerAliveInterval=30 $user_id@$host -t screen -d -R $session_id

```

### Explanation 

* -M 0: you can specify autossh to open extra TCP port to monitor your ssh session, but in practice I find that is not a must, can use -M 0 so we do not open extra ports in your server.

* -o TCPKeepAlive=yes -o ServerAliveCountMax=20 -o ServerAliveInterval=30 : This is the first layer of protection, the options specified here will try to keep your SSH session alive by sending heartbeat information at given interval. This is optional as we will have another two mechanism to make your session persistant.

* screen -d -R $session_id: -R $session_id will try to attach to a screen session named $session_id if exists, if not exists, it will create a session named $session_id. However, it may happen that one ssh connection is disconnected but the server is not refreshed (detached), when autossh reconnect, since it is still attached, it will start a new screen session with different process id and you will end up like below:
```console
4028118.4A435FE8-E9D4-42E8-A40F-FCCFF5C198C2    (07/29/2023 01:13:37 AM)        (Attached)
3802470.4A435FE8-E9D4-42E8-A40F-FCCFF5C198C2    (07/28/2023 05:43:22 PM)        (Detached)
```
you will have 2 processes using the same session id! That is why -d comes into the picture. with -d, it will first detach any existing one first. So you will eventually have 1 session binded to your local terminal window.

* now your session will be under the protection of screen, so you will never lose your session context. Lastly, if for whatever reason like network issue, your connection get lost, autossh will reconnect and reattach to the dedicated screen session. 

## 3. Usage

with the above script, if you want to ssh into your remote server, do:

```console
~/myssh.sh myserver.mydomain.com
```

## 4. Tips

Now you will have your SSH session persisted, and you will never (hopefully) have your SSH session disconnected unexpectedly. However, if you closed your terminal window, your screen session may still be there. So you may need to constantly monior your screen sessions. Put below lines in your server's ~/.bashrc

```bash
alias sls='screen -ls'
function sk(){
    sname=$1
    if [ -z $sname ]; then
        echo "require session name"
        return
    fi
    screen -S $sname -X quit
}
```

With the setting, you can use `sls` to list existing screen sessions. If you do not want any of them, do `sk session_name` to quit the session.

Now you are enpowered to use your remote server like they are your local machine without need to worry about the annoying disconnection issue.
