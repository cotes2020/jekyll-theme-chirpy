---
title: 配置多个 github 账号
author: jiap
date: 2021-02-18 
categories: [Skills]
tags: [Github]
---

一般人使用 github 只用一个账号，但对于开发人员而言，会涉及到不同的 github 账号，那一台电脑如何连接两个 github 账号呢？本文记录了多个 github 是如何配置的。

本文使用两个 github 账号为 `jiap` 和 `physicsay`。使用的 ssh key 仅为文章所用。 

> 本文基于 OSX Big Sur。

## 0. 预备知识

Secure Shell (SSH) 是一个允许两台电脑之间通过安全的连接进行数据交换的网络协议。通过加密保证了数据的保密性和完整性。SSH采用公钥加密技术来验证远程主机，以及(必要时)允许远程主机验证用户。

配置 github 的 SSH keys 可以方便登陆到 github 服务器，无需输入密码。具体方法为：在本地电脑生成 SSH key (包含公钥和密钥)，再在 github 中配置上生成的公钥，这样就可以在本地无密码登陆 github 了。

## 1. 创建 ssh key

打开终端 (Terminal)，输入以下语句，该语句是列出 `~/.ssh` 文件夹下有哪些文件，`~/.ssh` 是存放 ssh 连接配置的文件夹。

```bash
$ ls ~/.ssh
$ cd ~/.ssh
```
> 终端中代码均以 `$` 开始，实际运行不需要输入。

如果 `~/.ssh` 下有 `id_rsa`(私钥) 和 `id_rsa.pub` (公钥)，说明之前已生成过 SSH key 了，第二行语句是进入 `~/.ssh` 文件夹。

在终端输入

```bash
$ ssh-keygen -t rsa -C "jiap@hotmail.com"
```

终端会提示输入 ssh 密钥名称，输入 `github_jiap` 就会生成 `github_jiap` 和 `github_jiap.pub` 两个文件。若直接按回车，则生成文件名默认为  `id_rsa` 和 `id_ras.pub`，其中 `.pub` 为公钥。


<img src='/assets/img/2021-02-18-config-sevreal-github-accounts/new_rsa.png' style='zoom:80%; margin: 0 auto; display: block;'>

若在 `~/.ssh` 文件夹下已存在，则会提示是否覆盖，如下图箭头处所示。

<img src='/assets/img/2021-02-18-config-sevreal-github-accounts/exist_rsa.png' style='zoom:80%; margin: 0 auto; display: block;'>

这样就完成了两个 ssh key 的创建。

---
> 如果不想输入密钥名称，可以使用以下语句，也会生成两个密钥文件。

```bash
$ ssh-keygen -t rsa -C "physicsay@outlook.com" -f "physicsay_github"
```


## 2. 在 github 添加公钥

在终端输入以下命令，输出公钥内容，并复制：

```bash
$ cat github_jiap.pub
```

打开 github 网页，在设置- SSH & GPG keys 处设置，如下图所示：

<img src='/assets/img/2021-02-18-config-sevreal-github-accounts/github_setting.png' style='zoom:60%; margin: 0 auto; display: block;'>

点击 `New SSH key` 将之前在终端处得到的公钥粘贴保存即可。

同样的方式，添加第二个 github 账号的公钥。

## 3. 使用 ssh-agent 注册新的 SSH 密钥

ssh-agent：是一个可以控制和保存公钥身份验证所使用的私钥的程序。如果没有正在运行的 ssh-agent，在终端执行以下代码，以确保 ssh-agent 运行。

```bash
$ eval "$(ssh-agent -s)" 
```
再使用 ssh-add 添加私钥：

```bash
$ ssh-add ~/.ssh/github_jiap
$ ssh-add ~/.ssh/github_physicsay
```

输入 `$ ssh-add -l` 可查看现有的私钥；输入 `$ ssh-add -D` 可在验证程序中删掉所有的密钥(原密钥不会删除，只删除验证程序 `ssh-add` 中的密钥)。

## 4. 设置配置文件

在终端新建 `config` 文件，并进行配置

```bash
$ touch config
```

在 `config` 中编辑以下内容

```
# github: jiap, email: jiap@hotmail.com
Host jiap.github.com
   HostName github.com
   User git
   IdentityFile ~/.ssh/github_jiap
   
# github: physicsay, email: physicsay@outlook.com
Host physicsay.github.com
   HostName github.com
   User git
   IdentityFile ~/.ssh/github_physicsay
```

+ Host: 别名，用于区分多个 git 账号，填写的是 `jiap.github.com`；  
+ HostName: 填写的是 `github.com`；  
+ IdentityFile: ssh 连接使用的私钥；  

规则就是：从上至下读取 config 的内容，在每个 Host 下寻找对应的私钥，可以根据需要添加更多的 github 账号。

## 5. 测试连接性

在终端输入以下语句，测试 ssh 的连接是否成功：

```bash
$ ssh -T git@jiap.github.com
$ ssh -T git@physicsay.github.com
```

会有如下结果：

<img src='/assets/img/2021-02-18-config-sevreal-github-accounts/ssh_t.png' style='zoom:60%; margin: 0 auto; display: block;'>

这里注意，会提示是否继续连接，输入 `yes` 即可，连接成功后会有提示。

此时会在 `~/.ssh` 文件夹下生成一个新的 `known_host` 文件。


## 6. clone 项目

这里分成几种情况，最简单的是将自己 github 的仓库 clone 到本地，进入该文件夹后，并在终端输入 `$ git remote -v` 查看 URL。打开 `.git/config` 文件 (注意 `.git` 是一个隐藏文件夹。)

如果使用了 sublime 编辑器的话，直接在终端输入：

```bash
subl .git/config
```

将 `remote - url` 修改为在 `config` 中设置的即可，如下图所示。

<img src='/assets/img/2021-02-18-config-sevreal-github-accounts/config.png' style='zoom:60%; margin: 0 auto; display: block;'>
/
当然，也可以通过终端直接修改，进入该文件夹下，在终端输入： 
```bash
$ git remote set-url origin git@physicsay.github.com:physicsay/physicsay.github.io.git
```

> 1. 这里只需要将上述语句中的 `physicsay` 全部替换为自己的 github 账户名。
> 2. 修改时，需要确认本地文件夹所属的 github 账号，不要混淆。

---

这个内容很多人都写过，在学习过程中，尤其是 `config` 文件的设置比较难以理解，就有了这篇文章。

---
参考：  
1. [SSH key的介绍与在Git中的使用](https://www.jianshu.com/p/1246cfdbe460)  
2. [Git在同一机器下配置多个github账号](https://wylu.me/posts/e186bfe8/) 
3. [Git多用户，不同项目配置不同Git账号](https://blog.csdn.net/onTheRoadToMine/article/details/79029331) 
4. [同一客户端下使用多个git账号](https://www.jianshu.com/p/89cb26e5c3e8) 
