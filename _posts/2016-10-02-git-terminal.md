---
layout: post
title: "Git命令"
date: 2016-10-02 20:05:00.000000000 +09:00
categories: [Git]
tags: [Git, Pods, 私有库]
---

## Git命令

主要记录工作中使用到的Git命令.

``` bash
git init  // 创建代码仓库
git status // 查看仓库状态
git add .  // 将代码添加到暂缓区
git commit -m '注释' // 添加到本地代码仓库
git remote // 查看本地代码仓库是否关联远程仓库
git remote --help // 帮助
git remote add origin + url  // eg: https://github.com/Jovins/VRPhoto.git
git push origin master // 提交到远程代码仓库

/// 打标签
git tag  // 查看是否已经打标签
git tag -a '1.0.0' -m '注释' // 打一个标签1.0.0 
git push --tags // 提交所有标签
git log // 查看标签日记 
git tag -d 1.0.0  //  删除本地标签
git push origin :1.0.0 // 删除远程标签

// pod 
pod spec create LoaderButton  // 创建LoaderButton.podspec描述文件
pod init // 初始化代码仓库
pod repo // 查看代码仓库url
pod repo --help 
pod lib lint // 本地验证描述文件
pod spec lint // 远程验证
pod repo push master xxx.podspec  // 提交到本地索引库(实质上提交到本地时会做提交到远程索引库)

/// cocoapods 
// 首先需要将本机注册到trunk
pod trunk register <E-mail> '<用户名>' --description='<设备别名>'
// 通过配置好的Podspec，我们可以将库提交到CocoaPods上。
pod trunk push <文件名>.podspec
// 查看是库是否已经在CocoaPods上
pod search <库名>
// 如果查找不到，可以尝试
pod search <库名> --simple
// 删除指定版本
pod trunk delete <库名> <版本号>
```

## Github问题

`Error: remote: Support for password authentication was removed on August 13, 2021.`

2021年8月13号，Github把密码换成token.

### `token`好处

令牌（token）与基于密码的身份验证相比，令牌提供了许多安全优势：

+ 唯一： 令牌特定于 GitHub，可以按使用或按设备生成
+ 可撤销：可以随时单独撤销令牌，而无需更新未受影响的凭据
+ 有限 ： 令牌可以缩小范围以仅允许用例所需的访问
+ 随机：令牌不需要记住或定期输入的更简单密码可能会受到的字典类型或蛮力尝试的影响

### 修改token

`Settings` -> `Developer setting` -> `Personal access tokens` -> `Generate new token` ，设置token的有效期，访问权限等。生成`token`后记得复制出来，不然下次页面刷新会消失。

### 设置token

把`token`直接添加远程仓库链接中，这样就可以避免同一个仓库每次提交代码都要输入`token`.

```
git remote set-url origin https://<your_token>@github.com/<USERNAME>/<REPO>.git
```

+ `<your_token>`：换成你自己得到的`token`
+ `<USERNAME>`：是你自己`github`的`用户名`
+ `<REPO>`：是你的`仓库名称`

```
https://<token>@github.com/Jovins/jovins.github.io.git/
```

## 创建pods库的模板库

``` bash
pod lib create [name] 
What platform do you want to use?? [ iOS / macOS ]
> iOS
what is your name?
> jovins
what is your email?
> jovinscoder@163.com
what language do you want to use?[Swift/Objc]
> Swift
Would you like to include a demo application with your library? [Yes/No]
> Yes
Which testing frameworks will you use?[Specta / Kiwi / None]
> None
Would you like to do view based testing? [Yes / No]
> No
What is your Class prefix?
> WM
```

删除Development Pods -> [name] -> [name] -> Classes中的ReplaceMe.m

然后把框架代码放到Classes文件中，cd 到example目录，执行pod install

## 私有库

``` bash
// 1、创建远程私有库(url)
pod repo add [master] url
// 2、将远程私有库clone到本地

// 本地创建一个pod 私有库的模板库
pod lib create [name]
// 将核心代码拖入Classes文件夹中
pod install
// 编写描述文件

// 上传代码/tag
git add .
git commit -m '注释'
git remote add origin + url 
git push origin master

git tag -a '1.0.0' -m '注释'
git push --tags

// 向私有索引库提交spec描述文件
pod lib lint
pod spec lint
pod repo push [master] xxx.podspec

```

## 常用 Git 命令清单

``` 
Workspace：工作区
Index / Stage：暂存区
Repository：仓库区（或本地仓库）
Remote：远程仓库
```

## 新建代码库

``` bash
# 在当前目录新建一个Git代码库
$ git init

# 新建一个目录，将其初始化为Git代码库
$ git init [project-name]

# 下载一个项目和它的整个代码历史
$ git clone [url]
```

## 配置

Git的设置文件为`.gitconfig`，它可以在用户主目录下（全局配置），也可以在项目目录下（项目配置）。

``` bash
# 显示当前的Git配置
$ git config --list

# 编辑Git配置文件
$ git config -e [--global]

# 设置提交代码时的用户信息
$ git config [--global] user.name "[name]"
$ git config [--global] user.email "[email address]"
```

## 增加/删除文件

``` bash
# 添加指定文件到暂存区
$ git add [file1] [file2] ...

# 添加指定目录到暂存区，包括子目录
$ git add [dir]

# 添加当前目录的所有文件到暂存区
$ git add .

# 添加每个变化前，都会要求确认
# 对于同一个文件的多处变化，可以实现分次提交
$ git add -p

# 删除工作区文件，并且将这次删除放入暂存区
$ git rm [file1] [file2] ...

# 停止追踪指定文件，但该文件会保留在工作区
$ git rm --cached [file]

# 改名文件，并且将这个改名放入暂存区
$ git mv [file-original] [file-renamed]
```

## 代码提交

``` bash
# 提交暂存区到仓库区
$ git commit -m [message]

# 提交暂存区的指定文件到仓库区
$ git commit [file1] [file2] ... -m [message]

# 提交工作区自上次commit之后的变化，直接到仓库区
$ git commit -a

# 提交时显示所有diff信息
$ git commit -v

# 使用一次新的commit，替代上一次提交
# 如果代码没有任何新变化，则用来改写上一次commit的提交信息
$ git commit --amend -m [message]

# 重做上一次commit，并包括指定文件的新变化
$ git commit --amend [file1] [file2] ...
```

## 分支

``` bash
# 列出所有本地分支
$ git branch

# 列出所有远程分支
$ git branch -r

# 列出所有本地分支和远程分支
$ git branch -a

# 新建一个分支，但依然停留在当前分支
$ git branch [branch-name]

# 新建一个分支，并切换到该分支
$ git checkout -b [branch]

# 新建一个分支，指向指定commit
$ git branch [branch] [commit]

# 新建一个分支，与指定的远程分支建立追踪关系
$ git branch --track [branch] [remote-branch]

# 切换到指定分支，并更新工作区
$ git checkout [branch-name]

# 切换到上一个分支
$ git checkout -

# 建立追踪关系，在现有分支与指定的远程分支之间
$ git branch --set-upstream [branch] [remote-branch]

# 合并指定分支到当前分支
$ git merge [branch]

# 选择一个commit，合并进当前分支
$ git cherry-pick [commit]

# 删除分支
$ git branch -d [branch-name]

# 删除远程分支
$ git push origin --delete [branch-name]
$ git branch -dr [remote/branch]
```

## 标签

``` bash
# 列出所有tag
$ git tag

# 新建一个tag在当前commit
$ git tag [tag]

# 新建一个tag在指定commit
$ git tag [tag] [commit]

# 删除本地tag
$ git tag -d [tag]

# 删除远程tag
$ git push origin :refs/tags/[tagName]

# 查看tag信息
$ git show [tag]

# 提交指定tag
$ git push [remote] [tag]

# 提交所有tag
$ git push [remote] --tags

# 新建一个分支，指向某个tag
$ git checkout -b [branch] [tag]
```

## 查看信息

``` bash
# 显示有变更的文件
$ git status

# 显示当前分支的版本历史
$ git log

# 显示commit历史，以及每次commit发生变更的文件
$ git log --stat

# 搜索提交历史，根据关键词
$ git log -S [keyword]

# 显示某个commit之后的所有变动，每个commit占据一行
$ git log [tag] HEAD --pretty=format:%s

# 显示某个commit之后的所有变动，其"提交说明"必须符合搜索条件
$ git log [tag] HEAD --grep feature

# 显示某个文件的版本历史，包括文件改名
$ git log --follow [file]
$ git whatchanged [file]

# 显示指定文件相关的每一次diff
$ git log -p [file]

# 显示过去5次提交
$ git log -5 --pretty --oneline

# 显示所有提交过的用户，按提交次数排序
$ git shortlog -sn

# 显示指定文件是什么人在什么时间修改过
$ git blame [file]

# 显示暂存区和工作区的代码差异
$ git diff

# 显示暂存区和上一个commit的差异
$ git diff --cached [file]

# 显示工作区与当前分支最新commit之间的差异
$ git diff HEAD

# 显示两次提交之间的差异
$ git diff [first-branch]...[second-branch]

# 显示今天你写了多少行代码
$ git diff --shortstat "@{0 day ago}"

# 显示某次提交的元数据和内容变化
$ git show [commit]

# 显示某次提交发生变化的文件
$ git show --name-only [commit]

# 显示某次提交时，某个文件的内容
$ git show [commit]:[filename]

# 显示当前分支的最近几次提交
$ git reflog

# 从本地master拉取代码更新当前分支：branch 一般为master
$ git rebase [branch]
```

## 远程同步

``` bash
# 下载远程仓库的所有变动
$ git fetch [remote]

# 显示所有远程仓库
$ git remote -v

# 显示某个远程仓库的信息
$ git remote show [remote]

# 增加一个新的远程仓库，并命名
$ git remote add [shortname] [url]

# 取回远程仓库的变化，并与本地分支合并
$ git pull [remote] [branch]

# 上传本地指定分支到远程仓库
$ git push [remote] [branch]

# 强行推送当前分支到远程仓库，即使有冲突
$ git push [remote] --force

# 推送所有分支到远程仓库
$ git push [remote] --all
```

## 撤销

``` bash
# 恢复暂存区的指定文件到工作区
$ git checkout [file]

# 恢复某个commit的指定文件到暂存区和工作区
$ git checkout [commit] [file]

# 恢复暂存区的所有文件到工作区
$ git checkout .

# 重置暂存区的指定文件，与上一次commit保持一致，但工作区不变
$ git reset [file]

# 重置暂存区与工作区，与上一次commit保持一致
$ git reset --hard

# 重置当前分支的指针为指定commit，同时重置暂存区，但工作区不变
$ git reset [commit]

# 重置当前分支的HEAD为指定commit，同时重置暂存区和工作区，与指定commit一致
$ git reset --hard [commit]

# 重置当前HEAD为指定commit，但保持暂存区和工作区不变
$ git reset --keep [commit]

# 新建一个commit，用来撤销指定commit
# 后者的所有变化都将被前者抵消，并且应用到当前分支
$ git revert [commit]

# 暂时将未提交的变化移除，稍后再移入
$ git stash
$ git stash pop
```

## 回滚

``` bash
$ git reset --hard HEAD^         回退到上个版本
$ git reset --hard HEAD~3        回退到前3次提交之前，以此类推，回退到n次提交之前
$ git reset --hard commit_id     退到/进到 指定commit的sha码
```

## 合并

``` bash
对于复杂的系统，我们可能要开好几个分支来开发，那么怎样使用git合并分支呢？

合并步骤：
1、进入要合并的分支（如开发分支合并到master，则进入master目录）
git checkout master
git pull

2、查看所有分支是否都pull下来了
git branch -a

3、使用merge合并开发分支
git merge 分支名

4、查看合并之后的状态
git status

5、有冲突的话，通过IDE解决冲突；

6、解决冲突之后，将冲突文件提交暂存区
git add 冲突文件

7、提交merge之后的结果
git commit

如果不是使用git commit -m "备注" ，那么git会自动将合并的结果作为备注，提交本地仓库；

8、本地仓库代码提交远程仓库
git push

git将分支合并到分支，将master合并到分支的操作步骤是一样的。
```

## git push origin master

遇到的错误

```
! [rejected] master -> master (fetch first)
```

原因:

问题是本地还没有更新远程仓库，比如README.md文件没同步

解决:

```bash
$ git pull --rebase origin master
```

执行这句后本地仓库就跟远程仓库同步了

```bash
$ git push origin master
```

忽略冲突，强制推送

```bash
$ git push origin master -f // 大多数情况不要这么做
```