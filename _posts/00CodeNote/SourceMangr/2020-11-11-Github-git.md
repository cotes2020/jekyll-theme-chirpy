---
title: GitHub - Git
date: 2020-11-11 11:11:11 -0400
categories: [00CodeNote, SourceMagr]
tags: [git]
toc: true
image:
---

- [`git add`](#git-add)
- [`git commit`](#git-commit)
- [`git show` 显示各种类型的对象。](#git-show-显示各种类型的对象)

---



---

### `git add`

git add [--verbose | -v] [--dry-run | -n] [--force | -f] [--interactive | -i] [--patch | -p]
      [--edit | -e] [--[no-]all | --[no-]ignore-removal | [--update | -u]]
      [--intent-to-add | -N] [--refresh] [--ignore-errors] [--ignore-missing]
      [--chmod=(+|-)x] [--] [<pathspec>…​]

将文件内容添加到索引(将修改添加到暂存区)。将要提交的文件的信息添加到索引库中。
- 将要提交的文件的信息添加到索引库中(将修改添加到暂存区)，以准备为下一次提交分段的内容。
- 它通常将现有路径的当前内容作为一个整体添加，但是通过一些选项，它也可以用于添加内容，只对所应用的工作树文件进行一些更改，或删除工作树中不存在的路径了。
- “索引”保存工作树内容的快照，并且将该快照作为下一个提交的内容。 因此，在任何更改之后，并且在运行`git commit`命令之前，必须使用`git add`命令将任何新的或修改的文件添加到索引。
- 该命令可以在提交之前多次执行。它只在运行git add命令时添加指定文件的内容;
- 如果希望随后的更改包含在下一个提交中，那么必须再次运行git add将新的内容添加到索引。

`git status` 可用于获取哪些文件具有为下一次提交分段的更改的摘要。

    $ git status
    On branch master
    Your branch is up-to-date with 'origin/master'.
    Changes to be committed:
      (use "git reset HEAD <file>..." to unstage)
    	new file:   key.txt

默认情况下，`git add`命令不会添加忽略的文件。
- 如果在命令行上显式指定了任何忽略的文件，git add命令都将失败，并显示一个忽略的文件列表。
- 由Git执行的目录递归或文件名遍历所导致的忽略文件将被默认忽略。
- `git add-f(force)`选项添加被忽略的文件。

```bash
示例
添加documentation目录及其子目录下所有*.txt文件的内容：
$ git add documentation/*.txt
# 星号*是从shell引用的;
# 允许命令包含来自 Documentation/目录和子目录的文件。

将所有 git-*.sh 脚本内容添加：
$ git add git-*.sh
# 这个例子让shell扩展星号(即明确列出文件)，所以它不考虑子目录中的文件，
# subdir/git-foo.sh 这样的文件不会被添加。
```

基本用法 `git add <path>`
- 把<path>添加到索引库中，<path>可以是文件也可以是目录。
- git不仅能判断出<path>中，修改(不包括已删除)的文件，还能判断出新添的文件，并把它们的信息添加到索引库中。

```bash
$ git add .             # 将所有修改添加到暂存区
$ git add *             # Ant风格添加修改
$ git add *Controller   # 将以Controller结尾的文件的所有修改添加到暂存区

$ git add Hello*        # 将所有以Hello开头的文件的修改添加到暂存区
                        # HelloWorld.txt,Hello.java,HelloGit.txt ...

$ git add Hello?        # 将以Hello开头后面只有一位的文件的修改提交到暂存区
                        # 如: Hello1.txt, HelloA.java
                        # HelloGit.txt, Hello.java是不会被添加的
```

- `git add -u [<path>]`: 把<path>中所有`跟踪文件中被修改过或已删除文件的信息`添加到索引库。
  - 它不会处理那些不被跟踪的文件。
  - 省略<path>表示 . ,即当前目录。
- `git add -A`: 把中所有`跟踪文件中被修改过或已删除文件`和`所有未跟踪的文件信息`添加到索引库。
  - 省略<path>表示 . ,即当前目录。
- `git add -i`
  - 通过git add -i [<path>]: 查看中被所有修改过或已删除文件但没有提交的文件，并通过其revert子命令可以查看<path>中所有未跟踪的文件，同时进入一个子命令系统。

        $ git add -i
                   staged     unstaged path
          1:        +0/-0      nothing branch/t.txt
          2:        +0/-0      nothing branch/t2.txt
          3:    unchanged        +1/-0 readme.txt

        *** Commands ***
          1: [s]tatus     2: [u]pdate     3: [r]evert     4: [a]dd untracked
          5: [p]atch      6: [d]iff       7: [q]uit       8: [h]elp

        这里的t.txt和t2.txt表示已经被执行了git add，待提交。即已经添加到索引库中。
        readme.txt表示已经处于tracked下，它被修改了，但是还没有执行git add。即还没添加到索引库中。


git rm 文件名(包括路径) 从git中删除指定文件

git clone git://github.com/schacon/grit.git 从服务器上将代码给拉下来

git config --list 看所有用户

git ls-files 看已经被提交的

git rm [file name] 删除一个文件


### `git commit`

`git commit -a` 提交当前repos的所有的改变

git commit [-a | --interactive | --patch] [-s] [-v] [-u<mode>] [--amend]
       [--dry-run] [(-c | -C | --fixup | --squash) <commit>]
       [-F <file> | -m <msg>] [--reset-author] [--allow-empty]
       [--allow-empty-message] [--no-verify] [-e] [--author=<author>]
       [--date=<date>] [--cleanup=<mode>] [--[no-]status]
       [-i | -o] [-S[<keyid>]] [--] [<file>…​]

用于将更改记录(提交)到存储库。
- 将索引的当前内容与描述更改的用户和日志消息一起存储在新的提交中。

要添加的内容可以通过以下几种方式指定：
- 用`git commit`之前，用`git add`对索引进行递增的“添加”更改(注意：修改后的文件的状态必须为“added”);
- 用`git rm`从工作树和索引中删除文件，再次用`git commit`命令;
- 通过将文件作为参数列出到`git commit`命令(不使用--interactive或--patch选项)，在这种情况下，提交将忽略索引中分段的更改，而是记录列出的文件的当前内容(必须已知到Git的内容) ;
- 通过使用带有`-a`选项的`git commit`命令来自动从所有已知文件(即所有已经在索引中列出的文件)中添加“更改”，并自动从已从工作树中删除索引中的“rm”文件 ，然后执行实际提交;
- 通过使用`--interactive`或`--patch`选项与`git commit`命令一起确定除了索引中的内容之外哪些文件或hunks应该是提交的一部分，然后才能完成操作。

`--dry-run`选项可用于通过提供相同的参数集(选项和路径)来获取上一个任何内容包含的下一个提交的摘要。
如果您提交，然后立即发现错误，可以使用 `git reset` 命令恢复。

示例
提交已经被git add进来的改动。

    $ git add .
    $ # 或者 ~
    $ git add newfile.txt
    $ git commit -m "the commit message" #
    $ git commit -a
    # 会先把所有已经track的文件的改动`git add`进来，然后提交(有点像svn的一次提交,不用先暂存)。
    # 对于没有track的文件,还是需要执行`git add <file>` 命令。
    $ git commit --amend
    # 增补提交，会使用与当前提交节点相同的父节点进行一次新的提交，旧的提交将会被取消。

录制自己的工作时，工作树中修改后的文件的内容将临时存储到使用git add命名为“索引”的暂存区域。 一个文件只能在索引中恢复，而不是在工作树中，使用git reset HEAD - <file>进行上一次提交的文件，这有效地恢复了git的添加，并阻止了对该文件的更改，以参与下一个提交在使用这些命令构建状态之后，git commit(没有任何pathname参数)用于记录到目前为止已经进行了什么更改。 这是命令的最基本形式。
例子：
$ vi hello.c
$ git rm goodbye.c
$ git add hello.c
$ git commit


可以在每次更改后暂存文件，而不是在git commit中关注工作树中跟踪内容的文件的更改，可使用相应的git add和git rm。 也就是说，如果工作树中没有其他更改(hello.c文件内容不变)，则该示例与前面的示例相同：
$ vi hello.c
$ rm goodbye.c
$ git commit -a

`git commit -a`首先查看您的工作树，注意您已修改hello.c并删除了goodbye.c，并执行必要的`git add`和`git rm`


在更改许多文件之后，可以通过给出`git commit`的路径名来更改记录更改的顺序。
当给定路径名时，该命令提交只记录对命名路径所做的更改：


$ edit hello.c hello.h # 修改了这两个文件的内容
$ git add hello.c hello.h
$ edit Makefile
$ git commit Makefile

这提供了一个记录Makefile修改的提交。
在hello.c和hello.h中升级的更改不会包含在生成的提交中。
然而，它们的变化并没有消失 - 他们仍然有更改，只是被阻止。
按照上述顺序执行：
$ git commit
这个第二个提交将按照预期记录更改为hello.c和hello.h。
合并后(由`git merge`或`git pull`发起)由于冲突而停止，干净合并的路径已经被暂存为提交，并且冲突的路径保持在未加载状态。必须先检查哪些路径与git状态冲突，并在手工将其固定在工作树中之后，要像往常一样使用`git add`：
$ git status | grep unmerged
unmerged: hello.c
$ edit hello.c
$ git add hello.c
解决冲突和暂存结果后，git ls-files -u将停止提及冲突的路径。完成后，运行git commit最后记录合并：
$ git commit


---

git add [file name] 添加一个文件到git index

git commit -v 当你用－v参数的时候可以看commit的差异

git commit -m "This is the message describing the commit" 添加commit信息

git commit -a -a是代表add，把所有的change加到git index里然后再commit

git commit -a -v 一般提交命令

`git log` 看你commit的日志

`git diff` 查看尚未暂存的更新

git rm a.a 移除文件(从暂存区和工作区中删除)

git rm --cached a.a 移除文件(只从暂存区中删除)

`git commit -m` "remove" 移除文件(从Git中删除)

git rm -f a.a 强行移除修改后文件(从暂存区和工作区中删除)

git diff --cached 或 $ git diff --staged 查看尚未提交的更新

git stash push 将文件给push到一个临时空间中

git stash pop 将文件从临时空间pop下来

－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－

git remote add origin git@github.com:username/Hello-World.git

git push origin master 将本地项目给提交到服务器中

－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－

git pull 本地与服务器端同步

－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－

git push (远程仓库名) (分支名) 将本地分支推送到服务器上去。

git push origin server fix:awesome branch

－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－

git fetch 相当于是从远程获取最新版本到本地，不会自动merge

git commit -a -m "log_message" (-a是提交所有改动，-m是加入log信息) 本地修改同步至服务器端 ：

git branch branch_0.1 master 从主分支master创建branch_0.1分支

git branch -m branch_0.1 branch_1.0 将branch_0.1重命名为branch_1.0

git checkout branch_1.0/master 切换到branch_1.0/master分支

du -hs

git branch 删除远程branch

git push origin:branch_remote_name

git branch -r -d branch_remote_name

－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－

初始化版本库，并提交到远程服务器端

mkdir WebApp

cd WebApp

git init本地初始化

touch README

git add README添加文件

git commit -m 'first commit'

git remote add origin git@github.com:daixu/WebApp.git增加一个远程服务器端

上面的命令会增加URL地址为'git@github.com:daixu/WebApp.git'，名称为origin的远程服务器库，以后提交代码的时候只需要使用 origin别名即可


### `git show` 显示各种类型的对象。
`git show [options] <object>…​`

- 描述显示一个或多个对象(blobs，树，标签和提交)。
- 对于提交，它显示日志消息和文本差异。 还以 `git diff-tree --cc` 生成的特殊格式呈现合并提交。
- 对于标签，它显示标签消息和引用对象。
- 对于树，它显示的名称(相当于使用git ls-tree和--name-only选项)。
- 对于简单的blobs，它显示了普通的内容。
- 该命令采用适用于 `git diff-tree` 命令的选项来控制如何显示提交引入的更改。

```bash
# 1. 显示标签v1，以及标签指向的对象
$ git show v1

# 2. 显示标签v1指向的树
$ git show v1^{tree}

# 3. 显示标签v1指向的提交的主题
$ git show -s --format=%s v1^{commit}

# 4. 显示 Documentation/README 文件的内容，它们是 next 分支的第10次最后一次提交的内容
$ git show next~10:Documentation/README

# 5. 将Makefile的内容连接到分支主控的头部
$ git show master:Makefile master:t/Makefile
```

git show-ref
- 可以现实本地存储库的所有可用的引用以及关联的提交ID

      $ git show-ref
      3aa4c239f729b07deb99a52f125893e162daac9e refs/heads/master
      3aa4c239f729b07deb99a52f125893e162daac9e refs/remotes/origin/HEAD
      3aa4c239f729b07deb99a52f125893e162daac9e refs/remotes/origin/master
      f17132340e8ee6c159e0a4a6bc6f80e1da3b1aea refs/tags/secret



查看文本内容（blob对象)
- 知道一个文本对象的sha-1值，那么查看方式如下：
$ git show 215ded5
蚂蚁部落

查看tree对象：
- 显示当前tree对象的目录结构，代码如下：
$ git show fac4ee5^{tree}
![015709g5zlfblo4r5ukug4](https://i.imgur.com/FhXEdD9.jpg)


查看tag标签：
- 看一下当前项目的提交历史，
$ git log --oneline
![015752py6e09yp66j5f9lp](https://i.imgur.com/DfZqlnr.jpg)

- 查看tagLearn标签
$ git show tagLearn
上面是一个轻量级标签，输出信息展示了它所指向的commit提交和所指向提交与上一次提交之间的差异。
![015831oufzo699dyudiiu9](https://i.imgur.com/WIcXoYc.jpg)

- 查看一下有附注标签信息
$ git show annotatedTag
除了显示轻量级标签相同的信息外，还显示有附注标签对象的一些信息，打标签这，打标签的时间等。
![015921x31au3liia0iz636](https://i.imgur.com/FfjSJc2.jpg)


查看commit对象：
- 显示commit对象的相关信息（提交者，提交时间和commit对象sha-1值等）和上一个提交对象的差异。
$ git show 5a97a20
![020011v16jwnbtbrpjvwwp](https://i.imgur.com/x0y5bwJ.jpg)


















.
