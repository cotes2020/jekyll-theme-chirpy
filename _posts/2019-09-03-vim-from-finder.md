---
layout: post
title: "把「终端下的 Vim」作为 macOS Finder 的打开方式"
subtitle: 'Open file with terminal Vim from the macOS Finder'
author: "Hux"
header-style: text
tags:
  - Vim
---

我的日常主力编辑器主要是：

- (Neo)Vim
- Spacemacs (via Emacs-plus)
- Visual Studio Code
- IntelliJ IDEA

这里面只有 (Neo)Vim 是存活在终端中的（我并不在终端内使用 Emacs），而由于我日常主要是从终端（via iTerm）来使用电脑，所以会把他们都加入到 `$PATH` 里以方便从终端中唤起，VSCode 和 IDEA 都有一键加入的功能， Emacs 我在 `~/.zshrc` 中放了一个 `alias emacs='open -n -a Emacs.app .'` 解决。

但是，偶尔也会有从 Finder 中打开文件的需求，这时候如果通常会打开拓展名所绑定的 `Open with...` 应用，在大部分时候我的默认绑定是 VSCode，但是今天心血来潮觉得有没有办法直接打开 Vim 呢？搜了一下还真有基于 AppleScript 的解决方案：

1. 打开 `Automator.app`
2. 选择 `New Document`
3. 找到 `Run AppleScript` 的 action 双击添加
4. 编写 AppleScript 脚本来唤起终端与 vim （下文给出了我的脚本你可以直接稍作修改使用）
5. 保存为 `Applications/iTermVim.app` （你可以自己随便取）
6. 找到你想要以这种方式打开的文件，比如 `<随便>.markdown`，`⌘ i` 获取信息然后修改 `Open with` 为这个应用然后 `Change All...`

效果超爽 ;)

```applescript
on run {input, parameters}
  set filename to POSIX path of input
  set cmd to "clear; cd `dirname " & filename & "`;vim " & quote & filename & quote
  tell application "iTerm"
    activate
    tell the current window
      create tab with default profile
      tell the current session
        write text cmd
      end tell
    end tell
  end tell
end run
```

我这里的代码是采取是用 `iTerm` 与唤起 `vim`、窗口置前、在新窗口中打开、同时 `cd` 到目录。你也可以改成用 macOS 自带的 `Terminal.app`、在新窗口而非新 tab 打开、应用不同的 profile、或是执行其他 executable 等……任你发挥啦。

### References

- [Open file in iTerm vim for MacOS Sierra](https://gist.github.com/charlietran/43639b0f4e0a01c7c20df8f1929b76f2)
- [Open file in Terminal Vim on OSX](https://bl.ocks.org/napcs/2d8376e941133ccfad63e33bf1b1b60c)
