---
layout: post
title: "Vim 与中文输入法"
subtitle: 'Using Vim with non-english input method'
author: "Hux"
header-style: text
tags:
  - Vim
---

Update: 我最后还是放弃把 Vim 作为主要编辑器来输入中文了，整体使用下来 mental model 的 cost 太重了。记笔记时用用中文呀或者改改博客时偶尔用一下还蛮去，这个时候这个功能至少能帮助你 Esc 之后不煞笔，所以也不算完全没有价值吧……

---

我相信很多中文世界的 Vimer 都遇到过这个烦恼，在 vim 的 insert 模式时可能突然想输个中文，输完之后会本能的直接 `esc` 接 normal 模式操作，结果发现跳出来的是中文输入法……对于 vscode，我一般会在几次错误之后被逼到退出 vscode vim 模式，而对于终端中用的 neovim，就只能尽量不输入中文了。

为了满足我 1% 用 vim 输入中文的场景（比如写博客），我还是想看看有没有什么解决方案，Google 出来的解决方案基本是：*在退出 insert 模式时记住当时的输入法，并自动切换到默认输入法（一般是英文）给 normal 模式用，并且在下一次进入 insert 模式时再切换回来。*

原生 vim 的话，可以使用 [smartim](https://github.com/ybian/smartim) 插件，原理是调用 [im-select](https://github.com/daipeihust/im-select) 这个 CLI 工具来切换输入法。

对于 VSCode-vim 的话，smartim 的移植也在近期的 PR 中被 merge 到了插件里，[详情见文档的这部分配置]( https://github.com/VSCodeVim/Vim#use-im-select)，需要指定一下默认输入法和 im-select 的 binary 路径就好。

---

不过实话说，在 vim 中编辑中文的效率和体验和英文比都是大打折扣的。因为中文分词难度太高，不像英文可以简单依靠一个 `split " "` 搞定。所以其实无论 vim（`w`ord，`b`egin，`e`nd），emacs 还是操作系统自带的（比如 macOS 中的 `alt + 箭头`） 「按词移动」功能对于中文都仅仅是跳转到下一个空格处而已，对于中文来说基本就是下一句了……其他常用操作诸如 `f`，`/`, `r`eplace, `t`ill 也都无法很好的工作，基本只能靠 `hjkl` 爬行……

不过也算聊胜于无吧，由于我的主力外置键盘是 HHKB，能用 vim 操作的一个子集（`hjkl`, `o`, `A`, `I`, `v` etc.）可能也比按住 `Fn` 的方向键好用……
