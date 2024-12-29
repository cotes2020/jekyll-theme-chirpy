---
layout: post
title: "My Spacemacs Workflow"
subtitle: 'From Vim to Spacemacs'
author: "Hux"
header-style: text
published: false
tags:
  - Vim
  - Emacs
---

Emacs tend to provide a good support for functional programming languages. Indeed, many FP language community exclusively use Emacs and give only first-party IDE supports to Emacs, such as Coq, Agda, Standard ML, Clojure, etc.

For the purpose of programming Coq with Proof General, I started to try with Emacs. I quickly found Spacemacs a good alternatives for me...someone had get used to Vim keybindings and want to get some thing useful ASAP w/o configuring a long list as my `.vimrc`.

Though the overall experience is pretty smooth, many quirks about Spacemacs are always being forgotten and had to look up again and again, so I decided to open a note for some specific "workflow" that I often used.

Yes this is more like a note publishing online for the purpose of "on-demand accessible". So don't expect good writing anyways.


### Vim-binding

Choose `evil`!


### Airline

It's there!


### Nerd Tree / File Sidebar

`SPC f t` for _file tree_. The keybindings for specific operations are very different w/ Vim NerdTree though.


### Shell / Terminal

I occasionally use [Neovim's terminal emulator](https://neovim.io/doc/user/nvim_terminal_emulator.html) but in most of the time I just `cmd + D` for iTerms splitted window. 

I even mappped `:D` into split-then-terminal to make the experience on par ;)

```vim
command! -nargs=* D  belowright split | terminal <args>
```

Anyways, Spacemacs does provide a `:shell` that naturally split a window below for terminal. The experience is not very good though.


### Tabs / Workspaces

I tend to open multiple _workspace_. Though people might found Vim tabs useful, I am exclusively use iTerm tabs for similar jobs. However Spacemacs is not living in a terminal.

[r/spacemacs - Vim-style tabs?](https://www.reddit.com/r/spacemacs/comments/5w5d2s/vimstyle_tabs/) gave me a good way to approximate the experience by using [Spacemacs Workspaces](http://spacemacs.org/doc/DOCUMENTATION.html#workspaces): `SPC l w <nth>` trigger a so-called "layout transient state" (I have no idea what's that mean) to open N-th workspaces, and use `gt`/`gT` to switch between.


### Fuzz File Name Search / Rg

`SPC f f`


### Buffers

`SPC b b`
