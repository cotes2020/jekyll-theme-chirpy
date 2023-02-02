---
layout: post
title: Improve Your VSCode Productivity
date: 2020-09-06 11:00:00 00:00
comments: true
author: ebmarquez
categories:
  - productivity
tags: 
  - development
  - ide
  - productivity
---
If your a users of Microsoft's Visual Studio Code (VSCode) then you may have noticed that there are a lot of useful extensions in it's marketplace.  As of September 2020, it had approximately 20,690 extensions. Thats a lot of extension to sift through. In this article I wanted to share 4 different extensions that will help improve your productivity with VSCode. These are pretty universal and will make your code a bit more readable.  If you work with JSON, YAML, Python, PowerShell, C#, C++ or other.  These extension will help your everyday experience better.

- [Indent Rainbow][Indent-Rainbow] is an extension that will color code tabs and spaces in the code your are viewing. For me, I find this extension particular useful with working with JSON and YAML files.  The files I'm working with are large and complex and this visual aid can be useful to determine if edits are landing in the correct location.  The extension can be configured for specific type of files, but I use it for all files.

- [Bracket Pair Colorizer][color-bracket] is similar to Indent Rainbow where color is used.  It uses color to expose matching brackets in your code.  Unlike Indent Rainbow where it will highlight the tabs and spaces in a file.  This will add colors to the start and end of a set of braces. This can be handy when reviewing your code and identify if you have placed your braces in the correct location.  This visual aid should help save you a few seconds when reviewing code.  A second here and a second there save a lot of time in the long run.

- [Better Comments][better-comment] is a comment enhancer where it exposes useful notes to the owner or teams maintaining the program. If you work independent or within a team this will help make your comment "better". It has a syntax parser that looks for key characters or words in the code and will highlights them with specific colors.  This will override the existing color scheme that is used for your comments and highlight these items with a different color.  These items may not rise to the level where you need a bug or task for them and you want additional notes in your code. This is worth checking out.

- [Todo Tree][todo-tree] works with ToDo comments in the source code.  If your using something like Better Comments to expose ToDo items in the comments of your code. This will identify all of the ToDo items and place them in a central location within VsCode. The extension will make it much easier to find all those ToDo's your don't want to track in a bug or task.

Like almost all extension in VSCode these are all freeware and should work with any language your are using. These are my favorite extension that I utilize them on a daily bases.  I hope they make your coding experience better too.

[indent-rainbow]: https://marketplace.visualstudio.com/items?itemName=oderwat.indent-rainbow
[color-bracket]: https://marketplace.visualstudio.com/items?itemName=CoenraadS.bracket-pair-colorizer
[better-comment]: https://marketplace.visualstudio.com/items?itemName=aaron-bond.better-comments
[todo-tree]: https://marketplace.visualstudio.com/items?itemName=Gruntfuggly.todo-tree
