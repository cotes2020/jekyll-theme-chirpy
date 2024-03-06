---
title: Linux - Executing programs via command line
date: 2020-07-16 11:11:11 -0400
categories: [30System, Sysadmin]
tags: [Linux, Sysadmin]
math: true
image:
---

# Executing programs via command line

three basic ways to run a 'command' in the Command Prompt.

1. builtins ("internal commands")
    - These are commands built into cmd itself, and do not require an external program invocation.
    - They also do not perform any searching, and will always be executed with the highest priority if matched.
    - bypass builtins by wrapping the executable name in quotes: `echo` calls the builtin, but `"echo"` would search following cmd rules.


2. Direct invocation
    - This is when you directly specify a program name (without a path). For example, if you run cmd (cmd.exe) or ipconfig (ipconfig.exe) at the prompt, you are directly calling the external command. This performs limited searching implemented entirely within the Command Prompt, in this order:
      - The current directory.
      - The directories that are listed in the PATH environment variable.


3. Through the `start` command
    - execute a file through the start command, Command Prompt does not perform any searching.
    Instead, it passes the file name (and arguments) over to Windows itself (via the ShellExecuteEx API call), which must then search for the file's location. There are several places it searches in the following order:
      - Current working directory
      - Windows directory
      - Windows\System32 directory
      - Directories listed in PATH environment variable
      - Registry defined App Paths
      - Note that the Run dialog also uses this search method.

    - start it with start any_program.exe, you have a couple of options:
      - put it in the Windows or System32 directories, or any directory in the `%PATH%` environment variable.
      - add the directory it is located in (D:\Any_Folder) to the `%PATH%` environment variable,


      - add it to the App Paths registry key, as Notepad and Firefox does. App Paths links a file keyword (such as firefox.exe) with the full path to the file, unlike the other options that deal with directories. See here for more information.
      - registered in registry in the key `HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths` or its `HKEY_CURRENT_USER` analogue.


navigate to the location of the file with `cd /d D:\Any_Folder` (/d means change drive) and just run any_program.exe. `start D:\Any_Folder\any_program.exe`
- when path or file contains spaces
  - `start "" "D:\Any_Folder\any_program.exe"`

`D:\Any_Folder\any_program.exe`.
- no action need

`any_program.exe`
- Adding any_program.exe to path:
  - "Control Panel" ->"Advanced System Settings"
  - Advanced tab
  - "Environment Variables" Add the folder in which any_program.exe resides.
  - Edit the PATH Variable and add the folder in the end, separated by a ;

`START programmname`
- Making Executable File Location Available In CMD i.e Creating a PATH Variable:
  - SET PATH : In CMD Type
  - `SET ACROBAT="C:\Program Files (x86)\Adobe\Acrobat 11.0\Acrobat"`
  - Executing the file From CMD:
  - `START ACROBAT`






















.
