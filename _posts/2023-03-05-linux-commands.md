---
title: Linux Commands
author: dyl4n
date: 2023-03-05 13:35:23 +0700
categories: [Operating System]
tags: [linux commands]
render_with_liquid: false
comments: true
image:
  path: /thumbnails/Linux-Commands.png
published: false
---

Linux is a powerful operating system that offers a wide range of features and tools to help users manage their systems effectively. One of the most important aspects of Linux is the command-line interface (CLI), which allows users to interact with the system using text-based commands.

## What is a Linux Command?

A Linux command is a text-based instruction that you can use to interact with your Linux system. Commands can be entered in the terminal, which is a text-based interface for interacting with the operating system. The Linux command consists of three main parts: the `command name`, `options`, and `arguments`.

## Linux Command's Components

### Command Name

The command name is the first part of the Linux command and tells the system which action to perform. For example, the "ls" command is used to list the files and directories in the current directory.

### Options

- Options are additional settings that can modify the behavior of the command.
- Options are usually represented by one or more characters preceded by a hyphen (-).
- For example, the "ls" command can take several options, including "-l" to display the files in a long format and "-a" to display hidden files. To use multiple options, you can combine them, like "ls -la" to display all files, including hidden ones, in a long format.

### Arguments

- Arguments are additional information that is required by the command to perform its action.
- Arguments can be filenames, directory names, or other information required by the command.
- For example, the "mkdir" command requires an argument that specifies the name of the directory you want to create, like "mkdir mydirectory".

## Syntax of Linux Command

The syntax of a Linux command is the set of rules that govern how you should enter the command in the terminal. The basic syntax of a Linux command is as follows:

```bash
command [options] [arguments]
```

Here, the command is the name of the Linux command you want to execute. Options are optional and modify the behavior of the command. Arguments are also optional and provide additional information required by the command.

Let's take a look at an example to see how this works in practice. To list the files and directories in the current directory, you can use the "ls" command with the "-l" option, like this:

```bash
ls -l
```

This command tells the system to list the files in the current directory in a long format. You can also specify the directory you want to list as an argument, like this:

```bash
ls -l /home/user/datpc/Documents
```

Some rules for typing command lines:

![rules](https://user-images.githubusercontent.com/98354414/222964711-e5dda57a-a323-46e9-b04a-8ea79e55f753.png)

## Some common Linux Command Lines:

| **Commands** |                             **Usage**                             |        **Example**        |
| :----------: | :---------------------------------------------------------------: | :-----------------------: |
|   **pwd**    |                    Print the working directory                    |            pwd            |
|    **ls**    |                 List the contents of a directory                  |          ls -la           |
|   **cat**    |                      Print the file contents                      |       cat file.txt        |
|    **mv**    |               Move and rename files and directories               |    mv cat.txt dog.txt     |
|  **touch**   |                        Create empty files                         |      touch hello.txt      |
|  **mkdir**   |                      Create new directories                       |       mkdir thisDir       |
|    **rm**    |           Remove files and directories (with option –r)           | rm cat.txt, rm demoDir -r |
|   **man**    | Provide reference information on commands, subroutines, and files |         man find          |
|   **find**   |     Search for files and directories in a directory hierarch      |   find . –name ‘\*.txt’   |

## Conclusion

Linux commands are a powerful tool for managing your Linux system. By understanding the syntax of Linux commands, you can interact with your system more effectively and efficiently. Remember, the syntax of a Linux command consists of the command name, options, and arguments. With this knowledge, you can start exploring the wide range of commands available in Linux and take control of your system.

---

## Command Cheatsheet

| **Command**              | **Description**                                                                                                                                            |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  `man <tool>`            | Opens man pages for the specified tool.                                                                                                                    |
| `<tool> -h`              | Prints the help page of the tool.                                                                                                                          |
| `apropos <keyword>`      | Searches through man pages' descriptions for instances of a given keyword.                                                                                 |
| `cat`                    | Concatenate and print files.                                                                                                                               |
| `whoami`                 | Displays current username.                                                                                                                                 |
| `id`                     | Returns users identity.                                                                                                                                    |
| `hostname`               | Sets or prints the name of the current host system.                                                                                                        |
| `uname`                  | Prints operating system name.                                                                                                                              |
| `pwd`                    | Returns working directory name.                                                                                                                            |
| `ifconfig`               | The `ifconfig` utility is used to assign or view an address to a network interface and/or configure network interface parameters.                          |
| `ip`                     | Ip is a utility to show or manipulate routing, network devices, interfaces, and tunnels.                                                                   |
| `netstat`                | Shows network status.                                                                                                                                      |
| `ss`                     | Another utility to investigate sockets.                                                                                                                    |
| `ps`                     | Shows process status.                                                                                                                                      |
| `who`                    | Displays who is logged in.                                                                                                                                 |
| `env`                    | Prints environment or sets and executes a command.                                                                                                         |
| `lsblk`                  | Lists block devices.                                                                                                                                       |
| `lsusb`                  | Lists USB devices.                                                                                                                                         |
| `lsof`                   | Lists opened files.                                                                                                                                        |
| `lspci`                  | Lists PCI devices.                                                                                                                                         |
| `sudo`                   | Execute command as a different user.                                                                                                                       |
| `su`                     | The `su` utility requests appropriate user credentials via PAM and switches to that user ID (the default user is the superuser). A shell is then executed. |
| `useradd`                | Creates a new user or update default new user information.                                                                                                 |
| `userdel`                | Deletes a user account and related files.                                                                                                                  |
| `usermod`                | Modifies a user account.                                                                                                                                   |
| `addgroup`               | Adds a group to the system.                                                                                                                                |
| `delgroup`               | Removes a group from the system.                                                                                                                           |
| `passwd`                 | Changes user password.                                                                                                                                     |
| `dpkg`                   | Install, remove and configure Debian-based packages.                                                                                                       |
| `apt`                    | High-level package management command-line utility.                                                                                                        |
| `aptitude`               | Alternative to `apt`.                                                                                                                                      |
| `snap`                   | Install, remove and configure snap packages.                                                                                                               |
| `gem`                    | Standard package manager for Ruby.                                                                                                                         |
| `pip`                    | Standard package manager for Python.                                                                                                                       |
| `git`                    | Revision control system command-line utility.                                                                                                              |
| `systemctl`              | Command-line based service and systemd control manager.                                                                                                    |
| `ps`                     | Prints a snapshot of the current processes.                                                                                                                |
| `journalctl`             | Query the systemd journal.                                                                                                                                 |
| `kill`                   | Sends a signal to a process.                                                                                                                               |
| `bg`                     | Puts a process into background.                                                                                                                            |
| `jobs`                   | Lists all processes that are running in the background.                                                                                                    |
| `fg`                     | Puts a process into the foreground.                                                                                                                        |
| `curl`                   | Command-line utility to transfer data from or to a server.                                                                                                 |
| `wget`                   | An alternative to `curl` that downloads files from FTP or HTTP(s) server.                                                                                  |
| `python3 -m http.server` | Starts a Python3 web server on TCP port 8000.                                                                                                              |
| `ls`                     | Lists directory contents.                                                                                                                                  |
| `cd`                     | Changes the directory.                                                                                                                                     |
| `clear`                  | Clears the terminal.                                                                                                                                       |
| `touch`                  | Creates an empty file.                                                                                                                                     |
| `mkdir`                  | Creates a directory.                                                                                                                                       |
| `tree`                   | Lists the contents of a directory recursively.                                                                                                             |
| `mv`                     | Move or rename files or directories.                                                                                                                       |
| `cp`                     | Copy files or directories.                                                                                                                                 |
| `nano`                   | Terminal based text editor.                                                                                                                                |
| `which`                  | Returns the path to a file or link.                                                                                                                        |
| `find`                   | Searches for files in a directory hierarchy.                                                                                                               |
| `updatedb`               | Updates the locale database for existing contents on the system.                                                                                           |
| `locate`                 | Uses the locale database to find contents on the system.                                                                                                   |
| `more`                   | Pager that is used to read STDOUT or files.                                                                                                                |
| `less`                   | An alternative to `more` with more features.                                                                                                               |
| `head`                   | Prints the first ten lines of STDOUT or a file.                                                                                                            |
| `tail`                   | Prints the last ten lines of STDOUT or a file.                                                                                                             |
| `sort`                   | Sorts the contents of STDOUT or a file.                                                                                                                    |
| `grep`                   | Searches for specific results that contain given patterns.                                                                                                 |
| `cut`                    | Removes sections from each line of files.                                                                                                                  |
| `tr`                     | Replaces certain characters.                                                                                                                               |
| `column`                 | Command-line based utility that formats its input into multiple columns.                                                                                   |
| `awk`                    | Pattern scanning and processing language.                                                                                                                  |
| `sed`                    | A stream editor for filtering and transforming text.                                                                                                       |
| `wc`                     | Prints newline, word, and byte counts for a given input.                                                                                                   |
| `chmod`                  | Changes permission of a file or directory.                                                                                                                 |
| `chown`                  | Changes the owner and group of a file or directory.                                                                                                        |
