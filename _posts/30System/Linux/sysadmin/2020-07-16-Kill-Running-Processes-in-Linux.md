---
title: Linux - Find and Kill Running Processes in Linux
date: 2020-07-16 11:11:11 -0400
categories: [30System, Sysadmin]
tags: [Linux, Sysadmin]
math: true
image:
---


# Find and Kill Running Processes in Linux

[toc]

## Process

process on a Linux system
- a running occurrence of an application/program
- processes as tasks executing in the operating system.

process states:
- Running: meaning the process is either executing or it is just set to be executed.
  - When a process is running, it keeps on shifting from one state to another
- Waiting: meaning that the process is waiting for an event or for a system resource to carry out a task.
  - two types of waiting process
  - interruptible: A waiting process that can be interrupted by signals is called Interruptible.
  - uninterruptible: a waiting process that is directly waiting on hardware conditions and cannot be interrupted under any conditions.
- Stopped: the process has been stopped, using a signal.
- Zombie: the process has been stopped abruptly and is dead.

When killing processes, the kill command is used to send a named signal to a named process or groups of processes. The default signal is the TERM signal.
- the kill command can be a built-in function in many modern shells or external located at /bin/kill.

## Find Process PID
In Linux every process on a system has a PID (Process Identification Number).

```c
$ pidof processname
$ pidof firefox
$ pidof chrome
```

## Kill Processes

`$ kill pid_number`

send a named signal to the process by using the signal name:

`$ kill -SIGTERM pid_number`

Using the signal number to kill a process:

`$ kill -9 pid_number`


## Kill Multiple Process

$ kill -9 pid_number1, pid_number2, pid_number3



.
