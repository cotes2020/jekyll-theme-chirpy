---
title: ShellScripts
date: 2020-07-16 11:11:11 -0400
categories: [00CodeNote]
tags: [Script]
math: true
image:
---


# Writing Shell Scripts

[toc]

# What are Shell Scripts?

ref
http://linuxcommand.org/lc3_wss0140.php


With the thousands of commands available for the command line user, how can you remember them all? The answer is, you don't. The real power of the computer is its ability to do the work for you. To get it to do that, we use `shell` to automate things. We write `shell scripts`.
- a file containing a series of commands.
- The `shell` reads this file and carries out the commands as though they have been entered directly on the command line.

The shell is somewhat unique, in that it is both a `powerful command line interface` to the system and a `scripting language interpreter`. As we will see, most of the things that can be done on the command line can be done in scripts, and most of the things that can be done in scripts can be done on the command line.

To successfully write a shell script, you have to do three things:
- `Write` a script
- Give the shell permission to `execute` it
- Put it somewhere the shell can `find` it

## Writing a Script
A shell script is a file that contains ASCII text.
To create a shell script, you use a text editor that reads and writes ASCII text files.

There are many, many text editors available for your Linux system, both for the command line environment and the GUI environment. Here is a list of some common ones:

`vi, vim` command line
- The granddaddy of Unix text editors
- infamous for its difficult, non-intuitive command structure. On the bright side, vi is powerful, lightweight, and fast. On most Linux distributions, an enhanced version of the traditional `vi` editor called `vim` is used.

`Emacs` command line
- The true giant in the world of text editors is Emacs by Richard Stallman.
- Emacs contains (or can be made to contain) every feature ever conceived for a text editor. It should be noted that vi and Emacs fans fight bitter religious wars over which is better.

`nano` command line
- nano is a free clone of the text editor supplied with the pine email program. nano is very easy to use but is very short on features. I recommend nano for first-time users who need a command line editor.

`gedit`  graphical
- gedit is the editor supplied with the Gnome desktop environment.

`kwrite`  graphical
- kwrite is the "advanced editor" supplied with KDE. It has syntax highlighting, a helpful feature for programmers and script writers.

use text editor and type in your first script as follows:

```
#!/bin/bash
# My first script

echo "Hello World!"
```

The first line of the script is important. This is `shebang`, given to the shell indicating what program is used to interpret the script. In this case, it is /bin/bash. Other scripting languages such as Perl, awk, tcl, Tk, and python also use this mechanism.

The second line is a `comment`. Everything that appears after a "#" symbol is ignored by bash. As your scripts become bigger and more complicated, comments become vital. They are used by programmers to explain what is going on so that others can figure it out.

The last line is the echo command. echo: prints its arguments on the display.

## Setting Permissions
The next thing, give the `shell` permission to execute your script.

$ chmod 755 hello_world

The "755" will give you read, write, and execute permission.
Everybody else will get only read and execute permission.
If you want your script to be private (i.e., only you can read and execute), use "700" instead.

## Putting It in Your Path
At this point, your script will run.
$ ./hello_world

You should see "Hello World!" displayed. If you do not, see what directory you really saved your script in, go there and try again.

## talk about paths.
- When you type in the name of a command, the system does not search the entire computer to find where the program is located.
- You have noticed that you don't usually have to specify a complete path name to the program you want to run, the shell just seems to know.
- The shell does know. Here's how:
  - the shell maintains a list of directories where executable files (programs) are kept, and only searches the directories in that list.
  - If it does not find the program after searching each directory in the list, it will issue the famous command not found error message.
  - This list of directories is called your path. You can view the list of directories with the following command: `$ echo $PATH`

This will return a colon separated list of directories that will be searched if a specific path name is not given when a command is attempted. In our first attempt to execute your new script, we specified a pathname ("./") to the file.

You can add directories to your path with the following command, where directory is the name of the directory you want to add:
`$ export PATH=$PATH:directory`

A better way would be to edit your `.bash_profile` or `.profile` file to include the above command. That way, it would be done automatically every time you log in.


Most Linux distributions encourage a practice in which each user has a specific directory for the programs he/she personally uses: `bin`.
If you do not already have one, create it: `$ mkdir bin`
Move your script into `bin` directory and you're all set.
Now you just have to type: `$ hello_world` and your script will run.

On some distributions, most notably Ubuntu, you will need to open a new terminal session before your newly created bin directory will be recognized.

## Editing the Scripts You Already Have
you have some scripts of your own already. These scripts were put into your home directory when your account was created, and are used to configure the behavior of your sessions on the computer.

You can edit these scripts to change things.

### environment
During your session, the system is holding a number of facts about the world in its memory. This information is called the `environment`. The environment contains such things as your path, your user name, the name of the file where your mail is delivered, and much more. You can see a complete list of what is in your environment with the `set` command.

Two types of commands are often contained in the `environment`.
They are `aliases` and `shell functions`.


#### How is the Environment Established?
When you log on to the system, the `bash` program starts, and reads a series of configuration scripts called `startup files`.
These define the default environment shared by all users.
This is followed by more startup files in your home directory that define your personal environment.
The exact sequence depends on the type of shell session being started.

There are two kinds: a `login shell session` and a `non-login shell session`.
- `login shell session`: one in which we are prompted for our user name and password; when we start a virtual console session, for example.
- `non-login shell session` typically occurs when we launch a terminal session in the GUI.

`Login shells` read one or more startup files as shown below:
- `/etc/profile` :
	- A global configuration script that applies to all users.
- `~/.bash_profile` :
	- A user's personal startup file.
  - Can be used to extend or override settings in the global configuration script.
- `~/.bash_login` :
	- If ~/.bash_profile is not found, bash attempts to read this script.
- `~/.profile` :
	- If neither ~/.bash_profile nor ~/.bash_login is found, bash attempts to read this file.
  - This is the default in Debian-based distributions, such as Ubuntu.

`Non-login shell sessions` read the following startup files:
- `/etc/bash.bashrc` :
  - A global configuration script that applies to all users.
- `~/.bashrc` :
  - A user's personal startup file.
  - Can be used to extend or override settings in the global configuration script.

In addition to reading the startup files above, non-login shells also inherit the environment from their parent process, usually a login shell.

Take a look at your system and see which of these startup files you have.
most of the file names listed above start with a period (hidden), need to use the “-a” option

```
$ cat profile
# System-wide .profile for sh(1)

if [ -x /usr/libexec/path_helper ]; then
	eval `/usr/libexec/path_helper -s`
fi

if [ "${BASH-no}" != "no" ]; then
	[ -r /etc/bashrc ] && . /etc/bashrc
fi
---------------------------------------------
$ cat bashrc
# System-wide .bashrc file for interactive bash(1) shells.
if [ -z "$PS1" ]; then
 return
fi

PS1='\h:\W \u\$ '
# Make bash check its window size after a process completes
shopt -s checkwinsize

[ -r "/etc/bashrc_$TERM_PROGRAM" ] && . "/etc/bashrc_$TERM_PROGRAM"
```

The `~/.bashrc` file is probably the most important startup file from the ordinary user’s point of view, since it is almost always read. Non-login shells read it by default and most startup files for login shells are written in such a way as to read the ~/.bashrc file as well.

look inside a typical .bash_profile
```py
# .bash_profile
# Get the aliases and functions
# Lines not read by the shell. for human readability.
if [ -f ~/.bashrc ]; then
	. ~/.bashrc
fi

- This is called an if compound command
- If the file "~/.bashrc" exists, then read the "~/.bashrc" file.
- this bit of code is how a login shell gets the contents of .bashrc.

# User specific environment and startup programs
PATH=$PATH:$HOME/bin
# The next thing in our startup file, set set PATH variable to add the ~/bin directory to the path.
export PATH
# The export command tells the shell to make the contents of PATH available to child processes of this shell.
```

### Aliases
An alias is an easy way to create a new command which acts as an abbreviation for a longer one.
`alias name=value`
- name: the name of the new command
- value: the text to be executed whenever name is entered on the command line.

```py
create an alias called "l" and make it an abbreviation for the command "ls -l".
# Make sure you are in your home directory.
# open the file .bashrc and add this line to the end of the file:
alias l='ls -l'
# we have created a new command called "l" which will perform "ls -l".
#To try out your new command, close your terminal session and start a new one. This will reload the .bashrc file.

# Using this technique, you can create any number of custom commands for yourself. Here is another one for you to try:
alias today='date +"%A, %B %-d, %Y"'
# This alias creates a new command called "today" that will display today's date with nice formatting.

By the way, the alias command is just another shell builtin.
You can create your aliases directly at the command prompt;
however they will only remain in effect during your current shell session.

[me@linuxbox me]$ alias l='ls -l'
```

### Shell Functions
Aliases are good for very simple commands, but to create something more complex, try shell functions.
Shell functions can be thought of as "scripts within scripts" or little sub-scripts.

- open .bashrc with your text editor again
- replace the alias for "today" with the following:

```py
today() {
    echo -n "Today's date is: "
    date +"%A, %B %-d, %Y"
}
```

- `()` is a shell builtin too, and as with `alias`
- enter shell functions directly at the command prompt.

```py
[me@linuxbox me]$ today() {
> echo -n "Today's date is: "
> date +"%A, %B %-d, %Y"
> }

# like alias, shell functions defined directly on the command line only last as long as the current shell session.
```

# Here Scripts
Beginning with this lesson, we will construct a useful application. This application will produce an `HTML document` that contains information about your system.

I spent a lot of time thinking about how to teach shell programming, and the approach I have chosen is very different from most others that I have seen. Most favor a systematic treatment of shell features, and often presume experience with other programming languages. Although I do not assume that you already know how to program, I realize that many people today know how to write HTML, so our program will produce a web page.

As we construct our script, we will discover step by step the tools needed to solve the problem at hand.

## Writing an HTML File with a Script
a well formed HTML file contains the following content:
```
<html>
<head>
    <title>
    The title of your page
    </title>
</head>

<body>
    Your page content goes here.
</body>
</html>
```

Now, write a script to produce the above content:

```shell
#!/bin/bash

# sysinfo_page - A script to produce an html file

echo "<html>"
echo "<head>"
echo "  <title>"
echo "  The title of your page"
echo "  </title>"
echo "</head>"
echo ""
echo "<body>"
echo "  Your page content goes here."
echo "</body>"
echo "</html>"
```
This script can be used as follows:
[me@linuxbox me]$ sysinfo_page > sysinfo_page.html

lever programmers write programs, they try to save themselves typing.

The first improvement
replace the repeated use of the echo with a single instance by using quotation more efficiently:

```py
#!/bin/bash
# sysinfo_page - A script to produce an HTML file

echo "<html>
 <head>
   <title>
   The title of your page
   </title>
 </head>

 <body>
   Your page content goes here.
 </body>
 </html>"
```

Using quotation, it is possible to embed carriage returns in our text and have the echo command's argument span multiple lines.

While this is certainly an improvement, it does have a limitation. Since many types of markup used in html incorporate quotation marks themselves, it makes using a quoted string awkward. A quoted string can be used but each embedded quotation mark will need to be escaped with a backslash character.

to avoid the additional typing, the shell provides `here script`.

```
#!/bin/bash
# sysinfo_page - A script to produce an HTML file

cat << _EOF_

<html>
<head>
    <title>
    The title of your page
    </title>
</head>

<body>
    Your page content goes here.
</body>
</html>

_EOF_
```

A `here script` (here document) is an additional form of I/O redirection.
- It provides a way to include content that will be given to the standard input of a command.
- In the case of the script above, the standard input of the cat command was given a stream of text from our script.
- A `here script` is constructed like:

```
command << token
content to be used as command's standard input
token
```

`token` can be any string of characters.
- "_EOF_" ("End Of File") or use anything, as long as it does not conflict with a bash reserved word.
- The token that ends must exactly match the one that starts, or else the remainder of your script will be interpreted as more standard input to the command.

one additional trick
to indent the content portion of the `here script` to improve the readability of your script.
- Changing the the `<<` to `<<-` causes bash to `ignore the leading tabs` (but not spaces) in the here script.
- The output from the cat command will not contain any of the leading tab characters.

```
#!/bin/bash
# sysinfo_page - A script to produce an HTML file

cat <<- _EOF_
    <html>
    <head>
        <title>
        The title of your page
        </title>
    </head>

    <body>
        Your page content goes here.
    </body>
    </html>
_EOF_
```



O.k. edit our page to:
```
#!/bin/bash
# sysinfo_page - A script to produce an HTML file

cat <<- _EOF_
    <html>
    <head>
        <title>
        My System Information
        </title>
    </head>

    <body>
    <h1>My System Information</h1>
    </body>
    </html>
_EOF_
```

## Variables
make some changes because we want to be lazy.
- the phrase "My System Information" is repeated. improve it:
- added a line to the beginning of the script and replaced the two occurrences of the phrase "My System Information" with `$title`.

```
#!/bin/bash
# sysinfo_page - A script to produce an HTML file

title="My System Information"

cat <<- _EOF_
    <html>
    <head>
        <title>
        $title
        </title>
    </head>

    <body>
    <h1>$title</h1>
    </body>
    </html>
_EOF_
```

Variables
Variables: areas of memory that can be used to store information and are referred to by a name.
- In the case of our script, variable "title" placed the phrase "My System Information" into memory. Inside the `here script` that contains our HTML, we use "$title" to tell the shell to perform `parameter expansion` and replace the name of the variable with the variable's contents.
- Whenever the shell sees a word that begins with a `$`, it find out what was assigned to the variable and substitutes it.

to Create a Variable
- `variable=assign the information you wish to store`
  - followed immediately by an equal sign ("=").
  - No spaces are allowed.

to choose the names for variables. few rules.
- Names must start with a letter.
- A name must not contain embedded spaces. Use underscores instead.
- cannot use punctuation marks.

How Does This Increase Our Laziness?
- First, it reduced the amount of typing.
- Second and more importantly, it made our script easier to maintain.

programs are rarely ever finished. They are modified and improved by their creators and others. After all, that's what open source development is all about.
- wanted to change the phrase "My System Information" to "Linuxbox System Information." you would have had to change this in two locations.
- In the new version with the title variable, you only have to change it in one place.
- as scripts get larger and more complicated, it becomes very important.

## Environment Variables
When you start your shell session, some variables are already set by the startup file we looked at earlier.
- To see all the variables that are in your environment, use the `printenv` command.
- variable `$HOSTNAME`: contains the host name for your system.
- add this variable to script:

```
#!/bin/bash
# sysinfo_page - A script to produce an HTML file

title="System Information for"

cat <<- _EOF_
    <html>
    <head>
        <title>
        $title $HOSTNAME
        </title>
    </head>

    <body>
    <h1>$title $HOSTNAME</h1>
    </body>
    </html>
_EOF_

# Now our script will always include the name of the machine on which we are running.
# by convention, environment variables names are uppercase.
```

## Command Substitution and Constants

### substitute the results from a command. `$( )`
our script, it could create an HTML page that contained a few simple lines of text, including the host name of the machine which we obtained from the environment variable `HOSTNAME`. Now, we will add a time stamp to the page to indicate when it was last updated, along with the user that did it.

```
#!/bin/bash
# sysinfo_page - A script to produce an HTML file

title="System Information for"

cat <<- _EOF_
    <html>
    <head>
        <title>
        $title $HOSTNAME
        </title>
    </head>

    <body>
    <h1>$title $HOSTNAME</h1>
    <p>Updated on $(date +"%x %r %Z") by $USER</p>
    </body>
    </html>
_EOF_
```

employed another environment variable, `$USER`, to get the user name.

In addition, we used this strange looking thing: $(date +"%x %r %Z")

The characters `$( )`
- tell the shell, "substitute the results of the enclosed command."
- In our script, the shell insert the results of the command date +"%x %r %Z" which expresses the current date and time.

an older, alternate syntax for `$()`, uses the backtick character " \` ".
- This older form is compatible with the original Bourne shell (sh).
- The bash shell fully supports scripts written for sh
- the following forms are equivalent:
  - $(command)
  - \`command`

You can also assign the results of a command to a variable:
`right_now=$(date +"%x %r %Z")`

You can nest the variables (place one inside another):
`right_now=$(date +"%x %r %Z")`
`time_stamp="Updated on $right_now by $USER"`

### Constants

`variable`: the content of a variable is subject to change.
- it is expected that during the execution of your script, a variable may have its content modified by something you do.

`constants`: values that, once set, should never be changed.
- Bash also has these facilities but, to be honest, I never see it used.
- Instead, if a value is intended to be a constant, it is given an `uppercase name` to remind the programmer that it should be considered a constant even if it's not being enforced.

`Environment variables` are usually considered `constants` since they are rarely changed. Like constants, environment variables are given uppercase names by convention.


use uppercase names for `constants` and lowercase names for `variables`.

```
#!/bin/bash
# sysinfo_page - A script to produce an HTML file

title="System Information for $HOSTNAME"
RIGHT_NOW=$(date +"%x %r %Z")
TIME_STAMP="Updated on $RIGHT_NOW by $USER"

cat <<- _EOF_
    <html>
    <head>
        <title>
        $title
        </title>
    </head>

    <body>
    <h1>$title</h1>
    <p>$TIME_STAMP</p>
    </body>
    </html>
_EOF_
```

## Shell Functions
As programs get longer and more complex, they become more difficult to design, code, and maintain.
- it is often useful to break a single, large task into a series of smaller tasks.
- to break our single monolithic script into a number of separate functions.

As our script continues to grow, we will use `top down design` to help us plan and code our script.

If we look at our script's top-level tasks, we find the following list:

```
Open page           <html>
Open head section     <head>
Write title              <title> $title </title>
Close head section    </head>
Open body section     <body>
Write title              <h1>$title</h1>
Write time stamp         <p>$TIME_STAMP</p>    task 7
Close body section    </body>
Close page          </html>
```

All of these tasks are implemented, but we want to add more. Let's insert some additional tasks after task 7:

    Write time stamp
    Write system release info
    Write up-time
    Write drive space
    Write home space
    Close body section
    Close page

---

1. if there were commands that performed these additional tasks. use command substitution to place them in our script like so:

```py
#!/bin/bash
# sysinfo_page - A script to produce a system information HTML file

##### Constants

TITLE="System Information for $HOSTNAME"
RIGHT_NOW=$(date +"%x %r %Z")
TIME_STAMP="Updated on $RIGHT_NOW by $USER"

##### Main

cat <<- _EOF_
  <html>
  <head>
      <title>$TITLE</title>
  </head>

  <body>
      <h1>$TITLE</h1>
      <p>$TIME_STAMP</p>
      $(system_info)
      $(show_uptime)
      $(drive_space)
      $(home_space)
  </body>
  </html>
_EOF_
```
---

2. While there are no commands that do exactly what we need, create them using `shell functions`.
- shell functions act as "little programs within programs" and allow us to follow top-down design principles.
- To add the shell functions to our script, we change it so:

```py
#!/bin/bash
# sysinfo_page - A script to produce an system information HTML file

##### Constants
TITLE="System Information for $HOSTNAME"
RIGHT_NOW=$(date +"%x %r %Z")
TIME_STAMP="Updated on $RIGHT_NOW by $USER"

##### Functions
system_info()
{
}

show_uptime()
{
}

drive_space()
{
}

home_space()
{
}

##### Main

cat <<- _EOF_
  <html>
  <head>
      <title>$TITLE</title>
  </head>

  <body>
      <h1>$TITLE</h1>
      <p>$TIME_STAMP</p>
      $(system_info)
      $(show_uptime)
      $(drive_space)
      $(home_space)
  </body>
  </html>
_EOF_
```

A couple of important points about functions:
- First, they must appear before you attempt to use them.
- Second, the function body (the portions of the function between the { and } characters) must contain at least one valid command. As written, the script will execute error, because the function bodies are empty.
- The simple way to fix this is to place a return statement in each function body. script will execute successfully again.

### Keep Your Scripts Working
When you are developing a program, it is is often a good practice to add a small amount of code, run the script, add some more code, run the script, and so on.
mistake into your code will be easier to find and correct.

As you add functions to your script, you can also use a technique called `stubbing` to help watch the logic of your script develop.
- Stubbing works like this: imagine that we are going to create a function called "system_info" but we haven't figured out all of the details of its code yet.
- Rather than hold up the development of the script until we are finished with system_info, we just add an `echo` command like this:
```
system_info()
{
    # Temporary function stub
    echo "function system_info"
}
```

This way, our script will still execute successfully, even though we do not yet have a finished system_info function. We will later replace the temporary stubbing code with the complete working version.

The reason we use an `echo` command is so we get some feedback from the script to indicate that the functions are being executed.

go ahead and write stubs for our new functions and keep the script working.

```py
#!/bin/bash
# sysinfo_page - A script to produce an system information HTML file

##### Constants
TITLE="System Information for $HOSTNAME"
RIGHT_NOW=$(date +"%x %r %Z")
TIME_STAMP="Updated on $RIGHT_NOW by $USER"

##### Functions
system_info()
{
    # Temporary function stub
    echo "function system_info"
}

show_uptime()
{
    # Temporary function stub
    echo "function show_uptime"
}

drive_space()
{
    # Temporary function stub
    echo "function drive_space"
}

home_space()
{
    # Temporary function stub
    echo "function home_space"
}

##### Main

cat <<- _EOF_
  <html>
  <head>
      <title>$TITLE</title>
  </head>

  <body>
      <h1>$TITLE</h1>
      <p>$TIME_STAMP</p>
      $(system_info)
      $(show_uptime)
      $(drive_space)
      $(home_space)
  </body>
  </html>
_EOF_
```

### Some Real Work

#### show_uptime
- The `show_uptime` function: display the output of the `uptime` command.
- The `uptime` command: outputs several interesting facts about the system, including the length of time the system has been "up" (running) since its last re-boot, the number of users and recent system load.

    $ uptime
    9:15pm up 2 days, 2:32, 2 users, load average: 0.00, 0.00, 0.00

To get the output of the uptime command into our HTML page:
```
show_uptime()
{
    echo "<h2>System uptime</h2>"
    echo "<pre>"
    uptime
    echo "</pre>"
}
```

As you can see, this function outputs a stream of text containing a mixture of HTML tags and command output. When the command substitution takes place in the main body of the our program, the output from our function becomes part of the here script.

#### drive_space
The `drive_space` function: use the `df` command to provide a summary of the space used by all of the mounted file systems.

    $ df
    Filesystem   1k-blocks      Used Available Use% Mounted on
    /dev/hda2       509992    225772    279080  45% /
    /dev/hda1        23324      1796     21288   8% /boot
    /dev/hda3     15739176   1748176  13832360  12% /home
    /dev/hda5      3123888   3039584     52820  99% /usr

the drive_space function is very similar to the show_uptime function:
```
drive_space()
{
    echo "<h2>Filesystem space</h2>"
    echo "<pre>"
    df
    echo "</pre>"
}
```

#### home_space
The `home_space` function: display the amount of space each user is using in his/her home directory. It will display this as a list, sorted in descending order by the amount of space used.
```
home_space()
{
    echo "<h2>Home directory space by user</h2>"
    echo "<pre>"
    echo "Bytes Directory"
    du -s /home/* | sort -nr
    echo "</pre>"
}
```
Note that in order for this function to successfully execute, the script must be run by the superuser, since the du command requires superuser privileges to examine the contents of the /home directory.

#### system_info
not ready to finish the system_info function yet.
improve the stubbing code so it produces valid HTML:
```
system_info()
{
    echo "<h2>System release info</h2>"
    echo "<p>Function not yet implemented</p>"
}
```
---

# Flow Control - Part 1
how to add intelligence to our scripts.
So far, our project script has only consisted of a sequence of commands that starts at the first line and continues line by line until it reaches the end.
Most programs do much more than this. They make decisions and perform different actions depending on conditions.

The shell provides several commands that we can use to `control the flow of execution` in our program.

## `if commands; then commands [elif commands; then commands] fi`
if command: makes a decision based on the exit status of a command.

```py
if commands; then
commands
[elif commands; then
commands...]
[else
commands]
fi
```

## Exit Status
Commands (including the scripts and shell functions we write) issue a value to the system when they terminate, called an `exit status`.
- This value, which is an integer in the range of 0 to 255, indicates the success or failure of the command’s execution.
- By convention, zero indicates success and any other value indicates failure.
- The shell provides a parameter to examine the exit status.

```py
Some commands use different exit status values to provide diagnostics for errors
- many commands simply exit with a `value of one` when they fail.
- zero always indicates success.

$ ls -d /usr/bin
/usr/bin
$ echo $?
0
# the command executes successfully

$ ls -d /bin/usr
ls: cannot access /bin/usr: No such file or directory
$ echo $?
2
# indicating that the command encountered an error.
```

```
The shell provides two extremely simple builtin commands that do nothing except terminate with either a zero or one exit status. The true command always executes successfully and the false command always executes unsuccessfully:

[me@linuxbox~]$ true
[me@linuxbox~]$ echo $?
0
[me@linuxbox~]$ false
[me@linuxbox~]$ echo $?
1
```

```py
use these commands to see how the if statement works.
What the if statement really does is evaluate the success or failure of commands:

The command echo "It's true."
- executed when the command following if executes successfully
- is not executed when the command following if does not execute successfully.


[me@linuxbox ~]$ if true; then echo "It's true."; fi
It's true.
[me@linuxbox ~]$ if false; then echo "It's true."; fi
[me@linuxbox ~]$
```

## `test`
used most often with the `if` command to perform true/false decisions.

The command is unusual in that it has two different syntactic forms:
- First form:   `test` expression
- Second form:  `[ expression ]`

The `test` command:
- If the given expression is true, test exits with a status of zero;
- otherwise it exits with a status of 1.

a partial list of the conditions that test can evaluate.
```
J:etc luo$ help test
test: test [expr]
    Exits with a status of 0 (true) or 1 (false) depending on
    the evaluation of EXPR.  Expressions may be unary or binary.  Unary
    expressions are often used to examine the status of a file.  There
    are string operators as well, and numeric comparison operators.

    File operators:

        -a FILE        True if file exists.
        -b FILE        True if file is block special.
        -c FILE        True if file is character special.
        -d FILE        True if file is a directory.
        -e FILE        True if file exists.
        -f FILE        True if file exists and is a regular file.
        -g FILE        True if file is set-group-id.
        -h FILE        True if file is a symbolic link.
        -L FILE        True if file is a symbolic link.
        -k FILE        True if file has its `sticky' bit set.
        -p FILE        True if file is a named pipe.
        -r FILE        True if file is readable by you.
        -s FILE        True if file exists and is not empty.
        -S FILE        True if file is a socket.
        -t FD          True if FD is opened on a terminal.
        -u FILE        True if the file is set-user-id.
        -w FILE        True if the file is writable by you.
        -x FILE        True if the file is executable by you.
        -O FILE        True if the file is effectively owned by you.
        -G FILE        True if the file is effectively owned by your group.
        -N FILE        True if the file has been modified since it was last read.

      FILE1 -nt FILE2  True if file1 is newer than file2 (according to
                       modification date).

      FILE1 -ot FILE2  True if file1 is older than file2.

      FILE1 -ef FILE2  True if file1 is a hard link to file2.

    String operators:

        -z STRING      True if string is empty.

        -n STRING
        STRING         True if string is not empty.

        STRING1 = STRING2
                       True if the strings are equal.
        STRING1 != STRING2
                       True if the strings are not equal.
        STRING1 < STRING2
                       True if STRING1 sorts before STRING2 lexicographically.
        STRING1 > STRING2
                       True if STRING1 sorts after STRING2 lexicographically.

    Other operators:

        -o OPTION      True if the shell option OPTION is enabled.
        ! EXPR         True if expr is false.
        EXPR1 -a EXPR2 True if both expr1 AND expr2 are true.
        EXPR1 -o EXPR2 True if either expr1 OR expr2 is true.

        arg1 OP arg2   Arithmetic tests.  OP is one of -eq, -ne,
                       -lt, -le, -gt, or -ge.

    Arithmetic binary operators return true if ARG1 is equal, not-equal,
    less-than, less-than-or-equal, greater-than, or greater-than-or-equal
    than ARG2.
```


example:

```
if [ -f .bash_profile ]; then
    echo "You have a .bash_profile. Things are fine."
else
    echo "Yikes! You have no .bash_profile!"
fi
```

In this example, use the expression " -f .bash_profile ".
- This expression asks, "Is .bash_profile a file?"
- If the expression is true, then `test` exits with a zero (indicating true) and the if command executes the command(s) following the word then.
- If the expression is false, then `test` exits with a status of one and the if command executes the command(s) following the word else.

the if command followed by the `test` command, followed by a semicolon, and finally the word then.
- use the `[ expression ]` form of the `test` command
  - Notice that `the spaces` required between the "[", expression expression, the trailing "]" are required.
- The `semicolon` is a command separator. Using it allows you to put more than one command on a line. For example:
  - `$ clear; ls` : will clear the screen and execute the ls command.


On the second line, there is our old friend echo. The only thing of note on this line is the `indentation`.
- traditional to indent all blocks of conditional code; that is, any code that will only be executed if certain conditions are met.
- The shell does not require this; it is done to make the code easier to read.

we could write the following and get the same results:
```py
# Alternate form

if [ -f .bash_profile ]
then
    echo "You have a .bash_profile. Things are fine."
else
    echo "Yikes! You have no .bash_profile!"
fi

# Another alternate form

if [ -f .bash_profile ]
then echo "You have a .bash_profile. Things are fine."
else echo "Yikes! You have no .bash_profile!"
fi
```

## exit
In order to be good script writers, we must set the `exit status` when our scripts finish.
To do this, use the `exit` command: causes the script to terminate immediately and set the exit status to whatever value is given as an argument.
```
For example:

exit 0
exits your script and sets the exit status to 0 (success), whereas

exit 1
exits your script and sets the exit status to 1 (failure).
```

## Testing for Root
When we last left our script, we required that it be run with superuser privileges. This is because the `home_space` function needs to examine the size of each user's home directory, and only the superuser is allowed to do that.

But what happens if a regular user runs our script? It produces error messages. What if we could put something in the script to stop regular user run it?


The `id` command: tell us who the current user is.
When executed with the "-u" option, it prints the numeric user id of the current user.

    J:etc luo$ sudo id -u
    Password:
    0
    J:etc luo$ id -u
    502

```py
# this code will detect if the user is the superuser
if [ $(id -u) = "0" ]; then
    echo "superuser"
fi

# to stop the script if the user is not the superuser
if [ $(id -u) != "0" ]; then
    echo "You must be the superuser to run this script" >&2
    exit 1
fi
```

if the output of the id -u command is not equal to "0",
- then the script prints a descriptive error message, exits,
- and sets the exit status to 1, indicating to the operating system that the script executed unsuccessfully.

`>&2` at the end of the echo command.
- another form of I/O direction. You will often notice this in routines that display error messages.
- If this redirection were not done, the error message would go to standard output.
- With this redirection, the message is sent to standard error.
- Since we are executing our script and redirecting its standard output to a file, we want the error messages separated from the normal output.

We could put this routine near the beginning of our script so it has a chance to detect a possible error before things get under way

but in order to run this script as an ordinary user, we will use the same idea and modify the `home_space` function to test for proper privileges instead, like so:

```
function home_space
{
    # Only the superuser can get this information

    if [ "$(id -u)" = "0" ]; then
        echo "<h2>Home directory space by user</h2>"
        echo "<pre>"
        echo "Bytes Directory"
            du -s /home/* | sort -nr
        echo "</pre>"
    fi

}   # end of home_space
```

This way, if an ordinary user runs the script, the troublesome code will be passed over, rather than executed and the problem will be solved.


# Stay Out of Trouble
Now that our scripts are getting a little more complicated, I want to point out some common mistakes that you might run into. To do this, create the following script called `trouble.bash`.
enter it exactly as written.

```py
#!/bin/bash

number=1

if [ $number = "1" ]; then
    echo "Number equals 1"
else
    echo "Number does not equal 1"
fi

# run
$ nano
$ chmod 755 trouble.bash
$ trouble.bash
Number equals 1
```

## Empty Variables
Edit the script to change line 3 from `number=1` to `number=`

```py
#!/bin/bash

number=

if [ $number = "1" ]; then
    echo "Number equals 1"
else
    echo "Number does not equal 1"
fi

# run the script again.
$ ./trouble.bash
"/trouble.bash: [: =: unary operator expected."    # bash displayed an error message when we ran the script.
Number does not equal 1
```

the error message:
`./trouble.bash: [: =: unary operator expected`

error is occurring on line 5 not line 3.
- there is nothing wrong with line 3. number= is perfectly good syntax. set a variable's value to nothing. You can confirm the validity of this by trying it on the command line:

    [me@linuxbox me]$ number=
    [me@linuxbox me]$
    no error message.

wrong with line 5
- In line 5, the shell expands the value of number where it sees $number.
- when number=1, the shell substituted 1 for $number like so: `if [ 1 = "1" ]; then`
- when number=, the shell saw this after the expansion: `if [ = "1" ]; then`
  - which is an error. It also explains the rest of the error message we received.
  - The "=" is a binary operator; that is, it expects two items to operate upon - one on each side. What the shell is trying to tell us is that there is only one item and there should be a unary operator (like "!") that only operates on a single item.

To fix this problem, change line 5: `if [ "$number" = "1" ]; then`
- Now when the shell performs the expansion it will see: `if [ "" = "1" ]; then`
- which correctly expresses our intent.

This brings up an important thing to remember when you are writing your scripts. Consider what happens if a variable is set to equal nothing.

## Missing Quotes
Edit line 6 to remove the trailing quote from the end of the line: echo "Number equals 1
```
#!/bin/bash

number=

if [ $number = "1" ]; then
    echo "Number equals 1
else
    echo "Number does not equal 1"
fi
```

run the script again. You should get this:

    [me@linuxbox me]$ ./trouble.bash
    ./trouble.bash: line 8: unexpected EOF while looking for matching "
    ./trouble.bash: line 10 syntax error: unexpected end of file

Here we have another case of a mistake in one line causing a problem later in the script. What happens is the shell keeps looking for the closing quotation mark to tell it where the end of the string is, but runs into the end of the file before it finds it.

These errors can be a real pain to find in a long script. This is one reason you should test your scripts frequently when you are writing them so there is less new code to test. I also find that text editors with syntax highlighting make these kinds of bugs easier to find.

## Isolating Problems
Finding bugs in your programs can sometimes be very difficult and frustrating.

couple of techniques useful:

1. Isolate blocks of code by "commenting them out."
    - This trick involves putting comment characters at the beginning of lines of code to stop the shell from reading them.
    - Frequently, you will do this to a block of code to see if a particular problem goes away. By doing this, you can isolate which part of a program is causing (or not causing) a problem.

          #!/bin/bash
          number=1
          if [ $number = "1" ]; then
              echo "Number equals 1
          #else
          #   echo "Number does not equal 1"
          fi

    - By commenting out the else clause and running the script, we could show that the problem was not in the else clause even though the error message suggested that it was.

2. Use echo commands to verify your assumptions.
    - As you gain experience tracking down bugs, you will discover that bugs are often not where you first expect to find them. A common problem will be that you will make a false assumption about the performance of your program. You will see a problem develop at a certain point in your program and assume that the problem is there. This is often incorrect, as we have seen.
    - To combat this, you should place echo commands in your code while you are debugging, to produce messages that confirm the program is doing what is expected. There are two kinds of messages that you should insert.
      - The first type: simply announces that you have reached a certain point in the program. We saw this in our earlier discussion on stubbing. It is useful to know that program flow is happening the way we expect.
      - The second type: displays the value of a variable (or variables) used in a calculation or test. You will often find that a portion of your program will fail because something that you assumed was correct earlier in your program is, in fact, incorrect and is causing your program to fail later on.

## Watching Your Script Run
It is possible to have bash show you what it is doing when you run your script.

1. add `-x` to the first line of your script:
    - `#!/bin/bash -x`
    - Now, when you run your script, bash will display each line (with expansions performed) as it executes it. This technique is called `tracing`.

          $ ./trouble.bash
          + number=1
          + '[' 1 = 1 ']'
          + echo 'Number equals 1'
          Number equals 1

2. use the `set` command within your script to turn tracing on and off.
    - `set -x` to turn tracing on
    - `set +x` to turn tracing off.

          #!/bin/bash
          number=1

          set -x
          if [ $number = "1" ]; then
              echo "Number equals 1"
          else
              echo "Number does not equal 1"
          fi
          set +x


# Keyboard Input and Arithmetic
Up to now, our scripts have not been interactive. did not require any input from the user.

## read
To get input from the keyboard, use the `read` command.
- takes input from the keyboard and assigns it to a variable.

```py
#!/bin/bash

echo -n "Enter some text > "
read text
echo "You entered: $text"

# the script in action:
$ read_demo.bash
Enter some text > this is some text
You entered: this is some text
```

`-n` given to the echo command causes it to keep the cursor on the same line; i.e., it does not output a linefeed at the end of the prompt.

invoke the `read` command with "text" as its argument: wait for the user to type something followed by a carriage return (the Enter key) and then assign whatever was typed to the variable text.

- If you don't give the read command the name of a variable to assign its input, it will use the environment variable REPLY.

The read command: command line options.
- The `-t` option followed by a number of seconds provides an automatic timeout for the read command. the read command will give up after the specified seconds if no response has been received from the user.
  - could be used in the case of a script that must continue (perhaps resorting to a default response) even if the user does not answer the prompts.

        #!/bin/bash

        echo -n "Hurry up and type something! > "
        if read -t 3 response; then
            echo "Great, you made it in time!"
        else
            echo "Sorry, you are too slow!"
        fi

- The `-s` option causes the user's typing not to be displayed.
  - useful when you are asking the user to type in a password or other confidential information.

## Arithmetic
The shell provides features for integer arithmetic.
- integer: whole numbers like 1, 2, 458, -2859.
- does not mean fractional numbers like 0.5, .333, or 3.1415.
  - If you must deal with fractional numbers, there is a separate program called `bc` which provides an arbitrary precision calculator language. It can be used in shell scripts

1. to use the command line as a primitive calculator: `$ echo $((2+2))`
    - surround an arithmetic expression with `the double parentheses`, the shell will perform `arithmetic expansion`.
    -  whitespace is ignored:

            [me@linuxbox me]$ echo $((2+2))
            4
            [me@linuxbox me]$ echo $(( 2+2 ))
            4
            [me@linuxbox me]$ echo $(( 2 + 2 ))
            4

2. The shell can perform a variety of common (and not so common) arithmetic operations.

        #!/bin/bash
        first_num=0
        second_num=0

        echo -n "Enter the first number --> "
        read first_num
        echo -n "Enter the second number -> "
        read second_num

        echo "first number + second number = $((first_num + second_num))"
        echo "first number - second number = $((first_num - second_num))"
        echo "first number * second number = $((first_num * second_num))"
        echo "first number / second number = $((first_num / second_num))"
        echo "first number % second number = $((first_num % second_num))"
        echo "first number raised to the"
        echo "power of the second number   = $((first_num ** second_num))"

- the leading `$` is not needed to reference `variables inside the arithmetic expression` such as "first_num + second_num".
- Numbers that get too large overflow like the odometer in a car when you exceed the number of miles it was designed to count. It starts over but first it goes through all the negative numbers because of how integers are represented in memory.
- Division by zero (which is mathematically invalid) does cause an error.

- `%` symbol represents remainder (also known as modulo), performs division but instead of returning a quotient like division, it returns the remainder.
  - For example, when a remainder operation returns zero, it indicates that the first number is an exact multiple of the second. This can be very handy:

          #!/bin/bash
          number=0

          echo -n "Enter a number > "
          read number

          echo "Number is $number"
          if [ $((number % 2)) -eq 0 ]; then
              echo "Number is even"
          else
              echo "Number is odd"
          fi

Or, in this program that formats an arbitrary number of seconds into hours and minutes:

    #!/bin/bash
    seconds=0
    echo -n "Enter number of seconds > "
    read seconds

    hours=$((seconds / 3600))
    seconds=$((seconds % 3600))
    minutes=$((seconds / 60))
    seconds=$((seconds % 60))

    echo "$hours hour(s) $minutes minute(s) $seconds second(s)"


# Flow Control

## `case word in patterns ) commands ;; esac`
- `if` command: alter program flow based on a command's exit status.
- In programming terms, this type of program flow is called `branching` because it is like traversing a tree. You come to a fork in the tree and the evaluation of a condition determines which branch you take.
- more complex kind of branching called `a case`. A case is multiple-choice branch. Unlike the simple branch, where you take one of two possible paths, a case supports several possible outcomes based on the evaluation of a value.

1. construct this type of branch with multiple `if` statements.

        #!/bin/bash
        echo -n "Enter a number between 1 and 3 inclusive > "
        read character

        if [ "$character" = "1" ]; then
            echo "You entered one."
        elif [ "$character" = "2" ]; then
            echo "You entered two."
        elif [ "$character" = "3" ]; then
            echo "You entered three."
        else
            echo "You did not enter a number between 1 and 3."
        fi


2. built-in command `case` can be used to construct an equivalent program:

        #!/bin/bash
        echo -n "Enter a number between 1 and 3 inclusive > "
        read character

        case $character in
            1 ) echo "You entered one."
                ;;
            2 ) echo "You entered two."
                ;;
            3 ) echo "You entered three."
                ;;
            * ) echo "You did not enter a number between 1 and 3."
        esac

The `case` command has the following form:

      case word in
          patterns ) commands ;;
      esac

- `case` selectively executes statements if `word` matches a `pattern`. You can have any number of patterns and statements.
- Patterns can be literal text or wildcards.
- You can have multiple patterns separated by the "|" character.

      #!/bin/bash
      echo -n "Type a digit or a letter > "
      read character

      case $character in
                                      # Check for letters
          [[:lower:]] | [[:upper:]] ) echo "You typed the letter $character"
                                      ;;

                                      # Check for digits
          [0-9] )                     echo "You typed the digit $character"
                                      ;;

                                      # Check for anything else
          * )                         echo "You did not type a letter or a digit"
      esac

pattern `*` :
- will match anything, so it is used to catch cases that did not match previous patterns.
- Inclusion of `*` at the end is wise, can be `used to detect invalid input`.

## Loops
Looping is repeatedly executing a section of your program based on the exit status of a command.
- The shell provides three commands for looping: `while`, `until` and `for`. We are going to cover while and until in this lesson and for in a upcoming lesson.

### `while [ true ]; do xx done`
causes a block of code to be executed over and over, `as long as the exit status of a specified command is true`.

- example of a program that counts from zero to nine:

      #!/bin/bash
      number=0
      while [ "$number" -lt 10 ]; do
          echo "Number = $number"
          number=$((number + 1))
      done

    - create a variable number, initialize its value to 0
    - start the while loop. specified a command to test the value of number, see if number has a value less than 10.
    - Notice the word `do` and `done`. These enclose the block of code that will be repeated as long as the exit status remains zero, true.

In most cases, the block of code that repeats must do something that will eventually change the exit status, otherwise you will have what is called an `endless loop`.

  - In the example, the repeating block of code outputs the value of number (the echo command on line 5) and increments number by one on line 6.
  - Each time the block of code is completed, the test command's exit status is evaluated again. After the tenth iteration of the loop, number has been incremented ten times and the test command will terminate with a non-zero exit status.
  - At that point, the program flow resumes with the statement following the word `done`. Since `done` is the last line of our example, the program ends.

###  `until [ false ]; do xx done`
works exactly the same way, except the block of code is repeated `as long as the specified command's exit status is false`.

    #!/bin/bash
    number=0

    until [ "$number" -ge 10 ]; do
        echo "Number = $number"
        number=$((number + 1))
    done

## Building a Menu
One common way of presenting a user interface for a text based program is by using a `menu`. A `menu` is a list of choices from which the user can pick.

    #!/bin/bash
    selection=
    until [ "$selection" = "0" ]; do
        echo "
        PROGRAM MENU
        1 - Display free disk space
        2 - Display free memory
        0 - exit program
    "
        echo -n "Enter selection: "
        read selection

        echo ""
        case $selection in
            1 ) df ;;
            2 ) free ;;
            0 ) exit ;;
            * ) echo "Please enter 1, 2, or 0"
        esac
    done

- The purpose of the `until` loop in this program is to re-display the menu each time a selection has been completed. The loop will continue until selection is equal to "0," the "exit" choice. Notice how we defend against entries from the user that are not valid choices.

To make this program better
- adding a function that asks the user to press the `Enter` key after each selection has been completed
- and clears the screen before the menu is displayed again. Here is the enhanced example:

      #!/bin/bash

      press_enter()
      {
          echo -en "\nPress Enter to continue"
          read
          clear
      }

      selection=
      until [ "$selection" = "0" ]; do
          echo "
          PROGRAM MENU
          1 - display free disk space
          2 - display free memory

          0 - exit program
      "
          echo -n "Enter selection: "
          read selection
          echo ""
          case $selection in
              1 ) df ; press_enter ;;
              2 ) free ; press_enter ;;
              0 ) exit ;;
              * ) echo "Please enter 1, 2, or 0"; press_enter
          esac
      done


## When your computer hangs...
Hanging is when a program suddenly seems to stop and become unresponsive.
in most cases, the program is still running but its program logic is stuck in an endless loop.

Imagine this situation: you have an external device attached to your computer, such as a USB disk drive but you forgot to turn it on. You try and use the device but the application hangs instead. When this happens, you could picture the following dialog going on between the application and the interface for the device:

    Application:    Are you ready?
    Interface:  Device not ready.

    Application:    Are you ready?
    Interface:  Device not ready.

    Application:    Are you ready?
    Interface:  Device not ready.

    Application:    Are you ready?
    Interface:  Device not ready.

and so on, forever.

- Well-written software tries to avoid this situation by instituting a `timeout`: the loop is counting `the number of attempts` or calculating `the amount of time it has waited` for something to happen.
- If the number of tries or the amount of time allowed is exceeded, the loop `exits` and the program generates an `error` and `exits`.


# Positional Parameters
When we last left our script, it looked something like this:

```py
#!/bin/bash
# sysinfo_page - A script to produce a system information HTML file

##### Constants
TITLE="System Information for $HOSTNAME"
RIGHT_NOW=$(date +"%x %r %Z")
TIME_STAMP="Updated on $RIGHT_NOW by $USER"

##### Functions
system_info()
{
    echo "<h2>System release info</h2>"
    echo "<p>Function not yet implemented</p>"
}   # end of system_info

show_uptime()
{
    echo "<h2>System uptime</h2>"
    echo "<pre>"
    uptime
    echo "</pre>"
}   # end of show_uptime

drive_space()
{
    echo "<h2>Filesystem space</h2>"
    echo "<pre>"
    df
    echo "</pre>"
}   # end of drive_space

home_space()
{
    # Only the superuser can get this information

    if [ "$(id -u)" = "0" ]; then
        echo "<h2>Home directory space by user</h2>"
        echo "<pre>"
        echo "Bytes Directory"
        du -s /home/* | sort -nr
        echo "</pre>"
    fi
}   # end of home_space


##### Main
cat <<- _EOF_
  <html>
  <head>
      <title>$TITLE</title>
  </head>
  <body>
      <h1>$TITLE</h1>
      <p>$TIME_STAMP</p>
      $(system_info)
      $(show_uptime)
      $(drive_space)
      $(home_space)
  </body>
  </html>
_EOF_
```

more features I want to add:
- want to specify the name of the output file on the command line, as well as set a default output file name if no name is specified.
- want to offer an interactive mode that will prompt for a file name and warn the user if the file exists and prompt the user to overwrite it.

Naturally, we want to have a help option that will display a usage message.
All of these features involve using command line options and arguments.
- To handle options on the command line, we use a facility in the shell called `positional parameters`: a series of special variables ($0 through $9) that contain the contents of the command line.

Let's imagine the following command line:

[me@linuxbox me]$ some_program word1 word2 word3

If some_program were a bash shell script, we could read each item on the command line because the positional parameters contain the following:

    $0 would contain "some_program"
    $1 would contain "word1"
    $2 would contain "word2"
    $3 would contain "word3"

Here is a script you can use to try this out:

    #!/bin/bash

    echo "Positional Parameters"
    echo '$0 = ' $0
    echo '$1 = ' $1
    echo '$2 = ' $2
    echo '$3 = ' $3

## Detecting Command Line Arguments
Often, you will want to check to see if you have arguments on which to act. There are a couple of ways to do this.

First, simply check to see if $1 contains anything like so:

    #!/bin/bash

    if [ "$1" != "" ]; then
        echo "Positional parameter 1 contains something"
    else
        echo "Positional parameter 1 is empty"
    fi

Second, the shell maintains a variable called `$#` that contains the number of items on the command line in addition to the name of the command ($0).

    #!/bin/bash

    if [ $# -gt 0 ]; then
        echo "Your command line contains $# arguments"
    else
        echo "Your command line contains no arguments"
    fi

## Command Line Options
construct a while loop relies on `shift`.
- `shift` is a `shell builtin 执行内建的函数；内键指令 ` that operates on the `positional parameters`.
  - Each time you invoke `shift`, it "shifts" all the positional parameters down by one.
  - $2 becomes $1, $3 becomes $2, $4 becomes $3, and so on.
  - shift(shift 1) 命令每执行一次，变量的个数($#)减一（之前的$1变量被销毁,之后的$2就变成了$1），而变量值提前一位。

```py
1.
#!/bin/bash
# run.sh
echo "You start with $# positional parameters"
while [ "$1" != "" ]; do                             # Loop until all parameters are used up
    echo "Parameter 1 equals $1"
    echo "You now have $# positional parameters"
    shift                                            # Shift all the parameters down by one
done

J:Desktop luo$ nano run.sh
J:Desktop luo$ chmod 755 run.sh
J:Desktop luo$ run.sh a b c

You start with 3 positional parameters
Parameter 1 equals a
You now have 3 positional parameters
Parameter 1 equals b
You now have 2 positional parameters
Parameter 1 equals c
You now have 1 positional parameters
J:Desktop luo$

2.
# 示例： 依次读取输入的参数并打印参数个数：
#!/bin/bash
# run.sh
while [ "$#" != 0 ] ; do

    echo "第一个参数为：$1, 参数个数为：$#"
    shift
done

# 输入命令
$ run.sh a b c d e f
# 结果：

第一个参数为：a,参数个数为：6
第一个参数为：b,参数个数为：5
第一个参数为：c,参数个数为：4
第一个参数为：d,参数个数为：3
第一个参数为：e,参数个数为：2
第一个参数为：f,参数个数为：1
```


many programs, particularly ones from the GNU Project, support both short and long command line options.
- For example:
  - to display a help message for many of these programs, either the `-h` option or the longer `--help` option.
- Long option names are typically preceded by a double dash.

Here is the code we will use to process our command line:

```py
interactive=
filename=~/sysinfo_page.html

while [ "$1" != "" ]; do
    case $1 in
        -f | --file )           shift
                                filename=$1
                                ;;
        -i | --interactive )    interactive=1
                                ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done
```

- The first two lines:
  - set the variable `interactive` to be empty: indicate that the `interactive mode` has not been requested.
  - set the variable `filename` to contain a default file name. If nothing else is specified on the command line, this file name will be used.
  - After these two variables are set, we have default settings, in case the user does not specify any options.
- loop with `shift`
- Getting an Option's Argument
  - "-f" option requires a valid file name as an argument. We use `shift` again to get the next item from the command line and assign it to filename. Later we will have to check the content of filename to make sure it is valid.

## Integrating the Command Line Processor into the Script
We will have to move a few things around and add a usage function to get this new routine integrated into our script. We'll also add some test code to verify that the command line processor is working correctly. Our script now looks like this:

```py
#!/bin/bash

# sysinfo_page - A script to produce a system information HTML file

##### Constants

TITLE="System Information for $HOSTNAME"
RIGHT_NOW=$(date +"%x %r %Z")
TIME_STAMP="Updated on $RIGHT_NOW by $USER"

##### Functions

system_info()
{
    echo "<h2>System release info</h2>"
    echo "<p>Function not yet implemented</p>"

}   # end of system_info


show_uptime()
{
    echo "<h2>System uptime</h2>"
    echo "<pre>"
    uptime
    echo "</pre>"

}   # end of show_uptime


drive_space()
{
    echo "<h2>Filesystem space</h2>"
    echo "<pre>"
    df
    echo "</pre>"

}   # end of drive_space


home_space()
{
    # Only the superuser can get this information

    if [ "$(id -u)" = "0" ]; then
        echo "<h2>Home directory space by user</h2>"
        echo "<pre>"
        echo "Bytes Directory"
        du -s /home/* | sort -nr
        echo "</pre>"
    fi

}   # end of home_space


write_page()
{
    cat <<- _EOF_
    <html>
        <head>
        <title>$TITLE</title>
        </head>
        <body>
        <h1>$TITLE</h1>
        <p>$TIME_STAMP</p>
        $(system_info)
        $(show_uptime)
        $(drive_space)
        $(home_space)
        </body>
    </html>
_EOF_

}

usage()
{
    echo "usage: sysinfo_page [[[-f file ] [-i]] | [-h]]"
}


##### Main

interactive=
filename=~/sysinfo_page.html

while [ "$1" != "" ]; do
    case $1 in
        -f | --file )           shift
                                filename=$1
                                ;;
        -i | --interactive )    interactive=1
                                ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done


# Test code to verify command line processing

if [ "$interactive" = "1" ]; then
	echo "interactive is on"
else
	echo "interactive is off"
fi
echo "output file = $filename"


# Write page (comment out until testing is complete)

# write_page > $filename
```

### Adding Interactive Mode
The `interactive mode` is implemented with the following code:

```py
if [ "$interactive" = "1" ]; then

    response=

    echo -n "Enter name of output file [$filename] > "
    read response
    if [ -n "$response" ]; then
        filename=$response
    fi

    if [ -f $filename ]; then
        echo -n "Output file exists. Overwrite? (y/n) > "
        read response
        if [ "$response" != "y" ]; then
            echo "Exiting program."
            exit 1
        fi
    fi
fi
```

- First, we check if the interactive mode is on, otherwise we don't have anything to do.
- Next, we ask the user for the file name. Notice the way the prompt is worded:
  - echo -n "Enter name of output file `[$filename] >` "
- We display the current value of filename since, the way this routine is coded
  - if the user just presses the enter key, the default value of filename will be used.
  - This is accomplished in the next two lines where the value of response is checked.
  - If response is not empty, then `filename` is assigned the value of response.
  - Otherwise, filename is left unchanged, preserving its default value.
- After we have the name of the output file, we check if it already exists.
  - If it does, we prompt the user.
  - If the user response is not "y," we give up and exit, otherwise we can proceed.

---

# Flow Control

## `for variable in words; do commands done`
the remaining `flow control` statement, `for`.
- Like while and until, for is used to construct loops. for works like this:

      for variable in words; do
          commands
      done

- `for` assigns a `word` from the list of `words` to the specified `variable`, executes the commands, and repeats this over and over until all the words have been used up.

```
#!/bin/bash

for i in word1 word2 word3; do
    echo $i
done
```

- the variable `i` is assigned the string "word1", then the statement `echo $i` is executed,
- then the variable `i` is assigned the string "word2", and the statement `echo $i` is executed,
- and so on, until all the words in the list of words have been assigned.

The interesting thing about for is the many ways you can construct the list of words.
- All kinds of expansions can be used.

```py
#!/bin/bash
count=0
for i in $(cat ~/.bash_profile); do
    count=$((count + 1))
    echo "Word $count ($i) contains $(echo -n $i | wc -c) characters"
done
# take the file .bash_profile and count the number of words in the file and the number of characters in each word.
```

So what's this got to do with positional parameters? Well, one of the features of for is that it can use the positional parameters as the list of words:

```
#!/bin/bash

for i in "$@"; do
    echo $i
done
```

The shell variable `$@` contains the list of command line arguments.
- This technique is often used to process a `list of files` on the command line.


an example:

```py
#!/bin/bash

for filename in "$@"; do
    result=
    if [ -f "$filename" ]; then
        result="$filename is a regular file"
    elif [ -d "$filename" ]; then
        result="$filename is a directory"
    else
        result="$filename is not a exist file or directory"
    fi

#    else
#        if [ -d "$filename" ]; then
#            result="$filename is a directory"
#        fi
#    fi

    if [ -w "$filename" ]; then
        result="$result and it is writable"
    else
        result="$result and it is not writable"
    fi
    echo "$result"
done

# Try this script
$ try *
1 is a regular file and it is writable
2017.pdf is a regular file and it is writable

$ try dy7 hryf
dy7 is not a exist file or directory and it is not writable
hryf is not a exist file or directory and it is not writable
```

another example script.
```py
This one compares the files in two directories and lists which files in the first directory are missing from the second.

#!/bin/bashbash
# cmp_dir - program to compare two directories

# Check for required arguments
#if [ $# -ne 2 ]; then
    echo "usage: $0 directory_1 directory_2" 1>&2
    exit 1
fi

# Make sure both arguments are directories
if [ ! -d $1 ]; then
    echo "$1 is not a directory!" 1>&2
    exit 1
fi

if [ ! -d $2 ]; then
    echo "$2 is not a directory!" 1>&2
    exit 1
fi

# Process each file in directory_1, comparing it to directory_2
missing=0
for filename in $1/*; do
    fn=$(basename "$filename")
    if [ -f "$filename" ]; then
        if [ ! -f "$2/$fn" ]; then
            echo "$fn is missing from $2"
            missing=$((missing + 1))
        fi
    fi
done
echo "$missing files missing"
```

to improve the home_space function in our script to output more information.
```py
home_space()
{
    # Only the superuser can get this information

    if [ "$(id -u)" = "0" ]; then
    echo "<h2>Home directory space by user</h2>"
    echo "<pre>"
    echo "Bytes Directory"
        du -s /home/* | sort -nr
    echo "</pre>"
    fi

}   # end of home_space
```

Here is the new version:

```py
home_space()
{
    echo "<h2>Home directory space by user</h2>"
    echo "<pre>"
    format="%8s%10s%10s   %-s\n"
    printf "$format" "Dirs" "Files" "Blocks" "Directory"
    printf "$format" "----" "-----" "------" "---------"
    if [ $(id -u) = "0" ]; then
        dir_list="/home/*"
    else
        dir_list=$HOME
    fi
    for home_dir in $dir_list; do
        total_dirs=$(find $home_dir -type d | wc -l)
        total_files=$(find $home_dir -type f | wc -l)
        total_blocks=$(du -s $home_dir)
        printf "$format" $total_dirs $total_files $total_blocks
    done
    echo "</pre>"
}   # end of home_space
```

- `printf`: to produce formatted output according to the contents of a format string. printf comes from the C programming language and has been implemented in many other programming languages including C++, Perl, awk, java, PHP, and of course, bash.
- `find`: to search for files or directories that meet specific criteria. In the home_space function, we use find to list the directories and regular files in each home directory.
- `wc`: count the number of files and directories found.

deal with the problem of superuser access.
- test for the superuser with id
- according to the outcome of the test, assign different strings to the `variable dir_list`, the list of words for the for loop.
- This way, if an ordinary user runs the script, only his/her home directory will be listed.


Another function that can use a for loop is our unfinished system_info function. We can build it like this:
```py
system_info()
{
    # Find any release files in /etc
    if ls /etc/*release 1>/dev/null 2>&1; then
        echo "<h2>System release info</h2>"
        echo "<pre>"
        for i in /etc/*release; do
            # Since we can't be sure of the length of the file,
            # only display the first line.
            head -n 1 $i
        done
        uname -orp
        echo "</pre>"
    fi
}   # end of system_info
```

- first determine if there are any `release files` to process.
  - The `release files` contain the name of the vendor and the version of the distribution.
  - They are located in the `/etc directory.`
  - To detect them, perform `ls` command and throw away all of its output.
  - in the exit status. It will be `true` if any files are found.
- Next, we output the HTML for this section of the page, since we now know that there are release files to process.
  - To process the files, we start a `for` loop to act on each one.
  - Inside the loop, we use the `head` command to return the first line of each file.
- Finally, we use the `uname` command with the `o`, `r`, and `p` options to obtain some additional information from the system.



http://linuxcommand.org/lc3_wss0140.php












.
