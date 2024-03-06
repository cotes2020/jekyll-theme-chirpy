<!-- ---
title: Linux - Creating Aliases
date: 2020-11-01 11:11:11 -0400
categories: [30System, Sysadmin]
tags: [Linux, Sysadmin]
math: true
image:
--- -->

[toc]

---

# Creating Aliases

save ourselves time and effort using an "alias command" to stop typing or copying the same command again and again
- `aliases` makes possible to execute `a command or set of commands` by using a `pre-defined "string"` that we can customize the way we want.


Creating Aliases
- Temporary aliases
  - created in the current terminal session
  - if we close this session or open a new window, aliases will not work.
- Permanent aliases
  - persistent no matter if the current terminal session is ended or even if the host machine is rebooted.
  - The aliases information has to be written and saved into the user’s `shell configuration profile file`
  - macOS: `~/.bash_profile`
  - Linux: `~/.bashrc`
  - For both Linux and MacOS, these user’s shell configuration profile files are typically located in the `$HOME` directory.



---

## Creating Aliases for Docker Commands in Linux


`alias alias_name="command_to_run"`


```bash
$  dir
-bash: dir: command not found

alias dir = "ls -ltr"

$  dir
-rw-r--r--   1 dba  staff  976 Sep  8  2018 README.md
drwxr-xr-x   6 dba  staff  192 Oct  8  2018 Directory_1


$  cd $HOME
$  vi .bash_profile
# Docker aliases (shortcuts)
# List all containers by status using custom format
alias dkpsa='docker ps -a --format "table {{.Names}}\t{{.Image}}\t{{.Status}}"'
# Removes a container, it requires the container name \ ID as parameter
alias dkrm='docker rm -f'
# Removes an image, it requires the image name \ ID as parameter
alias dkrmi='docker rmi'
# Lists all images by repository sorted by tag name
alias dkimg='docker image ls --format "table {{.Repository}}\t{{.Tag}}\t{{.ID}}" | sort'
# Lists all persistent volumes
alias dkvlm='docker volume ls'
# Displays a container log, it requires the image name \ ID as parameter
alias dklgs='docker logs'
# Streams a container log, it requires the image name \ ID as parameter
alias dklgsf='docker logs -f'
# Initiates a session within a container, it requires the image name \ ID as parameter followed by the word "bash"
alias dkterm='docker exec -it'
# Starts a container, it requires the image name \ ID as parameter
alias dkstrt='docker start'
# Stops a container, it requires the image name \ ID as parameter
alias dkstp='docker stop'
~
:wq


$  dkimg
REPOSITORY                       TAG                  IMAGE ID
mcr.microsoft.com/mssql/server   2017-CU11            885d07287041
mcr.microsoft.com/mssql/server   2017-CU12            4095d6d460cd
```
