---
title: Find command in Linux
author: dyl4n
date: 2023-03-06 00:30:23 +0700
categories: [Operating System]
tags: [linux commands]
render_with_liquid: false
comments: true
image:
  path: /thumbnails/find-command.png
published: false
---

There's a powerful command that helps us to search for files and directories in Linux called `find`. Let's discuss and learn about it!

## Basic syntax of the `Find` command

```bash
find /path/ -type f -name file-to-search
```

Where,

- `/path` is the path where file is expected to be found. This is the starting point to search files. The path can also be `/` or `.` which represent root and current directory, respectively.
- `-type` is the type of the file that you want to search for. Learn more about [Linux File Types](https://quocdat.me/2023/03/05/linux-file-types)
- `-name` is the name of the file type that you want to search.

![find manual](https://user-images.githubusercontent.com/98354414/223011836-225762f9-f131-4729-b671-7c361f810580.png)

## Some options of `Find` command

|  **Option**   | **Example**                               | **Description**                             |
| :-----------: | ----------------------------------------- | ------------------------------------------- |
|   **-type**   | find . -type d                            | Find only directories                       |
|   **-name**   | find . -type f -name "\*.txt"             | Find file by name                           |
|  **-iname**   | find . -type f -iname "hello"             | Find file by name (case-insensitive)        |
|   **-size**   | find . -size +1G                          | Find files larger than 1G                   |
|   **-user**   | find . -type d -user jack                 | Find jack's file                            |
|  **-regex**   | find /var -regex '._/tmp/._[0-9]\*.file'  | Using Regex with find. See regex            |
| **-maxdepth** | find . -maxdepth 1 -name "a.txt"          | In the current directory and subdirectories |
| **-mindepth** | find / -mindepth 3 -maxdepth 5 -name pass | Between sub-directory level 2 and 4         |

## Examples of `Find` command

### Names

Find files using name in current directory

```bash
$ find . -name foo.txt
```

Find files under home directory

```bash
$ find /opt /usr /var -name foo.scala -type f
```

Find directories using name

```bash
$ find / -type d -name foo
```

### Size

Find all bigger than 10MB files

```bash
$ find / -size +10M
```

Find all smaller than 10MB files

```bash
$ find / -size -10M
```

Find all files that are exactly 10M

```bash
$ find / -size 10M
```

Find Size between 100MB and 1GB

```bash
$ find / -size +100M -size -1G
```

### Permissions

Find the files whose permissions are 777.

```bash
$ find . -type f -perm 0777
```

Find Read Only files.

```bash
$ find / -perm /u=r
```

### Owners and Groups

Find single file based on user

```bash
$ find / -user root -name foo.txt
```

Find all files based on group

```bash
$ find /home -group developer
```

### Multiple filenames

Find files with .sh and .txt extensions

```bash
$ find . -type f \( -name "*.sh" -o -name "*.txt" \)
```

### Multiple dirs

Find files with multiple dirs

```bash
$ find /opt /usr /var -name foo.scala -type f
```

### Empty

Delete all empty files in a directory

```bash
$ find . -type f -empty -delete
```

## Find Date and Time

|               |                                                      Means |
| ------------- | ---------------------------------------------------------: |
| atime         |                        access time (last time file opened) |
| mtime         |       modified time (last time file contents was modified) |
| ctime         |            changed time (last time file inode was changed) |
|               |                                               **Examples** |
| -mtime +0     |                         Modified greater than 24 hours ago |
| -mtime 0      |                         Modified between now and 1 day ago |
| -mtime -1     |            Modified less than 1 day ago (same as -mtime 0) |
| -mtime 1      |                       Modified between 24 and 48 hours ago |
| -mtime +1     |                            Modified more than 48 hours ago |
| -mtime +1w    |                         Last modified more than 1 week ago |
| -atime 0      |                 Last accessed between now and 24 hours ago |
| -atime +0     |                            Accessed more than 24 hours ago |
| -atime 1      |                       Accessed between 24 and 48 hours ago |
| -atime +1     |                            Accessed more than 48 hours ago |
| -atime -1     |         Accessed less than 24 hours ago (same as -atime 0) |
| -ctime -6h30m | File status changed within the last 6 hours and 30 minutes |

## Find And

### Find and Delete

Find and remove multiple files

```bash
$ find . -type f -name "*.mp3" -exec rm -f {} \;
```

### Find and Replace

Find all files and modify the content const to let

```bash
$ find ./ -type f -exec sed -i 's/const/let/g' {} \;
```

### Find and Rename

Find and suffix (added .bak)

```bash
$ find . -type f -name 'file*' -exec mv {} {}.bak\;
```

### Find and move

Find and move it to a specific directory (/tmp/music)

```bash
$ find . -name '*.mp3' -exec mv {} /tmp/music \;
```

### Find and copy

Find matching files and copy to a specific directory (/tmp/backup)

```bash
$ find . -name '*2020*.xml' -exec cp -r "{}" /tmp/backup \;
```

### Find and concatenate

Merge all csv files in the download directory into merged.csv

```bash
$ find download -type f -iname '*.csv' | xargs cat > merged.csv
```

### Find and sort

Find and sort in ascending

```bash
$ find . -type f | sort
```

### Find and chmod

Find files and set permissions to 644.

```bash
$ find / -type f -perm 0777 -print -exec chmod 644 {} \;
```

### Find and compress

Find all .java files and compress it into java.tar

```bash
$ find . -type f -name "*.java" | xargs tar cvf java.tar
```
