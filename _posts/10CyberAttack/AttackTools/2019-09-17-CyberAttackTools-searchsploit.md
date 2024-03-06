---
title: Meow's Testing Tools - searchsploit
date: 2019-09-17 11:11:11 -0400
categories: [10CyberAttack, CyberAttackTools]
tags: [CyberAttack, CyberAttackTools]
toc: true
image:
---

[toc]

---

# searchsploit

command line search tool for Exploit-DB that also allows you to take a copy of Exploit Database with you, everywhere you go. SearchSploit gives you the power to perform detailed off-line searches through your locally checked-out copy of the repository. This capability is particularly useful for security assessments on segregated or air-gapped networks without Internet access.

Many exploits contain links to binary files that are not included in the standard repository but can be found in our Exploit Database Binary Exploits repository instead. If you anticipate you will be without Internet access on an assessment, ensure you check out both repositories for the most complete set of data.

---

## Install SearchSploit

```bash
# Install SearchSploit
$ sudo apt update && sudo apt -y install exploitdb

# Updating SearchSploit
$ searchsploit -u
$ sudo apt update && sudo apt -y full-upgrade
```


---


## Using SearchSploit

```bash
# Help Screen
# By using -h, see all the features and options that are available to you:

$ searchsploit -h
  Usage: searchsploit [options] term1 [term2] ... [termN]

==========
 Examples
==========
  searchsploit afd windows local
  searchsploit -t oracle windows
  searchsploit -p 39446
  searchsploit linux kernel 3.2 --exclude="(PoC)|/dos/"
  searchsploit -s Apache Struts 2.0.0
  searchsploit linux reverse password
  searchsploit -j 55555 | json_pp

  For more examples, see the manual: https://www.exploit-db.com/searchsploit

=========
 Options
=========
## Search Terms
   -c, --case     [Term]      Perform a case-sensitive search (Default is inSEnsITiVe)
   -e, --exact    [Term]      Perform an EXACT & order match on exploit title (Default is an AND match on each term) [Implies "-t"]
                                e.g. "WordPress 4.1" would not be detect "WordPress Core 4.1")
   -s, --strict               Perform a strict search, so input values must exist, disabling fuzzy search for version range
                                e.g. "1.1" would not be detected in "1.0 < 1.3")
   -t, --title    [Term]      Search JUST the exploit title (Default is title AND the files path)
       --exclude="term"       Remove values from results. By using "|" to separate, you can chain multiple values
                                e.g. --exclude="term1|term2|term3"

## Output
   -j, --json     [Term]      Show result in JSON format
   -o, --overflow [Term]      Exploit titles are allowed to overflow their columns
   -p, --path     [EDB-ID]    Show the full path to an exploit (and also copies the path to the clipboard if possible)
   -v, --verbose              Display more information in output
   -w, --www      [Term]      Show URLs to Exploit-DB.com rather than the local path
       --id                   Display the EDB-ID value rather than local path
       --colour               Disable colour highlighting in search results

## Non-Searching
   -m, --mirror   [EDB-ID]    Mirror (aka copies) an exploit to the current working directory
   -x, --examine  [EDB-ID]    Examine (aka opens) the exploit using $PAGER

## Non-Searching
   -h, --help                 Show this help screen
   -u, --update               Check for and install any exploitdb package updates (brew, deb & git)

## Automation
       --nmap     [file.xml]  Checks all results in Nmaps XML output with service version
                                e.g.: nmap [host] -sV -oX file.xml

=======
 Notes
=======
 * You can use any number of search terms
 * By default, search terms are not case-sensitive, ordering is irrelevant, and will search between version ranges
   * Use '-c' if you wish to reduce results by case-sensitive searching
   * And/Or '-e' if you wish to filter results by using an exact match
   * And/Or '-s' if you wish to look for an exact version match
 * Use '-t' to exclude the files path to filter the search results
   * Remove false positives (especially when searching using numbers - i.e. versions)
 * When using '--nmap', adding '-v' (verbose), it will search for even more combinations
 * When updating or displaying help, search terms will be ignored
```


## Basic Search
```bash
$ searchsploit afd windows local
--------------------------------------------------------------------------------------- ---------------------------------
 Exploit Title                                                                         |  Path
--------------------------------------------------------------------------------------- ---------------------------------
Microsoft Windows (x86) - 'afd.sys' Local Privilege Escalation (MS11-046)              | windows_x86/local/40564.c
Microsoft Windows - 'afd.sys' Local Kernel (PoC) (MS11-046)                            | windows/dos/18755.c
Microsoft Windows - 'AfdJoinLeaf' Local Privilege Escalation (MS11-080) (Metasploit)   | windows/local/21844.rb
Microsoft Windows 7 (x64) - 'afd.sys' Dangling Pointer Privilege Escalation (MS14-040) | windows_x86-64/local/39525.py
Microsoft Windows 7 (x86) - 'afd.sys' Dangling Pointer Privilege Escalation (MS14-040) | windows_x86/local/39446.py
Microsoft Windows XP - 'afd.sys' Local Kernel Denial of Service                        | windows/dos/17133.c
Microsoft Windows XP/2003 - 'afd.sys' Local Privilege Escalation (K-plugin) (MS08-066) | windows/local/6757.txt
Microsoft Windows XP/2003 - 'afd.sys' Local Privilege Escalation (MS11-080)            | windows/local/18176.py
--------------------------------------------------------------------------------------- ---------------------------------

# Note, SearchSploit uses an AND operator, not an OR operator.
# The more terms that are used, the more results will be filtered out.
# Pro Tip: Do not use abbreviations (use SQL Injection, not SQLi).
# Pro Tip: If you are not receiving the expected results, try searching more broadly by using more general terms (use Kernel 2.6 or Kernel 2.x, not Kernel 2.6.25).
```

---

# Title Searching

```bash
# By default, searchsploit will check BOTH the title of the exploit as well as the path.
# Depending on the search criteria, this may bring up false positives (especially when searching for terms that match platforms and version numbers).
# Searches can be restricted to the titles by using the -t option:

$ searchsploit -t oracle windows
--------------------------------------------------------------------------------------- ---------------------------------
 Exploit Title                                                                         |  Path
--------------------------------------------------------------------------------------- ---------------------------------
Oracle 10g (Windows x86) - 'PROCESS_DUP_HANDLE' Local Privilege Escalation             | windows_x86/local/3451.c
Oracle 9i XDB (Windows x86) - FTP PASS Overflow (Metasploit)                           | windows_x86/remote/16731.rb
Oracle 9i XDB (Windows x86) - FTP UNLOCK Overflow (Metasploit)                         | windows_x86/remote/16714.rb
Oracle 9i XDB (Windows x86) - HTTP PASS Overflow (Metasploit)                          | windows_x86/remote/16809.rb
Oracle MySQL (Windows) - FILE Privilege Abuse (Metasploit)                             | windows/remote/35777.rb
Oracle MySQL (Windows) - MOF Execution (Metasploit)                                    | windows/remote/23179.rb
Oracle MySQL for Microsoft Windows - Payload Execution (Metasploit)                    | windows/remote/16957.rb
Oracle VirtualBox Guest Additions 5.1.18 - Unprivileged Windows User-Mode Guest Code Do| multiple/dos/41932.cpp
Oracle VM VirtualBox 5.0.32 r112930 (x64) - Windows Process COM Injection Privilege Esc| windows_x86-64/local/41908.txt
--------------------------------------------------------------------------------------- ---------------------------------

$ searchsploit oracle windows | wc -l
100
# If we did not use -t, we would have 94 (6 lines are in the heading/footer) results, rather than 9.
```

---

# Removing Unwanted Results

```bash
# remove unwanted results
# remove multiple terms by separating the value with a | (pipe).

$ searchsploit linux kernel 3.2 --exclude="(PoC)|/dos/"
--------------------------------------------------------------------------------------- ---------------------------------
 Exploit Title                                                                         |  Path
--------------------------------------------------------------------------------------- ---------------------------------
Linux Kernel (Solaris 10 / < 5.10 138888-01) - Local Privilege Escalation              | solaris/local/15962.c
Linux Kernel 2.6.22 < 3.9 (x86/x64) - 'Dirty COW /proc/self/mem' Race Condition Privile| linux/local/40616.c
Linux Kernel 2.6.22 < 3.9 - 'Dirty COW /proc/self/mem' Race Condition Privilege Escalat| linux/local/40847.cpp
Linux Kernel 2.6.22 < 3.9 - 'Dirty COW PTRACE_POKEDATA' Race Condition (Write Access Me| linux/local/40838.c
Linux Kernel 2.6.22 < 3.9 - 'Dirty COW' 'PTRACE_POKEDATA' Race Condition Privilege Esca| linux/local/40839.c
Linux Kernel 2.6.22 < 3.9 - 'Dirty COW' /proc/self/mem Race Condition (Write Access Met| linux/local/40611.c
Linux Kernel 2.6.39 < 3.2.2 (Gentoo / Ubuntu x86/x64) - 'Mempodipper' Local Privilege E| linux/local/18411.c
Linux Kernel 2.6.39 < 3.2.2 (x86/x64) - 'Mempodipper' Local Privilege Escalation (2)   | linux/local/35161.c
Linux Kernel 3.0 < 3.3.5 - 'CLONE_NEWUSER|CLONE_FS' Local Privilege Escalation         | linux/local/38390.c
Linux Kernel 3.14-rc1 < 3.15-rc4 (x64) - Raw Mode PTY Echo Race Condition Privilege Esc| linux_x86-64/local/33516.c
Linux Kernel 3.2.0-23/3.5.0-23 (Ubuntu 12.04/12.04.1/12.04.2 x64) - 'perf_swevent_init'| linux_x86-64/local/33589.c
Linux Kernel 3.2.x - 'uname()' System Call Local Information Disclosure                | linux/local/37937.c
Linux Kernel 3.4 < 3.13.2 (Ubuntu 13.04/13.10 x64) - 'CONFIG_X86_X32=y' Local Privilege| linux_x86-64/local/31347.c
Linux Kernel 3.4 < 3.13.2 (Ubuntu 13.10) - 'CONFIG_X86_X32' Arbitrary Write (2)        | linux/local/31346.c
Linux Kernel 4.8.0 UDEV < 232 - Local Privilege Escalation                             | linux/local/41886.c
Linux Kernel < 3.16.1 - 'Remount FUSE' Local Privilege Escalation                      | linux/local/34923.c
Linux Kernel < 3.16.39 (Debian 8 x64) - 'inotfiy' Local Privilege Escalation           | linux/local/44302.c
Linux Kernel < 3.2.0-23 (Ubuntu 12.04 x64) - 'ptrace/sysret' Local Privilege Escalation| linux_x86-64/local/34134.c
Linux Kernel < 3.4.5 (Android 4.2.2/4.4 ARM) - Local Privilege Escalation              | arm/local/31574.c
Linux Kernel < 3.5.0-23 (Ubuntu 12.04.2 x64) - 'SOCK_DIAG' SMEP Bypass Local Privilege | linux/local/44299.c
Linux Kernel < 3.8.9 (x86-64) - 'perf_swevent_init' Local Privilege Escalation (2)     | linux_x86-64/local/26131.c
Linux Kernel < 3.8.x - open-time Capability 'file_ns_capable()' Local Privilege Escalat| linux/local/25450.c
Linux kernel < 4.10.15 - Race Condition Privilege Escalation                           | linux/local/43345.c
Linux Kernel < 4.11.8 - 'mq_notify: double sock_put()' Local Privilege Escalation      | linux/local/45553.c
Linux Kernel < 4.13.9 (Ubuntu 16.04 / Fedora 27) - Local Privilege Escalation          | linux/local/45010.c
Linux Kernel < 4.15.4 - 'show_floppy' KASLR Address Leak                               | linux/local/44325.c
Linux Kernel < 4.4.0-116 (Ubuntu 16.04.4) - Local Privilege Escalation                 | linux/local/44298.c
Linux Kernel < 4.4.0-21 (Ubuntu 16.04 x64) - 'netfilter target_offset' Local Privilege | linux/local/44300.c
Linux Kernel < 4.4.0-83 / < 4.8.0-58 (Ubuntu 14.04/16.04) - Local Privilege Escalation | linux/local/43418.c
Linux Kernel < 4.4.0/ < 4.8.0 (Ubuntu 14.04/16.04 / Linux Mint 17/18 / Zorin) - Local P| linux/local/47169.c
--------------------------------------------------------------------------------------- ---------------------------------


$ searchsploit linux kernel 3.2 | wc -l
47

# By doing this, we slim the results down to 30 rather than 41 (6 lines are for the heading/footer)!
# You may of also noticed, "3.2" isn't always visible in the results. That is because SearchSploit by default, will try to detect the version, and then search between any ranges in the title. This behaviour can be disabled by doing -s.
# Pro Tip: By doing: searchsploit linux kernel --exclude="(PoC)|/dos/" | grep ' 3.2' (space before the version), you'll get even "cleaner" output (sorted based on version without any heading/footers).
```

---

# Piping Output (Alternative Method of Removing Unwanted Results)

```bash
# The output from searchsploit can be piped into any other program, which is especially useful when outputting the results in JSON format (using the -j option). With this, it is possible to remove any unwanted exploits by using grep.
# In the following example, we use grep to filter out any "Denial of Service (DoS)" results.
$ searchsploit XnView | grep -v '/dos/'
--------------------------------------------------------------------------------------- ---------------------------------
 Exploit Title                                                                         |  Path
--------------------------------------------------------------------------------------- ---------------------------------
XnView 1.90.3 - '.xpm' Local Buffer Overflow                                           | windows/local/3777.c
XnView 1.92.1 - 'FontName' Slideshow Buffer Overflow                                   | windows/local/5346.pl
XnView 1.92.1 - Command-Line Arguments Buffer Overflow                                 | windows/remote/31405.c
XnView 1.93.6 - '.taac' Local Buffer Overflow                                          | windows/local/5951.c
XnView 1.97.4 - '.MBM' File Remote Heap Buffer Overflow                                | windows/remote/34143.txt
--------------------------------------------------------------------------------------- ---------------------------------

$ searchsploit XnView | wc -l
23
# By piping the search results into grep, we managed to filter the results down to 5 rather than 17 (6 lines are in the heading/footer)!
# Pro Tip: We recommend using "/dos/" with grep rather than "dos" so the filter is applied to the path, rather than the title. Although denial of service entries may not include "dos" in their title, they will nevertheless have "dos" in the path. Removing results based on the path will also ensure you don't inadvertently filter out results that legitimately contain "dos" in their title (i.e.: EDB-ID #24623).
```


---


# Colour Output


```bash
# By default, searchsploit highlights the search terms in the results when they are displayed to the user. This works by inserting invisible characters into the output before and after the colour changes.
# Now, if you were to pipe the output (for example, into grep) and try to match a phrase of both highlighted and non-highlighted text in the output, it would not be successful. This can be solved by using the --colour option (--color works as well).
$ searchsploit xxx --colour
```

---

# Copy To Clipboard

```bash
# So now that we have found the exploit we are looking for, there are various ways to access it quickly.
# By using -p, get some more information about the exploit, as well as copy the complete path to the exploit onto the clipboard:
$ searchsploit 39446
--------------------------------------------------------------------------------------- ---------------------------------
 Exploit Title                                                                         |  Path
--------------------------------------------------------------------------------------- ---------------------------------
Microsoft Windows 7 (x86) - 'afd.sys' Dangling Pointer Privilege Escalation (MS14-040) | windows_x86/local/39446.py
--------------------------------------------------------------------------------------- ---------------------------------

$ searchsploit -p 39446
  Exploit: Microsoft Windows 7 (x86) - 'afd.sys' Dangling Pointer Privilege Escalation (MS14-040)
      URL: https://www.exploit-db.com/exploits/39446/
     Path: /usr/share/exploitdb/exploits/windows_x86/local/39446.py
File Type: Python script, ASCII text executable, with CRLF line terminators
Copied EDB-ID #39446's path to the clipboard.

$ /usr/share/exploitdb/exploits/windows_x86/local/39446.py
```

---


# Copy To Folder

```bash
# We recommend that you do not alter the exploits in your local copy of the database.
# Instead, make a copy of ones that are of interest and use them from a working directory.
# By using the -m option, we are able to select as many exploits we like to be copied into the same folder that we are currently in:
$ searchsploit MS14-040
--------------------------------------------------------------------------------------- ---------------------------------
 Exploit Title                                                                         |  Path
--------------------------------------------------------------------------------------- ---------------------------------
Microsoft Windows 7 (x64) - 'afd.sys' Dangling Pointer Privilege Escalation (MS14-040) | exploits/windows_x86-64/local/39525.py
Microsoft Windows 7 (x86) - 'afd.sys' Dangling Pointer Privilege Escalation (MS14-040) | exploits/windows_x86/local/39446.py
--------------------------------------------------------------------------------------- ---------------------------------
Shellcodes: No Result

$ searchsploit -m 39446 win_x86-64/local/39525.py

  Exploit: Microsoft Windows 7 (x86) - 'afd.sys' Dangling Pointer Privilege Escalation (MS14-040)
      URL: https://www.exploit-db.com/exploits/39446/
     Path: /usr/share/exploitdb/exploits/windows_x86/local/39446.py
File Type: Python script, ASCII text executable, with CRLF line terminators

Copied to: /root/39446.py


  Exploit: Microsoft Windows 7 (x64) - 'afd.sys' Dangling Pointer Privilege Escalation (MS14-040)
      URL: https://www.exploit-db.com/exploits/39525/
     Path: /usr/share/exploitdb/exploits/windows_x86-64/local/39525.py
File Type: Python script, ASCII text executable, with CRLF line terminators

Copied to: /root/39525.py

# You do not have to give the exact EDB-ID value (such as "39446");
# SearchSploit is able to automatically extract it from a path given to it (such as "39525").

```

---

# Exploit-DB Online

```bash
# The Exploit Database repository is the main core of Exploit-DB, making SearchSploit efficient and easy to use.
# However, some of the exploit metadata (such as screenshots, setup files, tags, and vulnerability mappings) are not included.
# To access them, need to check the website. quickly generate the links to exploits of interest by using the -w option:

$ searchsploit WarFTP 1.65 -w
--------------------------------------------------------------------------------------- ------------------------------------------
 Exploit Title                                                                         |  URL
--------------------------------------------------------------------------------------- ------------------------------------------
WarFTP 1.65 (Windows 2000 SP4) - 'USER' Remote Buffer Overflow (Perl)                  | https://www.exploit-db.com/exploits/3482
WarFTP 1.65 (Windows 2000 SP4) - 'USER' Remote Buffer Overflow (Python)                | https://www.exploit-db.com/exploits/3474
WarFTP 1.65 - 'USER' Remote Buffer Overflow                                            | https://www.exploit-db.com/exploits/3570
--------------------------------------------------------------------------------------- ------------------------------------------
```


ref:
- [SearchSploit – The Manual](https://www.exploit-db.com/searchsploit)


.
