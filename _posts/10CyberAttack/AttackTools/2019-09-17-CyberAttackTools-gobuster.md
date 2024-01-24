---
title: Meow's Testing Tools - gobuster
date: 2019-09-17 11:11:11 -0400
categories: [10CyberAttack, CyberAttackTools]
tags: [CyberAttack, CyberAttackTools]
toc: true
image:
---

[toc]

---

# gobuster

---


Select list of possible directories and files
- normally located at `/usr/share/dirbuster/wordlists`


```bash
-e is used to print full path of the files.
-u is used to assign target URL, 192.168.1.105 is our target/DVWA.
-w is used to assign wordlist. /usr/share/wordlists/dirb/common.txt is the wordlist location.
-v is used for verbose mode.
-n is used to print with no status codes.



# gobuster
 Usage:
   gobuster [command]
 Available Commands:
   dir         Uses directory/file brutceforcing mode
   dns         Uses DNS subdomain bruteforcing mode
   help        Help about any command
   vhost       Uses VHOST bruteforcing mode
 Flags:
   -h, --help              help for gobuster
   -z, --noprogress        Don't display progress
   -o, --output string     Output file to write results to (defaults to stdout)
   -q, --quiet             Don't print the banner and other noise
   -t, --threads int       Number of concurrent threads (default 10)
   -v, --verbose           Verbose output (errors)
   -w, --wordlist string   Path to the wordlist



FINDING FILES/ DIRECTORIES
On Target side we will be using DVWA (Dam Vulnerable Web Application)
-u is used to assign target URL, 192.168.1.105 is our target/DVWA.
-w is used to assign wordlist. /usr/share/wordlists/dirb/common.txt is the wordlist location.

# gobuster dir -u https://192.168.1.105/dvwa -w /usr/share/wordlists/dirb/common.txt
 Gobuster v3.0.1
 by OJ Reeves (@TheColonial) & Christian Mehlmauer (@FireFart)
 [+] Url:            https://192.168.1.105/dvwa
 [+] Threads:        10
 [+] Wordlist:       /usr/share/wordlists/dirb/common.txt
 [+] Status codes:   200,204,301,302,307,401,403
 [+] User Agent:     gobuster/3.0.1
 [+] Timeout:        10s
 2019/11/01 01:20:19 Starting gobuster
 /.hta (Status: 403)
 /.svn (Status: 301)
 /.htpasswd (Status: 403)
 /.svn/entries (Status: 200)
 /.htaccess (Status: 403)
 /css (Status: 301)
 /images (Status: 301)
 /includes (Status: 301)
 /js (Status: 301)
 2019/11/01 01:20:25 Finished
Above query has scanned all the files & directories on the target URL.




PRINTING FILES WITH FULL PATH
Type gobuster dir -e -u https://192.168.1.105/dvwa -w /usr/share/wordlists/dirb/common.txt
-e is used to print full path of the files.
-u is used to assign target URL 192.168.1.105 is our target.
-w is used to assign wordlist. /usr/share/wordlists/dirb/common.txt is the wordlist location.

# gobuster dir -e -u  https://192.168.1.105/dvwa -w /usr/share/wordlists/dirb/common.txt
 Gobuster v3.0.1
 by OJ Reeves (@TheColonial) & Christian Mehlmauer (@FireFart)
 [+] Url:            https://192.168.1.105/dvwa
 [+] Threads:        10
 [+] Wordlist:       /usr/share/wordlists/dirb/common.txt
 [+] Status codes:   200,204,301,302,307,401,403
 [+] User Agent:     gobuster/3.0.1
 [+] Expanded:       true
 [+] Timeout:        10s
 2019/11/01 01:21:34 Starting gobuster
 https://192.168.1.105/dvwa/.hta (Status: 403)
 https://192.168.1.105/dvwa/.htpasswd (Status: 403)
 https://192.168.1.105/dvwa/.svn (Status: 301)
 https://192.168.1.105/dvwa/.htaccess (Status: 403)
 https://192.168.1.105/dvwa/.svn/entries (Status: 200)
 https://192.168.1.105/dvwa/css (Status: 301)
 https://192.168.1.105/dvwa/images (Status: 301)
 https://192.168.1.105/dvwa/includes (Status: 301)
 https://192.168.1.105/dvwa/js (Status: 301)
 2019/11/01 01:21:39 Finished
Above you can find the full path of the target URL. This query can help to prepare for the initial level of information gathering.



PRINTING OUTPUT USING VERBOSE
Type gobuster dir -u https://192.168.1.105/dvwa -w /usr/share/wordlists/dirb/common.txt -v
-u is used to assign target URL. 192.168.1.105 is our target.
-w is used to assign wordlist. /usr/share/wordlists/dirb/common.txt is the wordlist location.
-v is used for verbose mode.
# gobuster dir -u https://192.168.1.105/dvwa -w /usr/share/wordlists/dirb/common.txt -v
 Gobuster v3.0.1
 by OJ Reeves (@TheColonial) & Christian Mehlmauer (@FireFart)
 [+] Url:            https://192.168.1.105/dvwa
 [+] Threads:        10
 [+] Wordlist:       /usr/share/wordlists/dirb/common.txt
 [+] Status codes:   200,204,301,302,307,401,403
 [+] User Agent:     gobuster/3.0.1
 [+] Verbose:        true
 [+] Timeout:        10s
 2019/11/01 01:33:32 Starting gobuster
 Missed: /.bashrc (Status: 404)
 Missed: /.cvs (Status: 404)
 Missed: /.cvsignore (Status: 404)
...
 Missed: /_mm (Status: 404)
 Missed: /_mygallery (Status: 404)
Above query has try to find files in verbose mode. Showing HTTP status code on each request.






PRINTING FILES WITH NO STATUS
Type gobuster dir -u https://192.168.1.105/dvwa -w /usr/share/wordlists/dirb/common.txt -n
-u is used to assign target URL. 192.168.1.105 is our target URL.
-w is used to assign wordlist. /usr/share/wordlists/dirb/common.txt is the wordlist location.
-n is used to print with no status codes.

# gobuster dir -u https://192.168.1.105/dvwa -w /usr/share/wordlists/dirb/common.txt -n
 Gobuster v3.0.1
 by OJ Reeves (@TheColonial) & Christian Mehlmauer (@FireFart)
 [+] Url:            https://192.168.1.105/dvwa
 [+] Threads:        10
 [+] Wordlist:       /usr/share/wordlists/dirb/common.txt
 [+] Status codes:   200,204,301,302,307,401,403
 [+] User Agent:     gobuster/3.0.1
 [+] No status:      true
 [+] Timeout:        10s
 2019/11/01 02:36:35 Starting gobuster
 /.hta
 /.htpasswd
 /.svn
 /.svn/entries
 /.htaccess
 /css
 /images
 /includes
 /js
 2019/11/01 02:36:38 Finished
Above query has printed with data without any status codes.






FINDING LENGTH OF THE RESPONSE FILES
Type gobuster dir -u https://192.168.1.105/dvwa -w /usr/share/wordlists/dirb/common.txt -l
-u is used to assign target URL. 192.168.1.105 is our target URL.
-w is used to assign wordlist location. -w /usr/share/wordlists/dirb/common.txt is our wordlist location.
-l is used find length of response files.
# gobuster dir -u https://192.168.1.105/dvwa -w /usr/share/wordlists/dirb/common.txt -l
 Gobuster v3.0.1
 by OJ Reeves (@TheColonial) & Christian Mehlmauer (@FireFart)
 [+] Url:            https://192.168.1.105/dvwa
 [+] Threads:        10
 [+] Wordlist:       /usr/share/wordlists/dirb/common.txt
 [+] Status codes:   200,204,301,302,307,401,403
 [+] User Agent:     gobuster/3.0.1
 [+] Show length:    true
 [+] Timeout:        10s
 2019/11/01 02:57:45 Starting gobuster
 /.hta (Status: 403) [Size: 1108]
 /.htpasswd (Status: 403) [Size: 1108]
 /.svn/entries (Status: 200) [Size: 256]
 /.htaccess (Status: 403) [Size: 1108]
 /.svn (Status: 301) [Size: 416]
 /css (Status: 301) [Size: 415]
 /images (Status: 301) [Size: 418]
 /includes (Status: 301) [Size: 420]
 /js (Status: 301) [Size: 414]
 2019/11/01 02:57:48 Finished
Above shows the files size. By this attacker can obtain type of files target uses to maintain their website and as per digital forensics expert of International Institute of Cyber Security file size is also one of the parameters in analyzing the malware.




FINDING FILES WITH SPECIFIC EXTENSION
Type gobuster dir -u https://192.168.1.105/dvwa -w /usr/share/wordlists/dirb/common.txt -x .php
-u is used to assign URL. 192.168.1.105 is our target URL
-w is used to assign wordlist. -w /usr/share/wordlists/dirb/common.txt is wordlist location.
-x is used to extract specific extension files. .php will be extracted.

# gobuster dir -u https://192.168.1.105/dvwa -w /usr/share/wordlists/dirb/common.txt -x .php
 Gobuster v3.0.1
 by OJ Reeves (@TheColonial) & Christian Mehlmauer (@FireFart)
 [+] Url:            https://192.168.1.105/dvwa
 [+] Threads:        10
 [+] Wordlist:       /usr/share/wordlists/dirb/common.txt
 [+] Status codes:   200,204,301,302,307,401,403
 [+] User Agent:     gobuster/3.0.1
 [+] Extensions:     php
 [+] Timeout:        10s
 2019/11/01 03:32:20 Starting gobuster
 /.hta (Status: 403)
 /.hta.php (Status: 403)
 /.htpasswd (Status: 403)
 /.htpasswd.php (Status: 403)
 /.htaccess (Status: 403)
 /.htaccess.php (Status: 403)
 /.svn/entries (Status: 200)
 /.svn (Status: 301)
 /css (Status: 301)
 /images (Status: 301)
 /includes (Status: 301)
 /js (Status: 301)
 2019/11/01 03:32:25 Finished
Above query has found files with .php extension. This query can help attacker to create malicious files on specific extension.



FINDING USERNAME & PASSWORD
Type gobuster dir -u https://testphp.vulnweb.com/login.php -w /usr/share/wordlists/dirb/common.txt -U test -P test
-u is used to assign URL. 192.168.1.105 is our target URL
-w is used to assign wordlist. -w /usr/share/wordlists/dirb/common.txt is wordlist location.
-U is for username & -P is for password.
# gobuster dir  -u https://testphp.vulnweb.com/login.php -w /usr/share/wordlists/dirb/common.txt -U test -P test
 Gobuster v3.0.1
 by OJ Reeves (@TheColonial) & Christian Mehlmauer (@FireFart)
 [+] Url:            https://testphp.vulnweb.com/login.php
 [+] Threads:        10
 [+] Wordlist:       /usr/share/wordlists/dirb/common.txt
 [+] Status codes:   200,204,301,302,307,401,403
 [+] User Agent:     gobuster/3.0.1
 [+] Auth User:      test
 [+] Timeout:        10s
 2019/11/01 04:31:34 Starting gobuster
 /admin.php (Status: 200)
 /index.php (Status: 200)
 /info.php (Status: 200)
 /phpinfo.php (Status: 200)
 /xmlrpc.php (Status: 200)
 /xmlrpc_server.php (Status: 200)
 2019/11/01 04:32:54 Finished
```



























.
