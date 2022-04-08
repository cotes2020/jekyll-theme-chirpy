---
title: Meow's SecurityEvent - 2022 Apr 6 AWS lambda
date: 2021-10-4 11:11:11 -0400
categories: [10CyberAttack, SecurityEvent]
tags: [SecurityEvent]
toc: true
image:
---

- [Meow's SecurityEvent - 2022 Apr 6 AWS lambda](#meows-securityevent---2022-apr-6-aws-lambda)
  - [basic](#basic)
  - [Analysing Lambda Malware](#analysing-lambda-malware)
  - [configruation](#configruation)
  - [Communication with the Monero server at `116.203.4[.]0`](#communication-with-the-monero-server-at-11620340)
- [history](#history)
- [solution](#solution)

- ref
  - https://www.cadosecurity.com/cado-discovers-denonia-the-first-malware-specifically-targeting-lambda/
  - https://14518100.fs1.hubspotusercontent-na1.net/hubfs/14518100/Playbooks/Playbook_%20Securing%20&%20Investigating%20AWS%20ECS%20.pdf?__hstc=185812470.304aab62b8d53253769a91c31b636844.1649351215522.1649356371552.1649361678913.3&__hssc=185812470.1.1649361678913&__hsfp=2845785784

---

# Meow's SecurityEvent - 2022 Apr 6 AWS lambda


## basic

**malware Denonia**
- Denonia: the name the attackers gave the domain it communicates with.

- The malware uses `newer address resolution techniques` for command and control traffic to <font color=red> evade typical detection measures and virtual network access controls </font>.

- this sample is fairly innocuous in that it only runs `crypto-mining software`, it demonstrates how attackers exploit complex cloud infrastructure, and is indicative of potential future, more nefarious attacks.

- From the telemetry we have seen, the distribution of Denonia so far has been limited.

Denonia with the following SHA-256 hash:
- a31ae5b7968056d8d99b1b720a66a9a1aeee3637b97050d95d96ef3a265cbbca


During dynamic analysis, the malware quickly halted execution and logged the following error:

![Screen Shot 2022-04-07 at 12.26.52](https://i.imgur.com/WpdxJz9.png)

- the environment in which this malware is expected to execute.
- it was a 64-bit ELF executable targeting the x86-64 architecture and that it uses a number of third-party libraries, including one specifically to enable execution inside AWS Lambda environments.





## Analysing Lambda Malware

Analysing a binary designed to run in AWS Lambda poses some interesting challenges.

Whilst Denonia is clearly designed to execute inside of Lambda environments
- may simply be a matter of **compromising AWS Access and Secret Keys**
- then manually deploying into compromised Lambda environments


Using the redress tool we identified some interesting `third-party Go libraries` that the malware embeds. This gave us some clues about its functionality:

- `github.com/aws/aws-lambda-go/lambda` – libraries, samples and tools for writing Lambda functions in Go
- `github.com/aws/aws-lambda-go/lambdacontext` – helpers for retrieving contextual information from a Lambda invoke request
- `github.com/aws/aws-sdk-go/aws` – general AWS SDK for Golang
- `github.com/likexian/doh-go` – DNS over HTTPS in Go, supports providers such as Quad9, Cloudflare etc



Lambda function handler expects certain data to be set:

![Screen Shot 2022-04-07 at 12.39.04](https://i.imgur.com/LKE1t1q.png)


- Despite the presence of this, we discovered during dynamic analysis that `the sample will continue execution outside a Lambda environment` (i.e. on a vanilla Amazon Linux box).
- We suspect this is likely due to Lambda “serverless” environments using Linux under the hood, so the malware believed it was being run in Lambda (after we manually set the required environment variables) despite being run in our sandbox.


**DNS over HTTPS**

- Normally when you request a domain name such as google.com
  - you send out an `unencrypted DNS request` to find the IP Address the domain resolves to
  - then your machine can connect to the domain.
- A relatively modern replacement for **traditional DNS** is **DNS over HTTPS (DoH)**.
  - DoH encrypts `DNS queries`, and sends the requests out as regular HTTPS traffic to DoH resolvers.

- Using DoH
  - a fairly unusual choice for the Denonia authors
  - provides two disadvantages:
    - **AWS cannot see the dns lookups** for the malicious domain, reducing the likelihood of triggering a detection
    - **Some Lambda environments may be unable to perform DNS lookups**, depending on VPC settings.



**The HTTPS Request to the Google DoH Server**
- the malware sending these requests using the “doh-go” library to URLs such as:
- https://cloudflare-dns.com/dns-query?name=gw.denonia.xyz&type=A
- https://dns.google.com/resolve?name=gw.denonia.xyz&type=A


![Screen Shot 2022-04-07 at 12.46.29](https://i.imgur.com/FDwmFTN.png)

- And the DoH server (in this case from Google) responds with the `IP the domain` resolves to in a JSON format:

![Screen Shot 2022-04-07 at 12.48.11](https://i.imgur.com/pxmLYlP.png)



## configruation

- The attacker controlled domain `gw.denonia[.]xyz` resolves to `116.203.4[.]0`
- written into a config file for **xmrig** at `/tmp/.xmrig.json`:

> on AWS Lambda, the only directory that can write to is /tmp.
> The binary also sets the HOME directory to /tmp with “HOME=/tmp”.
> XMRig itself is executed from memory.



```json
"pools":[
  {
    "url":"116.203.4.0:333",
    "user":"echonet.amd64",
    "pass":null,
    "rig0id":"echonet.amd64"
  }
]
```




## Communication with the Monero server at `116.203.4[.]0`

- Denonia then starts XMRig from memory
- XMRig communicates with the `attacker controlled Mining pool` at `116.203.4[.]0:3333`

- XMRig also writes to the console as it executes:





# history


a second Denonia sample was uploaded to VirusTotal in January 2022:
- 739fe13697bc55870ceb35003c4ee01a335f9c1f6549acb6472c5c3078417eed




# solution


- investigate and remediate both AWS ECS and AWS Lambda environments


**Indicators of Compromise**

```json
rule lambda_malware
{
    meta:
        description = "Detects AWS Lambda Malware"
        author = "cdoman@cadosecurity.com"
        license = "Apache License 2.0"
        date = "2022-04-03"
        hash1 = "739fe13697bc55870ceb35003c4ee01a335f9c1f6549acb6472c5c3078417eed"
        hash2 = "a31ae5b7968056d8d99b1b720a66a9a1aeee3637b97050d95d96ef3a265cbbca"
    strings:
        $a = "github.com/likexian/doh-go/provider/"
        $b = "Mozilla/5.0 (compatible; Ezooms/1.0; help@moz.com)"
        $c = "username:password pair for mining server"
    condition:
        filesize < 30000KB and all of them
}

```


```bash
# **Domains**
denonia[.]xyz
ctrl.denonia[.]xyz
gw.denonia[.]xyz
1.gw.denonia[.]xyz
www.denonia[.]xyz
xyz.denonia[.]xyz
mlcpugw.denonia[.]xyz

# **IP Addresses**
116.203.4[.]0
162.55.241[.]99
148.251.77[.]55
```
















.
