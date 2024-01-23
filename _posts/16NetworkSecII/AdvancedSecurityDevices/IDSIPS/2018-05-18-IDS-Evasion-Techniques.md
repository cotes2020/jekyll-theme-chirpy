---
title: NetworkSec - Advanced SecDevices - IDS Evasion Techniques
# author: Grace JyL
date: 2018-05-18 11:11:11 -0400
# description:
# excerpt_separator:
categories: [15NetworkSec, AdvancedDev]
tags: [NetworkSec]
math: true
# pin: true
toc: true
# image: /assets/img/note/tls-ssl-handshake.png
---

[toc]

---

# IDS Evasion Techniques

> "A look at whisker's anti-IDS tactics" by Rain Forest Puppy (http://www.apachesecurity.net/archive/whiskerids.html)
> "IDS Evasion Techniques and Tactics" by Kevin Timm (https://www.securityfocus.com/printable/infocus/1577)

---


## Brief Overview
- From CIDF to RFC 4765
  - `Common Intrusion Detection Framework (CIDF)`
    - old (late 90s) attempt by DARPA (US govt's Defense Advanced Research Projects Agency)
    - to develop an IDS interchange format
    - Started as a research project, currently dormant
    - CIDF components that together define an Intrusion Detection System:
      - E-boxes - event generators (sniffers, monitors)
      - A-boxes - analysis engines (signature matchers)
      - D-boxes - storage mechanisms (loggers)
      - C-boxes - countermeasures (alarms, firewalls)
    - ￼![Pasted Graphic](https://i.imgur.com/5uPQW9e.png)
  - The IETF (Internet Engineering Task Force) recently (2007) started work on a common format (RFC 4765):

- Physical, Network and Host IDS/IPS
  - Physical: Security Guards, Security Cameras, Access Control Systems (Card, Biometric), Firewalls, Man Traps, Motion Sensors
- NIDS design considerations & problems


---


## Simple Evasion Techniques

### Using mixed case characters
- This technique can be useful for attackers when attacking platforms (e.g., Windows) where filenames are not case sensitive;
- otherwise, it is useless.
- Its usefulness rises, however, if the target Apache includes mod_speling as one of its modules.
- This module tries to `find a matching file on disk, ignoring case and allowing up to one spelling mistake`.

### Character escaping
- escape any character by preceding the character with a backslash character `(\)`,
- and if the character does not have a special meaning, the escaped character will convert into itself.
  - `\d converts to d`.
- It is not much but it is enough to fool an IDS.
- For example
  - an IDS looking for the pattern `id` would not detect a string `i\d`, which has essentially the same meaning.


### Using whitespace
- Using excessive whitespace, especially the less frequently thought of characters such as TAB and new line, can be an evasion technique.
- example
  - SQL injection attempt using `DELETE  FROM` (two spaces in between the words instead of one)
  - the attack will be undetected by an IDS looking for `DELETE FROM` (with just one space in between).



---


## Path Obfuscation

Many evasion techniques are used in attacks against the filesystem.
- For example, many methods can obfuscate paths to make them less detectable:


### Self-referencing directories
- When a `./` combination is used in a path
- it does not change the meaning but it breaks the sequence of characters in two.
- example, 
- `/etc/passwd = obfuscated: /etc/./passwd`.


### Double slashes
- Using double slashes is one of the oldest evasion techniques.
- example, 
- `/etc/passwd may be written as /etc//passwd`.


### Path traversal
- Path traversal occurs when a backreference is used to back out of the current folder, but the name of the folder is used again to advance.
- For example, 
- `/etc/passwd may be written as /etc/dummy/../passwd`, and both versions are legal.
- This evasion technique can be used against application code that performs a file download to make it disclose an arbitrary file on the filesystem.
- Another use of the attack is to evade an IDS system looking for well-known patterns in the traffic
- (`/etc/passwd` is one example).


### Windows folder separator
- When the web server is running on Windows, the Windows-specific folder separator `\` can be used.
- For example, 
- `../../cmd.exe = ..\..\cmd.exe`.


### IFS evasion
- Internal Field Separator (IFS)
- a feature of some UNIX shells (sh and bash, for example) that allows the user to change the field separator (normally, a whitespace character) to something else.
- After execute an `IFS=X` command on the shell command line
- type `CMD=X/bin/catX/etc/passwd;eval$CMD` to display the contents of the `/etc/passwd` file on screen.


---

## Null-Byte Attacks

- Using `URL-encoded null bytes` is an evasion technique and an attack at the same time.
  - This attack is effective against applications developed using C-based programming languages.
  - Even with scripted applications, the application engine they were developed to work with is likely to be developed in C and possibly vulnerable to this attack.
  - Even Java programs eventually use native file manipulation functions, making them vulnerable, too.

- Internally, all C-based programming languages use the null byte for string termination.
  - When a URL-encoded null byte is planted into a request,
  - it often fools the receiving application, which decodes the encoding and plants the null byte into the string.
  - The planted null byte will be treated as the end of the string during the program's operation,
  - and `the part of the string that comes after it and before the real string terminator will practically vanish`.

- We looked at how a URL-encoded null byte can be used as an attack when we covered source code disclosure vulnerabilities in the "Source Code Disclosure" section.
- This vulnerability is rare in practice though Perl programs can be in danger of null-byte attacks, depending on how they are programmed.


- Null-byte encoding is used as an evasion technique mainly against web application firewalls when they are in place.
- These systems are almost exclusively C-based (they have to be for performance reasons), making the null-byte evasion technique effective.
- Web application firewalls trigger an error when a dangerous signature (pattern) is discovered.
  - They may be configured not to forward the request to the web server, the attack attempt will fail.
  - However, if the signature is hidden after an encoded null byte, the firewall may not detect the signature, allowing the request through and making the attack possible.
- To see how this is possible, we will look at a single POST request, representing an attempt to exploit a vulnerable `form-to-email script` and retrieve the passwd file:

```
POST /update.php HTTP/1.0
Host: www.example.com
Content-Type: application/x-form-urlencoded
Content-Length: 78

firstname=Ivan&lastname=Ristic%00&email=ivanr@webkreator.com;cat%20/etc/passwd
```


- A web application firewall configured to watch for the `/etc/passwd` string will normally easily prevent such an attack.
- But by embedded a null byte at the end of the lastname parameter.
  - firstname=Ivan&lastname=Ristic`%00`&email=ivanr@webkreator.com;cat%20/etc/passwd
- If the firewall is vulnerable to this type of evasion, it may miss our command execution attack, enabling us to continue with compromise attempts.

---


## Some Advanced Evasion Techniques (AETs)

### Obfuscation: (Insertion, Evasion, Session Splicing, Fragmentation)
---

#### Insertion:

Stuffing the analyzer with “invalid” packets
- 看到碎成1-character packet秒選 **Insertion Attack**)
- (1-character packet, 利用TTL-1=0讓部分packet死在IDS
- ![Pasted Graphic](https://i.imgur.com/4JszTLs.png)

---

#### Evasion:
Slipping “valid” packets past the analyzer
- ![Pasted Graphic 1](https://i.imgur.com/GJR5dmR.png)

![Pasted Graphic 2](https://i.imgur.com/iOjcYTQ.png)

![Pasted Graphic 3](https://i.imgur.com/GIlqSqA.png)

---

#### Fragmentation

Breaking the attack into multiple packets
- Similar to Session Splicing
  - Attacker send packets in blocks that do not trigger IDS signatures or cause alerts
  - Generally more powerful than Session Splicing
- Two common fragmentation methods
  - overwrites a section of a previous fragment
  - overwrites a complete fragment
- Enables attackers to write an entire packet of garbage information and craft their attack to blend in with standard protocols
- Some IDSs do have ways to handle these attacks through reassembly

Examples:
- Attack 1: **Overlap Method**
  - Packet 1: `GET /cgi-bin/`
  - Packet 2: `aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/../phxx`
  - Packet 3: `f?`
- This fragmentation can `overwrite the 'xx' portion of Packet 2 with the data in Packet 3`, making the information resemble the following:
  - `GET /cgi-bin/ aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/../phf?`

- Attack 2: **Overwrite Method**
  - Packet 1: `GET /cgi-bin/`
  - Packet 2: `some_normal_filename.cgi`
  - Packet 3: `/aaa/../aaa/../aaa/../phxx`
  - Packet 4: `f?`
- similar to the first
- but, the `'xx' portion is overwritten` and the `some_normal_filename.cgi packet is completely overwritten with the last two packets`
  - This leaves `GET /cgi-bin/phf?` as the end result.

---

#### Session Splicing


attacker `delivers data in multiple, smallsized packets` to the target computer
- `deliver the payload over multiple packets over long periods of time`
  - effective against IDS that `do not reconstruct packets before checking them again through intrusion signatures`
  - Many IDS `stop reassembly if they do not receive packets within a certain time`
  - If attackers are aware of delays in packet reassembly at the IDS, they add delays between packet transmissions to bypass the reassembly
  - IDS become useless if the target host keeps sessions active for a longer time than the IDS reassembly time

- defeating simple pattern matching in IDS systems without session reconstruction
- characteristic of the attack: a continuous stream of small packets.
  - making it very difficult for an IDS to detect the attack signatures.
  - `no single packet triggers the IDS`

- Any attack attempt after a successful splicing attack will not be logged by the IDS


Tool: Whisker
- One basic technique is to split the attack payload into multiple small packets, so that the IDS must reassemble the packet stream to detect the attack.
- way of splitting packets:
  - fragmenting them, but an adversary can also simply craft packets with small payloads.
- The 'whisker' evasion tool calls crafting packets with small payloads 'session splicing'.

---

### Denial of Service (DoS) & False Positive Generation
- Causing resource starvation and overloading the IDS
- Basic problem
  - NIDS needs to simulate the operation of all protected end-systems and internal network
  - Scarce Resources (CPU cycles, memory, disk space, bandwidth)
  - Usually working in a “fail-open” state
- CPU DoS (target computationally expensive operations)
  - Fragment/Segment reassembly
  - Encryption/Decryption
- Memory DoS
  - `(target state management operations)`
  - TCP 3-way Handshake (TCP Control Block - TCB)
  - Fragment/Segment reassembly
- Network Bandwidth DoS
  - `(target NIDS’s inability to process packets at line speed)`
- Reactive Systems DoS
  - Trigger lots of alarms (false positives)
  - Prevent valid access by spoofed addresses
  - Hide real attacks

---

### Pattern-Matching Weaknesses
- Exploiting pattern-based detection approach employed by most IDS
- Most IDS solutions employ a `pattern-based detection component`
  - This approach is problematic, because not all input needs to be the same to trigger vulnerabilities

- Example pattern:
  - GET `/cgi-bin/phf?`
- Obfuscation:
  - GET `/cgi-bin/aaaaaaaaaaaaaaaaaaaaaaaaaa/..%25%2fp%68f?`
- Both versions result in the same output, yet look very different
- Unicode evasion can also be used here
  - e.g. `\ can be represented as 5C, C19C and E0819C`

extremely common IDS evasion technique in the web world? unicode characters
- Unicode attacks can be effective against applications that understand it.
- Unicode: represent every character needed by every written human language as a single integer number.
- Unicode evasion, referenced as UTF-8 evasion.
- Unicode characters are normally represented with two bytes, but this is impractical in real life.
- One aspect of UTF-8 encoding causes problems:
  - non-Unicode characters can be represented encoded.
  - What is worse is multiple representations of each character can exist.
  - Non-Unicode character encodings are known as overlong characters, and may be signs of attempted attack.

---

### Protocol Violation,
- Attacks targeted at complex protocol
  - e.g. SMB (Server Message Block), MSRPC, SunRPC
- In order to provide protection to a complex protocol, the IDS has to have a deep understanding of it
- The IDS implementation also needs to be
  - Fault-tolerant
  - Resilient
  - Able to cope with excessive and unexpected connections and requests

---

### TTL Attacks
- attackers have some knowledge of the internal network topology
  - Attackers must know the distance to the end host and whether an IDS is placed in front of the end host
- By using a small TTL flag in a TCP packet, attackers can send packets that will only reach the IDS and not the end host
  - The IDS, in turn, will think the packet addressed to the end host will make it there
  - This allows attackers to inject garbage packets into the IDS stream processing
- Example:
  - Packet 1: GET /cgi-bin/p       TTL 15
  - Packet 2: some_file.cgi?=     TTL 10
  - Packet 3: hf?                         TTL 15
- This assumes that the end host is beyond the 15 TTL limit and will receive the data
- It also assumes that the IDS is within the 10-14 TTL limit, and any data lower than that will not reach the destination host
  - The IDS receives: GET /cgi-bin/psome_file.cgi?=hf?
  - The end host will receive: GET /cgi-bin/phf?

---

### Urgency Flag
- The urgency flag is used within the TCP protocol to mark data as urgent
  - When the urgency flag is set, `all data before the urgency pointer is ignored`,
  - and `the data to which the urgency pointer points is processed`

- Some IDSs do not take into account the TCP protocol's urgency feature
  - Attackers can place garbage data before the urgency pointer
  - The IDS reads that data without consideration for the end host's urgency flag handling
  - This means the IDS has more data than the end host actually processed

- Example of an urgency flag attack:
  - "1 Byte data, next to Urgent data, will be lost, when Urgent data and normal data are combined."
  - Packet 1: `ABC`
  - Packet 2: `DEF Urgency Pointer: 3`
  - Packet 3: `GHI`
  - End result: `ABCDEFHI`
- According to the 1122 RFC,
  - the urgency pointer causes `one byte of data next to the urgent data to be lost` when urgent data is combined with normal data


---

### Polymorphic Shellcode
- Most IDSs contain signatures for commonly used strings within shellcode
  - This is easily bypassed by using encoded shellcode containing a stub that decodes the shellcode that follows
  - This means that shellcode can be completely different each time it is sent
- Polymorphic shellcode allows attackers to hide their shellcode by encrypting it in a simplistic form
  - It is difficult for IDSs to identify this data as shellcode
- This method also hides the commonly used strings within shellcode, making shellcode signatures useless


![Pasted Graphic 1](https://i.imgur.com/lfPxe57.png)


---

### ASCII Shellcode
- Similar to polymorphic shellcode
  - ASCII shellcode contains only characters contained within the ASCII standard
- This helps attackers bypass IDS pattern matching signatures as strings are hidden within the shellcode in a similar fashion to polymorphic shellcode
- The following is an ASCII shellcode example:
  - char shellcode[] =
  - "LLLLYhb0pLX5b0pLHSSPPWQPPaPWSUTBRDJfh5tDS"
  - "RajYX0Dka0TkafhN9fYf1Lkb0TkdjfY0Lkf0Tkgfh"
  - "6rfYf1Lki0tkkh95h8Y1LkmjpY0Lkq0tkrh2wnuX1"
  - "Dks0tkwjfX0Dkx0tkx0tkyCjnY0LkzC0TkzCCjtX0"
  - "DkzC0tkzCj3X0Dkz0TkzC0tkzChjG3IY1LkzCCCC0"
  - "tkzChpfcMX1DkzCCCC0tkzCh4pCnY1Lkz1TkzCCCC"
  - "fhJGfXf1Dkzf1tkzCCjHX0DkzCCCCjvY0LkzCCCjd"
  - “X0DkzC0TkzCjWX0Dkz0TkzCjdX0DkzCjXY0Lkz0tk"
  - "zMdgvvn9F1r8F55h8pG9wnuvjrNfrVx2LGkG3IDpf"
  - "cM2KgmnJGg`bin`Y`sh`dvD9d";
- When executed, the shellcode above executes a "/bin/sh" shell

---

### Encryption and Tunneling
- When the attacker manages to establish an encrypted tunnel to the target, IDS-es are evaded completely
- Any sort of encrypted connection/tunneling works
  - SSH, SSL, IPSec, RDP, etc

---

### Application Hijacking
- If done correctly `few HIDS will detect it`, while `NIDS usually skip the application layer completely`
- Application layer attacks enable many different forms of evasion
- Many applications that deal with media such as images, video and audio employ some form of compression
- When a flaw is found in these applications, the entire attack can occur within compressed data, and the IDS will have no way to check the compressed file format for signatures

---

### File Locations and Integrity
- Circumventing triggers in HIDS



---


## to evade IDS during a Port Scan:
- Use fragmented IP packets
- Spoof your IP address when launching attacks and sniff responses from the server
- Use source routing (if possible)
- Connect to proxy servers or compromised Trojaned machines to launch attacks


---

## Potential Solutions
- Normalization
  - Normalization takes obfuscated input and attempts to translate it into what the end host will eventually see
  - This usually entails encoding in formats such as Unicode and UTF8
  - The normalization process allows for encoding, translation and the application of pattern matching to the normalized data
  - Prevents obfuscating the attack strings using Unicode or UTF8 strings
  - Polymorphic shellcode & ASCII shellcode could circumvent this
  - Some IDSs are attempting to apply normalization to polymorphic shellcode
  - CPU intensive, which affects monitoring for the remaining network traffic
  - Normalization also applies to network data
  - Some IDSs normalize fragmented packets and reassemble them in the proper order
  - This enables the IDS to look at the information just as the end host will see it
  - In addition, some IDSs change the TTL field to a large number
  - This ensures that packets reach the end host

- Packet Interpretation Based on Target Host
  - IDS are at a disadvantage
  - These systems attempt to recreate what the end host will see and handle
  - There are a lot of disparate methods of communicating data over a network
  - The end host's TCP/IP stack should be used (host-based IDS)
  - Better than trying to recreate the stream in a way that the stream may be handled
  - In using the host to do the work, the guessing portion of the task is eliminated
  - Another option: using modular TCP/IP stacks within an IDS
  - Using the stacks based on the targeted host's operating system
  - Specific OS handling of anomalous traffic must be thoroughly reviewed
  - Effective in mitigating fragmentation, RST packet handling and Urgency Flags

---

## Tools & Resources
- Some free IDS:
  - ACARM-ng, AIDE (Advanced Intrusion Detection Environment)
  - Bro NIDS, Fail2ban, OSSEC (Open Source Host-based IDS)
  - Prelude SIEM (Security Information & Event Management)
  - Samhain
  - Snort, Suricata
  - Tripwire

- some IDS evasion tools:
  - Evader
  - Nmap
  - Nmap reference on IDS evasion
  - libemu
  - Kali Linux
  - Fragroute
  - Fragrouter
  - InTrace
  - SniffJoke
  - Other tools
  - Wireshark, HxD (hex editor/viewer)
