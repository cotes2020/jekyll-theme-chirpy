---
title: Meow's Testing Tools - John the Ripper
date: 2019-09-17 11:11:11 -0400
categories: [10CyberAttack, CyberAttackTools]
tags: [CyberAttack, CyberAttackTools]
toc: true
image:
---

# John the Ripper

[toc]

---

# John the Ripper

[toc]

---

# John the Ripper

- offline password cracking tool
- works on files that have been grabbed from original source.

## different modes used to crack passwords

1. single crack mode:
   - takes information from the different fields in the file, applying mangling rules to them, to try as passwords.
   - Because the list of inputs is comparatively small, there are extensive mangling rules to vary the source text to generate potential passwords
   - fastest mode

![Pasted Graphic 9](https://i.imgur.com/dzDUyv4.png)

1. wordlist mode: takes a wordlist as input, comparing the hash of each word against the password hash.
   - You can apply mangling rules to your wordlist, which will generate variants on the words, since people often use variations on known words as their passwords.
   - The longer your wordlist, the better chance you will have of cracking passwords, the longer it will take.
   - will only identify passwords in the wordlist.
   - If long passphrase or truly random characters, wordlist won't work. need to try another mode.

![Screen Shot 2020-09-22 at 23.22.15](https://i.imgur.com/wrJzW6C.png)

```
locate rockyou.txt
john --format=raw-md5 /usr/wordlist/rockyou.txt /usr/Desktop/passwdhash.txt
```

3. incremental mode: try every possible combination of characters .
   - needs to be told what characters to try.
     - may be all ASCII characters, uppercase characters, numbers…
   - need to know the password length.
     - Because of the number of possible variants, this mode will need to be stopped because John can't get through all the variants in a reasonable time, unless specified a short password length.

---

## different passwd

1. Windows passwords:
   - hash from hashdump in Meterpreter — John

2. Linux passwords:
   - captured shadow file, passwd file — unshadow program
   - early days of Unix, there was a single file stored user info and passwords were
   - Problem:
   - With information regular users needed to obtain from that file, permissions is anyone could read it.
   - Anyone could read the hashes, obtain the passwords using cracking strategies.
   - As a result:
   - the public information was stored in one file: still named passwd for backward compatibility
   - the passwords and the necessary information (usernames, user IDs): the shadow file.

unshadow program
- combine the two files so all the needed information is together and consolidated
- merges the shadow file and the passwd file.
- run unshadow with a captured shadow file and passwd file.


Once you have the two files merged by unshadow, run John against it to acquire the password.
- John will identify the format of the file and the hash algorithm used to generate it.
- This information is stored in the file.
  - `$6$`: hashed using the secure hash algorithm with 512 bits for the output (SHA-512 ).
- What comes after that is the hashed password that John will be comparing against.
- John isn't the only way to obtain passwords from local files.
