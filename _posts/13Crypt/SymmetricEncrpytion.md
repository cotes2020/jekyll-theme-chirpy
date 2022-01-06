---
title: Cryptography - Symmetric Encryption
date: 2018-05-18 11:11:11 -0400
categories: [13Cryptography]
tags: [cryptography]
toc: true
image:
---

[toc]

--- 
 

# Symmetric Encryption

> shared-key, secret key, session-key cryptography.

- both parties have a same/shared key to encrypt/decrypt packet during a session (session key).

![Pasted Graphic 2](https://i.imgur.com/kMj7s4U.png)


---


## The major strength:
- Faster
  - Symmetric key encryption is very fast,
  - often 1,000 to 10,000 times faster than asymmetric algorithms.
- hardware implementations
  - By nature of the mathematics involved, symmetric key cryptography also naturally lends itself to hardware implementations, creating the opportunity for even higher-speed operations.


---


## weaknesses

- Key distribution
  - major problem.
  - Parties must have a secure method of exchanging the secret key before establishing communications with a symmetric key protocol.
  - If a secure electronic channel is not available, offline key distribution method must be used (out-of-band exchange).

- does not implement nonrepudiation.
  - Because any communicating party can encrypt and decrypt messages with the shared secret key
  - no way to prove where a given message originated.

- not scalable.
  - Extremely difficult for large groups to communicate using symmetric key cryptography.
  - Secure private communication between individuals in the group could be achieved only if each possible combination of users shared a private key.
  - example:
    - n parties using symmetric cryptosystem
    - a distinct secret key is needed for each pair of parties
    - total: n(n − 1)/2 keys
  - rise problem of key management.
  - When large-sized keys are used, symmetric encryption is very difficult to break.
  - It is primarily employed to
    - perform bulk encryption
    - and provides only for the security service of confidentiality.



- Keys must be regenerated often.
  - Each time a participant leaves the group, all keys known by that participant must be discarded.

















.
