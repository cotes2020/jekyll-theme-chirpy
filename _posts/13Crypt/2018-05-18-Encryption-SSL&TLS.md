---
title: Cryptography - SSL/TLS Encryption
date: 2018-05-18 11:11:11 -0400
categories: [13Cryptography]
tags: [cryptography]
toc: true
image:
---

- [Cryptography - SSL/TLS Encryption](#cryptography---ssltls-encryption)
  - [overall](#overall)
  - [SSL/TLS vs SSH](#ssltls-vs-ssh)

---

# Cryptography - SSL/TLS Encryption

---

## overall

SSL/TLS uses both asymmetric and symmetric encryption to protect the confidentiality and integrity of data-in-transit. Asymmetric encryption is used to establish a secure session between a client and a server, and symmetric encryption is used to exchange data within the secured session.




---


## SSL/TLS vs SSH

SSH = Secure SHell

![SSH-Handshake](https://i.imgur.com/Ka4IjRZ.jpg)

SSL = Secure Sockets Layer

![ssl2buy-tls12-13](https://i.imgur.com/i06MCnK.jpg)

similar:
- **SSL** and **SSH** both provide the cryptographic elements to build a tunnel for confidential data transport with checked integrity. For that part, they use similar techniques, and may suffer from the same kind of attacks, so they should provide similar security (i.e. good security) assuming they are both properly implemented.
- That both exist is a kind of NIH syndrome: the **SSH** developers should have reused **SSL** for the tunnel part (the **SSL** protocol is flexible enough to accommodate many variations, including not using certificates).

differ:
- They differ on the things which are around the tunnel.
- 1
  - **SSL** traditionally uses `X.509 certificates` for announcing server and client public keys;
  - **SSH** has its own format.
- 2
  - **SSH** comes with a set of protocols for what goes inside the tunnel (multiplexing several transfers, performing password-based authentication within the tunnel, terminal management...)
  - while there is no such thing in **SSL**,
    - or, more accurately, when such things are used in SSL, they are not considered to be part of SSL
    - for instance, when doing password-based HTTP authentication in a **SSL** tunnel, we say that it is part of "HTTPS", but it really works in a way similar to what happens with SSH
- 3
  - **SSL** is a general method for `protecting data transported` over a network,
  - whereas **SSH** is a network application for `logging in and sharing data` with a remote computer.

  - The transport layer protection in **SSH** is similar in capability to SSL, so which is "more secure" depends on what your specific threat model calls for and whether the implementations of each address the issues you're trying to deal with.

  - **SSH** then has a `user authentication layer` which **SSL** lacks (because it doesn't need it - **SSL** just needs to authenticate the two connecting interfaces which **SSH** can also do).

  - Regarding the issue of which there are more potential attacks against, it seems clear that **SSH** has a larger attack surface. But that's just because **SSH** has a whole application built into it: the attack surface of **SSL** + whatever application you need to provide cannot be compared because we don't have enough information.


```
      **SSL**              SSH
+-------------+ +-----------------+
| Nothing     | | RFC4254         | Connection multiplexing
+-------------+ +-----------------+
| Nothing     | | RFC4252         | User authentication
+-------------+ +-----------------+
| RFC5246     | | RFC4253         | Encrypted data transport
+-------------+ +-----------------+
```

- 4
  - From a strict cryptographic point of view, they both provide authenticated encryption, but in two different ways.
  - **SSH** uses `Encrypt-and-MAC`, the ciphered message is juxtaposed to a message authentication code (MAC) of the clear message to add integrity. This is not proven to be always fully secure (even if in practical cases it should be enough).
  - **SSL** uses `MAC-then-Encrypt`: a MAC is juxtaposed to the clear text, then they are both encrypted. This is not the best either, as with some block cipher modes parts of the MAC can be guessable and reveal something on the cipher. This led to vulnerabilities in TLS 1.0 (BEAST attack).
  - So they have both potential theoretical weaknesses.
  - The strongest method is `Encrypt-then-MAC` (add a MAC of the ciphered message), which is implemented, e.g., in IPsec ESP.



Conceptually,
- you could take **SSH** and replace the tunnel part with the one from SSL.
- You could also take HTTPS and replace the **SSL** thing with SSH-with-data-transport and a hook to extract the server public key from its certificate.
- There is no scientific impossibility and, if done properly, security would remain the same. However, there is no widespread set of conventions or existing tools for that.


- So we do not use **SSL** and **SSH** for the same things, but that's because of what tools historically came with the implementations of those protocols, not due to a security related difference. And whoever implements **SSL** or **SSH** would be well advised to look at what kind of attacks were tried on both protocols.



- While there are other applications for these protocols, the basic differences are clear. **SSH** is generally a tool for technicians, and **SSL/TLS** is a mechanism for securing websites that is transparent to the user.
- Of course, these two are not mutually exclusive. **SSH** may use SSL/TLS as part of its secure solution. There are a variety of possible implementations for these versatile protocols.




---
