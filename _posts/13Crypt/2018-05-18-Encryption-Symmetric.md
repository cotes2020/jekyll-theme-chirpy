---
title: Cryptography - Symmetric Encryption
date: 2018-05-18 11:11:11 -0400
categories: [13Cryptography]
tags: [cryptography]
toc: true
image:
---

- [Cryptography - Symmetric Encryption](#cryptography---symmetric-encryption)
  - [strength](#strength)
  - [weaknesses](#weaknesses)
- [Symmetric Block Cipher / stream cipher](#symmetric-block-cipher--stream-cipher)
  - [Block Cipher](#block-cipher)
    - [Data Encryption Standard (DES) `64-bit text, 56-bit keys`](#data-encryption-standard-des-64-bit-text-56-bit-keys)
    - [Triple DES (3DES) `64-bit text, 3 x 56 (168)-bit DES keys`](#triple-des-3des-64-bit-text-3-x-56-168-bit-des-keys)
    - [IDEA International Data Encryption Algorithm `64-bit text, 128-bit key`](#idea-international-data-encryption-algorithm-64-bit-text-128-bit-key)
    - [Blowfish `64-bit text, 32-448 bits key`](#blowfish-64-bit-text-32-448-bits-key)
    - [Twofish `128-bit blocks, 128, 192, 256-bit keys`](#twofish-128-bit-blocks-128-192-256-bit-keys)
    - [Advanced Encryption Standard (AES) `128-bit, 128, 192, 256 bit key`](#advanced-encryption-standard-aes-128-bit-128-192-256-bit-key)
      - [AES Round Structure](#aes-round-structure)
    - [Skipjack algorithm `64-bit`](#skipjack-algorithm-64-bit)
  - [stream cipher](#stream-cipher)
    - [Ron’s Code / Rivest Cipher.](#rons-code--rivest-cipher)
      - [RC4 `40-2048 bits`](#rc4-40-2048-bits)
      - [RC5 `32, 64, or 128 bits`](#rc5-32-64-or-128-bits)
      - [RC6](#rc6)

---


# Cryptography - Symmetric Encryption

> shared-key, secret key, session-key cryptography.

- both parties have a same/shared key to encrypt/decrypt packet during a session (session key).

![Pasted Graphic 2](https://i.imgur.com/kMj7s4U.png)


---


## strength

- Faster
  - Symmetric key encryption is very fast,
  - often 1,000 to 10,000 times faster than asymmetric algorithms.
- hardware implementations
  - By nature of the mathematics involved, symmetric key cryptography also naturally lends itself to hardware implementations, creating the opportunity for even higher-speed operations.


---


## weaknesses

- Key distribution
  - major problem
  - need secure method of exchanging the secret key before establishing communications with a symmetric key protocol.
  - If a secure electronic channel is not available, offline key distribution method must be used  (out-of-band exchange).

- does not implement non-repudiation.
  - Because any communicating party can encrypt and decrypt messages with the shared secret key
  - no way to prove where a given message originated.

> Non-repudiation is the assurance that someone cannot deny the validity of something.
> Non-repudiation is a legal concept

- not scalable
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






---


# Symmetric Block Cipher / stream cipher

> Symmetric encryption algorithm is recommended by the U.S. National Institute of Standards and Technology (NIST):

Block Cipher
- Data Encryption Standard (DES) `64-bit text, 56-bit keys` 已能被暴力破解
- Triple DES (3DES) `64-bit text, 3 x 56 (168)-bit DES keys` isn’t used as often as AES today.
- IDEA International Data Encryption Algorithm `64-bit text, 128-bit key` Pretty Good Privacy (PGP) secure email package
- Blowfish `64-bit text, 32-448 bits key` widely use today
- Twofish `128-bit blocks, 128, 192, 256-bit keys`
- Advanced Encryption Standard (AES) `128-bit text, 128, 192, 256 bit key` attack not currently possible

stream cipher
- Ron’s Code / Rivest Cipher.
  - RC4 `40-2048 bits`
  - RC5 `32, 64, or 128 bits`
  - RC6



---

## Block Cipher

### Data Encryption Standard (DES) `64-bit text, 56-bit keys`

The US government published the Data Encryption Standard in 1977 as a proposed standard cryptosystem for all government communications. was developed in response to the NBS/NIST, issuing a request for proposals for a standard cryptographic algorithm in 1973.

- Developed by IBM and adopted by NIST in 1977
- Due to flaws in the algorithm, cryptographers no longer consider DES secure
- DES 已能被暴力破解
  - onsidered adequate to resist a brute-force attack for up to 90 years.
  - but now small 56 bits key and can be broken with brute force in minutes.
- DES was superseded by the AES in December 2001.
- It is still important to understand DES because it is the building block of Triple DES (3DES).



- 64-bit blocks and 56-bit keys
  - 64 bits of plain text generate 64-bit of ciphertext.
  - Small key space makes exhaustive search attack feasible since late 90s

- DES uses a long series of `exclusive OR (XOR) operations` to generate the ciphertext.
- This process is repeated 16 times for each encryption/decryption operation.
- Each repetition is commonly referred to as a round of encryption, explaining the statement that DES performs 16 rounds of encryption.


DES uses a 56-bit key to drive the encryption and decryption process.
- However, you may read in some literature that DES uses a 64-bit key. This is not an inconsistency 矛盾, logical explanation:
- The DES specification calls for a 64-bit key.
  - only 56 bits actually contain keying information.
  - remaining 8 bits contain parity information to ensure that the other 56 bits are accurate.
  - 每7个比特会设置一个用于错误检查的比特，因此实质上其密钥长度是56比特。
- In practice, however, those parity bits are rarely used.
- You should commit the 56-bit figure to memory.

DES is a 64-bit block cipher that has 5 modes of operation:
- Electronic Codebook (ECB) mode,
- Cipher Block Chaining (CBC) mode,
- Cipher Feedback (CFB) mode,
- Output Feedback (OFB) mode,
- Counter (CTR) mode.


---


### Triple DES (3DES) `64-bit text, 3 x 56 (168)-bit DES keys`

the Data Encryption Standard 56-bit key is no longer considered adequate in the face of modern cryptanalytic techniques and supercomputing power.

Triple DES
- a symmetric block cipher
- Improve he known weaknesses of DES.
- uses the same algorithm to produce a more secure encryption.
  - multiple encryption.
  - it encrypts data using the DES algorithm in three separate passes and uses multiple keys.
  - Just as DES encrypts data in 64-bit blocks, 3DES also encrypts data in 64-bit blocks.

- Although 3DES is a strong algorithm, it isn’t used as often as AES today.
  - AES is much less resource intensive.
  - if hardware doesn’t support AES, 3DES is a suitable alternative.
  - 3DES uses key sizes of 56 bits, 112 bits, or 168 bits.

Nested application of DES with three different keys KA, KB, and KC,
- three 56-bit DES keys (total 168 bits), a strong encryption algorithm.
- Effective key length is 168 bits, making exhaustive search attacks unfeasible
- C = EKC(DKB(EKA(P)));
- P = DKA(EKB(DKC(C)))


```py
pip install pyDES

import pyDes

data = "DES Algorithm Implementation"
k = pyDes.des(
      "DESCRYPT",
      pyDes.CBC,
      "\0\0\0\0\0\0\0\0",
      pad=None,
      padmode=pyDes.PAD_PKCS5
    )
d = k.encrypt(data)

print "Encrypted: %r" % d
print "Decrypted: %r" % k.decrypt(d)
assert k.decrypt(d) == data
```

- Equivalent to DES when KA=KB=KC (backward compatible)
  - the security of 3DES varies based on the way it is implemented.
  - 3 keying options:
    - 3 keys are different (keying option 1);
    - 2 keys are the same (keying option 2);
    - 3 keys are the same (keying option 3) to maintain backwards compatibility with DES.

---

### IDEA International Data Encryption Algorithm `64-bit text, 128-bit key`
The International Data Encryption Algorithm (IDEA)
- developed in response to complaints about the insufficient key length of the DES algorithm.
- 国际数据加密算法
- block cipher
- Like DES, 64-bit blocks of plain text/ciphertext.
  - However, operation with a 128-bit key.
  - This key is broken up in a series of operations into 52 16-bit subkeys.

- The subkeys then act on the input text using a combination of XOR and modulus operations to produce the encrypted/decrypted version of the input message.
- IDEA can operate in same 5 modes used by DES: ECB, CBC, CFB, OFB, and CTR.

The IDEA algorithm is patented by its Swiss developers.
- However, they have granted an unlimited license to anyone who wants to use IDEA for noncommercial purposes.
- One popular implementation of IDEA is found in Phil Zimmerman’s popular `Pretty Good Privacy (PGP) secure email package`.

---

### Blowfish `64-bit text, 32-448 bits key`

- Bruce Schneier: to replace DES.
- strong symmetric block cipher
  - still widely used today.
- another alternative to DES and IDEA.
- encrypts data in 64-bit blocks of text.
- supports key sizes between 32 and 448 bits.

- However, it extends IDEA’s key strength even further by allowing the use of variable-length keys, ranging from insecure 32 bits to extremely strong 448 bits.
- Obviously, the longer keys will result in a corresponding increase in encryption/decryption time.
- Blowfish faster than both IDEA and DES in some instances.
- Part of the reason is that Blowfish encrypts data in smaller 64-bit blocks, AES encrypts data in 128-bit blocks.
- Mr. Schneier released Blowfish for public use with no license required.
- Blowfish encryption is built into a number of commercial software products and operating systems.
- A number of Blowfish libraries are also available for software developers.

	https://wemedia.ifeng.com/76674795/wemedia.shtml
	https://my.oschina.net/u/3664884/blog/1607425
	https://blog.csdn.net/findmyself_for_world/article/details/50222081
	https://blog.51cto.com/yinghao/563302

---


### Twofish `128-bit blocks, 128, 192, 256-bit keys`
developed by Bruce Schneier (also the creator of Blowfish) was another one of the AES finalists.
- a block cipher.
- related to Blowfish, but encrypts data in 128-bit blocks
- supports 128-, 192-, or 256-bit keys.

It was one of the finalist algorithms evaluated by NIST for AES.
- However, NIST selected Rijndael as AES instead.

Twofish uses two techniques not found in other algorithms:
- Prewhitening involves XORing the plain text with a separate subkey before the first round of encryption.
- Postwhitening uses a similar operation after the 16th round of encryption.


---


### Advanced Encryption Standard (AES) `128-bit, 128, 192, 256 bit key`

U.S. National Institute for Standards and Technology (NIST)
- 1997, NIST want replace DES.
- five finalists, and ultimately chose Advanced Encryption Standard (AES).
- October 2000, NIST announced that the Rijndael block cipher had been chosen as the replacement for DES.
- November 2001, NIST released FIPS 197, which mandated the use of AES/Rijndael for the encryption of all sensitive but unclassified data by the US government.


- a <font color=red> strong symmetric block cipher </font>
  - Same key
  - typically considered the preferred symmetric encryption algorithm.
  - Exhaustive search attack not currently possible
- Because of its strengths, AES has been adopted in a wide assortment of applications.
  - example
  - many applications that encrypt data on USB drives use AES.


- <font color=red> encrypts data in 128-bit blocks </font>
  - several possible AES key sizes:
    - 128-bit keys require 10 rounds of encryption.
    - 192-bit keys require 12 rounds of encryption.
    - 256-bit keys require 14 rounds of encryption.
    - (AES-128, AES-192, or AES-256)
  - Plain and cipher text, same size.
  - more bits are used, more difficult to discover the key and decrypt the data.
  - Longer keys for a specific algorithm result in stronger key strength.


strengths
- Fast:
  - uses elegant mathematical formulas and only requires one pass to encrypt and decrypt data.
  - In contrast, 3DES requires multiple passes to encrypt and decrypt data.
- Efficient
  - less resource intensive than other encryption algorithms such as 3DES.
  - encrypts and decrypts quickly even when ciphering data on small devices,
  - such as USB flash drives.
- Strong:
  - trong encryption of data, high level of confidentiality.


---

#### AES Round Structure

The 128-bit version of the AES encryption algorithm proceeds in `ten rounds`.
- Each round performs an invertible transformation on a 128-bit array, called state.
- The initial state X0 = <font color=blue> XOR of the plaintext P with the key K </font>
  - X0 = P XOR K.
- The ciphertext C is the output of the final round:
  - C = X10.

Each round is built from 4 basic steps:
1. SubBytes step: an S-box substitution step.
2. ShiftRows step: a permutation step.
3. MixColumns step: a matrix multiplication 矩阵乘法 step.
4. AddRoundKey step: an XOR step with a round key derived from the 128-bit encryption key.

![Screen Shot 2018-11-17 at 15.36.40](https://i.imgur.com/BwNSJAW.png)

Java AES Encryption Example
- [Source](https://java.sun.com/javase/6/docs/technotes/guides/security/crypto/CryptoSpec.html)

```java
// Generate an AES
keyKeyGenerator keygen = KeyGenerator.getInstance("AES");
SecretKey aesKey = keygen.generateKey();

// Create a cipher object for AES in ECB mode and PKCS5 padding
Cipher aesCipher;
aesCipher = Cipher.getInstance("AES/ECB/PKCS5Padding");

// Encrypt
aesCipher.init(Cipher.ENCRYPT_MODE, aesKey);
byte[] plaintext = "My secret message".getBytes();
byte[] ciphertext = aesCipher.doFinal(plaintext);


// Decrypt
aesCipher.init(Cipher.DECRYPT_MODE, aesKey);
byte[] plaintext1 = aesCipher.doFinal(ciphertext);
```


---


### Skipjack algorithm `64-bit`


- block ciphers
  - 64-bit blocks of text.
  - 80-bit key
- supports the same 4 modes of operation supported by DES.

- Skipjack was quickly embraced by the US government
  - provides the cryptographic routines supporting the Clipper and Capstone encryption chips.
  - approved for use by the US government in Federal Information Processing Standard (FIPS) 185, the Escrowed Encryption Standard (EES).

- However, Skipjack has an added twist—it supports the escrow of encryption keys.
- Two government agencies, NIST and the Department of the Treasury, hold a portion of the information required to reconstruct a Skipjack key.
- When law enforcement authorities obtain legal authorization, they contact the two agencies, obtain the pieces of the key, and are able to decrypt communications between the affected parties.
- Skipjack and the Clipper chip not used by the cryptographic community at large because of its mistrust 不信任 of the escrow procedures in place within the US government.



---


## stream cipher

---

### Ron’s Code / Rivest Cipher.

Ron Rivest invented several versions of RC



#### RC4 `40-2048 bits`
The most commonly used version is RC4 (ARC4)
- a symmetric stream cipher
- optimized for confidential communications,
  - like bidirectional voice and video

- can use between 40 and 2,048 bits.

- enjoyed a long life as a strong cipher.
  - For many years, it has been the recommended encryption mechanism in SSL and TLS, when encrypting HTTPS connections on the Internet.

> However, experts have speculated since 2013 that agencies such as the U.S. National Security Agency (NSA) can break RC4, even when implemented correctly such as in TLS.
> - Because of this, companies such as Microsoft recommend disabling RC4 and using AES instead.
> - Even though AES is a block cipher and RC4 is a stream cipher, TLS can implement either one.



#### RC5 `32, 64, or 128 bits`
- a symmetric algorithm
  - patented by Rivest, Shamir, and Adleman (RSA) Data Security, the people who developed the RSA asymmetric algorithm.
- a block cipher
- variable block sizes (32, 64, or 128 bits)
- uses key sizes between 0 length and 2,040 bits.
- faster, larger key size.


#### RC6
- cipher derived from RC5.



---











.










.
