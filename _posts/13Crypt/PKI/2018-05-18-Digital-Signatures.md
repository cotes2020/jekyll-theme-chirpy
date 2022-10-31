---
title: Cryptography - Digital certificates
date: 2018-05-18 11:11:11 -0400
categories: [13Cryptography, PKI]
tags: [cryptography]
toc: true
image:
---


[toc]

---


# Digital certificates

---

## Basic

Digital certificates
- electronic credentials
  - Most commonly they contain a public key and the identity of the owner.
  - used to assert the online identities of individuals, computers, and other entities on a network.
- They are issued by certification authorities (CAs)
  - who validate the identity of the certificate-holder both before the certificate is issued and when the certificate is used.
- Common uses include business scenarios requiring authentication, encryption, and digital signing.


1. <font color=red> combine asymmetric cryptography and hashing </font>

2. Implement a digital signature system for 2 distinct goals:
   - assure the recipient that <font color=red> the message truly came from the claimed sender. </font>
     - enforce nonrepudiation
     - preclude 阻止 the sender from claiming the message is forgery

   - assure the recipient that <font color=red> the message was not altered </font> while in transit between the sender and recipient.
     - against malicious modification
       - a third party altering the meaning of the message
     - and against unintentional modification
       - because of faults in the communications process, such as electrical interference
   ￼

3. Digital signatures are used for more than just messages.
   - Software vendors often use digital signature technology to authenticate code distributions that you download from the Internet, such as applets and software patches.
   - digital signature process does not provide any privacy in and of itself.
   - It only ensures that the cryptographic goals of <font color=red> integrity, authentication, and nonrepudiation </font>


4. Essentially are endorsed copies of an individual’s public key.
   - digital certificate binds an individual’s identity to a public key, contains all info a receiver needs to be assured of the identity of the public key owner.



---

## Purposes

The certificate purpose can be one of four settings:

- <font color=red> Encryption.</font> A certificate with this purpose will contain cryptographic keys for encryption and decryption.

- <font color=red> Signature.</font> A certificate with this purpose will contain cryptographic keys for signing data only.

- <font color=red> Signature and encryption.</font> A certificate with this purpose covers all primary uses of a certificate’s cryptographic key, including encryption of data, decryption of data, initial logon, or digitally signing data.

- <font color=red> Signature and smartcard logon.</font> A certificate with this purpose allows for initial logon with a smart card, and digitally signing data; it cannot be used for data encryption.

SSL is probably the first protocol to use digital certificates.
- Now a days they are widely used where ever there is a need for signing and encryption.


Certificate:
- a mechanism that associates the public key with an individual,
- contains a great deal of information about the user.
- Each user of a PKI system has a certificate used to verify their authenticity.


1:1 correspondence does not necessarily exist between identities and certificates.
- An entity can have multiple key pairs for separate purposes.
  - multiple certificates,
  - each attesting to separate public key ownership.
- It is also possible to have different classes of certificates, again with different keys.
- This flexibility allows entities total discretion in how they manage their keys, and the PKI manages the complexity by using a unified process that allows key verification through a common interface.


If an application creates a key store that can be accessed by other app, it will provide a standardized interface, application programming interface (API).
- In Mozilla and Linux systems,
  - this interface is usually PKCS #11,
- in Microsoft applications
  - the interface is Cryptography API: Next Generation (CNG) for Microsoft Vista and later.
- example
- Application A went through the process of registering a certificate and generating a key pair.
- It created a key store that provides an interface to allow other applications to communicate with it and use the items held within the store.

The local key store is just one location where these items can be held.
- Often the digital certificate and public key are also stored in a certificate repository so that they are available to a subset of individuals.


---

## Step

![Screen Shot 2021-01-08 at 14.42.04](https://i.imgur.com/VnhkqSr.png)

If Alice wants to digitally sign a message she’s sending to Bob, she performs the following actions:
1. Alice <font color=red> generates a hashing message digest </font> of the original plaintext message </font>
   - such as SHA-512.
2. Alice <font color=red> encrypts the hash using private key </font>
   - This encrypted message digest is the digital signature.
3. Alice <font color=red> appends the digital signature with the plaintext message </font>
4. Alice transmits the appended message to Bob.
5. When Bob receives the digitally signed message, he reverses the procedure:
   - decrypts the digital signature using Alice’s public key.
     - got hash and plaintext.
   - uses the same hashing function to create a message digest of the plaintext message received.
   - compares the decrypted hash received with the hash he computed.
     - If match, the message he received was sent by Alice.
     - If do not match, either the message was not sent by Alice or the message was modified while in transit.

![Pasted Graphic 43](https://i.imgur.com/FO7VBvz.png)


If Alice wanted to ensure the privacy of her message to Bob, she could add a step to the message creation process.

- After appending the signed message digest to the plaintext message, Alice could encrypt the entire message with Bob’s public key.
- When Bob received the message
  - decrypt it with his own private key
  - following the steps just outlined.


---


## DSS Digital Signature Standard

Digital Signature Standard (DSS).
- The National Institute of Standards and Technology specifies the digital signature algorithms acceptable for federal government use in Federal Information Processing Standard (FIPS) 186-4
- This document
  - specifies all federally approved digital signature algorithms must use the SHA-2 hashing functions.
  - specifies the 3 currently approved standard encryption algorithms that can be used to support a digital signature infrastructure:
    - The <font color=red> Digital Signature Algorithm (DSA) </font> as specified in FIPS 186-4
    - The <font color=red> Rivest, Shamir, Adleman (RSA) algorithm </font> as specified in ANSI X9.31
    - The <font color=red> Elliptic Curve DSA (ECDSA) </font> as specified in ANSI X9.62

2 other digital signature algorithms you should recognize,
- Schnorr’s signature algorithm > 比特币
- Nyberg-Rueppel’s signature algorithm.



---


## Certificate Issues

Before clients use a certificate, they first verify it is valid with some checks.

different certificate issues that can result in an invalid certificate.
- Browsers typically display an error describing the issue and encouraging users not to use the certificate.
- Applications that detect a certificate issue might display an error using a certificate, but they are typically coded to not use it.

Some of the common issues are:
- Expired.
  - The first check is to ensure that it isn’t expired.
  - If the certificate is expired, the computer system typically gives the user an error indicating the certificate is not valid.
- Certificate not trusted.
  - The next check is to see if the certificate was issued by a trusted CA.

- Example
  - Windows system will look in the `Trusted Root Certification Authority store`
  - If the system doesn’t have a copy of the CA’s certificate, it will indicate the certificate is not trusted.
  - Users can override this warning, though there are often warnings encouraging users not to continue.
- Improper certificate and key management.
  - Private keys should remain private.
  - If the certificates holding the private keys aren’t managed properly, it can compromise the certificate.
