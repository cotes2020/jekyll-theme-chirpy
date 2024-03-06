---
title: Cryptography - Digital Certificates Verification
date: 2018-05-18 11:11:11 -0400
categories: [13Cryptography, PKI]
tags: [cryptography]
toc: true
image:
---


[toc]

---


# Digital Certificates Verification


---

## Trust and Certificate Verification

- When a user chooses to trust a CA
  - download that CA’s digital certificate and public key, stored on her local computer.
  - Most browsers have a list of CAs configured to be trusted by default
  - when a user installs a new web browser
    - most well-known trusted CAs will be trusted without any change of settings.

- In the Microsoft CNG environment,
  - the user can add and remove CAs from this list as needed.

- In production environments
  - require a higher degree of protection,
  - this list will be pruned, and possibly the only CAs listed will be the company’s internal CAs.
  - ensures that digitally signed software will be automatically installed only if it was signed by the company’s CA.
  - Other products, such as Entrust, use centrally controlled policies to determine which CAs are to be trusted, instead of expecting the user to make these critical decisions.


The use of digital signatures
- allows certificates to be saved in public directories without the concern of them being accidentally or intentionally altered.
- client's message digest value does not match the digital signature embedded in the certificate itself
  - the certificate has been modified.
  - not to accept the validity of the corresponding public key.
- Similarly, an attacker could not create a new message digest, encrypt it, and embed it within the certificate because he would not have access to the CA’s private key.


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



## Obtaining a Certificate From CA

obtain a certificate for your business from commercial CAs.
- The Issuing entities of commercial CAs provide certificate with a cost.

- User without key
  - directly approach to a issuing CA
  - issuing CA will generate a Key pair on user's behalf.
  - give the private key and certificate containing the public key with Issuing CA's signature to user
    - after all necessary validations as per CA's policy.

- User with key
  - User generate key and CSR then send to Issuing CA for a certificate.
    - a Key pair
      - tool like Keytool in Java
    - a Certificate Signing Request (CSR)
      - tool like Keytool
      - CSR contains the public key of the user and user identity information

![obtain-cert](https://i.imgur.com/nhpVaWr.gif)

User must keep the private key secret.
- If private key is compromised or lost then issuing CA must be informed.
- CAs keep the certificates in Certificate Revocation List whose private keys believed to have been compromised or lost.

self signed certificates
- You can yourself be a CA and issue your own certificates
- but for commercial purpose your self signed certificated will not be trusted.
- Only established and well known CAs self signed certificates are trusted.
- Root certificate of a CA is always self signed.

---

### certification practices statement (CPS)

Every CA should have a <font color=red> certification practices statement (CPS) </font>
- outlines
  - how identities are verified;
    - the steps the CA follows to generate, maintain, and transmit certificates;
  - why the CA can be trusted to fulfill its responsibilities.
  - describes how keys are secured,
  - what data is placed within a digital certificate,
  - and how revocations will be handled.

- If a company is going to use and depend on a public CA
  - security officers and legal department should review the CA’s entire CPS
  - to ensure that it will properly meet the company’s needs,
  - and to make sure that the level of security claimed by the CA is high enough for their use and environment.

- A critical aspect of a PKI is the trust between the users and the CA
  - so the CPS should be reviewed and understood to ensure that this level of trust is warranted.

The certificate server:
- the actual service that issues certificates
  - based on the data provided during the initial registration process.
  - <font color=red> constructs and populates the digital certificate </font> with the necessary information
  - and <font color=red> combines the user’s public key with the resulting certificate </font>
  - The certificate is then <font color=red> digitally signed with the CA’s private key </font>
    - private key assures the recipient that the certificate came from the CA.

---

### certificate signing request (CSR):
- First steps in getting a certificate, submit CSR
  - a request formatted for the CA.
  - have the public key that you wish to use and your fully distinguished name (often a domain name).
  - The CA will then use this to process your request for a digital certificate.


---

## steps involved in checking the validity of a message.

1. Receiver receives a digitally signed message from Sender
   - digitally signed message = digital certificate (public key + hash encrypted with private key) + message

2. <font color=red> Compare the CA </font>
   - To sure the authenticity of this message:
   - Receiver compares the <font color=red> CA signed Sender’s certificate </font> to <font color=red> the list of CAs he has configured/loaded in computer </font>
   - He trusts the CAs in his list and no others.
     - If the certificate was signed by a CA he does not have in the list,
       - not accept the certificate as being valid.
     - If the certificate was signed by a CA in his list of trusted CAs,
       - accepted.

3. Use the <font color=red>  CA’s public key to decrypt the digital signature </font>
   - recover the original message digest embedded within the certificate (validating the digital signature).

4. <font color=red> Calculate a message digest for the certificate </font>

5. <font color=red> Compare the two resulting message digest values </font> to ensure the integrity of the certificate.
   - verify that the certificate has not been altered.
   - Using the CA’s public key and the digest certificate,
     - Receiver can verify the integrity of the certificate.
     - this CA did actually create the certificate
   - he now trust the origin of Sender’s certificate.

6. Review the identification information within the certificate
   - such as the e-mail address.

7. Review the validity dates.

8. Check a revocation list to see if the certificate has been revoked.
   - not done yet
   - now verify not revoked this certificate.
     - If the start date hasn’t happened yet
     - the stop date has been passed, the certificate is not valid.
     - Receiver reviews to make sure it is still deemed valid.
   - now verify whether this certificate has been revoked for any reason
     - refer to a list of revoked certificates see if Sender’s certificate is listed.
     - checked online: Online Certificate Status Protocol (OCSP).

9. Receiver now trusts that this certificate is legitimate and that it belongs to Sender.


10. now <font color=red> verify the integrity of the message </font>
    - Receiver runs the message
      - calculates a message digest value of X.
    - The certificate holds <font color=blue> Sender’s public key </font>
       - Receiver extracts Sender’s public key from certificate.
       - <font color=blue> uses Sender’s public key to decrypt digital signature </font>
       - a digital signature:
         - message digest encrypted with a private key.
         - decryption get a hash of value Y.
    - compares values X and Y,
       - if they are the same, the message has not been modified during transmission.
        - decrypt the digital signature by sender public key,
        - Receiver know that the message actually came from Sender.

11. reads her message, useful message.
    - Fortunately, all of this PKI work is performed without user intervention and happens behind the scenes.
    - Receiver didn’t have to exert any energy.
    - He simply replies, “Fine. How are you?”









 .
