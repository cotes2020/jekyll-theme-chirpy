---
title: Cryptography - Asymmetric Encryption
date: 2018-05-18 11:11:11 -0400
categories: [13Cryptography]
tags: [cryptography]
toc: true
image:
---

- [Cryptography - Asymmetric Encryption](#cryptography---asymmetric-encryption)
  - [major advantage](#major-advantage)
  - [disadvantage:](#disadvantage)
- [Algorithms:](#algorithms)
  - [pretty good privacy (PGP):](#pretty-good-privacy-pgp)
  - [Diffie-Hellman (DH)](#diffie-hellman-dh)
  - [RSA (Rivest–Shamir–Adleman)](#rsa-rivestshamiradleman)
    - [Disadvantages:](#disadvantages)
    - [RSA code](#rsa-code)
    - [brute force](#brute-force)
    - [key generation:](#key-generation)
  - [El Gamal](#el-gamal)

---

# Cryptography - Asymmetric Encryption

- also called
  - **Public-Key Encryption**
  - Two-key systems
  - **public key cryptography (PKC)**
- In 1976, Whitfield Diffie and Martin R. Hellman published the first public key exchange protocol


- Asymmetric key cryptography relies on `NP-hard problem`:
  - a math problem is considered NP-hard if it cannot be solved in polynomial time.
  - X^2, X^3,
  - NP-hard problem: 2^X.


- Asymmetric cryptography relies on types of problem that are relatively easy to solve one way but are extremely difficult to solve the other way. (Trapdoor function)
  - 233x347=80851
  - 80851=?x? (Hard)


- adders the most serious problem of symmetric encryption: **key distribution**.
  - no fear of authorized key disclosure.

---

## major advantage

- <font color=red> Key distribution </font> is a simple process
  - The addition of new users requires the generation of only one public-private keypair.
    - makes the algorithm extremely scalable.
    - Users who want to participate in the system simply make their public key available to anyone with whom they want to communicate.
    - There is no method the private key can be derived from the public key.
  - Users can be removed far more easily from asymmetric systems.
    - Asymmetric cryptosystems provide a key revocation mechanism that allows a key to be canceled, effectively removing a user from the system.

- <font color=red> Key regeneration </font>
  - required only when a user’s private key is compromised.
  - If a user leaves the community, the system administrator simply needs to invalidate that user’s keys.
  - No other keys are compromised, therefore key regeneration is not required for any other user.

- <font color=red> provide integrity, authentication, and nonrepudiation </font>
  - If a user does not share their private key with other individuals, a message signed by that user can be shown to be accurate and from a specific source and cannot be later repudiated.

- <font color=red> No preexisting communication link needs to exist </font>
  - individuals can begin communicating securely from the moment they start communicating.
  - does not require a preexisting relationship to provide a secure mechanism for data exchange.


## disadvantage:
- very strong, but very <font color=red> resource intensive </font>
  - takes a significant amount of processing power to encrypt and decrypt data,
  - especially when compared with symmetric encryption.


- higher security but <font color=red> slower </font>
  - not for large quantities of real-time data.
  - might be used to encrypt a small chunk of data.

- Most cryptographic protocols use asymmetric encryption only for <font color=red> key exchange </font>
  - secure transmission of large amounts of data use <font color=blue> public key cryptography to establish a connection and then exchange a symmetric secret key </font>
  - The remainder of the session then uses symmetric cryptography.
  - Key exchange:
    - share cryptographic keys between two entities.
    - use asymmetric encryption to key exchange, share a symmetric key.
    - uses the symmetric encryption to encrypt and decrypt data (much more efficient).


![Pasted Graphic](https://i.imgur.com/E8U83A3.png)

- uses asymmetric (different) keys for the sender and the receiver.
  - C = EPB (M)
  - M = DSB (C)

- Once the sender encrypts the message with the recipient’s public key, no user (including the sender) can decrypt that message without knowing the recipient’s private key.
  - public keys can be freely shared using insecured communications
  - create secure communications between users previously unknown to each other.

- public key cryptography entails a higher degree of computational complexity. Keys must be longer than those used in private key systems to produce cryptosystems of equivalent strengths.
- example:
  - authenticate the other party in a conversation
  - or to exchange a shared key to be used during a session (after which, the parties in the conversation could start using symmetric encryption).


---

# Algorithms:

---


## pretty good privacy (PGP):
often used to encrypt e-mail traffic. A free variant of PGP is GNU Privacy Guard (GPC).

example:
- when client A wants to communicate securely with server 1.
  - Client A requests server 1’s digital certificate.
  - Server 1 sends its digital certificate,
  - client A knows the received certificate is really from server 1.
    - because the certificate has been authenticated (signed) by a trusted third party, called a **certificate authority**.
  - Client A extracts 提取 server 1’s `public key` from server 1’s digital certificate.
    - Data encrypted using server 1’s public key can only be decrypted with server 1’s private key.
  - Client A generates a random string of data called a session key.
    - The session key is then encrypted using server 1’s public key and sent to server 1.
  - Server 1 decrypts the session key using its private key.
	At this point, both client A and server 1 know the session key, use it to symmetrically encrypt traffic during the session.


---


## Diffie-Hellman (DH)

Whitfield Diffie and Martin Hellman first published the Diffie-Hellman scheme in 1976.
- Interestingly, Malcolm J. Williamson secretly created a similar algorithm while working in a British intelligence agency. It is widely believed that the work of these three provided the basis for public-key cryptography.

a key exchange algorithm
- to privately share a symmetric key between two parties.
- Once the two parties know the symmetric key, they use symmetric encryption to encrypt the data.


Diffie-Hellman methods
- uses large integers and modular arithmetic
- support both static keys and ephemeral keys.
  - RSA is based on the Diffie-Hellman key exchange concepts using static keys.
- Diffie-Hellman methods that use ephemeral keys are:
  - <font color=red> Diffie-Hellman Ephemeral (DHE) </font>
    - uses ephemeral keys, generating different keys for each session.
    - Some documents list this as Ephemeral Diffie-Hellman (EDH).
  - <font color=red> Elliptic Curve Diffie-Hellman Ephemeral (ECDHE) </font>
    - uses ephemeral keys generated using ECC.
  - <font color=red> Elliptic Curve Diffie-Hellman (ECDH) </font>
    - uses static keys.

When Diffie-Hellman is used, the two parties negotiate the strongest group that both parties support.
- There are currently more than 25 DH (Diffie Hellman) groups in use
- defined as DH Group 1, DH Group 2, and so on.
- Higher group more secure.
- Example
  - DH Group 1 uses 768 bits in the key exchange process
  - DH Group 15 uses 3,072 bits.


---

## RSA (Rivest–Shamir–Adleman)

inventors: Ronald L. Rivest, Adi Shamir, and Leonard M. Adleman.
- They patented their algorithm and formed a commercial venture known as RSA Security to develop mainstream implementations of their security technology.
- early public key encryption system that uses large integers as the basis for the process
- patented in 1977, released its patent to the public about 48 hours before it expired in 2002.

RSA algorithm:
- asymmetric encryption
- security backbone of many well-known security infrastructures: Microsoft, Nokia, and Cisco.
- RSA works with both encryption and digital signatures, used in many environments,
  - like Secure Sockets Layer (SSL), and key exchange.
  - commonly used as part of a `public key infrastructure (PKI) system`.
    - PKI uses <font color=blue> digital certificates and a certificate authority (CA) </font> to allow secure communication across a public network.
    - But ECDH better in PKI for key agreement.

- based on <font color=red> factoring 2 larger primes </font>
  - the computational difficulty inherent in factoring large prime numbers.
  - uses the mathematical properties of prime numbers to generate secure public and private keys.
  - it is difficult to factor the product of two large prime numbers.

- RSA is secure if sufficient key sizes are used.
  - RSA laboratories recommend a key size of 2,048 bits to protect data through the year 2030.
  - If data needs to be protected beyond 2030, they recommend a key size of 3,072 bits.


### Disadvantages:
- much slower than the those for existing symmetric encryption schemes.
- Require in practice a key length larger than that for symmetric cryptosystems.
  - RSA: 2048-bit keys
  - AES: 256-bit keys.



### RSA code

```py
# a 除以 m 所得的余数记作 a mod m.
# 如果 a mod m = b mod m
# 即 a, b 除以 m 所得的余数相等，那么我们记作：a≡b(mod m)

from Crypto import Random
from Crypto.PublicKey import RSA
import base64

def generate_keys():
   # key length must be a multiple of 256 and >= 1024
   key_length = 256*4
   privatekey = RSA.generate(key_length, Random.new().read)
   publickey = privatekey.publickey()
   return privatekey, publickey

def encrypt_message(plaintext , publickey):
   encrypted_msg = publickey.encrypt(plaintext, 32)[0]
   encoded_encrypted_msg = base64.b64encode(encrypted_msg)
   return encoded_encrypted_msg

def decrypt_message(encoded_encrypted_msg, privatekey):
   decoded_encrypted_msg = base64.b64decode(encoded_encrypted_msg)
   decoded_decrypted_msg = privatekey.decrypt(decoded_encrypted_msg)
   return decoded_decrypted_msg

plaintext = "This is the illustration of RSA algorithm of asymmetric cryptography"
privatekey , publickey = generate_keys()
encrypted_msg = encrypt_message(plaintext , publickey)
decrypted_msg = decrypt_message(encrypted_msg, privatekey)

print "%s - (%d)" % (privatekey.exportKey() , len(privatekey.exportKey()))
print "%s - (%d)" % (publickey.exportKey() , len(publickey.exportKey()))
print "Original content: %s - (%d)" % (plaintext, len(plaintext))
print "Encrypted message: %s - (%d)" % (encrypted_msg, len(encrypted_msg))
print "Decrypted message: %s - (%d)" % (decrypted_msg, len(decrypted_msg))
```


```py
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
from Crypto.Signature import PKCS1_v1_5
from Crypto.Hash import SHA512, SHA384, SHA256, SHA, MD5
from Crypto import Random
from base64 import b64encode, b64decode
hash = "SHA-256"

def newkeys(keysize):
   random_generator = Random.new().read
   key = RSA.generate(keysize, random_generator)
   private, public = key, key.publickey()
   return public, private

def importKey(externKey):
   return RSA.importKey(externKey)

def getpublickey(priv_key):
   return priv_key.publickey()

def encrypt(message, pub_key):
   cipher = PKCS1_OAEP.new(pub_key)
   return cipher.encrypt(message)

def decrypt(ciphertext, priv_key):
   cipher = PKCS1_OAEP.new(priv_key)
   return cipher.decrypt(ciphertext)

def sign(message, priv_key, hashAlg = "SHA-256"):
   global hash
   hash = hashAlg
   signer = PKCS1_v1_5.new(priv_key)
   if (hash == "SHA-512"): digest = SHA512.new()
   elif (hash == "SHA-384"): digest = SHA384.new()
   elif (hash == "SHA-256"): digest = SHA256.new()
   elif (hash == "SHA-1"): digest = SHA.new()
   else: digest = MD5.new()
   digest.update(message)
   return signer.sign(digest)

def verify(message, signature, pub_key):
   signer = PKCS1_v1_5.new(pub_key)
   if (hash == "SHA-512"): digest = SHA512.new()
   elif (hash == "SHA-384"): digest = SHA384.new()
   elif (hash == "SHA-256"): digest = SHA256.new()
   elif (hash == "SHA-1"): digest = SHA.new()
   else: digest = MD5.new()
   digest.update(message)
   return signer.verify(digest, signature)
```


### brute force

```py
def p_and_q(n):
   data = []
   for i in range(2, n):
      if n % i == 0:
         data.append(i)
   return tuple(data)

def euler(p, q):
   return (p - 1) * (q - 1)

def private_index(e, euler_v):
   for i in range(2, euler_v):
      if i * e % euler_v == 1:
         return i

def decipher(d, n, c):
   return c ** d % n

def main():
    e = int(input("input e: "))
    n = int(input("input n: "))
    c = int(input("input c: "))
    # t = 123
    # private key = (103, 143)
    p_and_q_v = p_and_q(n)
    # print("[p_and_q]: ", p_and_q_v)
    euler_v = euler(p_and_q_v[0], p_and_q_v[1])
    # print("[euler]: ", euler_v)
    d = private_index(e, euler_v)
    plain = decipher(d, n, c)
    print("plain: ", plain)

if __name__ == "__main__":
   main()
```



### key generation:

```bash
# 1. Choose two large prime numbers p,q (approximately 200 digits each): p and q, of approximately equal size such that their product, n = pq, is of the required bit length (such as 2,048 bits, 4,096 bits, and so forth).

# 2. Compute the product of those two numbers:
# Generate the RSA modulus
   N = p * q.
   m = (p-1)(q-1)

# 3. Select a number, e, that satisfies the following two requirements:
# Derived Number (e)
   1< e < N
   e  co-prime to m.
  #  (2 numbers have no common factors other than 1.)

# Step 3: Public key
# The specified pair of numbers n and e forms the RSA public key and it is made public.


# 4. Find d: Private Key d is calculated from the numbers p, q and e. The mathematical relationship between the numbers is as follows −

ed = 1 mod (p-1) (q-1)

  (ed – 1) mod (p – 1)(q – 1) = 0.
  de mod m ≡ 1

# 5. Distribute e and n as the public key to all cryptosystem users.

# 6. Keep d secret as the private key.

# Encryption Formula
# Consider a sender who sends the plain text message to someone whose public key is (n,e). To encrypt the plain text message in the given scenario, use the following syntax −
C = Pe mod n

# Decryption Formula
# The decryption process is very straightforward and includes analytics for calculation in a systematic approach. Considering receiver C has the private key d, the result modulus will be calculated as −
Plaintext = Cd mod n



If Alice wants to send an encrypted message to Bob,
Alice generates the ciphertext (C) from the plain text (P):
C = P^e mod n
# e is Bob’s public key
# n is the product of p and q created during the key generation process
When Bob receives the message, he retrieve the plaintext message:
P = C^d mod n
	https://www.youtube.com/watch?v=wXB-V_Keiu8



To make this even more clear, let’s look at an example:
1. Select primes: p = 17 and q =11
1. Compute n = pq =17×11 = 187
1. Compute ø(n) = (p–1)(q-1) = 16×10 = 160
1. Select e = 7
1. Find d, such that de mod m ≡ 1, d = 23
1. Since 23×7 = 161 mod M (160) = 1
1. Publish public key 7
1. Keep secret private key 23
1. Now let’s use these keys. Use the number 3 as the plain text. Remember e = 7, d = 23, and n = 187.
1. Cipher text = Plaintexte  mod n = 37 mod 187                   = 2187 mod 187 = 130
1. Plaintext = Cipher textd  mod nPlaintext = 13023 mod 187Plaintext = 4.1753905413413116367045797e+48mod 187
1. Plaintext = 3
```


---


## El Gamal

Diffie-Hellman: uses large integers and modular arithmetic to facilitate the secure exchange of secret keys over insecure communications channels.

In 1985, Dr. T. El Gamal
- how the mathematical principles behind the Diffie-Hellman key exchange algorithm could be extended to support an entire public key cryptosystem used for encrypting/decrypting messages.
- Dr. El Gamal did not obtain a patent on his extension of Diffie-Hellman, and it is freely available for use, unlike the then-patented RSA technology. (RSA released its algorithm into the public domain in 2000.)


el Gamal:
- based on Diffie-Hellman, relies on discrete logarithms.


one of the major advantages of El Gamal over the RSA algorithm:
- it was released into the public domain.

Major disadvantage:
- the algorithm doubles the length of any message it encrypts.
- Hard when encrypting long messages/data then transmitte over a narrow bandwidth communications circuit.

---
