---
title: Cryptography - Certification Authority
date: 2018-05-18 11:11:11 -0400
categories: [13Cryptography, PKI]
tags: [cryptography]
toc: true
image:
---

[toc]


- ref:
  <!-- - [Basics of Digital Certificates and Certificate Authority](https://sites.google.com/site/ddmwsst/digital-certificates#TOC-PEM-Privacy-Enhanced-Mail-Encoding) -->


---

# Authority


---

## Registration authorities (RAs):
- assist Certificate authorities (CAs)
  - RA help offload work from the CA.
- verify users’ identities prior to issuing digital certificates.
  - providing certificates to users, requires server, CAs can become overloaded and need assistance.
- operates as an intermediary in the process:
  - <font color=red> distribute keys </font>
  - <font color=red> accept registrations for a digital certificate for CA </font>
  - <font color=red> register, authenticate and validate identities, allowing CAs to remotely validate user identities </font>
    - require proof of identity from the individual requesting a certificate
    - validate this information
    - then advise the CA to generate a certificate
      - The CA will
      - digitally sign the certificate using its private key.
      - use the RA-provided information to generate a digital certificate,
      - integrate the necessary data into the certificate fields and send a copy of the certificate to the user.
  - <font color=red> The RA doesn’t directly issue certificates, CA do it </font>


---

## CA - Certificate Authority

- offer notarization services for digital certificates.
  - issuing, revoking, and distributing certificates.
  - To obtain a digital certificate from a reputable CA, you must prove your identify to the satisfaction of the CA.

- issues digital certificates (contain a public key and the identity of the owner)
  - private key is not made available publicly,
  - but kept secret by the end user who generated the key pair.

- The certificate is also a confirmation or validation by the CA that the public key contained in the certificate belongs to the person, organization, server or other entity noted in the certificate.

- A CA's obligation in such schemes is to verify an applicant's credentials, so that users and relying parties can trust the information in the CA's certificates.

- CAs use a variety of standards and tests to do so.

- In essence, the Certificate Authority is responsible for saying "yes, this person is who they say they are, and we, the CA, verify that".

If the user trusts the CA and can verify the CA's signature, then he can also verify that a certain public key does indeed belong to whoever is identified in the certificate.
- Browsers maintain list of well known CAs root certificates.
- Aside from commercial CAs, some providers issue digital certificates to the public at no cost.
- Large institutions or government entities may have their own CAs.

- can be online (common) / offline.
  - Online CA:
    - most. always connected and accessible.
  - Offline CA:
    - usually for a root CA that has been isolated from network access.
  - since isolated, compromised are reduced.

- can be either private or public,
  - public CAs:
    - very large. make money by selling certificates.
    - providing certificates to the general public.
    - the public CA must be trusted.
  - Private CAs:
    - Many operating system providers allow their systems to be configured as CA systems
    - to generate internal certificates used within business/large external settings.
    - single service running on a server within a private network


```
The major CAs:
- Symantec
- Thawte
- GeoTrust
- GlobalSign
- Comodo Limited
- Starfield Technologies
- GoDaddy
- DigiCert
- Network Solutions, LLC
- Entrust
```


Nothing is preventing any organization from simply setting up shop as a CA.
- the certificates issued by a CA are only as good as the trust placed in the CA that issued them.
- This is an important item to consider:
  - receiving a digital certificate from a third party.
  - If don’t recognize and trust the name of the CA that issued the certificate,
  - shouldn’t place any trust in the certificate at all.

- PKI relies on a hierarchy of trust relationships:
  - If you configure your browser to trust a CA,
  - it will automatically trust all of the digital certificates issued by that CA.
- Browser developers preconfigure browsers to trust the major CAs to avoid placing this burden on users.




---

### CA Hierarchy

CAs are hierarchical in structure. There are generally three types of hierarchies, and they are denoted by the number of tiers.

---

#### Single/One Tier Hierarchy


<!-- <img src="https://sites.google.com/site/ddmwsst/digital-certificates/ca-1.png?attredirects=0" width="250"> -->

A single tier Hierarchy
- consists of one CA.
- The single CA is both a <font color=red> Root CA and an Issuing CA </font>
  - Root CA
    - is the term for the trust anchor of the PKI.
    - Any applications, users, or computers that trust the Root CA trust any certificates issued by the CA hierarchy.
  - Issuing CA
    - a CA that issues certificates to end entities.
- For security reasons, these two roles are normally separated.
- When using a single tier hierarchy they are combined.



---

#### Two Tier Hierarchy

<!-- <img src="https://sites.google.com/site/ddmwsst/digital-certificates/ca-2.png?attredirects=0" width="200"> -->


A two tier hierarchy
- most common.
- a compromise between the One and Three Tier hierarchies.
- Root CA that is offline, and a subordinate issuing CA that is online.
- <font color=red> The level of security is increased </font>
  - the Root CA and Issuing CA roles are separated.
  - <font color=red> the Root CA is offline </font>
  - so the private key of the Root CA is better protected from compromise.
- <font color=red> increases scalability and flexibility </font>
  - there can be multiple Issuing CA’s that are subordinate to the Root CA.
  - allows CA in different geographical location, and with different security levels.


---

#### Three Tier Hierarchy

<!-- <img src="https://sites.google.com/site/ddmwsst/digital-certificates/ca-3.png?attredirects=0" width="300"> -->

Three Tier Hierarchy
- a second tier is placed between the Root CA and the issuing CA.
- use the second tier CA as a <font color=red> Policy CA </font>
  - Policy CA
    - configured to issue certificates to the Issuing CA that is restricted in what type of certificates it issues.
    - can also be used as an administrative boundary.
      - only issue certain certificates from subordinates of the Policy CA, and perform a certain level of verification before issuing certificates,
      - but the policy is only enforced from an administrative not technical perspective.

- when revoke a number of CAs due to a key compromise,
  - can perform it at the Second Tier level, leaving other “branches from the root” available.

- Second Tier CAs in this hierarchy can, like the Root, be kept offline.


---

## Certificate Chain / trust / Path

When you get a certificate for your public key from a commercial CA
- then <font color=red> your certificate is associated with a chain of certificates / trust </font>
- The number of certificates in the chain depends on the CA's hierarchical structure.
- The following image shows a certificate chain for a two tier CA.
- The owners/users certificate is signed by a Issuing CA and issuing CA's certificate is signed by the Root CA. Root CA's certificate is self signed.


<!-- ![Certificate Chain](https://sites.google.com/site/ddmwsst/digital-certificates/chain-of-trust.gif?attredirects=0) -->


During a User's certificate validation by a browser or a program,
- browser needs to validate the signature by finding the public key of the next issuing CA or intermediate CA.
- The process will continue until the root certificate is reached.
- Root CA is self signed and must be trusted by the browser at the end.
- Browsers keep all well known CAs root certificates in their trust store.


---

### The most common trust model:

hierarchical / centralized trust model.
- the public CA creates the first CA, root CA.
- If the organization is large, it can create intermediate and child CAs.
  - it includes a section used to store intermediate CA certificates.
- A large trust chain works like this:
  - The root CA issues certificates to intermediate CAs.
  - Intermediate CAs issue certificates to child CAs.
  - Child CAs issue certificates to devices or end users.


web of trust / decentralized trust model
- sometimes used with PGP and GPG.
- A web of trust uses self-signed certificates, and a third party vouches for these certificates.
- Example
  - five of your friends trust a certificate, you can trust the certificate.
- If the third party is a reliable source,
  - the web of trust provides a secure alternative.
- if the third party does not adequately verify certificates,
  - the use of certificates that shouldn’t be trusted.



---

### AIA - Authority Information Access Locations

When a client or application is validating a certificate
- it needs to validate the certificate that is being used and also the entire chain of the certificate.
- the application or client needs a certificate from each CA in the chain
- beginning with the issuing CA and ending with the Root CA.


- If the application or client does not have access to the certificates in the chain
  - locally the application or client needs a place from which to obtain the certificates.
  - This location is Authority Information Access - AIA.

AIA location is
- the repository where the CA certificate is stored
- so that it can be downloaded by clients or applications validating a certificate.
- The AIA location is included in the AIA extension of a certificate.



---

### CDP - CRL Distribution Point Locations

A CRL Distribution Point (CDP)
- where clients or applications that are validating a certificate download the certificate revocation list (CRL) to obtain revocation status.
- CA’s periodically publish CRLs to allow clients and applications to determine if a certificate has been revoked.
- CRLs contain the serial number of the certificate that has been revoked, a timestamp indicating when the certificate was revoked, as well as the reason for revocation.

---

## Anatomy of a Certificate

A digital certificate
- binds a user, computer, or service’s identity to a public key by providing information about the subject of the certificate, the validity of the certificate, and applications and services that can use the certificate.

- Certificates issued in PKIs are structured to meet these objectives based on standards established by the Public-Key Infrastructure (X.509) Working Group (PKIX) of the Internet Engineering Tasks Force (IETF).



---

## the certificate chain engine

Certificate Validation Process
Before it trusts certificates, Browsers/applications perform a validation check to ensure that certificates are valid and that they have a valid certification path.
- The status of a public key certificate is determined through three distinct, but interrelated, processes.
- But this may vary slightly based on implementations.

---

### Certificate Discovery or Chain Building

The chain-building process
- validate the certification path by checking each certificate in the certification path
- from the end certificate to the root CA’s certificate.
- The certificates are retrieved from
  - the Intermediate Certification Authorities store,
  - the Trusted Root Certification Authorities store,
  - or from a URL specified in the AIA attribute of the certificate.
- If it discovers a problem with one of the certificates in the path, or if it cannot find a certificate, the certification path is discarded as a nontrusted certification path.

To improve performance, Browsers/Operating Systems may store subordinate CA certificates in the Intermediate Certification Authorities store
- so the future requests for the certificate can be satisfied from the store, rather than accessing the certificate through a URL.

---

#### Certificate Storage

A certificate store
- often contain numerous certificates, possibly issued from a number of different CAs.
- In Windows systems
  - there are separate stores known as the machine store, used by the computer,
  - and the user store or My store used by the currently logged-on user.

- In Java Environment certificates are stored in JKS files and are pointed by System Properties

```java
\-Djavax.net.ssl.keyStore=${some path}/keystore.jks
\-Djavax.net.ssl.trustStore=${some path}/cacerts.jks
\-Djavax.net.ssl.keyStorePassword=key-store-password
```

---

#### Purpose

The certificate chain engine builds all possible certificate chains.
- The entire graph of certificate chains is constructed and then ordered by the “quality” of the chain.
- The best-quality chain for a given end certificate is returned to the calling application as the default chain.

Each chain is built by using a combination of the certificates available in the certificate stores and certificates available from published URL locations.
- Each certificate in the chain is assigned a status code.
- The status code indicates whether the individual certificate is:

  - Signature-valid
    - Is the signature valid?

  - Time-valid
    - Are the certificate start and expiration dates properly configured,
    - has the start date not occurred yet,
    - or has the certificate expired?

  - Expired
    - Has the certificate expired?

  - Revoked
    - Has the certificate been revoked?

  - Time-nested
    - Have any of the certificates that are higher in the PKI hierarchy expired?

  - Any other restrictions on the certificate
    - For example, is the certificate being used for a purpose other than has been intended?


- Each status code has a <font color=red> precedence </font> assigned to it.
  - For example,
  - an expired certificate has a higher precedence than a revoked certificate.
  - This is because an expired certificate should not be checked for revocation status.

- If different status codes are assigned to the certificates in a certificate chain, the status code with the highest precedence is applied to the certificate chain and propagated into the certificate chain status.

---

### Path validation

path validation
- For each certificate in the chain, the certificate chain engine must select a certificate of the issuing CA.
- This process is repeated until a self-signed certificate is reached (typically, this is a root CA certificate).

There are different processes that can be used to select the certificate for an issuing CA.
- The actual process that is used is based on whether the certificate currently being investigated has the Authority Key Identifier (AKI) extension defined.
- Inspection of the AKI extension will lead to one of three matching processes being implemented:

  - Exact match
    - If the AKI extension contains the issuer’s user name and issuer serial number, only certificates that match on user name and serial number will be chosen in the chain-building process.
    - As a further test, the issuer name on the issued certificate must match the subject name on the issuer certificate.

  - Key match
    - If the AKI extension only contains public key information, only certificates that contain the indicated public key in the Subject Key Identifier (SKI) extension will be chosen as valid issuers.

  - Name match.
    - If no information exists in the AKI, or if the AKI does not exist in the certificate, the certificate will be marked as “name match.” In name matching, the subject name of a certificate must match the issuer name in the current certificate in order for the certificate to be chosen as a valid issuer.
    - Because the data is stored in a binary format, the name-matching process is case-sensitive.

In all cases, even if a matching certificate is not found in the store, the current certificate will still be marked as “exact match”, “key match” or “name match,” because this describes the attempted match rather than the attained match.

---

#### Caching

Certificate chaining:
- combines all the certificates from the root CA down to the certificate issued to the end user.
  - In a small organization, the root CA can simply issue certificates to the devices and end users.
  - not necessary to have intermediate / child CAs.



<font color=red> least recently used (LRU) caching scheme </font>
- To increase performance, the **certificate chain engine** uses a least recently used (LRU) caching scheme.
  - This scheme
  - creates a cache entry for each certificate it encounters as it builds the certificate chain.
  - Each cache entry includes the status of the certificate, so the best certificate chain can be built from cached items on subsequent calls to the chaining API without having to determine the status of each certificate again.
- After a certificate has been added to the cache, it will not be removed until it expires or is revoked.

During the path validation process, valid cached certificates will always be selected.
- If valid cached certificates are not found, a store search will be performed.
- For issuer certificates and CRLs, URL retrieval can be required to download the certificates and CRLs from the distribution point indicated in the URL.

- All certificates are stored in the cache when the certificates are selected from a store or from a URL.
- The only difference is the location where the cached certificates are stored.


![Pasted Graphic 5](https://i.imgur.com/IqwMZKK.png)


---

### Revocation

The certificate chain engine will check each certificate in the chain to determine whether the certificate has been revoked and the reason for the revocation.
- The revocation checking can take place either in conjunction with the chain building process or after the chain is built.
- If a revoked certificate is discovered in the chain, the chain is assigned a lower quality value.

---
