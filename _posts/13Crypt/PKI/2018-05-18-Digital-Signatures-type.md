---
title: Cryptography - Digital certificates Type
date: 2018-05-18 11:11:11 -0400
categories: [13Cryptography, PKI]
tags: [cryptography]
toc: true
image:
---


[toc]

---

# Digital Certificate

---


## type of Digital Certificate

Certificate Attributes
- Four main types of certificates are used:
  - End-entity certificates
  - CA certificates
  - Cross-certification certificates
  - Policy certificates
￼

![Pasted Graphic 45](https://i.imgur.com/z0v2okV.png)


- <font color=red> End-entity certificates </font>
  - issued by a CA to a specific subject, Joyce, the Accounting department, or a firewall.
  - An end-entity certificate is the identity document provided by PKI implementations.

- <font color=red> CA certificate </font>
  - can be self-signed, in the case of a stand-alone or root CA,
  - or can be issued by a superior CA within a hierarchical model.
    - the superior CA gives the authority
    - and allows the subordinate CA to accept certificate requests and generate the individual certificates itself.
  - may be necessary when a company needs to have multiple internal CAs,
    - different departments within an organization need to have their own CAs servicing their specific end-entities (users, network devices, and applications) in their sections.
    - In these situations, a representative from each department requiring a CA registers with the more highly trusted CA and requests a CA certificate.

- <font color=red> Cross-certification certificates </font>
  - used when independent CAs establish peer-to-peer trust relationships.
  - a mechanism
  - one CA can issue a certificate allowing its users to trust another CA.

- <font color=red> Policy certificates </font>
  - Within sophisticated CAs used for high-security applications, a mechanism is required to provide centrally controlled policy information to PKI clients.
  - This is often done by placing the policy information in a policy certificate.


---


## Types of Certificates


The authentication requirements differ depending on the type of certificate being requested.
- usually at least 3 types.
  - higher class of certificate carry out more powerful and critical tasks than the one below it.

- Class 1 certificate
  - for verify an individual’s identity through e-mail.
    - A person use his public / private key pair to digitally sign e-mail and encrypt message contents.
  - may only be asked to provide your name, e-mail address, and physical address.
    - In most situations, require the user to enter specific information into a web-based form.
    - The web page will have a section that accepts the user’s public key, or step the user through creating a public/private key pair, which will allow the user to choose the size of the keys to be created.

- Class 2 certificate:
  - for software signing.
    - software vendor would register for this type of certificate to digitally sign its software.
    - provides integrity for the software after it is developed and released,
    - allows the receiver of the software to verify from whom the software actually came.
  - may need to provide the RA with more data, such as driver’s license, passport, and company information that can be verified.

- Class 3 certificate:
  - for company to set up its own CA,
  - allow it to carry out its own identification verification and generate certificates internally.
  - will be asked to provide even more information and most likely will need to go to the RA’s office for a face-to-face meeting.




Types of X.509 certificates: slightly different purpose, same general structure, but the application is different.

- <font color=red> Wildcard certificates </font>
  - starts with an asterisk (`*`)
  - more widely, with subdomains of a given domain.
    - wildcard certificate for all subdomains.
    - No different X.509 certificate for each subdomain
    - can be used for multiple domains with same root domain.
  - reduce the administrative burden associated with managing multiple certificates.
    - To use a single certificate for a subdomain and “entirely different domain”, a SAN must be used.
  - Example
  - Google
    - wildcard certificate issued to `*.google.com`.
      - same certificate can be used for Google domains :
      - close alternative that supports any sub domain (e.g. `*.google.com`)
        - `accounts.google.com`
        - `support.google.com`
      - But could not be used for `gmail.com`.

- <font color=red> Subject Alternative Name (SAN) certificates </font>
  - not so much a type of certificate as a special field in X.509.
  - used for multiple domains that have different names, but owned by the same organization.
    - Example:
      - Google uses SANs of `*.google.com, *.android.com, *.cloud.google.com`, and more.
  - commonly used for systems with the same base domain names, but different top-level domains.
    - Example
      - `google.com` and `google.net`,
      - it could use a single SAN certificate for both domain names.
  - specify additional items (IP addresses, domain names…) to be protected by this single certificate.
    - with a list of alternative domains, sub domains, IP addresses that can also use the certificate. 
    - Provides extended site validation.
      - Example:
      - CrucialExams.com,
      - www.CrucialExams.com,
      - api.CrucialExams.com,
      - IP 4.5.4.5           (all in a single cert).

- <font color=red> X-509-compliant certificate </font>
  - Examaple:
  - hardening a web server,
  - which should allow a secure certificate-based session using the organization's PKI infrastructure.
  - The web server should also utilize the latest security techniques and standards.

- <font color=red> self-signed certificate </font>
  - not issued by a trusted CA.
    - Private CAs within an enterprise often create self-signed certificates.
    - Self-signed certificates from private CAs
    - eliminate the cost of purchasing certificates from public CAs.
  - They aren’t trusted by default.
    - However, administrators can use automated means to place copies of the self-signed certificate into the trusted root CA store for enterprise computers.
  - using Microsoft Internet Information Services (IIS).
    - The certificate will be X.509, digitally signed by you, can be used to transmit your public key
    - but it won’t be trusted by browsers,
    - will instead generate a certificate error message.

- <font color=red> Code signing certificates </font>
  - certificates digitally sign computer code.
  - validate the authentication of executable applications or scripts.
  - verifies the code has not been modified.
  - Ensuring no altered since the developer created it.
  - digitally sign a program's executable or script files.
  - This allows the person/computer running the application or script to verify it's authenticity


- <font color=red> Machine/computer certificates </font>
  - certificates assigned to a specific machine.
  - often used in authentication schemes.
  - The certificate is typically used to identify the computer within a domain.
  - Example:
    - for machine to sign in to the network, it must authenticate using its machine certificate.

- <font color=red> Email certificates </font>
  - uses of email certificates are for encryption of emails and digital signatures.
  - `Secure Multipurpose Internet Mail Extensions (S/MIME)` uses X.509 certificates to secure email communications.

- <font color=red> User certificates </font>
  - for individual users, often used for authentication.
  - Users must present their certificate to authenticate prior to accessing some resource.
  - Example
    - Microsoft systems can create user certificates
    - allowing the user to encrypt data using `Encrypting File System (EFS)`.


- <font color=red> Root certificates </font>
  - for root authorities, usually self-signed by that authority.


- <font color=red> Domain validation certificates </font>
  - (most common certificates)
  - to secure communication with a specific domain.
  - a low-cost certificate that website administrators use to provide TLS for a given domain.
  - indicates that the certificate requestor has some control over a DNS domain.
  - The CA takes extra steps to contact the requestor such as by email or telephone.
  - The intent is to provide additional evidence to clients that the certificate and the organization are trustworthy.

- <font color=red> Extended validation certificates </font>
  - require more validation of the certificate holder, provide more security.
  - use additional steps beyond domain validation.
  - a domain with an extended validation certificate, the address bar includes the name of the company before the actual URL.
  - This helps prevent impersonation from phishing attacks.
  - Example
    - PayPal
      - uses an extended validation certificate.
      - it shows `PayPal, Inc [US] |` before the URL.
      - Imagine an attacker sends a phishing email with a link to `paypa1.com` (with 1 in paypa1 instead of the letter).
      - If a user clicks the link, she will be able to see that the site isn’t truly a PayPal site,
      - assuming she understands extended validation certificates.

---

## example of Digital Certificate

X.509: not officially accepted as a standard, and implementations can vary from vendor to vendor.
- Microsoft and Mozilla have adopted X.509 as their defacto standard for Secure Sockets Layer (SSL) communication between web clients and servers.

Purchase the complete official X.509 standard from the International Telecommunications Union (ITU). It’s part of the Open Systems Interconnection (OSI) series of communication standards and can be purchased electronically on the ITU website at www.itu.int.


---


### Classification

Commercial CAs uses the concept of classes for different types of digital certificates.
- For example [VeriSign](http://www.verisign.com/) has the following classification
  - Class 1 for individuals, intended for email.
  - Class 2 for organizations, for which proof of identity is required.
  - Class 3 for servers and software signing, for which independent verification and checking of identity and authority is done by the issuing certificate authority.
  - Class 4 for online business transactions between companies.
  - Class 5 for private organizations or governmental security.

Other vendors may choose to use different classes or no classes at all as this is not specified in the specification, though, most do opt to use classes in some form.


---


### X.509 standard:

After an RA verifies an individual’s identity, the CA generates the digital certificate, but how does the CA know what type of data to insert into the certificate?


The certificates are created and formatted based on the <font color=red> X.509 standard </font>
- X.509 standard
  - outlines the necessary fields of a certificate
  - and the possible values that can be inserted into the fields.

- X.509 version 3 is the most current version of the standard.
  - a standard of the International Telecommunication Union (www.itu.int).
  - The IETF’s Public-Key Infrastructure (X.509), or PKIX, working group has adapted the X.509 standard to the more flexible organization of the Internet, as specified in RFC 3280, and is commonly referred to as PKIX for Public Key Infrastructure (X.509).



- identifying information governed by international standard X.509.
  - defines the <font color=red> certificate formats </font> and fields for public keys.
  - defines the <font color=red> procedures should used </font> to distribute public keys.
    - comes in two basic types:
      - End-Entity Certificate:
        - The most common
        - issued by a CA to an end entity.
        - An end entity is a system that doesn’t issue certificates but merely uses them.
      - CA Certificate
        - issued by one CA (root certificate) to another CA (intermedia CA).
        - The second CA, in turn, can then issue certificates to an end entity.


---

#### contents of X.509 standard:

The following figure shows the contents of X.509 version 3 certificates

<!-- <img alt="pic" src="https://sites.google.com/site/ddmwsst/digital-certificates/x509v3.gif?attredirects=0" width="450"> -->

Certificates that conform to X.509 contain the following data:
- <font color=red> _Version_ </font>
  - the version of the X.509 standard that was followed to create the certificate;
  - indicates the format and fields that can be used.
- <font color=red> _Certificate Serial number_ </font>
  - uniquely identifies the certificate, from the CA
  - The CA uses this serial number to validate a certificate.
  - If the CA revokes the certificate, it publishes this serial number in a certificate revocation list (CRL).

- <font color=red> _Certificate Algorithm for Certificate issuer's Signature_ </font>

- <font color=red> _Issuer_ </font>
  - identification of the CA that issued the certificate


- <font color=red> _Validity period_ </font>
  - the dates and times
  - valid starting and ending date and time

- <font color=red> _Subject_ </font>
  - Subject’s name:
  - contains the distinguished name, or DN, of the entity that owns the public key contained in the certificate.
  - example:
  - subject as Google, Inc, a wildcard certificate, used for all web sites with the google.com root domain name.
- <font color=red> _Subject’s public key Information_ </font>
  - the meat of the certificate
  - Identifies the public key being bound to the certified subject;
  - the actual public key the certificate owner used to set up secure communications
  - identifies the algorithm used to create the private/public key pair.
  - RSA asymmetric encryption uses the public key in combination with the matching private key.
  - <font color=red> _Signature Algorithm identifier_ </font>
    - the algorithm used by the CA to digitally sign the contents of the certificate.
  - <font color=red> _Signature Value_ </font>:
    - Bit string containing the digital signature.

- <font color=red> _Issuer unique identifier_ </font>
  - (versions 2 and 3 only)

- <font color=red> _Subject unique identifier_ </font>
  - (versions 2 and 3 only)

- <font color=red> _Extensions_ </font>
  - (version 3 only).
  - additional customized variables containing data inserted into the certificate by the certificate authority
  - to support tracking of certificates or various applications.
  - to expand the functionality of the certificate.
  - Companies can customize the use of certificates within their environments by using these extensions.
  - X.509 v3 has extended the extension possibilities.

  - <font color=red> _Subject alternative name_ </font>
    - A subject can be presented in many different formats.
    - For example,
    - if the certificate must include a user’s account name in the format of an LDAP distinguished name, e-mail name, and a user principal name (UPN),
    - you can include the e-mail name or UPN in a certificate by adding a subject alternative name extension that includes these additional name formats.

  - <font color=red> _CRL distribution points (CDP)_ </font>
    - When a user, service, or computer presents a certificate, an application or service must determine whether the certificate has been revoked before its validity period has expired.
    - The CDP extension provides one or more URLs where the application or service can retrieve the certificate revocation list (CRL) from.

  - <font color=red> _Authority Information Access (AIA)_ </font>
    - After an application or service validates a certificate, the certificate of the CA that issued the certificate (parent CA) must also be evaluated for revocation and validity.
    - The AIA extension provides one or more URLs from where an application or service can retrieve the issuing CA certificate.

  - <font color=red> _Enhanced Key Usage (EKU)_ </font>
    - This attribute includes an object identifier (OID) for each application or service a certificate can be used for.
    - Object identifiers/OIDs:
    - a unique sequence of numbers from a worldwide registry.
    - help identify objects. usually dot separated numbers.
    - Example, OID 2.5.4.6 might correspond to the country-name value


  - <font color=red> _Certificate policies_ </font>
    - Describes what measures an organization takes to validate the identity of a certificate requestor before it issues a certificate.
    - An OID is used to represent the validation process and can include a policy-qualified URL that fully describes the measures taken to validate the identity.


---

## Certificate Format and Encoding

The X.509 Digital certificate formats are defined using <font colot=red> ASN.1 - Abstract Syntax Notation One </font>
- an International Standards Organization (ISO) data representation format
- used to achieve interoperability between platforms.
- It does define how certificate contents should be encoded to store in files.

```json
Certificate  ::=  SEQUENCE  {
        tbsCertificate       TBSCertificate,
        signatureAlgorithm   AlgorithmIdentifier,
        signatureValue       BIT STRING  }

TBSCertificate  ::=  SEQUENCE  {
        version         \[0\]  EXPLICIT Version DEFAULT v1,
        serialNumber         CertificateSerialNumber,
        signature            AlgorithmIdentifier,
        issuer               Name,
        validity             Validity,
        subject              Name,
        subjectPublicKeyInfo SubjectPublicKeyInfo,
        issuerUniqueID  \[1\]  IMPLICIT UniqueIdentifier OPTIONAL,
                             -- If present, version MUST be v2 or v3
        subjectUniqueID \[2\]  IMPLICIT UniqueIdentifier OPTIONAL,
                             -- If present, version MUST be v2 or v3
        extensions      \[3\]  EXPLICIT Extensions OPTIONAL
                             -- If present, version MUST be v3
        }
AlgorithmIdentifier  ::=  SEQUENCE  {
        algorithm               OBJECT IDENTIFIER,
        parameters              ANY DEFINED BY algorithm OPTIONAL  }


More definitions will follow for CertificateSerialNumber, Name, Validity etc
```


Most certificates use one of the X.509 v3 formats.
certificates used to distribute certificate revocation lists use the X.509 v2 format.

Certificates are typically stored as:
- binary files: stored as 1s and 0s.
- BASE64 ASCII encoded files: converts the binary data into an ASCII string format.
- Additionally, some certificates are also encrypted to provide additional confidentiality.

2 commonly used encoding schemas are used to store X.509 certificates in files:
- DER and PEM


The base format of certificates is (CER) or (DER).
- CER and DER formats are defined by the International Telegraph Union Telecommunication Standardization Sector (ITU-T) in the X.690 standard.
- They use a variant of the Abstract Syntax Notation One (ASN.1) format, which defines data structures commonly used in cryptography.
  - CER is a binary format
  - DER is an ASCII format.

---

#### Canonical Encoding Rules `.cer`
- binary format
- alternate form of .crt (Microsoft Convention).
- can use Microsoft crypto API to convert `.crt` to `.cer`
- (both DER-encoded .cer, or base64 [PEM]-encoded .cer).
- also recognized by IE as a command to run an MS cryptoAPI command (specifically rundll32.exe cryptext.dll, CryptExtOpenCER).

---

#### PEM (Privacy Enhanced Mail) Encoding

- The most commonly used encoding schema for X.509 certificate files
  - supported by almost all applications
  - the extension of the certificate is `.pem`

- The full specification of PEM is in [RFC 1421](https://tools.ietf.org/html/rfc1421).
- But the idea of PEM encoding on X.509 certificates is very simple:
  - Encode the content with Base64 encoding.
  - Enclose the Base64 encoding output between two lines:
  - `"-----BEGIN CERTIFICATE-----" and "-----END CERTIFICATE-----"`


sample of a PEM encoded X.509 certificate:

```
-----BEGIN CERTIFICATE-----
MIIDODCCAvagAwIBAgIERqplETALBgcqhkjOOAQDBQAwfzELMAkGA1UE
...
Cgfs2kXj/IQCFDC5GT5IrLTIFxAyPUo1tJo2DPkK
-----END CERTIFICATE-----
```

PEM-based certificates can be used for just about anything.
- can be formatted as CER (binary files) or DER (ASCII files).
- can also be used to share public keys within a certificate, request certificates from a CA as a CSR, install a private key on a server, publish a CRL, or share the full certificate chain.

- Example
- PEM-encoded file holding the certificate with the public key typically uses the `.cer or.crt extension`.
- A PEM file holding just the private key typically uses the `.key extension`.



---

#### DER (Distinguished Encoding Rules) Encoding


- another popular encoding used to store X.509 certificate files.
- ASCII format.
- used for binary DER-encoded certificates.
  - These files may also bear the .cer or .crt.
- The Distinguished Encoding Rules of ASN.1 is an International Standard drawn from the constraints placed on BER encodings by X.509.
- DER encodings are valid BER encodings.
- DER is the same thing as BER with all but one sender's options removed.
- For example,
- in BER a boolean value of true can be encoded in 255 ways,
- while in DER there is only one way to encode a boolean value of true.
- The full specification of DER is in [RFC 1421](http://www.itu.int/ITU-T/studygroups/com17/languages/X.690-0207.pdf).

X.509 certificate files encode in DER are binary files, which can not be view with text editors. DER encoded certificate files are supported by almost all applications. The file extensions for DER encoded certificates are .cer, .der, .crt


---

#### PKCS Formats

- a group of public-key cryptography standards devised and published by RSA Security.
- As such, RSA Security and its research division, RSA Labs, had an interest in promoting and facilitating the use of public-key techniques.
- To that end, they developed (from the early 1990s onwards) the PKCS standards.
- They retained control over them, announcing that they would make changes/improvements as they deemed necessary, and so the PKCS standards were not, in a significant sense, actual industry standards, despite the name.
- Some, but not all, have in recent years begun to move into standards organizations like the IETF PKIX working group.

The file extensions in this case are .p7b, .p7c, .p12, .pfx etc.


- .pfx: Personal Information Exchange
  - (PKCS 12 archive) archive file for PKCS#12 standard certificate information.
  - commonly used to store private keys. includes the private key, should never be shared!
  - a predecessor to the P12 certificate and it has the same usage.
  - Administrators often use this format on Windows systems to import and export certificates.

- P12: the use of PKCS#12 standard.
  - commonly used to store private keys
  - use the PKCS version 12 (PKCS#12) format and they are CER-based (binary).
  - They are commonly used to hold certificates with the private key.
  - Example
  - installing a certificate on a server to supports HTTPS sessions, you might install a P12 certificate with the private key.
  - Because it holds the private key, it’s common to encrypt P12 certificates. It’s also possible to include the full certificate chain in a P12 certificate.

- P7b: base 64 encoded ASCII files. They actually include several variations: P7b, P7C, etc.
  - use the PKCS version 7 (PKCS#7) format and they are DER-based (ASCII).
  - commonly use: share public keys with proof of identity of the certificate holder.
  - Recipients use the public keys to encrypt or decrypt data.
  - Example
  - web server might use a P7B certificate to share its public key.
  - P7B certificates can also contain a certificate chain or a CRL.
  - However, they never include the private key.


---

## Examples

Let us check a real certificate, its details and its chain.

view certificate link

<!-- ![link](https://sites.google.com/site/ddmwsst/digital-certificates/view-cert.png?attredirects=0) -->



- the certificate owned by State Bank of India.
- "Issued by" field is issued by VeriSign Class 3 Extended Validation SSL SGC CA.



<!-- ![link](https://sites.google.com/site/ddmwsst/digital-certificates/cert-gen.png?attredirects=0) -->


- details > "Show" dropdown filters them for better viewing.
- seeing the subject, SBI, and its detail Distinguished Name (DN). On the right issuer's DN.


<!-- ![link](https://sites.google.com/site/ddmwsst/digital-certificates/cert-det-ver1.png?attredirects=0) -->



- "Extensions Only" from the show dropdown.
- seeing CRL Distribution point and on the right AIA location.
  - Every certificate normally points to a CRL given by its issuer.
- click to download the CRL


<!-- ![link](https://sites.google.com/site/ddmwsst/digital-certificates/cert-det-ext.png?attredirects=0) -->

<!-- ![link](https://sites.google.com/site/ddmwsst/digital-certificates/cert-revo-list.png?attredirects=0) -->



- "Certificate Path" tab to see the certificate chain.
- Certificate viewer allows you see other certificates in the chain by highlighting a certificate and click on the "View Certificate" button as shown on the right below.
- VeriSign is a two tier CA,
  - where "VeriSign" is the Root
  - "VeriSign Class 3 Extended Validation SSL SGC CA" is a Issuing CA.


<!-- ![link](https://sites.google.com/site/ddmwsst/digital-certificates/cert-path.png?attredirects=0) -->


see Issuer CA's certificate.
- for root certificate the "Issued to" or "Subject" and "Issued by" or "Issuer" fields are same.
- So this is a self signed certificate.

<!-- ![link](https://sites.google.com/site/ddmwsst/digital-certificates/issuer-cert.png?attredirects=0) -->

<!-- ![link](https://sites.google.com/site/ddmwsst/digital-certificates/root-cert.png?attredirects=0) -->



---
