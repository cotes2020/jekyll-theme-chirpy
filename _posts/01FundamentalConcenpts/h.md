---
title: Security Fundamental Concepts
date: 2020-02-10 11:11:11 -0400
categories: [01FundamentalConcenpts]
tags: [FundamentalConcenpts, CIA]
math: true
image:
---

# Security Fundamental Concepts

- [Security Fundamental Concepts](#security-fundamental-concepts)
  - [Security Fundamentals](#security-fundamentals)
    - [Network Security Goals](#network-security-goals)
  - [The CIA Triad](#the-cia-triad)
    - [Confidentiality 机密性](#confidentiality-机密性)
      - [To support Confidentiality](#to-support-confidentiality)
    - [Integrity 诚实](#integrity-诚实)
      - [To support Integrity](#to-support-integrity)
    - [Availability](#availability)
      - [To support Availability](#to-support-availability)

---

## Security Fundamentals

- this section identifies several categories of network attacks.

### Network Security Goals

- Today’s corporate networks, the demands of ecommerce and customer contact require connectivity between internal corporate networks and the outside world.

- Two basic assumptions about modern corporate networks:
  - Today’s corporate networks are large, interconnect with other networks, and run both standards-based and proprietary protocols.
  - The devices and applications connecting to and using corporate networks are continually increasing in complexity.

- Computers and networks are being misused at a growing rate.

- Spam, phishing, and computer viruses are becoming multibillion-dollar problems, as is identity theft, which poses a serious threat to the personal finances and credit ratings of users, and creates liabilities for corporations.

- Thus, there is a growing need for broader knowledge of computer security in society as well as increased expertise among information technology professionals. Society needs more security-educated computer professionals, who can successfully defend against and prevent computer attacks, as well as security-educated computer users, who can safely manage their own information and the systems they use.

- computer security's concepts and terms. 3 primary goals of network security:
  - Confidentiality
  - Integrity
  - Availability

---

## The CIA Triad

### Confidentiality 机密性

- Data confidentiality: `keeping data private, avoidance of the unauthorized disclosure of information`. offers a high level of assurance that data, objects, or resources are restricted from unauthorized subjects.

- (entail physically or logically restricting access to sensitive data or encrypting traffic traversing a network)

- the heart of information security

Events that lead to confidentiality breaches include:

- failing to properly encrypt a transmission,
- failing to fully authenticate a remote system before transferring data,
- leaving open otherwise secured access points,
- accessing malicious code that opens a back door,
- misrouted faxes,
- documents left on printers,
- or even walking away from an access terminal while data is displayed on the monitor.

#### To support Confidentiality

Tools for protecting sensitive information:

- **Encryption**:

  - `Encrypt traffic, package using encryption/decryption key`
  - extremely difficult to determine the original information without the decryption key.
  - encryption/decryption algorithms uses a key in its mathematical calculation,

- **network-security mechanisms**

  - like firewalls, access control lists [ACL]
  - prevent unauthorized access to network resources.

- **Authentication**: 鉴定

  - `identity or role that someone has`.
  - Require appropriate credentials (like usernames and passwords):
  - something the person has (smart card/radio key fob storing secret keys),
  - something the person knows (password),
  - something the person is (fingerprint).

- **Authorization**: 授权

  - `Determine if a person or system is allowed access to resources`, based on an access control policy.
  - `rules and policies that limit access` to confidential information to those people and/or systems with a “need to know.”
  - may be determined by identity (like name or serial number, or by a role that a person has)
  - prevent attacker from tricking the system to have access to protected resources.

- **Physical security, personnel training**:

  - the establishment of `physical barriers to limit access` to protected computational resources.
  - locks on cabinets and doors, windowless rooms, sound dampening materials,
  - rooms with walls incorporating copper meshes (Faraday cages) so that electromagnetic signals cannot enter or exit the enclosure.

Example:

- little lock icon in website:
- browser perform an **authentication** procedure to verify the `web site is indeed who it says it is`.
- the web site check that `our browser is authentic and we have the appropriate authorizations to access this web page` according to its access control policy.
- Our browser then asks the web site for an encryption key to encrypt our credit card information in encrypted form.
- Finally, our credit card number reaches the server that is providing this web site, `the data center where the server is located should have appropriate levels of physical security`, access policies, and authorization and authentication mechanisms to keep our credit card number safe.
- There are a number of real demonstrated risks to physical eavesdropping.

**Confidentiality** and **integrity** depend on each other.

- Without object integrity, confidentiality cannot be maintained.
- Other concepts, conditions, and aspects of confidentiality include the following:

  - **Sensitivity**: the quality of information could cause harm or damage if disclosed. Maintaining confidentiality of sensitive information helps to prevent harm or damage.

  - **Discretion** 慎重: an act of decision where an operator can influence or control disclosure in order to minimize harm or damage.

  - **Criticality** 危险程度: The higher criticality, the more likely the need to maintain the confidentiality of the information. High levels of criticality are essential to the operation or function of an organization.

  - **Concealment**: the act of hiding or preventing disclosure. Often concealment is viewed as a means of cover, obfuscation, or distraction.

  - **Secrecy**: the act of keeping something a secret or preventing the disclosure of information.

  - **Privacy**: keeping information confidential that is personally identifiable or that might cause harm, embarrassment, or disgrace to someone if revealed.

  - **Seclusion** 隔绝:  storing something in an out-of-the-way location. This location can also provide strict access controls. Seclusion can help enforcement confidentiality protections.

  - **Isolation**: keeping something separated from others. Isolation can be used to prevent commingling of information or disclosure of information.

- Each organization needs to evaluate the nuances of confidentiality they wish to enforce. Tools and technology that implements one form of confidentiality might not support or allow other forms.

### Integrity 诚实

the property that `information has not be altered/modified` in an unauthorized way. objects must retain their veracity and be intentionally modified by only authorized subjects.

A lot of ways that data integrity can be compromised in computer systems and networks (benign / malicious)

- **benign compromise**:

  - Example: a storage device being hit with a stray cosmic ray that flips a bit in an important file, or a disk drive might simply crash, completely destroying some of its files.

- **malicious compromise**:

  - Example: a computer virus that infects our system and deliberately changes some the files of our operating system, so that our computer then works to replicate the virus and send it to other computers.

- Examples of integrity violations:

  - Modifying the appearance of a corporate website
  - Intercepting and altering an e-commerce transaction
  - Modifying financial records that are stored electronically

Integrity can be examined from 3 perspectives:

- `Preventing unauthorized subjects` from making modifications
- `Preventing authorized subjects from making unauthorized modifications`, such as mistakes
- `Maintaining the internal and external consistency of objects` so that their data is a correct and true reflection of the real world and any relationship with any child, peer, or parent object is valid, consistent, and verifiable

Numerous attacks focus on the violation of integrity, include:

- viruses,
- logic bombs,
- unauthorized access,
- errors in coding and applications,
- malicious modification,
- intentional replacement,
- system back doors.
- not limited to intentional attacks. Human error, oversight, or ineptitude accounts for many instances of unauthorized alteration of sensitive information.

#### To support Integrity

There are several tools specifically for support integrity:

- **Backups**: the `periodic archiving 存档 of data`.

- **Checksums**:

  - A checksum function depends on the entire contents of a file, the computation 计算 of a function that `maps the contents of a file to a numerical value`.
  - Even a small change to the input file (such as flipping a single bit) will result in a different output value.
  - Checksums are like trip-wires, they are used to detect when a breach to data integrity has occurred.

- **Data correcting codes**:

  - methods for storing data in such a way that small changes can be easily detected and automatically corrected.
  - These codes are typically applied to small units of storage (e.g., at the byte level or memory word level), but there are also data-correcting codes that can be applied to entire files as well.

These tools for achieving data integrity all possess a common trait, they use `redundancy`.

- they involve the **replication** 复制 of some information content or functions of the data
- so that we can detect and sometimes even correct breaches in data integrity.

not just the `content of data file`, also need to protect the `metadata` 元数据 for data file (attributes of the file or information about access to the file, not strictly a part of its content)

- Examples of metadata:
  - the user who is the owner of the file,
  - the last user who has modified the file,
  - the last user who has read the file,
  - the dates and times when the file was created and last modified and accessed,
  - the name and location of the file in the file system,
  - the list of users or groups who can read or write the file.

- Thus, changing any metadata of a file should be considered a violation of its integrity.

- Example:

  - intruder might not modify 修改 the content of files in a system he has infiltrated,
  - `but nevertheless 仍然 be modifying metadata`
  - Like access time stamps, by looking at our files (and thereby compromising their confidentiality if they are not encrypted).
  - So system has integrity checks for metadata, may be able to detect an intrusion that would have otherwise gone unnoticed.

### Availability

A measure of the `data’s accessibility`. the property that information is accessible and modifiable in a timely fashion by those authorized to do so.

- availability of 99.999% (five nines of availability): down only 5 minutes per year,
Availability includes efficient uninterrupted access to objects and prevention of denial-of-service (DoS) attacks.

There are numerous threats to availabilit, include:

- device failure,
- software errors,
- environmental issues (heat, static, flooding, power loss, and so on).

- some forms of attacks that focus on the violation of availability:

  - denial of service (DoS) attack:
  - Send improperly formatted data to a networked device,
  - resulting in an unhandled exception error.
  - Flood a network system with an excessive amount of traffic or requests,
  - consume a system’s processing resources
  - prevent the system from responding to many legitimate requests.
  - object destruction,
  - communication interruptions.

#### To support Availability

a number of tools for providing availability:

- **Physical protections**:

  - infrastructure meant to keep information available even in the event of physical challenges.
  - Like buildings housing critical computer systems withstand storms, earthquakes, and bomb blasts, outfitted with generators and other electronic equipment to be able to cope with power outages and surges.

- **Computational redundancies**:

  - computers and storage devices that serve as fallbacks in the case of failures.
  - Example:
    - redundant arrays of inexpensive disks (RAID) use storage redundancies to keep data available to their clients.
    - Also, web servers are often organized in multiples called “farms”
    - so that the failure of any single computer can be dealt with without degrading the availability of the web site.

Numerous countermeasures can ensure availability against possible threats, include:

- designing intermediary delivery systems properly,
- using access controls effectively,
- monitoring performance and network traffic,
- using firewalls and routers to prevent DoS attacks,
- implementing redundancy for critical systems,
- maintaining and testing backup systems.

Most security policies, as well as **business continuity planning (BCP)**, focus on the use of fault tolerance features at the various levels of access/storage/security (that is, disk, server, or site) with the goal of eliminating single points of failure to maintain availability of critical systems.
