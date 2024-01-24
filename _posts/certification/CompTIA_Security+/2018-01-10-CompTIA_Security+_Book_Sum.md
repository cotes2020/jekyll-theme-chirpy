---
title: Certification - CompTIA Security+ Book Sum
date: 2018-01-10 11:11:11 -0400
description: CompTIA Security+ Book Sum
categories: [Certification, Security+]
# img: /assets/img/sample/rabbit.png
tags: [Certification, Security+]
---

- [Certification - CompTIA Security+ Book Sum](#certification---comptia-security-book-sum)
  - [Chapter 1 Exam Topic Review](#chapter-1-exam-topic-review)
    - [Understanding Core Security Goals](#understanding-core-security-goals)
    - [Introducing Basic Risk Concepts](#introducing-basic-risk-concepts)
    - [Understanding Control Types](#understanding-control-types)
    - [Implementing Virtualization](#implementing-virtualization)
    - [Using Command-Line Tools](#using-command-line-tools)
  - [Chapter 2 Exam Topic Review](#chapter-2-exam-topic-review)
    - [Exploring Authentication Concepts](#exploring-authentication-concepts)
    - [Comparing Authentication Services](#comparing-authentication-services)
    - [Managing Accounts](#managing-accounts)
    - [Comparing Access Control Models](#comparing-access-control-models)
  - [Chapter 5 Exam Topic Review](#chapter-5-exam-topic-review)
    - [Implementing Secure Systems](#implementing-secure-systems)
    - [Summarizing Cloud Concepts](#summarizing-cloud-concepts)
    - [Deploying Mobile Devices Securely](#deploying-mobile-devices-securely)
    - [Exploring Embedded Systems](#exploring-embedded-systems)
    - [Protecting Data](#protecting-data)
  - [Chapter 6 Exam Topic Review](#chapter-6-exam-topic-review)
    - [Understanding Threat Actors](#understanding-threat-actors)
    - [Determining Malware Types](#determining-malware-types)
  - [Chapter 7 Exam Topic Review](#chapter-7-exam-topic-review)
    - [Comparing Common Attacks](#comparing-common-attacks)
    - [Summarizing Secure Coding Concepts](#summarizing-secure-coding-concepts)
    - [Identifying Application Attacks](#identifying-application-attacks)
    - [Understanding Frameworks and Guides](#understanding-frameworks-and-guides)
  - [Chapter 8 Exam Topic Review](#chapter-8-exam-topic-review)
  - [Implementing Controls to Protect Assets](#implementing-controls-to-protect-assets)
  - [Chapter 10 Exam Topic Review](#chapter-10-exam-topic-review)
  - [Chapter 11 Implementing Policies to Mitigate Risks](#chapter-11-implementing-policies-to-mitigate-risks)

---

# Certification - CompTIA Security+ Book Sum

---

## Chapter 1 Exam Topic Review

### Understanding Core Security Goals

- A use case helps professionals identify and clarify requirements to achieve a goal.

- `Confidentiality` ensures that data is only viewable by authorized users.
  - Encryption is the best choice to provide confidentiality.
  - Access controls also protect the confidentiality of data.

- `Steganography` (hiding data inside other data) is one method of supporting obfuscation by making the hidden data harder to see.

- `Integrity` provides assurances that data has not been modified, tampered with, or corrupted through unauthorized or unintended changes. Data can be a message, a file, or data within a database.
  - Hashing is a common method of ensuring integrity.

- `Non-repudiation` prevents entities from denying they took an action.
  - Digital signatures and audit logs provide non-repudiation.
  - Digital signatures also provide integrity for files and email.

- `Availability` ensures that data and services are available when needed. A common goal is to remove single points of failure.
  - Methods used to increase or maintain availability include fault tolerance, failover clusters, load balancing, backups, virtualization, HVAC systems, and generators.


### Introducing Basic Risk Concepts

- `Risk` is the possibility of a threat exploiting a vulnerability and resulting in a loss.

- A `threat` is any circumstanceor event that has the potential to compromise confi integrity, or availability.

- A `vulnerability` is a weakness. It can be a weakness in the hardware, software, configuration, or users operating the system.


- Risk mitigation reduces risk by reducing the chances that a threat will exploit a vulnerability or by reducing the impact of the risk.

- Security controls reduce risks. example, antivirus software is a security control that reduces the risk of virus infection.


### Understanding Control Types

- The three primary security control types are
  - `technical` (implemented with technology),
  - `administrative` (using administrative or management methods),
  - and `physical` (controls physically touch).

- A `technical` control is one that uses technology to reduce vulnerabilities.
  - Encryption, antivirus software, IDSs, firewalls, and the principle of least privilege are technical controls.

- `Administrative` controls are primarily administrative and include items such as risk and vulnerability assessments.
  - Some administrative controls help ensure that day-to-day operations of an organization comply with their overall security plan.
  - Some examples include security awareness and training, configuration management, and change management.

- `Preventive controls` attempt to **prevent security incidents**.
  - Examples include system hardening, user training, guards, change management, and account disablement policies.

- `Detective controls` attempt to **detect when a vulnerability has been exploited**.
  - Examples include log monitoring, trend analysis, security audits (such as a periodic review of user rights), video surveillance systems, and motion detection systems.

- `Corrective controls` attempt to **reverse the impact of an incident or problem after it has occurred**.
  - Examples include intrusion prevention systems (IPSs), backups, and system recovery plans.

- `Deterrent controls` attempt to **prevent incidents** by discouraging threats. Warning.

- `Compensating controls` are **alternative controls** used when it isn’t feasible or possible to use the primary control.


### Implementing Virtualization

- `Virtualization` allows multiple servers to operate on a single physical host. They provide increased availability with various tools such as snapshots and easy restoration.

- `Type I hypervisors` run directly on the system hardware. They are often called bare-metal hypervisors because they don’t need to run within an operating system.

- `Type II hypervisors` run as software within a host operating system.

- `Container virtualization` is a specialized version of a Type II hypervisor.
  - It allows services or applications to run within their own isolated cells or containers.
  - Containers don’t have a full operating system but instead use the kernel of the host.

- `Snapshots` capture the state of a VM at a moment in time. Administrators often take a snapshot before performing a risky operation. If necessary, they can revert the VM to the snapshot state.

- `VM sprawl` can occur if personnel within the organization don’t manage the VMs.

- `VM escape` attacks allow an attacker to access the host system from the VM.
  - The primary protection is to keep the host and guests up to date with current patches.


### Using Command-Line Tools

- You run command-line tools in the Command Prompt window (in Windows) and the terminal (in Linux).

- The `ping` command can be used to check connectivity; check name resolution; and verify that routers, firewalls, and intrusion prevention systems block ICMP.

- The `ipconfig` command on Windows allows you to view the configuration of network interfaces.

- Linux uses `ifconfig` and/or `ip` to view and manipulate the configuration of network interfaces. You can enable promiscuous mode on a NIC with ifconfig.

- `Netstat` allows you to view statistics for TCP/IP protocols and view all active network connections. This can be useful if you suspect malware is causing a computer to connect with a remote computer.

- `Tracert` lists the routers (also called hops) between two systems. It can be used to verify a path has not changed.

- The `arp` command allows you to view and manipulate the ARP cache. This can be useful if you suspect a system’s ARP cache has been modified during an attack.


---


## Chapter 2 Exam Topic Review

### Exploring Authentication Concepts

- `Authentication` allows entities to prove their identity by using credentials known to another entity.

- `Identification` occurs when a user claims or professes an identity, such as with a username, an email address, a PIV card, or by using biometrics.

- `Authentication` occurs when an entity provides proof of an identity (such as a password). A second identity is the authenticator and it verifies the authentication.

- `Authorization` provides access to resources based on a proven identity.

- `Accounting` methods track user activity and record the activity in logs.

- Five factors of authentication are:

  - Something you `know`, such as a username and password

  - Something you `have`, such as a smart card, CAC, PIV, or token

  - Something you `are`, using biometrics, such as fingerprints or retina scans

  - Somewhere you `are`, using geolocation, a computer name, or a MAC address

  - Something you `do`, such as gestures on a touch screen


- Something you `know`

  - The something you know factor typically refers to a shared secret, such as a password or a PIN. This is the least secure form of authentication.

  - `Passwords` should be strong and changed often. Complex passwords include multiple character types. Strong passwords are complex and at least 14 characters long.

  - Administrators should verify a user’s identity before resetting the user’s password. When resetting passwords manually, administrators should configure them as temporary passwords that expire after the first use, requiring users to create a new password the first time they log on. Self-service password systems automate password recovery.

  - `Password policies` provide a technical means to ensure users employ secure password practices.
    - Password policies should apply to any entity using a password. This includes user accounts and accounts used by services and applications.
    - Applications with internally created passwords should still adhere to the organization’s password policy.

  - `Password length` specifies the minimum number of characters in the password.

  - `Password complexity` ensures passwords are complex and include at least three of the four character types, such as special characters.

  - `Password history` remembers past passwords and prevents users from reusing passwords.

  - `Minimum password age` is used with password history to prevent users from changing their password repeatedly to get back to the original password.

  - `Maximum password age` or `password expiration` forces users to change their password periodically. When administrators reset user passwords, the password should expire upon first use.

  - `Account lockout policies` lock out an account after a user enters an incorrect password too many times.


- Something you `have`

  - `Smart cards` are credit card-sized cards that have embedded certificates used for authentication. They require a PKIto issue certificates.

  - `Common Access Cards (CACs) and Personal Identity Verification (PIV) cards` can be used as photo IDs and as smart cards (both identification and authentication).

  - `Tokens (or key fobs) display numbers in an LCD`. These numbers provide rolling, one-time use passwords and are synchronized with a server. USB tokens include an embedded chip and a USB connection. Generically, these are called hardware tokens.

  - `HOTP and TOTP` are open source standards used to create one-time-use passwords.
    - HOTP creates a one-time-use password that does not expire
    - TOTP creates a one-time password that expires after 30 seconds.

- Something you `are`


- - `Biometric` methods are the most difficult to falsify. Physical methods include voice and facial recognition, fingerprints, retina scans, iris scans, and palm scans. Biometric methods can also be used for identification.


- - The `false acceptance rate (FAR), or false match rate`, identifies the percentage of times false acceptance occurs.

- - The `false rejection rate (FRR), or false nonmatch rate`, identifies the percentage of times false rejections occur.

- - The `crossover error rate (CER)` indicates the quality of the biometric system. Lower CERs are better.


- `Single-factor authentication` includes one or more authentication methods in the same factor, such as a PIN and a password. Dual-factor (or two-factor) authentication uses two factors of authentication, such as a USB token and a PIN.

- `Multifactor authentication` uses two or more factors. Multifactor authentication is stronger than any form of single-factor authentication

- Authentication methods using two or more methods in the same factor are single-factor authentication. example, a password and a PIN are both in the something you know factor, so they only provide single-factor authentication.


### Comparing Authentication Services

- `Kerberos` is a network authentication protocol using tickets issued by a KDC or TGT server.
  - If a ticket-granting ticket expires, the user might not be able to access resources.
  - Microsoft Active Directory domains and Unix realms use Kerberos for authentication.

- `LDAP` specifies formats and methods to query directories.
  - It provides a single point of management for objects, such as users and computers, in an Active Directory domain or Unix realm.
  - The following is an example of an LDAP string: LDAP:// CN=Homer,CN=Users,DC=GetCertifiedGetAhead,DC=com

- `LDAP Secure (LDAPS)` encrypts transmissions with SSL or TLS.

- `Single sign-on (SSO)` allows users to authenticate with a single user account and access multiple resources on a network without authenticating again.

  - SSO can be used to provide central authentication with a federated database and use this authentication in an environment with different operating systems (nonhomogeneous environment).

- `SAML` is an XML-based standard used to exchange authentication and authorization information between different parties.
  - SAML is used with web-based applications.

- A `federated identity` links a user’s credentials from different networks or operating systems, but the federation treats it as one identity.

- `Shibboleth` is an open source federated identity solution that includes Open SAML libraries.

- `OAuth and OpenID Connect` are used by many web sites to streamline the authentication process for users. They allow users to log on to many web sites with another account, such as one they’ve created with Google, Facebook, PayPal, Microsoft, or Twitter.


### Managing Accounts

- The principle of `least privilege` is a technical control that uses access controls. It specifies that individuals or processes are granted only the rights and permissions needed to perform assigned tasks or functions, but no more.

- Users should not share accounts. It prevents effective identification, authentication, authorization, and accounting. Most organizations ensure the Guest account is disabled.

- `Account policies` often require administrators to have two accounts (an administrator account and a standard user account) to prevent privilege escalation and other attacks.

- An `account disablement policy` ensures that inactive accounts are disabled. Accounts for employees who either resign or are terminated should be disabled as soon as possible. Configuring expiration dates on temporary accounts ensures they are disabled automatically.

- `Time restrictions` can prevent users from logging on or accessing network resources during specific hours.

- `Location-based` policies prevent users from logging on from certain locations.

- Accounts should be `recertified` to verify they are still required.
  - example, if the organization extends a contract, it’s a simple matter to recertify the account. Administrators verify that the contract has been extended, change the expiration date, and enable the account.

- Administrators routinely perform `account maintenance`. This is often done with scripts to automate the processes and includes deleting accounts that are no longer needed.

- `Credential management systems` store and simplify the use of credentials for users.
  - When users access web sites needing credentials, the system automatically retrieves the stored credentials and submits them to the web site.


### Comparing Access Control Models

- The `role-based access control (role-BAC)` model `uses roles to grant access` by placing users into roles based on their assigned jobs, functions, or tasks. A matrix matching job titles with required privileges is useful as a planning document when using role-BAC.

- `Group-based privileges` are a form of role-BAC. Administrators create groups, add users to the groups, and then assign permissions to the groups. This simplifies administration because administrators do not have to assign permissions to users individually.

- The `rule-based access control (rule-BAC)` model is `based on a set of approved instructions`, such as ACL rules in a firewall. Some rule-BAC implementations use rules that trigger in response to an event, such as modifying ACLs after detecting an attack.

- In the `discretionary access control (DAC)` model, every object has an owner.
  - The owner has explicit access and establishes access for any other user. Microsoft NTFS uses the DAC model, with every object having a discretionary access control list (DACL). The DACL identifies who has access and what access they are granted. A major flaw of the DAC model is its susceptibility to Trojan horses.

- `Mandatory access control (MAC)` uses `security or sensitivity labels to identify objects (what you’ll secure) and subjects (users)`.
  - It is often used when access needs to be restricted based on a need to know.
  - The administrator establishes access based on predefined security labels.
  - These labels are often defined with a lattice to specify the upper and lower security boundaries.

- An `attribute-based access control (ABAC)` evaluates attributes and grants access based on the value of these attributes.
  - It is used in many `software defined networks (SDNs)`.


---

## Chapter 5 Exam Topic Review

### Implementing Secure Systems

- `Least functionality` is a core secure system design principle.
  - It states that systems should be deployed with only the applications, services, and protocols they need to function.

- A trusted operating system meets a set of predetermined requirements such as those defined in the Common Criteria. It typically uses the `mandatory access control (MAC) model`. (Label)

- A `master image` provides a secure starting point for systems.
  - Master images are typically created with templates or other baselines to provide a secure starting point for systems.
  - Integrity measurement tools detect when a system deviates from the baseline.

- `Patch management` procedures ensure operating systems and applications are kept up to date with current patches. This ensures they are protected against known vulnerabilities.

- `Change management` policies define the process for making changes and help reduce unintended outages from changes.

- `Application whitelisting` allows authorized software to run, but blocks all other software.
- `Application blacklisting` blocks unauthorized software, but allows other software to run.

- `Sandboxing` provides a high level of flexibility for testing security controls and testing patches. You can create sandboxes in virtual machines (VMs) and with the chroot command on Linux systems.

- `Electromagnetic interference (EMI)` comes from: motors, power lines, and fluorescent lights and can be prevented with shielding.

- `Electromagnetic pulse (EMP)` is a short burst of electromagnetic energy. Mild forms such as electrostatic discharge and lightning can be prevented but EMP damage from military weapons may not be preventable.

- `Full disk encryption (FDE)` encrypts an entire disk. A self-encrypting drive (SED) includes the hardware and software necessary to automatically encrypt a drive.

- `Trusted Platform Module (TPM)` is a chip included with many laptops and some mobile devices and it provides full disk encryption, a secure boot process, and supports remote attestation. TPMs have an encryption key burned into them that provides a hardware root of trust.

- `hardware security module (HSM)` is a removable or external device used for encryption. An HSM generates and stores RSA encryption keys and can be integrated with servers to provide hardware-based encryption.



### Summarizing Cloud Concepts

- Cloud computing provides an organization with additional resources. Most cloud services are provided via the Internet or a hosting provider. On-premise clouds are owned and maintained by an organization.

- `Software as a Service (SaaS)` includes web-based applications such as web-based email.

- `Infrastructure as a Service (IaaS)` provides hardware resources via the cloud. It can help an organization limit the size of their hardware footprint and reduce personnel costs.

- `Platform as a Service (PaaS)` provides an easy-to-configure operating system and on- demand computing for customers.

- `cloud access security broker (CASB)` is a software tool or service deployed between an organization’s network and the cloud provider. It monitors all network traffic and can enforce security policies acting as Security as a Service.

- `Private cloud` are only available for a specific organization.
- `Public cloud` services are provided by third-party companies and available to anyone.
- `community cloud` is shared by multiple organizations.
- `hybrid cloud` is a combination of two or more clouds.


### Deploying Mobile Devices Securely

- Mobile devices include smartphones and tablets and run a mobile operating system.

- `Corporate-owned, personally enabled (COPE)` mobile devices are owned by the organization, but employees can use them for personal reasons.

- `Bring your own device (BYOD)` policies allow employees to connect their mobile device to the organization’s network. Choose your own device (CYOD) policies include a list of acceptable devices and allow employees with one of these devices to connect them to the network.

- A `virtual desktop infrastructure (VDI)` is a virtual desktop and these can be created so that users can access them from a mobile device.

- Mobile devices can connect to the Internet, networks, and other devices using ANT, infrared, cellular, wireless, satellite, Bluetooth, near field communication (NFC), and USB connections.

- `Mobile device management (MDM)` tools help ensure that devices meet minimum security requirements. They can monitor devices, enforce security policies, and block network access if devices do not meet these requirements.

  - `MDM tools` can restrict applications on devices, segment and encrypt data, enforce strong authentication methods, and implement security methods such as screen locks and remote wipe.

- A `screen lock` is like a password-protected screen saver on desktop systems that automatically locks the device after a period of time. A remote wipe signal removes all the data from a lost phone.

- `Geolocation` uses `Global Positioning System (GPS)` to identify a device’s location. Geofencing uses GPS to create a virtual fence or geographic boundary. Organizations use geofencing to enable access to services or devices when they are within the boundary, and block access when they are outside of the boundary. Geotagging uses GPS to add geographical information to files (such as pictures) when posting them on social media sites.

- A `third-party app store` is something other than the primary store for a mobile device.
  - Apple’s App Store is the primary store for Apple devices.
  - Google Play is a primary store for Android devices.

- `Jailbreaking` removes all software restrictions on Apple devices.
- `Rooting` provides users with root-level access to an Android device.
- `Custom firmware` can also root an Android device.
- `MDM tools` block network access for jailbroken or rooted devices.

- `Sideloading` is the process of copying an application to an Android device instead of installing it from an online store.

- `Universal Serial Bus On-The-Go (USB OTG)` cable allows you to connect mobile devices.

- `Tethering` allows one mobile device to share its Internet connection with other devices.
- `Wi-Fi Direct` allows you to connect devices together without a wireless router.


### Exploring Embedded Systems

- An `embedded system` is any device that has a dedicated function and uses a computer system to perform that function.
  - A security challenge with embedded systems is keeping them up to date.
  - Embedded systems include smart devices sometimes called the `Internet of things (IoT)`, such as wearable technology and home automation devices.

  - Embedded systems are found in many common and special-purpose devices. This includes multi-function devices (MFDs), such as printers; heating, ventilation, and air conditioning (HVAC) systems; medical devices; automotive vehicles; aircraft; and unmanned aerial vehicles (UAVs).

- A `system on a chip (SoC)` is an integrated circuit that includes a full computing system.

- A `supervisory control and data acquisition (SCADA) system` controls an `industrial control system (ICS)`.
  - The ICS is used in large facilities such as power plants or water treatment facilities.
  - SCADA and ICS systems are typically in isolated networks without access to the Internet, and are sometimes protected by `network intrusion prevention systems (NIPSs)`.

- A `real-time operating system (RTOS)` is an operating system that reacts to input within a specific time.


### Protecting Data

- The primary method of protecting the confidentiality of data is with encryption and strong access controls. File system security includes the use of encryption to encrypt files and folders.

- You can encrypt individual columns in a database, entire databases, individual files, entire disks, and removable media.

- Users should be given only the permissions they need. When they have too much access, it can result in access violations or the unauthorized access of data.

- You can use the `chmod` command to change permissions on a Linux system.

- `Data exfiltration` is the unauthorized transfer of data outside an organization.

- `Data loss prevention (DLP)` techniques and technologies help prevent data loss. They can block transfer of data to USB devices and analyze outgoing data via email to detect unauthorized transfers. Cloud-based DLP systems can enforce security policies for any data stored in the cloud.


---

## Chapter 6 Exam Topic Review


### Understanding Threat Actors

- `Script kiddies` use existing computer scripts or code to launch attacks. They typically have very little expertise or sophistication, and very little funding.

- `hacktivist` launches attacks as part of an activist movement or to further a cause.

- `Insiders` (such as employees of a company) have legitimate access to an organization’s internal resources. They sometimes become malicious insiders out of greed or revenge.

- `Competitors` sometimes engage in attacks to gain proprietary information about another company.

- `Organized crime` is an enterprise that employs a group of individuals working together in criminal activities. Their primary motivation is money.

- Some attackers are organized and sponsored by a `nation-state or government`.

- `advanced persistent threat (APT)` is a targeted attack against a network. An APT group has both the capability and intent to launch sophisticated and targeted attacks. They are sponsored by a nation-state and often have a significant amount of resources and funding.

- A common method attackers often use before launching an attack is to gather information from open-source intelligence, including any information available via web sites and social media.

### Determining Malware Types

- Malware includes several different types of malicious code, including `viruses, worms, logic bombs, backdoors, Trojans, ransomware, rootkits, and more`.

- `virus` is malicious code that attaches itself to a host application. The code runs when the application is launched.

- `worm` is self-replicating malware that travels throughout a network without user intervention.

- `logic bomb` executes in response to an event, such as a day, time, or condition. Malicious insiders have planted logic bombs into existing systems, and these logic bombs have delivered their payload after the employee left the company.

- `Backdoors` provide another way of accessing a system. Malware often inserts backdoors into systems, giving attackers remote access to systems.

- `Trojan` appears to be one thing, such as pirated software or free antivirus software, but is something malicious. A remote access Trojan (RAT) is a type of malware that allows attackers to take control of systems from remote locations.

- `Drive-by downloads` often attempt to infect systems with Trojans.

- `Ransomware` is a type of malware that takes control of a user’s system or data. Criminals attempt to extort payment as ransom combined to return control to the user. Crypto-malware is ransomware that encrypts the user’s data. Attackers demand payment to decrypt the data.

- `Spyware` is software installed on user systems without the user’s knowledge or consent and it monitors the user’s activities. It sometimes includes a keylogger that records user keystrokes.

- `botnet` is a group of computers called zombies controlled through a command-and-control server. Attackers use malware to join computers to botnets. Bot herders launch attacks through botnets.

- `Rootkits` take root-level or kernel-level control of a system. They hide  their  processes  to  avoid   detection.  They can remove user privileges and modify system files.
Recognizing Common Attacks

- `Social engineering` is the practice of using social tactics to gain information or trick users into performing an action they wouldn’t normally take.

  - Social engineering attacks can occur in person, over the phone, while surfing the Internet, and via email. Many social engineers attempt to impersonate others.

  - Social engineers and other criminals employ several psychology-based principles to help increase the effectiveness of their attacks. They are authority, intimidation, consensus, scarcity, urgency, familiarity, and trust.

- `Shoulder surfing` is an attempt to gain unauthorized information through casual observation, such as looking over someone’s shoulder, or monitoring screens with a camera. Screen filters can thwart shoulder surfing attempts.

- `hoax` is a message, often circulated through email, that tells of impending doom from a virus or other security threat that simply doesn’t exist.

- `Tailgating` is the practice of one person following closely behind another without showing credentials. Mantraps help prevent tailgating.

- `Dumpster divers` search through trash looking for information. Shredding or burning documents reduces the risk of dumpster diving.

- `Watering hole` attacks discover sites that a targeted group visits and trusts. Attackers then modify these sites to download malware. When the targeted group visits the modified site, they are more likely to download and install infected files.

- `Spam` is unwanted or unsolicited email. Attackers often use spam in different types of attacks.

- `Phishing` is the practice of sending email to users with the purpose of tricking them into revealing sensitive information, installing malware, or clicking on a link.

- `Spear phishing` and whaling are types of phishing. Spear phishing targets specific groups of users and whaling targets high-level executives.

- `Vishing` is a form of phishing that uses voice over the telephone and often uses Voice over IP (VoIP). Some vishing attacks start with a recorded voice and then switch over to a live person.
Blocking Malware and Other Attacks

- `Antivirus software` can detect and block different types of malware, such as worms, viruses, and Trojans. Antivirus software uses signatures to detect known malware.

  - Antivirus software typically includes a file integrity checker to detect files modified by a rootkit.

- `hashes` can verify the integrity of signature files when downloading signatures manually.

- `Data execution prevention (DEP)` prevents code from executing in memory locations marked as nonexecutable. The primary purpose of DEP is to protect a system from malware.

- `Advanced malware tools` monitor files and activity within the network.

- `Anti-spam software` attempts to block unsolicited email. You can configure a spam filter to block individual email addresses and email domains.

- `Security-related awareness and training programs` help users learn about new threats a n d security trends, such as new viruses, new phishing attacks, and zero-day exploits. Zero- d a y exploits take advantage of vulnerabilities that are not known by trusted sources.


---

## Chapter 7 Exam Topic Review

### Comparing Common Attacks

- `DoS` attack is an attack launched from a single system and attempts to disrupt services.

- `DDoS` attacks are DoS attacks from multiple computers. typically include sustained, abnormally high network traffic.

- `Spoofing` attacks attempt to impersonate another system.

  - `MAC address spoofing` changes the source MAC address
  - `IP spoofing` changes the source IP address.

- `ARP poisoning` attacks attempt to mislead computers or switches about the actual MAC address of a system.

  - They can be used to launch a man-in-the-middle attack.

- `DNS poisoning` attacks modify DNS data and can redirect users to malicious sites.

  - Many DNS servers use DNSSEC to protect DNS records and prevent DNS poisoning attacks.

- `Amplification` attacks send increased traffic to, or request additional traffic from, a victim.

- `Password` attacks attempt to discover passwords.

  - A `brute force attack` attempts to guess all possible character combinations

  - a `dictionary attack` uses all the words and character combinations stored in a file.

    - Account lockout policies thwart online brute force attacks and complex passwords thwart offline password attacks.

  - a `hash` attack, the attacker discovers the hash of the user’s password and then uses it to log on to the system as the user.

    - Passwords are often stored as a hash.
    - Weak hashing algorithms are susceptible to collisions, which allow different passwords to create the same hash.

  - a `birthday` attack, an attacker is able to create a password that produces the same hash as the user’s actual password.

    - This is also known as a hash collision.

    - A hash collision occurs when the hashing algorithm creates the same hash from different passwords.

- `Password salting` adds additional characters to passwords before hashing

  - prevents many types of attacks, including dictionary, brute force, and rainbow table attacks.

- `Replay` attacks capture data in a session with the intent of using information to impersonate one of the parties. Timestamps and sequence numbers thwart replay attacks.

- A `known plaintext` attack is possible if an attacker has both the plaintext and the ciphertext created by encrypting the plaintext. It makes it easier to decrypt other data using a similar method.


- Attackers buy domain names with minor typographical errors in typo squatting (also called URL hijacking) attacks. The goal is to attract traffic when users enter incorrect URLs. Attackers can configure the sites with malware to infect visitors or configure the site to generate ad revenue for the attacker.

- `Clickjacking` tricks users into clicking something other than what they think they’re clicking.

- `Session ID attack`, utilize the user’s session ID to impersonate the user.

- `Domain hijacking` attacks allow an attacker to change the registration of a domain name without permission from the owner.

- A `man-in-the-browser` is a proxy Trojan horse that exploits vulnerable web browsers. When successful, it allows attacks to capture keystrokes and all data sent to and from the browser.

- A `driver shim` is additional code that can be run instead of the original driver.

- Attackers `exploiting unknown or undocumented vulnerabilities` are taking advantage of zero-day vulnerabilities. The vulnerability is no longer a zero-day vulnerability after the vendor releases a patch to fix it.

- `Buffer overflows` occur when an application receives more data, or unexpected data, than it can handle and exposes access to system memory.

  - Buffer overflow attacks exploit buffer overflow vulnerabilities. A common method uses NOP instructions or NOP sleds such as a string of x90 commands.
  - Two primary protection methods against buffer overflow attacks are `input validation` and `keeping a system up to date`.

- `Integer overflow` attacks attempt to use or create a numeric value bigger than the application can handle.


### Summarizing Secure Coding Concepts

- `Compiled code` has been optimized by an application and converted into an executable file.
- `Runtime code` is code that is evaluated, interpreted, and executed when the code is run.


- `Input validation` checks the data before passing it to the application

  - A common coding error in web-based applications is the lack of input validation.
  - prevents: `buffer overflow, SQL injection, command injection, and cross-site scripting attacks`.

- `Server-side input validation` is the most secure. Attackers can bypass client-side input validation, but not server-side input validation.

- `Race conditions` allow two processes to access the same data at the same time, causing inconsistent results. Problems can be avoided by locking data before accessing it.

- `Error-handling` routines within applications can prevent application failures and protect the integrity of the operating systems. Error messages shown to users should be generic, but the application should log detailed information on the error.

- `Code signing` uses a digital signature within a certificate to authenticate and validate software code.

- `Code quality and testing techniques` include static code analysis, dynamic analysis (such as fuzzing), stress testing, sandboxing, and model verification.

- `Software development life cycle (SDLC) models` provide structure for software development projects.

- `Waterfall` uses multiple stages with each stage feeding the next stage.
- `Agile` is a more flexible model and it emphasizes interaction with all players in a project.

- `Secure DevOps` is an agile-aligned methodology. It stresses security throughout the lifetime of the project.


### Identifying Application Attacks

- Common web servers are `Apache` (running on Linux) and `Internet Information Services` (running on Microsoft servers).

- Databases are optimized using a process called `normalization`. A database is considered normalized when it conforms to the first three normal forms.

- `SQL injection` attacks provide information about a database and can allow an attacker to read and modify data within a database. Input validation and stored procedures provide the best protection against SQL injection attacks.

- `Cross-site scripting (XSS)` allows an attacker to redirect users to malicious web sites and steal cookies. It uses HTML and JavaScript tags with < and > characters.

- `Cross-site request forgery (XSRF)` causes users to perform actions on web sites without their knowledge and allows attackers to steal cookies and harvest passwords.

- XSS and XSRF attacks are mitigated with input validation techniques.

### Understanding Frameworks and Guides

- `Frameworks` are references that provide a foundation.

- `Cybersecurity frameworks` typically use a structure of basic concepts and provide guidance on how to implement security.
- `Regulatory frameworks` are based on relevant laws and regulations.
- A `non-regulatory framework` is not required by any law.

- Some frameworks are used within a single country (and referred to as national frameworks), while others are used internationally.

- Some frameworks only apply to certain industries. example, organizations that handle credit cards typically comply with the Payment Card Industry Data Security Standard (PCI DSS).

- Vendor-specific guides should be used when configuring specific systems.

--

## Chapter 8 Exam Topic Review

- When preparing for the exam, make sure you understand these key concepts covered in this chapter.
Understanding Risk Management

- A risk is the likelihood that a threat will exploit a vulnerability. A threat is a potential danger that can compromise confidentiality, integrity, or availability of data or a system. A vulnerability is a weakness.

- Impact refers to the magnitude of harm that can be caused if a threat exercises a vulnerability.

- Threat assessments help an organization identify and categorize threats. An environmental threat assessment evaluates the likelihood of an environmental threat, such as a natural disaster, occurring. Manmade threat assessments evaluate threats from humans.

- Internal threat assessments evaluate threats from within an organization. External threat assessment evaluates threats from outside an organization.

- A vulnerability is a flaw or weakness in software or hardware, or a weakness in a process that a threat could exploit, resulting in a security breach.

- Risk management attempts to reduce risk to a level that an organization can accept, and the remaining risk is known as residual risk.

- Senior management is responsible for managing risk and the losses associated from residual risk.

- You can avoid a risk by not providing a service or participating in a risky activity. Purchasing insurance, such as fire insurance, transfers the risk to another entity. Security controls mitigate, or reduce, risks. When the cost of a control outweighs a risk, it is common to accept the risk.

- A risk assessment quantifies or qualifies risks based on different values or judgments. It starts by identifying asset values and prioritizing high-value items.

- Quantitative risk assessments use numbers, such as costs and asset values. The single loss expectancy (SLE) is the cost of any single loss. The annual rate of occurrence (ARO) indicates how many times the loss will occur annually. You can calculate the annual loss expectancy (ALE) as SLE × ARO.

- Qualitative risk assessments use judgments to prioritize risks based on likelihood of occurrence and impact. These judgments provide a subjective ranking.

- Risk assessment results are sensitive. Only executives and security professionals should be granted access to risk assessment reports.

- A risk register is a detailed document listing information about risks. It typically includes risk scores along with recommended security controls to reduce the risk scores.

- A supply chain assessment evaluates a supply chain needed to produce and sell a product. It includes raw materials and all the processes required to create and distribute a finished product.
Comparing Scanning and Testing Tools

- A port scanner scans systems for open ports and attempts to discover what services and protocols are running.

- Network mapping identifies the IP addresses of hosts within a network. Network scanners expand on network mapping. They identify the operating system running on each host. They can also identify services and protocols running on each host.

- Wireless scanners can detect rogue access points (APs) in a network. Many can also crack passwords used by the APs.

- Banner grabbing queries remote systems to detect their operating system, along with services, protocols, and applications running on the remote system.

- Vulnerability scanners passively test security controls to identify vulnerabilities, a lack of security controls, and common misconfigurations. They are effective at discovering systems susceptible to an attack without exploiting the systems.

- A false positive from a vulnerability scan indicates the scan detected a vulnerability, but the vulnerability doesn’t exist.

- Credentialed scans run under the context of an account and can be more accurate than non- credentialed scans, giving fewer false positives.

- Penetration testers should gain consent prior to starting a penetration test. A rules-of- engagement document identifies the boundaries of the test.

- A penetration test is an active test that attempts to exploit discovered vulnerabilities. It starts with a vulnerability scan and then bypasses or actively tests security controls to exploit vulnerabilities.

- Passive reconnaissance gathers information from open-source intelligence.

- Active reconnaissance uses scanning techniques to gather information.

- After initial exploitation, a penetration tester uses privilege escalation techniques to gain more access.

- Pivoting during a penetration test is the process of using an exploited system to access other systems.

- In black box testing, testers perform a penetration test with zero prior knowledge of the environment. White box testing indicates that the testers have full knowledge of the environment, including documentation and source code for tested applications. Gray box testing indicates some knowledge of the environment.

- Scans can be either intrusive or non-intrusive. Penetration testing is intrusive (also called invasive) and can potentially disrupt operations. Vulnerability testing is non-intrusive (also called non-invasive).

- Exploitation frameworks store information about security vulnerabilities. They are often used by penetration testers (and attackers) to detect and exploit software.
Using Security Tools

- Protocol analyzers (sniffers) can capture and analyze data sent over a network. Testers (and attackers) use protocol analyzers to capture cleartext data sent across a network.

- Administrators use protocol analyzers for troubleshooting communication issues by inspecting protocol headers to detect manipulated or fragmented packets.

- Captured packets show the type of traffic (protocol), source and destination IP addresses, source and destination MAC addresses, and flags.

- Tcpdump is a command-line protocol analyzer. Captured packet files can be analyzed in a graphical protocol analyzer such as Wireshark.

- Nmap is a sophisticated network scanner run from the command line.

- Netcat is a command-line tool used to remotely administer servers. Netcat can also be used for banner grabbing.

- Logs record events and by monitoring logs, administrators can detect event anomalies.

- Security logs track logon and logoff activity on systems.

- System logs identify when services start and stop.

- Firewall and router logs identify the source and destination of traffic.

- A security information and event management (SIEM) system can aggregate and correlate logs from multiple sources in a single location. A SIEM also provides continuous monitoring and automated alerting and triggers.

- Continuous security monitoring helps an organization maintain its security posture, by verifying that security controls continue to function as intended.

- User auditing records user activities. User auditing reviews examine user activity.

- Permission auditing reviews help ensure that users have only the rights and permissions they need to perform their jobs, and no more.

---

## Implementing Controls to Protect Assets

CompTIA Security+ objectives covered in this chapter:
1.2	Compare and contrast types of attacks.
Social engineering (Tailgating)
1.6	Explain the impact associated with types of vulnerabilities.
Vulnerable business processes, System sprawl/undocumented assets, Architecture/ design weaknesses
2.1	Install and configure network components, both hardware- and software-based, to support organizational security.
Load balancer (Scheduling [Affinity, Round-robin], Active- passive, Active-active, Virtual IPs)
2.2	Given a scenario, use appropriate software tools to assess the security posture of an organization.
Backup utilities
2.3	Given a scenario, troubleshoot common security issues.
Asset management
3.1	Explain use cases and purpose for frameworks, best practices and secure configuration guides.
Defense-in-depth/layered security (Vendor diversity, Control diversity [Administrative, Technical], User training)
3.8	Explain how resiliency and automation strategies reduce risk.
Distributive allocation, Redundancy, Fault tolerance, High availability, RAID
3.9	Explain the importance of physical security controls.
Lighting, Signs, Fencing/gate/cage, Security guards, Alarms, Safe, Secure cabinets/ enclosures, Protected distribution/Protected cabling, Airgap, Mantrap, Faraday cage, Lock



types, Biometrics, Barricades/bollards, Tokens/cards, Environmental controls (HVAC, Hot and cold aisles, Fire suppression), Cable locks, Cameras, Motion detection, Logs, Infrared detection, Key management
4.3	Given a scenario, implement identity and access management controls.
Physical access control (Proximity cards, Smart cards)
5.2	Summarize business impact analysis concepts.
RTO/RPO, MTBF, MTTR, Mission-essential functions, Identification of critical systems, Single point of failure, Impact (Life, Property, Safety, Finance, Reputation), Privacy impact assessment, Privacy threshold assessment
5.6	Explain disaster recovery and continuity of operation concepts.
Recovery sites (Hot site, Warm site, Cold site), Order of restoration, Backup concepts (Differential, Incremental, Snapshots, Full), Geographic considerations (Off-site backups, Distance, Location selection, Legal implications, Data sovereignty), Continuity of operation planning (Exercises/tabletop, After-action reports, Failover, Alternate processing sites, Alternate business practices)
**
You can’t eliminate risk to an organization’s assets. However, you can reduce the impact of many threats by implementing security controls. It’s common to implement several controls using a defense-in-depth strategy. Physical security controls help protect access to secure areas. Redundancy and fault-tolerance strategies help eliminate single points of failure for critical systems. Backups ensure that data remains available even after data is lost. More in-depth business continuity strategies help ensure mission-critical functions continue to operate even if a disaster destroys a primary business location. This chapter covers these concepts.
Implementing Defense in Depth
Defense in depth (also known as layered security) refers to the security practice of implementing several layers of protection. You can’t simply take a
single action, such as implementing a firewall or installing antivirus software, and consider yourself protected. You must implement security at several different layers. This way, if one layer fails, you still have additional layers to protect you.
If you drive your car to a local Walmart, put a five-dollar bill on the dash, and leave the keys in the car and the car running, there is a very good chance the car won’t be there when you come out of the store. On the other hand, if you ensure nothing of value is visible from the windows, the car is locked, it has an alarm system, and it has stickers on the windows advertising the alarm system, it’s less likely that someone will steal it. Not impossible, but less likely.
You’ve probably heard this as “there is no silver bullet.” If you want to kill a werewolf, you can load your gun with a single silver bullet and it will find its mark. The truth is that there is no such thing as a silver bullet. (Of course, there is no such thing as a werewolf either.)
Applied to computers, it’s important to implement security at every step, every phase, and every layer. Information technology (IT) professionals can never rest on their laurels with the thought they have done enough and no longer need to worry about security.
Control diversity is the use of different security control types, such as technical controls, administrative controls, and physical controls. example, technical security controls such as firewalls, intrusion detection systems (IDSs), and proxy servers help protect a network. Physical security controls can provide extra protection for the server room or other areas where these



devices are located. Administrative controls such as vulnerability assessments and penetration tests can help verify that these controls are working as expected.

￼
Vendor diversity is the practice of implementing security controls from different vendors to increase security. example, Chapter 3, “Exploring Network Technologies and Tools,” describes a demilitarized zone (DMZ). Many DMZs use two firewalls and vendor diversity dictates the use of firewalls from different vendors. example, one firewall could be a Cisco firewall and the other one could be a Check Point firewall. If a vulnerability is discovered in one of these firewalls, an attacker might be able to exploit it. However, it’s unlikely that both firewalls would develop a vulnerability at the same time.




User training also helps provide defense in depth. If users engage in risky behaviors, such as downloading and installing files from unknown sources or responding to phishing emails, they can give attackers a path into an organization’s network. However, providing regular training to users on common threats, and emerging threats, helps them avoid these types of attacks.
Comparing Physical Security Controls
A physical security control is something you can physically touch, such as a hardware lock, a fence, an identification badge, and a security camera. Physical security access controls attempt to control entry and exits, and organizations commonly implement different controls at different boundaries, such as the following:
Perimeter. Military bases and many other organizations erect a fence around the entire perimeter of their land. They often post security guards at gates to control access. In some cases, organizations
install barricades to block vehicles.
Buildings. Buildings commonly have additional controls for both safety and security. example, guards and locked doors restrict entry so only authorized personnel enter. Many buildings include lighting and video cameras to monitor the entrances and exits.
Secure work areas. Some companies restrict access to specific work areas when employees perform classified or restricted access tasks. In some cases, an organization restricts access to all internal work areas. In other words, visitors can enter the lobby of a building, but they are not able to enter internal work areas without an escort.
Server and network rooms. Servers and network devices such as routers and switches are normally stored in areas where only the appropriate IT personnel can access them. These spaces may be designated as server rooms or wiring closets. I t ’s common for an organization to provide additional physical security for these rooms to prevent attackers from accessing the equipment. example, locking a wiring closet prevents an attacker from installing illicit monitoring hardware, such as a protocol analyzer, to capture network traffic.
Hardware. Additional physical security controls protect individual systems. example, server rooms often have locking cabinets to protect servers and other equipment



installed in the equipment bays. Cable locks protect laptop computers, and smaller devices can be stored in safes.
Airgap. An airgap is a physical security control that ensures that a computer or network is physically isolated from another computer or network. example, you can isolate a computer from a network by ensuring that it is not connected to any other system in the network. This lack of connectivity provides an airgap. This is often done to separate classified networks from unclassified networks.
Using Signs
A simple physical security control is a sign. F o r example, an “Authorized Personnel Only” sign will deter many people from entering a restricted area. Similarly, “No Trespassing” signs let people know they shouldn’t enter. Of course, these signs won’t deter everyone, so an organization typically uses additional physical security measures.
Comparing Door Lock Types
It’s common to secure access to controlled areas of a building with door locks, and there are many different lock types. A door access system is one that only opens after some access control mechanism is used. Some common door access systems are cipher locks, proximity cards, and biometrics.
When implementing door access systems, i t’s important to limit the number of entry and exit points. example, if a data center has only one entrance and exit, it is much easier to monitor this single access point. You can control it with door locks, video surveillance, and guards. On the other hand, if the data center has two entry/exit points, you need another set of controls to control access in both places.
Another important consideration with door access systems is related to personnel safety and fire. In the event of a fire, door access systems should allow personnel to exit the building without any form of authentication.
￼
Securing Door Access with Cipher Locks
Cipher locks often have four or five buttons labeled with numbers. Employees press the numbers in a certain order to unlock the door. example, the cipher code could be 1, 3, 2, 4. Users enter the code in the correct order to gain access. Cipher locks can be electronic or manual. An electronic cipher lock automatically unlocks the door after you enter the correct code into the keypad. A manual cipher lock requires the user to turn a handle after entering the code.
To add complexity and reduce brute force attacks, many manual cipher locks include a code that requires two numbers entered at the same time. Instead of just 1, 3, 2, 4, the code could be 1/3 (entered at the same time), then 2, 4, 5.
One challenge with cipher locks is that they don’t identify the users. Further, uneducated



users can give out the cipher code to unauthorized individuals without understanding the risks. Shoulder surfers might attempt to discover the code by watching users as they enter. Security awareness training can help reduce these risks.
Securing Door Access with Cards
It’s also possible to secure access to areas with proximity cards or smart cards. Proximity cards are small credit card-sized cards that activate when they are in close proximity to a card reader. Many organizations use these for access points, such as the entry to a building or the entry to a controlled area within a building. The door uses an electronic lock that only unlocks when the user passes the proximity card in front of a card reader.
Similarly, it’s possible to use smart cards or physical tokens (described in Chapter 2, “Understanding Identity and Access Management”) for door access. In some scenarios, the smart cards include proximity card electronics. In other scenarios, users must insert the smart card into a smart card reader to gain access.
You’ve probably seen proximity card readers implemented with credit card readers. Many self-serve gasoline stations and fast-food restaurants use them. Instead of swiping your credit card through a magnetic reader, you simply pass it in front of the reader (in close proximity to the reader), and the reader extracts your credit card’s information.
These are becoming popular elsewhere, too. example, if you stay at a Walt Disney World property, they can issue you a bracelet that includes the functionality of a proximity card. To enter your hotel room, you wave your bracelet in front of the door. If you want to buy food or souvenirs or pay for almost anything, you can simply wave your bracelet in front of a card reader to complete your purchase.
The card (and bracelet) doesn’t require its own power source. Instead, the electronics in the card include a capacitor and a coil that can accept a charge from the proximity card reader. When you pass the card close to the reader, the reader excites the coil and stores a charge in the capacitor. Once charged, the card transmits the information to the reader using a radio frequency. When used with door access systems, the proximity card can send just a simple signal to unlock the door. Some systems include details on the
user and record when the user enters or exits the area. When used this way, it’s common to combine the proximity card reader with a keypad requiring the user to enter a personal identification number (PIN). This identifies and authenticates the user with multifactor authentication. The user has something
(the proximity
card) and knows something (a PIN).
Many organizations use proximity cards with turnstiles to provide access for a single person at a time. These are the same type of turnstiles used as entry gates in subways, stadiums, and amusement parks.






Securing Door Access with Biometrics
It’s also possible to use biometric methods as an access control system.
One of the benefits is that some biometric methods provide both identification and authentication. When connected



to a back-end database, these systems can easily record the activity, such as who entered the area and when.
example, you can install a retina scanner at the entrance to a secure server room. When individuals want to enter, the biometric scanner identifies and authenticates them. It’s important to ensure you use an accurate biometric system and configure it to use a low false acceptance rate, as described in Chapter 2. Otherwise, it might falsely identify unauthorized individuals and grant them access.
￼
Tailgating
Chapter 6, “Comparing Threats, Vulnerabilities, and Common Attacks,” discusses several types of social engineering attacks and tailgating is another one. Tailgating (also called piggybacking) occurs when one user follows closely behind another user without using credentials. example, if Lisa opens a door with her proximity card and Bart follows closely behind her without using a proximity card, Bart is tailgating. If authorized users routinely do this, it indicates the environment is susceptible to a social engineering attack where an unauthorized user follows closely behind an authorized user.
example, an organization hired a security company to perform a vulnerability assessment. The company sent one of its top security professionals (who happened to be an attractive woman) to see if she could get into the building. She saw that employees were using proximity cards to get into the building, but she didn’t have one. Instead, she loaded herself up with a book bag and a laptop—ensuring her hands weren’t free. She timed her approach carefully and followed closely behind an employee with a proximity card. She flashed a friendly smile, and sure enough, the employee held the door open for her.
Most of us learn to be polite and courteous and social engineers take advantage of this. It’s polite to hold a door open for people who have their
hands full. In contrast, it’s rude to slam the door in the face of someone following behind us. However, most users don’t want to help criminals. Security awareness programs and training help users understand how criminals use tactics such as tailgating. Educated users are less likely to be tricked, even by a friendly smile from an attractive woman.
High-traffic areas are most susceptible to tailgating attacks. Security guards can be an effective preventive measure at access points, but they need to be vigilant to ensure that tailgating does not occur. The best solution is a mantrap.
Preventing Tailgating with Mantraps
A mantrap is a physical security mechanism designed to control access to a secure area through a buffer zone. Personnel use something like a proximity card to gain access, and the mantrap allows one person, and only one person, to pass through. Because they only allow one person through at a time, mantraps prevent tailgating. Mantraps get their name due to their



ability to lock a person between two areas, such as an open access area and a secure access area, but not all of them are that sophisticated.
An example of a simple mantrap is a turnstile similar to what you see in many public transport systems. Even if you’ve never ridden the subway in one of many U.S. cities or the Tube in London, you’ve probably seen turnstiles in movies such as While You Were Sleeping. When customers present a token, the turnstile unlocks and allows a single person through at a time. Similarly, users unlock the turnstile mantrap with something like a proximity card.

￼
A sophisticated mantrap is a room, or even a building, that creates a large buffer area between the secure area and the unsecured area. Access through the entry door and the exit door is tightly controlled, either with guards or with an access card such as a proximity card.



It’s also possible to require identification and authentication before allowing passage through a mantrap. F o r example, a retina scanner can identify individuals  and restrict access	to only authorized individuals. Similarly, some card reader systems support the use of unique PINs assigned to the user. Users present their card and enter their PIN to gain access before the mantrap opens.
Increasing Physical Security with Guards
Many organizations use security guards to control access to buildings and secure spaces. If employees have ID badges, guards can check these badges prior to granting the employees access. Even if ID badges aren’t used, guards can still verify people’s identity using other identification. Similarly, the security guards can restrict access by checking people’s identity against a preapproved access control list. In some cases, guards record all access in an access log. Security guards can also take a less-active role to deter security
incidents. example, a security guard can deter tailgating incidents by observing personnel when
they use their proximity card to gain access to a secure area.
Monitoring Areas with Cameras
Organizations are increasingly using security cameras in the workplace and surrounding areas for video surveillance. This includes areas outside of a building, such as a parking lot, and all building entrances and exits. Additionally, many organizations use cameras to monitor internal entrances of high-security areas, such as the entrance of a data center or server room.
Cameras are connected to a closed-circuit television (CCTV) system, which transmits signals from video cameras to monitors that are similar to TVs. In addition to providing security, CCTV can also enhance safety by deterring threats.
Organizations often use video cameras within a work environment to protect employees and enhance security in the workplace. In addition to live monitoring, most systems include a recording element, and they can verify if someone is stealing the company’s assets. By recording activity, videos can be played back later for investigation and even prosecution.
Video surveillance provides the most reliable proof of a person’s location and
activity.



Access logs provide a record, but it’s possible to circumvent the security of an access log. example, if Bart used your proximity card to gain access to a secure space, the log will indicate you entered, not Bart. In contrast, if the video shows that Bart entered the room at a certain time of day, it’s not easy for Bart to refute the video.
￼
When using video surveillance in a work environment, it’s important to respect privacy and to be aware of privacy laws. Some things to consider are:
Only record activity in public areas. People have a reasonable expectation of privacy in certain areas, such as locker rooms and restrooms, and it is often illegal to record activity in these areas.
Notify employees of the surveillance. If employees aren’t notified of the surveillance, legal issues related to the video surveillance can arise. This is especially true if the recordings are used when taking legal and/or disciplinary actions against an employee.
Do not record audio. Recording audio is illegal in many jurisdictions, without the express consent of all parties being recorded. Many companies won’t even sell surveillance cameras that record audio.
Fencing, Lighting, and Alarms
Fences provide a barrier around a property and deter people from entering. When using a fence, it’s common to control access to the area via specific gates. Guards often monitor these gates and ensure only authorized individuals can enter. When additional security is required, organizations sometimes configure dual gates, allowing access into one area where credentials are checked before allowing full access. This effectively creates a cage preventing full access, but also prevents unauthorized individuals from escaping.
Installing lights at all the entrances to a building can deter attackers
from trying to break in. Similarly, lighting at the entrances of any internal restricted areas can deter people from trying to enter. Many organizations use a combination of automation, light dimmers, and motion sensors to save on electricity costs without sacrificing security. The lights automatically turn on at dusk, but in a low, dimmed mode. When the motion sensors detect any movement, the lights turn on at full capacity. They automatically turn off at dawn.
It’s important to protect the lights. example, if an attacker can remove the light bulbs, it defeats the control. Either place the lights high enough so that they can’t be easily reached, or protect them with a metal cage.
Alarms provide an additional physical security protection. This includes alarms that detect fire and alarms that detect unauthorized access. Fire alarms detect smoke and/or heat and trigger fire suppression systems. Burglary prevention systems monitor entry points such as doors and windows, detecting when someone opens them.
Y o u can also combine motion detection systems with burglary prevention systems. They detect movement within monitored areas and trigger alarms. Obviously, you wouldn’t have



motion detection systems turned on all the time. Instead, you’d turn them on when people will not be working in the area, such as during nights or weekends.

￼
You might have noticed that fencing, lighting, and alarms can all be combined with motion detection. At the most basic level, motion detection methods detect moving objects. Many motion detectors use microwave technologies to detect movement. This is like the technology used in some police radar speed guns.




A
more advanced method is infrared detection. Infrared detectors sense infrared radiation, sometimes called infrared light, which effectively sees a difference between objects of different temperatures. example, a person is much warmer than objects in a room and easily stands out using an infrared detector. This can help eliminate false alarms by sensing more than just motion, but motion from objects of different temperatures.
Securing Access with Barricades
In some situations, fencing isn’t enough to deter potential attackers. To augment fences and other physical security measures, organizations erect stronger barricades. example, military bases often erect strong, zigzag barricades that require vehicles to slow down to navigate through them. This prevents attackers from trying to ram through the gates.
Businesses and organizations need to present an inviting appearance, so they can’t use such drastic barricades. However, they often use bollards, which are short vertical posts, composed of reinforced concrete and/or steel. They often place the bollards in front of entrances about three or four feet apart. They typically paint them with colors that match their store. You’ve probably walked through a set of bollards multiple times without giving them a second thought. However, thieves who are contemplating driving a car or truck through the entrance see them.
￼
Many thieves have driven vehicles right through the front of buildings, and then proceeded to steal everything in sight. Depending on the strength of the walls, criminals might even be able to drive through a wall with a truck. Strategically placed bollards will prevent these types of attacks.
Using Hardware Locks
You can implement simple physical security measures to prevent access to secure areas. example, you can use hardware locks—similar to what you use to secure your home—to secure buildings as well as rooms within buildings. Companies that don’t have the resources to employ advanced security systems often use these types of hardware locks.



Instead of allowing free access to wiring closets or small server rooms, small organizations use these types of locks to restrict access. Although these locks aren’t as sophisticated as the ones used by large organizations, they are much better than leaving the rooms open and the equipment exposed.
Key management is an important concept to consider when using hardware locks. Proper key management ensures that only authorized personnel can access the physical keys. This might be done by locking keys within a safe or locking cabinet.
Securing Mobile Computers with Cable Locks
Cable locks are a great theft deterrent for mobile computers, and even many desktop computers at work. Computer cable locks work similar to how a bicycle cable lock works. However, instead of securing a bicycle to a bike rack or post, a computer cable lock secures a computer to a piece of furniture.
The user wraps the cable around a desk, table, or something heavy, and then plugs it into an opening in the laptop specifically created for this purpose. Most cable locks have a four-digit combo. If you (or anyone) remove the cable lock without the combo, it will likely destroy the laptop.
Another common use of cable locks is for computers in unsupervised labs. example, you can secure laptop or desktop computers with cable locks in a training lab. This allows you to leave the room open so that students can use the equipment, but the cable locks prevent thieves from stealing the equipment.
Securing Servers with Locking Cabinets
Larger companies often have large server rooms with advanced security to restrict access. Additionally, within the server room, administrators use locking cabinets or enclosures to secure equipment mounted within the bays. An equipment bay is about the size of a large refrigerator and can hold servers, routers, and other IT equipment. These bays have doors in the back and many have doors in the front, too. Administrators lock these doors to prevent unauthorized personnel from accessing the equipment.
￼
Offices often have file cabinets that lock, too, so i t ’s important to pay attention to the context when referring to locking cabinets. example, if you want to secure equipment within a server room, a locking cabinet is one of many physical security controls you can use. If you want to secure unattended smartphones in an office space, you can also use a locking cabinet, but this is an office file cabinet that locks.
Securing Small Devices with a Safe
Locking file cabinets or safes used in many offices help prevent the theft of smaller devices. example, you can store smaller devices such as external USB drives or USB flash drives in an



office safe or locking cabinet when they aren’t in use. Depending on the size of the office safe and office cabinet, you might also be able to secure laptops within them.
Asset	Management
Asset management is the process of tracking valuable assets throughout their life cycles. example, organizations commonly implement processes to track hardware such as servers, desktop computers, laptop computers, routers, and switches. An effective asset management system can help reduce several vulnerabilities:
Architecture and design weaknesses. Asset management helps reduce architecture and design weaknesses by ensuring that purchases go through an approval process. The approval process does more than just compare costs. It also evaluates the purchase to ensure it fits in the overall network architecture. Unapproved assets often weaken security by adding in additional resources that aren’t managed.
System sprawl and undocumented assets. System sprawl occurs when an organization has more systems than it needs, and systems it owns are underutilized. Asset management begins before the hardware is purchased and helps prevent system sprawl by evaluating the purchase. Additionally, after the purchase is completed, asset management processes ensure hardware is added into the asset management tracking system. This ensures that the assets are managed and tracked from cradle to grave.
Many organizations use automated methods for inventory control. example, radio- frequency identification (RFID) methods can track the movement of devices. These are the same types of devices used in stores to prevent shoplifting. If someone exits without paying, the RFID device transmits when the shoplifter gets close to the exit door and sounds an alarm. Organizations won’t necessarily have an alarm, but they can track the movement of devices.
Mobile devices are easy to lose track of, so organizations often use asset- tracking methods to reduce losses. example, when a user is issued a mobile device, asset-tracking methods record it. Similarly, if the user leaves
the company, asset-tracking methods ensure the user returns the device.
Implementing Environmental Controls
Although environmental controls might not seem security related, they directly contribute to the availability of systems. This includes ensuring temperature and humidity controls are operating properly, fire suppression systems are in place, and proper procedures are used when running cables.
Heating, Ventilation, and Air Conditioning
Heating, ventilation, and air conditioning (HVAC) systems are important physical security controls that enhance the availability of systems. Quite simply, computers and other electronic equipment can’t handle drastic changes in temperatures, especially hot temperatures. If systems overheat, the chips can actually burn themselves out.
The cooling capacity of HVAC systems is measured as tonnage. This has nothing to do with weight, but instead refers to cooling capacity. One ton of cooling equals 12,000 British thermal units per hour (Btu/hour), and typical home HVAC systems are three-ton units. Higher-tonnage HVAC systems can cool larger areas or areas with equipment generating more heat.
The amount of air conditioning needed to cool a massive data center is much greater than



you need to cool your home, primarily because of all the heat generated by the equipment. If your home air conditioner fails in the middle of summer, you might be a little uncomfortable for a while, but if the data center HVAC system fails, it can result in loss of availability and a substantial loss of money.
I worked in several environments where we had a policy of shutting down all electronics when the room temperature reached a certain threshold. When we didn’t follow the policy, the systems often developed problems due to the heat and ended up out of commission for a lot longer than the AC.
Most servers aren’t in cases like a typical desktop computer. Instead, they are housed in rack-mountable cases. These rack-mountable servers are installed in equipment cabinets (also called racks or bays) about the size of tall refrigerators. A large data center will have multiple cabinets lined up beside each other in multiple rows.
These cabinets usually have locking doors in the front and rear for physical security. The doors are perforated with cold air coming in the front, passing over and through the servers to keep them cool, and warmer air exiting out the rear. Additionally, a server room has raised flooring with air conditioning pumping through the space under the raised floor.
￼
Hot and Cold Aisles
Hot and cold aisles help regulate the cooling in data centers with multiple rows of cabinets. The back of all the cabinets in one row faces the back of all the cabinets in an adjacent row. Because the hot air exits out the back of the cabinet, the aisle with the backs facing each other is the hot aisle.
Similarly, the front of the cabinets in one row is facing the front of the cabinets in the adjacent row. Cool air is pumped through the floor to this cool aisle using perforated floor tiles in the raised flooring. This is the cold aisle. In some designs, cool air is also pumped through the base of the cabinets.
This depends on the design of the cabinets and the needs of the equipment. Consider what happens if all the cabinets had their front facing the same way
without a hot/cold aisle design. The hot air pumping out the back of one row of cabinets would be sent to the front of the cabinets behind them. The front
row would have very cold air coming in the front, but other rows would have warmer air coming in the front.
Of course, an HVAC also includes a thermostat as a temperature control and additional humidity controls. The thermostat ensures that the air temperature is controlled and maintained. Similarly, humidity controls ensure that the humidity is controlled. High humidity can cause condensation on the equipment, which causes water damage. Low humidity allows a higher incidence of electrostatic discharge (ESD).
HVAC and Fire
HVAC systems are often integrated with fire alarm systems to help prevent a fire from spreading. One of the core elements of a fire is oxygen. If the HVAC system continues to operate normally while a fire is active, it continues to pump oxygen, which feeds the fire. When the HVAC system is integrated



with the fire alarm system, it controls the airflow to help prevent the rapid spread of the fire. Many current HVAC systems have dampers that can control airflow to specific areas of a building. Other HVAC systems automatically turn off when fire suppression systems detect a fire.
￼
Fire Suppression
Yo u can fight fires with individual fire extinguishers, with fixed systems, or both. Most organizations included fixed systems to control fires and place portable fire extinguishers in different areas around the organization. A fixed system can detect a fire and automatically activate to extinguish the fire. Individuals use portable fire extinguishers to extinguish or suppress small fires.
The different components of a fire are heat, oxygen, fuel, and a chain reaction creating the fire. Fire suppression methods attempt to remove or disrupt one of these elements to extinguish a fire. You can extinguish a fire using one of these methods:
Remove the heat. Fire extinguishers commonly use chemical agents or water to remove the heat. However, water should never be used on an electrical fire.
Remove the oxygen. Many methods use a gas, such as carbon dioxide (CO2) to displace the oxygen. This is a common method of fighting electrical fires because CO2 and similar gasses are harmless to electrical equipment.
Remove the fuel. Fire-suppression methods don’t typically fight a fire this way, but of course, the fire will go out once all the material is burned.
Disrupt the chain reaction. Some chemicals can disrupt the chain reaction of fires to stop them.
When implementing any fire suppression system, it’s important to
consider the safety of personnel. example, if a fire suppression system uses a gas such as carbon dioxide (CO2) to displace the oxygen, it’s important to ensure that personnel can get out before the oxygen is
displaced.
Similarly, consider an exit door secured with a proximity card. Normally, employees open the door with the proximity card and the system records their exit. What happens if a fire starts and power to the building is lost? The proximity card reader won’t work, and if the door can’t open, employees will be trapped. It’s important to ensure that an alternative allows personnel to exit even if the proximity card reader loses power. Of course, this might introduce a vulnerability to consider. You don’t want an attacker to access a secure data center just by removing power to the proximity reader.
Environmental Monitoring
Environmental monitoring includes temperature and humidity controls. From a very basic perspective, an HVAC system monitors the current temperature and humidity and makes adjustments as necessary to keep the temperature and humidity constant.



Large-scale data centers often have sophisticated logging capabilities for environmental monitoring. The HVAC system still attempts to keep the temperature and humidity constant. However, the logs record the actual temperature and humidity at different times during the day. This allows administrators to review the performance of the HVAC system, to see if it is able to keep up with the demands within the data center.
Shielding
Shielding helps prevent electromagnetic interference (EMI) and radio frequency interference (RFI) from interfering with normal signal transmissions. It also protects against unwanted emissions and helps prevent an attacker from capturing network traffic.
Although you might see EMI and RFI in the same category as EMI/RFI, they are different. EMI comes from different types of motors, power lines, and even fluorescent lights. RFI comes from radio frequency (RF) sources such as AM or FM transmitters. However, shielding used to block interference from both EMI and RFI sources is often referred to as simply EMI shielding.
Attackers often use different types of eavesdropping methods to capture network traffic. If the data is emanating outside of the wire or outside of an enclosure, attackers may be able to capture and read the data. EMI shielding fulfills the dual purpose of keeping interference out and preventing attackers from capturing network traffic.
Protected Cabling
Twisted-pair cable, such as CAT5e and CAT6 cable, comes in both shielded twisted-pair (STP) and unshielded twisted-pair (UTP) versions. The shielding helps prevent an attacker from capturing network traffic and helps block interference from corrupting the data.
When data travels along a copper wire (such as twisted-pair), it creates an induction field around the wire. If you have the right tools, you can simply place the tool around the wire and capture the signal. The shielding in STP cable blocks this. Fiber-optic cable is not susceptible to this type of attack. Signals travel along a fiber-optic cable as light pulses, and they do not create
an induction field.
Protected Distribution of Cabling
Physical security includes planning where you route cables and how you route them. Skilled network administrators can cut a twisted-pair cable, attach an RJ-45 connector to each end, and connect them back together with an adapter in less than 5 minutes. Experienced fiber-optic cable technicians can do the same thing with a fiber-optic cable within 10 minutes.
If an attacker did this, he could connect the cut cable with a hub, and then capture all the traffic going through the hub with a protocol analyzer. This represents a significant risk.
One method of reducing this risk is to run cables through cable troughs or wiring ducts. A cable trough is a long metal container, typically about 4 inches wide by 4 inches high. If you run data cables through the cable trough, they aren’t as accessible to potential attackers. In contrast, many organizations simply run the cable through a false ceiling or a raised floor.
In addition to considering physical security, it’s important to keep the cables away from EMI sources. example, if technicians run cables over or through fluorescent lighting fixtures, the EMI from the lights can disrupt the signals on the cables. The result is intermittent connectivity for users.



Faraday Cage
A Faraday cage is typically a room that prevents signals from emanating beyond the room. It includes electrical features that cause RF signals that reach the boundary of the room to be reflected back, preventing signal emanation outside the Faraday cage. A Faraday cage can also be a small enclosure.
In addition to preventing signals from emanating outside the room, a Faraday cage also provides shielding to prevent outside interference such as EMI and RFI from entering the room. At a very basic level, some elevators act as a Faraday cage (though I seriously doubt the designers were striving to do so). You might have stepped into an elevator and found that your cell phone stopped receiving and transmitting signals. The metal shielding around the elevator prevents signals from emanating out or signals such as the cell phone
tower signal from entering
the elevator.
On a smaller scale, electrical devices such as computers include shielding to prevent signals from emanating out and block interference from getting in.
￼
Adding Redundancy and Fault Tolerance
One of the constants with computers, subsystems, and networks is that they will fail. It’s one of the few things you can count on. It’s not a matter of if they will fail, but when. However, by adding redundancy into your systems and networks, you can increase the reliability of your systems even when they fail. By increasing reliability, you increase one of the core security goals: availability.
Redundancy adds duplication to critical system components and networks and provides fault tolerance. If a critical component has a fault, the duplication provided by the redundancy allows the service to continue as if a fault never occurred. In other words, a system with fault tolerance can suffer a fault, but it can tolerate it and continue to operate. Organizations often add redundancies to eliminate single points of failure.
You can add redundancies at multiple levels:
Disk redundancies using RAID
Server redundancies by adding failover clusters
Power redundancies by adding generators or an UPS
Site redundancies by adding hot, cold, or warm sites
Single Point of Failure
A single point of failure is a component within a system that can cause the entire system to fail if the component fails. When designing redundancies, an organization will examine different



components to determine if they are a single point of failure. If so, they take steps to provide a redundancy or fault-tolerance capability. The goal is to increase reliability and availability of the systems.
Some examples of single points of failure include:
Disk. If a server uses a single drive, the system will crash if the single drive fails. Redundant array of inexpensive disks (RAID) provides fault tolerance for hard drives and is a relatively inexpensive method of adding fault tolerance to a system.
Server. If a server provides a critical service and its failure halts the service, it is a single point of failure. Failover clusters (discussed later in this chapter) provide fault tolerance for critical servers.
Power. If an organization only has one source of power for critical systems, the power is a single point of failure. However, elements such as uninterruptible power supplies (UPSs) and power generators provide fault tolerance for power outages.
Although IT personnel recognize the risks with single points of failure, they often overlook them until a disaster occurs. However, tools such as business continuity plans (covered later in this chapter) help an organization identify critical services and address single points of failure.
￼
Disk Redundancies
Any system has four primary resources: processor, memory, disk, and the network interface. Of these, the disk is the slowest and most susceptible to failure. Because of this, administrators often upgrade disk subsystems to improve their performance and redundancy.
Redundant array of inexpensive disks (RAID) subsystems provide fault tolerance for disks and increase the system availability. Even if a disk fails, most RAID subsystems can tolerate the failure and the system will
continue to operate. RAID systems are becoming much more affordable as the price of drives steadily falls and disk capacity steadily increases. While i t’s expected that you are familiar with RAID subsystems, the following sections provide a short summary to remind you of the important details.
RAID-0
RAID-0 (striping) is somewhat of a misnomer because it doesn’t provide any redundancy or fault tolerance. It includes two or more physical disks. Files stored on a RAID-0 array are spread across each of the disks.
The benefit of a RAID-0 is increased read and write performance. Because a file is spread across multiple physical disks, the different parts of the file can be read from or written to each of the disks at the same time. If you have three 500 GB drives used in a RAID-0, you have 1,500 GB (1.5 TB) of storage space.



RAID-1
RAID-1 (mirroring) uses two disks. Data written to one disk is also written to the other disk. If one of the disks fails, the other disk still has all the data, so the system can continue to operate without any data loss. With this in mind, if you mirror all the drives in a system, you can actually lose half of the drives and continue to operate.
You can add an additional disk controller to a RAID-1 configuration to remove the disk controller as a single point of failure. In other words, each of the disks also has its own disk controller. Adding a second disk controller to a mirror is called disk duplexing.
If you have two 500 GB drives used in a RAID-1, you have 500 GB of storage space. The other 500 GB of storage space is dedicated to the fault- tolerant, mirrored volume.
RAID-2, RAID 3, and RAID-4 are rarely used.
RAID-5 and RAID-6
A RAID-5 is three or more disks that are striped together similar to RAID-0. However, the equivalent of one drive includes parity information. This parity information is striped across each of the drives in a RAID-5 and is used for fault tolerance. If one of the drives fails, the system can read the information on the remaining drives and determine what the actual data should be. If two of the drives fail in a RAID-5, the data is lost.
￼
RAID-6 is an extension of RAID-5, and it includes an additional parity block. A huge benefit is that the RAID-6 disk subsystem will continue to operate even if two disk drives fail. RAID-6 requires a minimum of four disks.
RAID-10
A RAID-10 configuration combines the features of mirroring (RAID-1) and striping (RAID-0). RAID-10 is sometimes called RAID 1+0. A variation is
RAID-01 or RAID 0+1 that also combines the features of mirroring and striping but implements the drives a little differently.
The minimum number of drives in a RAID-10 is four. When adding more drives, you add two (or multiples of two such as four, six, and so on). If you have four 500 GB drives used in a RAID-10, you have 1 TB of usable storage.
Server Redundancy and High Availability
High availability refers to a system or service that needs to remain operational with almost zero downtime. Utilizing different redundancy and fault-tolerance methods, it’s possible to achieve 99.999 percent uptime, commonly called five nines. This equates to less than 6 minutes of downtime a year: 60 minutes × 24 hours × 365 days ×.00001 = 5.256 minutes. Failover clusters are a key component used to achieve five nines.
Although five nines is achievable, it’s expensive. However, if the potential cost of an outage is high, the high cost of the redundant technologies is justified. example, some web sites



generate a significant amount of revenue, and every minute a web site is unavailable represents lost money. High-capacity failover clusters ensure the service is always available even if a server fails.
Distributive allocation is another option to provide both high availability and scalability, though it is typically used primarily in scientific applications. In a distributed application model, multiple computers (often called nodes) are configured to work together to solve complex problems. These computers are configured within a local network. A central processor divides the complex problem into smaller tasks. It then coordinates tasking of the individual nodes and collecting the results. If any single nodes fail, the central processor doesn’t task it anymore, but overall processing continues, providing high availability. This also provides high scalability because it is relatively easy to add additional nodes and task them when they come online.
Failover Clusters for High Availability
The primary purpose of a failover cluster is to provide high availability for a service offered by a server. Failover clusters use two or more servers in a cluster configuration, and the servers are referred to as nodes. At least one server or node is active and at least one is inactive. If an active node fails, the inactive node can take over the load without interruption to clients.
￼Consider Figure 9.1, which shows a two-node active-passive failover cluster. Both nodes are individual
servers, and they both have access to external data storage used by the active server. Additionally, the two nodes have a monitoring connection to each other used to check the health or heartbeat of each other.
Figure 9.1: Failover cluster
Imagine that Node 1 is the active node. When any of the clients connect, the cluster software (installed on both nodes) ensures that the clients connect to the active node. If Node 1 fails, Node 2 senses the failure through the heartbeat connection and configures itself as the active node. Because both nodes have access to the shared storage, there is no loss of data for the client. Clients may notice a momentary hiccup or pause, but the service continues.
You might notice that the shared storage in Figure 9.1 represents a single point of failure. It’s not uncommon for this to be a robust hardware RAID-10. This ensures that even if a hard drive in the shared storage fails, the service will continue. Additionally, if both nodes are plugged into the same power grid, the power represents a single point of failure. They can each be protected with a separate UPS, and use a separate power grid.



It’s also possible to configure the cluster as an active-active cluster. Instead of one server being passive, the cluster balances the load between both servers.
Cluster configurations can include many more nodes than just two. However, nodes need to have close to identical hardware and are often quite expensive, but if a company truly needs to achieve 99.999 percent uptime, it’s worth the expense.
Load Balancers for High Availability
A load balancer can optimize and distribute data loads across multiple computers or multiple networks. example, if an organization hosts a popular web site, it can use multiple servers hosting the same web site in a web farm. Load-balancing software distributes traffic equally among all the servers in the web farm, typically located in a DMZ.
The term load balancer makes it sound like it’s a piece of hardware, but a load balancer can be hardware or software. A hardware-based load balancer accepts traffic and directs it to servers based on factors such as processor utilization and the number of current connections to the server. A software- based load balancer uses software running on each of the servers in the load- balanced cluster to balance the load.
Load balancing primarily provides scalability, but it also contributes to high availability. Scalability refers to the ability of a service to serve more clients without any decrease in performance. Availability ensures that systems are up and operational when needed. By spreading the load among multiple systems, it ensures that individual systems are not overloaded, increasing overall availability.
Consider a web server that can serve 100 clients per minute, but if more than 100 clients connect at a time, performance degrades. You need to either scale up or scale out to serve more clients. You scale the server up by adding additional resources, such as processors and memory, and you scale out by adding additional servers in a load balancer.














￼
Figure 9.2 shows an example of a load balancer with multiple web servers configured in a web farm. Each web server includes the same web application. A load balancer uses a scheduling technique to determine where to send new requests. Some load balancers simply send new requests to the servers in a round-robin fashion. The load balancer sends the first request to Server 1, the second request to Server 2, and so on. Other load balancers automatically detect the load on individual servers and send new clients to the least used server.
Figure 9.2: Load balancing



Some load balancers use source address affinity to direct the requests. Source affinity sends requests to the same server based on the requestor’s IP address. example, imagine that Homer sends a request to retrieve a web page. The load balancer records his IP address and sends his request to Server 3. When he sends another request, the load balancer identifies his IP address and sends his request to Server 3 again. Source affinity effectively sticks users to a specific server for the duration of their sessions.
A software-based load balancer uses a virtual IP. example, imagine the IP address of the web site is 72.52.206.134. This IP address isn’t assigned to a specific server. Instead, clients send requests to this IP address and the load-balancing software redirects the request to one of the three servers in the web farm using their private IP addresses. In this scenario, the actual IP address is referred to as a virtual IP.
An added benefit of many load balancers is that they can detect when a server fails. If a server stops responding, the load-balancing software no longer sends clients to this server. This contributes to overall high availability for the load balancer.
￼
Clustering Versus Load Balancing
It’s worth mentioning that CompTIA has grouped both clustering and load balancing into the same category of load balancing in the objectives. Many IT professionals do the same thing, though technically they are different concepts. In general, failover clusters are commonly used for applications such as database applications. Load balancers are often used for services, such as web servers in a web farm.
Power Redundancies
Power is a critical utility to consider when reviewing redundancies. For mission-critical systems, you can use uninterruptible power supplies and generators to provide both fault tolerance and high availability. An UPS provides fault tolerance for power and can protect against power fluctuations. It provides short-term power. Generators provide long-term power in extended outages.
Protecting Data with Backups
Backups are copies of data created to ensure that if the original data is lost or corrupted, it can be restored. Maybe I should restate that. Backups are copies of data created to ensure that when the original data is lost or corrupted, it can be restored. The truth is, if you work with computers long enough, you will lose data. The difference between a major catastrophe and a minor inconvenience is the existence of a usable backup.




￼

It’s important to realize that redundancy and backups are not the same thing. Protecting data with a RAID-1 or RAID-10 does not negate the need for backups. If a fire destroys a server, it also destroys the data on the RAID. Without a backup, all of the data is gone. Forever.
Comparing Backup Types
Backup utilities support several different types of backups. Even though third-party backup programs can be quite sophisticated in what they do and how they do it, you should have a solid understanding of the basics.
The most common media used for backups is tape. Tapes store more data and are cheaper than other media, though some organizations use hard drives for backups. However, the type of media doesn’t affect the backup type.
The following backup types are commonly used:
Full backup. A full (or normal backup) backs up all the selected data.
Differential backup. This backs up all the data that has changed or is different since the last full backup.
Incremental backup. This backs up all the data that has changed since the last full or incremental backup.
Snaphots. A snapshot backup captures the data at a point in time. It is sometimes referred to as an image backup.
Full Backups
A full backup backs up all data specified in the backup. example, you could have several folders on the D: drive. If you specify these folders in the backup program, the backup program backs up all the data in these folders.



Although it’s possible to do a full backup on a daily basis, it’s rare to do so in most production environments. This is because of two limiting factors:
Time. A full backup can take several hours to complete and can interfere with operations. However, administrators don’t always have unlimited time to do backups and other system maintenance. example, if a system is online 24/7, administrators might need to limit the amount of time for full backups to early Sunday morning to minimize the impact on users.

- Money. Backups need to be stored on some type of media, such as tape or hard drives. Performing full backups every day requires more media, and the cost can be prohibitive. Instead, organizations often combine full
backups with differential or incremental backups.
However, every backup strategy must start with a full backup.
Restoring a Full Backup
A full backup is the easiest and quickest to restore. You only need to restore the single full backup and you’re done. If you store backups on tapes, you only need to restore a single tape. However, most organizations need to balance time and money and use either a full/differential or a full/incremental backup strategy.
Differential Backups
A differential backup strategy starts with a full backup. After the full backup, differential backups back up data that has changed or is different since the last full backup.
example, a full/differential strategy could start with a full backup on Sunday night. On Monday night, a differential backup would back up all files that changed since the last full backup on Sunday. On Tuesday night, the differential backup would again back up all the files that changed since the last full backup. This repeats until Sunday, when another full backup starts the process again. As the week progresses, the differential backup steadily grows in size.
Order of Restoration for a Full/Differential
Backup Set
Assume for a moment that each of the backups was stored on different tapes. If the system crashed on Wednesday morning, how many tapes would you need to recover the data?
The answer is two. You would first recover the full backup from Sunday. Because the differential backup on Tuesday night includes all the files that changed after the last full backup, you would restore that tape to restore all the changes up to Tuesday night.
Incremental Backups
An incremental backup strategy also starts with a full backup. After the full backup, incremental backups then back up data that has changed since the last backup. This includes either the last full backup, or the last incremental backup.
example, a full/incremental strategy could start with a full backup on Sunday night. On Monday night, an incremental backup would back up all the files that changed since the last full backup. On Tuesday night, the incremental backup would back up all the files that changed since the incremental backup on Monday night. Similarly, the Wednesday night backup would back up all files that changed since the last incremental backup on Tuesday night. This repeats until Sunday when another full backup starts the process again. As the week progresses, the incremental backups stay about the same size.



Order of Restoration for a Full/Incremental Backup Set
Assume for a moment that each of the backups were stored on different tapes. If the system crashed on Thursday morning, how many tapes would you need to recover the data?
The answer is four. You would first need to recover the full backup from Sunday. Because the incremental backups would be backing up different data each day of the week, each of the incremental backups must be restored—and must be restored in chronological order.
Sometimes, people mistakenly think the last incremental backup would have all the relevant data. Although it might have some relevant data, it doesn’t have everything.
example, imagine you worked on a single project file each day of the week, and the system crashed on Thursday morning. In this scenario, the last incremental backup would hold the most recent copy of this file. However, what if you compiled a report every Monday but didn’t touch it again until the following Monday? Only the incremental backup from Monday would include the most recent copy. An incremental backup from Wednesday night or another day of the week wouldn’t include the report.
Choosing Full/Incremental or Full/Differential
A logical question is, “Why are there so many choices for backups?” The answer is that different organizations have different needs. example, imagine two organizations perform daily backups to minimize losses. They each do a full backup on Sunday, but are now trying to determine if they should use a full/incremental or a full/differential strategy.
The first organization doesn’t have much time to perform maintenance throughout the week. In this case, the backup administrator needs to minimize the amount of time required to complete backups during the week. An incremental backup only backs up the data that has changed since the last backup. In other words, it includes changes only from a single day. In contrast, a differential backup includes all the changes since the last full
backup. Backing up the changes from a single day takes less time than backing up changes from multiple days, so a full/ incremental backup is the best choice.
In the second organization, recovery of failed systems is more important. If a failure requires restoring data, they want to minimize the amount of time needed to restore the data. A full/ differential is the best choice in this situation because it only requires the restoration of two backups, the full and the most recent differential backup. In contrast, a full/incremental can require the restoration of several different backups, depending on when the failure occurs.






Snapshot Backup
A snapshot backup captures the data at a moment in time. It is commonly used with virtual machines and sometimes referred to as a checkpoint. Chapter 1, “Mastering Security Basics,” discusses virtual machines (VMs) and administrators often take a snapshot of a VM before a risky operation such as an update. If the update causes problems, it’s relatively easy to revert the VM to the state it was in before the update.


Testing Backups
I’ve heard many horror stories in which personnel are regularly performing backups thinking all is well. Ultimately, something happens and they need to restore some data. Unfortunately, they discover that none of the backups hold valid data. People have been going through the motions, but something in the process is flawed.
The only way to validate a backup is to perform a test restore. Performing a test restore is nothing more than restoring the data from a backup and verifying its integrity. If you want to verify that you can restore the entire backup, you perform a full restore of the backup. If you want to verify that you can restore individual files, you perform a test restore of individual files. It’s common to restore data to a different location other than the original source location, but in such a way that you can validate the data.
As a simple example, an administrator can retrieve a random backup and attempt to restore it. There are two possible outcomes of this test, and both are good:
The test succeeds. Excellent! You know that the backup process works. You don’t necessarily know that every backup tape is valid, but at least you know that the process is sound and at least some of your backups work.
The test fails. Excellent! You know there’s a problem that you can fix before a crisis. If you discovered the problem after you actually lost data, it wouldn’t help you restore the data.
An additional benefit of performing regular test restores is that it allows administrators to become familiar with the process. The first time they do a restore shouldn’t be in the middle of a crisis with several high-level managers peering over their shoulders.
Protecting Backups
If data is important enough to be backed up, it’s important enough to protect. Backup media should be protected at the same level as the data that it holds. In other words, if proprietary data enjoys the highest level of protection within an organization, then backups of this data should also have the highest level of protection.
Protecting backups includes:
Storage. This includes using clear labeling to identify the data and physical security protection to prevent others from easily accessing it while it’s stored.
Transfer. Data should be protected any time it is transferred from one location to another. This is especially true when transferring a copy of the backup to a separate geographical location.
Destruction. When the backups are no longer needed, they should be destroyed. This can be accomplished by degaussing the media, shredding or burning the media, or scrubbing the media by repeatedly writing varying patterns of 1s and 0s onto the media.
Backups and Geographic Considerations
Organizations typically create a backup policy to answer critical questions related to backups. The backup policy is a written document and will often identify issues such as what data to back up, how often to back up the data, how to test the backups, and how long to retain the backups.
Additionally, it’s important to address special geographic considerations, such as the following:



Off-site backups. A copy of a backup should be stored in a separate geographic location. This protects against a disaster such as a fire or flood. Even if a disaster destroys the site, the organization will still have another copy of the critical data.
Distance. Many organizations have specific requirements related to the distance between the main site and the off-site location. In some scenarios, the goal is to have the off-site location relatively close so that backups can be easily retrieved. However, in other scenarios, the off- site location must be far away, such as 25 miles or further away.
Location selection. The location is often dependent on environmental issues. example, consider an organization located in California near the San Andreas fault. The off-site backup location should be far enough away that an earthquake at the primary location doesn’t affect the off-site location.
Legal implications. The legal implications related to backups depends on the data stored in the backups. example, if the backups include Personally Identifiable Information (PII) or Protected Health Information (PHI), the backups need to be protected according to governing laws.
Data sovereignty. Data sovereignty refers to the legal implications when data is stored off-site. If the backups are stored in a different country, they are subject to the laws of that country. This can be a concern if the backups are stored in a cloud location, and the cloud servers are in a different country. example, imagine that an organization is located in the United States. It routinely does backups and stores them with a cloud provider. The cloud provider has some servers in the United States, some in Canada, and some in Mexico. If the organization’s backups are stored in the other countries, it can be subject to additional laws and regulations.
￼
Comparing Business Continuity Elements
Business continuity planning helps an organization predict and plan for potential outages of critical services or functions. The goal is to ensure that critical business operations continue and the organization can survive the outage. Organizations often create a business continuity plan (BCP). This plan includes disaster recovery elements that provide the steps used to return critical functions to operation after an outage.
Disasters and outages can come from many sources, including:
Fires
Attacks
Power outages
Data loss from any cause
Hardware and software failures
Natural disasters, such as hurricanes, floods, tornadoes, and earthquakes



Addressing all of these possible sources takes a lot of time and effort. The goal is to predict the relevant disasters, their impact, and then develop recovery strategies to mitigate them. One of the first things an organization completes is a business impact analysis.
Business Impact Analysis Concepts
A business impact analysis (BIA) is an important part of a BCP. It helps an organization identify critical systems and components that are essential to the organization’s success. These critical systems support mission-essential functions. The BIA also helps identify vulnerable business processes. These are processes that support mission-essential functions.
example, imagine an organization has an online e-commerce business. Some basic mission-essential functions might include serving web pages, providing a shopping cart path, accepting purchases, sending email confirmations, and shipping purchases to customers. The shopping cart path alone is a business process and because it is essential to the mission of e- commerce sales, management will likely consider it a vulnerable business process to protect. The customer needs to be able to view products, select a product, enter customer information, enter credit card data, and complete the purchase. Some critical systems that support the web site are web servers and a back-end database application hosted on one or more database servers.
If critical systems and components fail and cannot be restored quickly, mission-essential functions cannot be completed. If this lasts too long, it’s very possible that the organization will not survive the disaster.
example, if a disaster such as a hurricane hit, which services must the organization restore to stay in business? Imagine a financial institution. It might decide that customers must have uninterrupted access to account data through an online site. If customers can’t access their funds online, they might lose faith with the company and leave in droves.
However, the company might decide to implement alternate business practices in other elements of the business. example, management might decide that accepting and processing loan applications is not important enough to continue during a disaster. Loan processing is still important to the company’s bottom line, but a delay will not seriously affect its ability to stay
in business. In this scenario, continuous online access is a mission-essential function, but processing loan applications during a disaster is not mission- essential.
The time to make these decisions is not during a crisis. Instead, the organization completes a BIA in advance. The BIA involves collecting information from throughout the organization and documenting the results. This documentation identifies core business or mission requirements. The BIA does not recommend solutions. However, it provides management with valuable information so that they can focus on critical business functions. It helps them address some of the following questions:
What are the critical systems and functions?
Are there any dependencies related to these critical systems and functions?
What is the maximum downtime limit of these critical systems and functions?
What scenarios are most likely to impact these critical systems and functions?
What is the potential loss from these scenarios?
example, imagine an organization earns an average of $5,000 an hour through online sales. In this scenario, management might consider online sales to be a mission-essential function and all systems that support online sales are critical systems. This includes web servers



and back-end database servers. These servers depend on the network infrastructure connecting them, Internet access, and access to payment gateways for credit card charges.
After analysis, they might determine that the maximum allowable outage for online sales is five hours. Identifying the maximum downtime limit is extremely important. It drives decisions related to recovery objectives and helps an organization identify various contingency plans and policies.
Impact
The BIA evaluates various scenarios, such as fires, attacks, power outages, data loss, hardware and software failures, and natural disasters. Additionally, the BIA attempts to identify the impact from these scenarios.
When evaluating the impact, a BIA looks at multiple items. example, it might attempt to answer the following questions related to any of the scenarios:
Will a disaster result in loss of life? Is there a way to minimize the risk to personnel?
Will a disaster result in loss of property?
Will a disaster reduce safety for personnel or property?
What are the potential financial losses to the organization?
What are the potential losses to the organization’s reputation?
example, a database server might host customer data, including credit card information. If an attacker was able to access this customer data, the cost to the organization might exceed millions of dollars.
You might remember the attack on retail giant Target during November and December 2013. Attackers accessed customer data on more than 110 million customers, resulting in significant losses for Target. Estimates of the total cost of the incident have ranged from $600 million to over $1 billion. This includes loss of sales—Target suffered a 46 percent drop in profits during the last quarter of 2013, compared with the previous year. Customers were afraid to use their credit cards at Target and simply stayed away. It also includes the cost to repair their image, the cost of purchasing credit monitoring for affected customers, fines from the payment-card industry, and an untold number of lawsuits. Target reportedly has $100 million in cyber insurance that helped them pay claims related to the data breach.
￼
Privacy Impact and Threshold Assessments
Two tools that organizations can use when completing a BIA are a privacy threshold assessment and a privacy impact assessment. National Institute of Standards and Technology (NIST) Special Publication (SP) 800- 122, “Guide to Protecting the Confidentiality of Personally Identifiable Information (PII),” covers these in more depth, but refers to a privacy threshold assessment as a privacy threshold analysis.



The primary purpose of the privacy threshold assessment is to help the organization identify PII within a system. Typically, the threshold assessment is completed by the system owner or data owner by answering a simple questionnaire.
If the system holds PII, then the next step is to conduct a privacy impact assessment. The impact assessment attempts to identify potential risks related to the PII by reviewing how the information is handled. The goal is to ensure that the system is complying with applicable laws, regulations, and guidelines. The impact assessment provides a proactive method of addressing potential risks related to PII throughout the life cycle of a computing system.
￼
Recovery Time Objective
The recovery time objective (RTO) identifies the maximum amount of time it can take to restore a system after an outage. Many BIAs identify the maximum acceptable outage or maximum tolerable outage time for mission- essential functions and critical systems. If an outage lasts longer than this maximum time, the impact is unacceptable to the organization.
example, imagine an organization that sells products via a web site generates $10,000 in revenue an hour. It might decide that the maximum acceptable outage for the web server is five minutes. This results in an RTO of five minutes, indicating any outage must be limited to less than five minutes. This RTO of five minutes only applies to the mission-essential function of online sales and the critical systems supporting it.
Imagine that the organization has a database server only used by internal employees, not online sales. Although the database server may be valuable, it is not critical. Management might decide they can accept an outage for as long as 24 hours, resulting in an RTO of less than 24 hours.
Recovery Point Objective
A recovery point objective (RPO) identifies a point in time where data loss is acceptable. example, a server may host archived data that has very few changes on a weekly basis. Management might decide that some data loss is acceptable, but they always want to be able to recover data from at least the previous week. In this case, the RPO is one week.
With an RPO of one week, administrators would ensure that they have at least weekly backups. In the event of a failure, they will be able to restore
recent backups and meet the RPO. In some cases, the RPO is up to the minute of the failure. example,
any data loss from an online database recording customer transactions might be unacceptable. In this case, the organization can use a variety of techniques to ensure administrators can restore data up to the moment of failure.
Comparing MTBF and MTTR
When working with a BIA, experts often attempt to predict the possibility of a
failure.



￼
example, what is the likelihood that a hard disk within a RAID configuration will fail? The following two terms are often used to predict potential failures:
Mean time between failures (MTBF). The mean time between failures (MTBF) provides a measure of a system’s reliability and is usually represented in hours. More specifically, the MTBF identifies the average (the arithmetic mean) time between failures. Higher MTBF numbers indicate a higher reliability of a product or system. Administrators and security experts attempt to identify the MTBF for critical systems with a goal of predicting potential outages.
Mean time to recover (MTTR). The mean time to recover (MTTR) identifies the average (the arithmetic mean) time it takes to restore a failed system. In some cases, people interpret MTTR as the mean time to repair, and both mean essentially the same thing. Organizations that have maintenance contracts often specify the MTTR as a part of the contract. The supplier agrees that it will, on average, restore a failed system within the MTTR time. The MTTR does not provide a guarantee that it will restore the system within the MTTR every time. Sometimes, it might take a little longer and sometimes it might be a little quicker, with the average defined by the MTTR.
Continuity of Operations Planning
Continuity of operations planning focuses on restoring mission- essential functions at a recovery site after a critical outage. example, if a hurricane or other disaster prevents the company from operating in the primary location, the organization can continue to operate the mission- essential functions at an alternate location that management previously identified as a recovery site. Failover is the process of moving mission-
essential functions to the alternate site.
Recovery Sites
A recovery site is an alternate processing site that an organization can use after a disaster. The three primary types of recovery sites are hot sites, cold sites, and warm sites. These alternate locations could be office space within a building, an entire building, or even a group of buildings. Two other types of recovery sites are mobile sites and mirrored sites. The following sections provide more details on these sites.
Hot Site
A hot site would be up and operational 24 hours a day, seven days a week and would be able to take over functionality from the primary site quickly after a primary site failure. It would include all the equipment, software, and communication capabilities of the primary site, and all the data would be up to date. In many cases, copies of backup tapes are stored at the hot site as the off-site location.
In many cases, a hot site is another active business location that has the capability to



assume operations during a disaster. example, a financial institution could have locations in two separate cities. The second location provides noncritical support services, but also includes all the resources necessary to assume the functions of the first location.
Some definitions of hot sites indicate they can take over instantaneously, though this isn’t consistent. In most cases, it takes a little bit of time to transfer operations to the hot site, and this can take anywhere from a few minutes to an hour.
￼
Clearly, a hot site is the most effective disaster recovery solution for high- availability requirements. If an organization must keep critical systems with high-availability requirements, the hot site is the best choice. However, a hot site is the most expensive to maintain and keep up to date.
Cold Site
A cold site requires power and connectivity but not much else. Generally, if it has a roof, electricity, running water, and Internet access, you’re good to go. The organization brings all the equipment, software, and data to the site when it activates it.
I often take my dogs for a walk at a local army base and occasionally see soldiers activate an extreme example of a cold site. On most weekends, the fields are empty. Other weekends, soldiers have transformed one or more fields into complete operational sites with tents, antennas, cables, generators, and porta-potties.
Because the army has several buildings on the base, they don’t need to operate in the middle of fields, but what they’re really doing is testing their ability to stand up a cold site wherever they want. If they can do it in the field, they can do it in the middle of a desert, or anywhere else they need to.
A cold site is the cheapest to maintain, but it is also the most difficult to test.
Warm Site
You can think of a warm site as the Goldilocks solution—not too hot and not too cold, but just right. Hot sites are generally too expensive for most organizations, and cold sites sometimes take too long to configure for full operation. However, the warm site provides a compromise that an organization can tailor to meet its needs.
example, an organization can place all the necessary hardware at the warm site location but not include up-to-date data. If a disaster occurs, the organization can copy the data to the warm site and take over operations. This is only one example, but there are many different possibilities of warm site configurations.
Site Variations
Although hot, cold, and warm sites are the most common, you might also come across two additional alternate site types: mobile and mirrored.



Amobilesiteisaself- containedtransportableunitwithalltheequipmentneededforspecific requirements. example, you can outfit a semitrailer with everything needed for operations, including a satellite dish for connectivity. Trucks, trains, or ships haul it to its destination and it only needs power to start operating.
￼
Mirrored sites are identical to the primary location and provide 100 percent availability. They use real-time transfers to send modifications from the primary location to the mirrored site. Although a hot site can be up and operational within an hour, the mirrored site is always up and operational.
Order of Restoration
After the disaster has passed, you will want to return all the functions to the primary site. As a best practice, organizations return the least critical functions to the primary site first. Remember, the critical functions are operational at the alternate site and can stay there as long as necessary. If a site has just gone through a disaster, it’s very likely that there are still some unknown problems. By moving the least critical functions first, undiscovered problems will appear
and can be resolved without significantly affecting mission-essential functions.
Disaster Recovery
Disaster recovery is a part of an overall business continuity plan. Often, the organization will use the business impact analysis to identify the critical systems and components and then develop disaster recovery strategies and disaster recovery plans (DRPs) to address the systems hosting these functions.
In some cases, an organization will have multiple DRPs within a BCP, and in other cases, the organization will have a single DRP. example, it’s
possible to have individual DRPs that identify the steps to recover individual critical servers and other DRPs that detail the recovery steps after different types of disasters such as hurricanes or tornadoes. A smaller organization might have a single DRP that simply identifies all the steps used to respond to any disruption.
A DRP or a BCP will include a hierarchical list of critical systems. This list identifies what systems to restore after a disaster and in what order. example, should a server hosting an online web site be restored first, or a server hosting an internal application? The answer is dependent on how the organization values and uses these servers. In some cases, systems have interdependencies requiring systems to be restored in a certain order.
If the DRP doesn’t prioritize the systems, individuals restoring the systems will use their own judgment, which might not meet the overall needs of the organization. example, Nicky New Guy might not realize that a web server is generating $5,000 an hour in revenue but does know that he’s responsible for keeping a generic file server operational. Without an ordered list of critical systems, he might spend his time restoring the file server and not the web server.
This hierarchical list is valuable when using alternate sites such as warm or
cold sites, too.



When the organization needs to move operations to an alternate site, the organization will want the most important systems and functions restored first.
Similarly, the DRP often prioritizes the services to restore after an outage.
As a rule, critical business functions and security services are restored first. Support services are restored last.
The different phases of a disaster recovery process typically include the following steps:
Activate the disaster recovery plan. Some disasters, such as earthquakes or tornadoes, occur without much warning, and a disaster recovery plan is activated after the disaster. Other disasters, such as hurricanes, provide a warning, and the plan is activated when the disaster is imminent.
Implement contingencies. If the recovery plan requires implementation of an alternate site, critical functions are moved to these sites. If the disaster destroyed on-site backups, this step retrieves the off- site backups from the off-site location.
Recover critical systems. After the disaster has passed, the organization begins recovering critical systems. The DRP documents which		systems	to	recover	and	includes detailedstepsonhowtorecoverthem.Thisalsoincludesreviewingchangema documentation to ensure that recovered systems include approved changes.
Test recovered systems. Before bringing systems online, administrators test and verify them. This may include comparing the restored system with a performance baseline to verify functionality.
After-action report. The final phase of disaster recovery includes a review of the disaster, sometimes called an after-action review. This often includes a lessons learned review to identify what went right and what went wrong. After reviewing the after-action report, the organization often updates the plan to incorporate any lessons learned.
￼
Testing Plans with Exercises
Business continuity plans and disaster recovery plans include testing. Testing validates that the plan works as desired and will often include testing redundancies and backups. There are several different types of testing used with BCPs and DRPs.
NIST SP 800-34, “Guide to Test, Training, and Exercise Programs for IT Plans and Capabilities,” provides detailed guidance on testing BCP and DRP plans. SP 800-34 identifies two primary types of exercises: tabletop exercises and functional exercises.
A tabletop exercise (also called a desktop exercise or a structured walk- through) is discussion-based. A coordinator gathers participants in a classroom or conference room, and leads them through one or more scenarios. As the coordinator introduces each stage of an incident, the participants identify what they’ll do based on the plan. This generates discussion about team members’ roles and responsibilities and the decision- making process during an incident. Ideally, this validates that the plan is valid. However, it sometimes reveals flaws. The BCP coordinator ensures the plans are rewritten if necessary.



Functional exercises provide personnel with an opportunity to test the plans in a simulated operational environment. There is a wide range of functional exercises, from simple simulations to full-blown tests. In a simulation, the participants go through the steps in a controlled manner without affecting the actual system. example, a simulation can start by indicating that a server failed. Participants then follow the steps to rebuild the server on a test system. A full- blown test goes through all the steps of the plan. In addition to verifying that the test works, this also shows the amount of time it will take to execute the plan.
Some of the common elements of testing include:
Backups. Backups are tested by restoring the data from the backup, as discussed in the “Testing Backups” section earlier in this chapter.
Server restoration. A simple disaster recovery exercise rebuilds a server. Participants follow the steps to rebuild a server using a test system without touching the live system.
Server redundancy. If a server is within a failover cluster, you can test the cluster by taking a primary node offline. Another node within the cluster should automatically assume the role of this offline node.
￼
Alternate sites. You can test an alternate site (hot, cold, or warm) by moving some of the functionality to the alternate site and ensuring the alternate site works as desired. It’s also possible to test individual elements of an alternate site, such as Internet connectivity, or the ability to obtain and restore backup media.
Chapter 9 Exam Topic Review
When preparing for the exam, make sure you understand these key concepts covered in this chapter.
Implementing Defense in Depth
Layered security (or defense in depth) employs multiple layers of
security to protect against threats. Personnel constantly monitor, update, add to, and improve existing security controls.
Control diversity is the use of different security control types, such as technical controls, administrative controls, and physical controls.
Vendor diversity is the practice of implementing security controls from different vendors to increase security.
Comparing Physical Security Controls
Physical security controls are controls you can physically touch. They often control entry and exit points, and include various types of locks.
An airgap is a physical security control that ensures that a computer or network is physically isolated from another computer or network.
Controlled areas such as data centers and server rooms should only have a single



entrance and exit point. Door lock types include cipher locks, proximity cards, and biometrics.
A proximity card can electronically unlock a door and helps prevent unauthorized personnel from entering a secure area. By themselves, proximity cards do not identify and authenticate users. Some systems combine proximity cards with PINs for identification and authentication.
Tailgating occurs when one user follows closely behind another user without using credentials. A mantrap can prevent tailgating.
Security guards are a preventive physical security control and they can prevent unauthorized personnel from entering a secure area. A benefit of guards is that they can recognize people and compare an individual’s picture ID for people they don’t recognize.
Cameras and closed-circuit television (CCTV) systems provide video surveillance. They provide reliable proof of a person’s identity and activity.
Fencing, lighting, and alarms are commonly implemented with motion detection systems for physical security. Infrared motion detection systems detect human activity based on the temperature.
Barricades provide stronger physical security than fences and attempt to deter attackers. Bollards are effective barricades that allow people through, but block vehicles.
Cable locks secure mobile computers such as laptop computers in a training lab. Server bays include locking cabinets or enclosures within a server room. Small devices can be stored in safes or locking office cabinets to prevent the theft of unused resources.
Asset management processes protect against vulnerabilities related to architecture and design weaknesses, system sprawl, and undocumented assets.
Heating, ventilation, and air conditioning (HVAC) systems control airflow for data centers and server rooms. Temperature controls protect systems from damage due to overheating.
Hot and cold aisles provide more efficient cooling of systems within a data center.
EMI shielding prevents problems from EMI sources such as
fluorescent lighting fixtures. It also prevents data loss in twisted-pair cables. A Faraday cage prevents signals from emanating beyond a room or enclosure.
Adding Redundancy and Fault Tolerance
A single point of failure is any component that can cause the entire system to fail if it fails.
RAID disk subsystems provide fault tolerance and increase availability. RAID-1 (mirroring) uses two disks. RAID-5 uses three or more disks and can survive the failure of one disk. RAID-6 and RAID-10 use four or more disks and can survive the failure of two disks.
Load balancers spread the processing load over multiple servers. In an active- active configuration, all servers are actively processing requests. In an active-passive configuration, at least one server is not active, but is instead monitoring activity ready to take over for a failed server. Software-based load balancers use a virtual IP.
Affinity scheduling sends client requests to the same server based on the client’s IP address. This is useful when clients need to access the same server for an entire online session. Round-robin scheduling sends requests to servers using a predefined order.



Protecting Data with Backups
Backup strategies include full, full/differential, full/incremental, and snapshot strategies. A full backup strategy alone allows the quickest recovery time.
Full/incremental backup strategies minimize the amount of time needed to perform daily backups.
Test restores verify the integrity of backups. A test restore of a full backup verifies a backup can be restored in its entirety.
Backups should be labeled to identify the contents. A copy of backups should be kept off-site.
It’s important to consider the distance between the main site and the off-site location.
The data contained in the backups can have legal implications. If it includes Personally Identifiable Information (PII) or Protected Health Information (PHI), it must be protected according to governing laws.
The location of the data backups affects the data sovereignty. If backups are stored in a different country, the data on the backups is now subject to the laws and regulations of that country.
Comparing Business Continuity Elements
A business impact analysis (BIA) is part of a business continuity plan (BCP) and it identifies mission-essential functions, critical systems, and vulnerable business processes that are essential to the organization’s success.
The BIA identifies maximum downtimes for these systems and components. It considers various scenarios that can affect these systems and components, and the impact to life, property, safety, finance, and reputation from an incident.
A privacy threshold assessment identifies if a system processes data that exceeds the threshold for PII. If the system processes PII, a privacy impact assessment helps identify and reduce risks related to potential loss of the PII.
A recovery time objective (RTO) identifies the maximum amount of time it should take to restore a system after an outage. The recovery point objective (RPO) refers to the amount of data you can afford to
lose.
Mean time between failures (MTBF) identifies the average (the arithmetic mean) time between failures. The mean time to recover (MTTR) identifies the average (the arithmetic mean) time it takes to restore a failed system.
Continuity of operations planning identifies alternate processing sites and alternate business practices. Recovery sites provide alternate locations for business functions after a major disaster.
A hot site includes everything needed to be operational within 60 minutes. It is the most effective recovery solution and the most expensive. A cold site has power and connectivity requirements and little else. It is the least expensive to maintain. Warm sites are a compromise between hot sites and cold sites.
Periodic testing validates continuity of operations plans. Exercises validate the steps to restore individual systems, activate alternate sites, and document other actions within a plan. Tabletop exercises are discussion-based only. Functional exercises are hands-on exercises.



Online References
Do you know how to answer performance-based questions? Check out the online extras at https://gcgapremium.com/501-extras.

---

## Chapter 10 Exam Topic Review


- When preparing for the exam, make sure you understand these key concepts covered in this chapter.
Introducing Cryptography Concepts


- Integrity provides assurances that data has not been modified. Hashing ensures that data has retained integrity.

- Confidentiality ensures that data is only viewable by authorized users. Encryption protects the confidentiality of data.

- Symmetric encryption uses the same key to encrypt and decrypt data.

- Asymmetric encryption uses two keys (public and private) created as a matched pair.

- A digital signature provides authentication, non-repudiation, and integrity.

- Authentication validates an identity.

- Non-repudiation prevents a party from denying an action.

- Users sign emails with a digital signature, which is a hash of an email message encrypted with the sender’s private key.

- Only the sender’s public key can decrypt the hash, providing verification it was encrypted with the sender’s private key.
Providing Integrity with Hashing

- Hashing verifies the integrity of data, such as downloaded files and email messages.

- A hash (sometimes listed as a checksum) is a fixed-size string of numbers or hexadecimal characters.

- Hashing algorithms are one-way functions used to create a hash. You cannot reverse the process to re-create the original data.

- Passwords are often stored as hashes instead of the actual password. Salting the password thwarts many password attacks.

- Two commonly used key stretching techniques are bcrypt and Password-Based Key Derivation Function 2 (PBKDF2). They protect passwords against brute force and rainbow table attacks.

- Common hashing algorithms are Message Digest 5 (MD5), Secure Hash Algorithm (SHA), and Hash-based Message Authentication Code (HMAC).

- HMAC provides both integrity and authenticity of a message.
Providing Confidentiality with Encryption

- Confidentiality ensures that data is only viewable by authorized users.

- Encryption provides confidentiality of data, including data-at-rest (any type of data stored on disk) or data-in-transit (any type of transmitted data).

- Block ciphers encrypt data in fixed-size blocks. Advanced Encryption Standard (AES) and Twofish encrypt data in 128-bit blocks.

- Stream ciphers encrypt data 1 bit or 1 byte at a time. They are more efficient than block ciphers when encrypting data of an unknown size or when sent in a continuous stream. RC4 is a commonly used stream cipher.

- Cipher modes include Electronic Codebook (ECB), Cipher Block Chaining (CBC), Counter (CTM) mode, and Galois/Counter Mode (GCM).

  - ECB should not be used.

  - GCM is widely used because it is efficient and provides data authenticity.

- Data Encryption Standard (DES), Triple DES (3DES), and Blowfish are block ciphers that encrypt data in 64-bit blocks. AES is a popular symmetric block encryption algorithm, and it uses 128, 192, or 256 bits for the key.

- Asymmetric encryption uses public and private keys as matched pairs.

- If the public key encrypted information, only the matching private key can decrypt it.

- If the private key encrypted information, only the matching public key can decrypt it.

- Private keys are always kept private and never shared.

- Public keys are freely shared by embedding them in a certificate.

- RSA is a popular asymmetric algorithm. Many cryptographic protocols use RSA to secure data such as email and data transmitted over the Internet. RSA uses prime numbers to generate public and private keys.

- Elliptic curve cryptography (ECC) is an encryption technology commonly used with small wireless devices.

- Diffie-Hellman provides a method to privately share a symmetric key between two parties. Elliptic Curve Diffie-Hellman Ephemeral (ECDHE) is a version of Diffie-Hellman that uses ECC to re-create keys for each session.

- Steganography is the practice of hiding data within a file. You can hide messages in the white space of a file without modifying its size. A more sophisticated method is by modifying bits within a file. Capturing and comparing hashes of files can discover steganography attempts.
Using Cryptographic Protocols

- When using digital signatures with email: The sender’s private key encrypts (or signs). The sender’s public key decrypts.

- A digital signature provides authentication (verified identification) of the sender, non- repudiation, and integrity of the message.

- Senders create a digital signature by hashing a message and encrypting the hash with the sender’s private key.

- Recipients decrypt the digital signature with the sender’s matching public key.

- When encrypting email: The recipient’s public key encrypts. The recipient’s private key decrypts.

- Many email applications use the public key to encrypt a symmetric key, and then use the symmetric key to encrypt the email contents.

- S/MIME and PGP secure email with encryption and digital signatures. They both use RSA, certificates, and depend on a PKI. They can encrypt email at rest (stored on a drive) and in transit (sent over the network).

- TLS is the replacement for SSL. SSL is deprecated and should not be used.

- When encrypting web site traffic with TLS: The web site’s public key encrypts a symmetric key. The web site’s private key decrypts the symmetric key. The symmetric key encrypts data in the session.

- Weak cipher suites (such as those supporting SSL) should be disabled to prevent downgrade attacks.
Exploring PKI Components

- A Public Key Infrastructure (PKI) is a group of technologies used to request, create, manage, store, distribute, and revoke digital certificates. A PKI allows two entities to privately share symmetric keys without any prior communication.

- Most public CAs use a hierarchical centralized CA trust model, with a root CA and intermediate CAs. A CA issues, manages, validates, and revokes certificates.

- Root certificates of trusted CAs are stored on computers. If a CA’s root certificate is not in the trusted store, web users will see errors indicating the certificate is not trusted or the CA is not recognized.

- You request a certificate with a certificate signing request (CSR). You first create a private/ public key pair and include the public key in the CSR.

- CAs revoke certificates when an employee leaves, the private key is compromised, or the CA is compromised. A CRL identifies revoked certificates as a list of serial numbers.

- The CA publishes the CRL, making it available to anyone. Web browsers can check certificates they receive from a web server against a copy of the CRL to determine if a received certificate is revoked.

- Public key pinning provides clients with a list of hashes for each public key it uses.

- Certificate stapling provides clients with a timestamped, digitally signed OCSP response. This is from the CA and appended to the certificate.

- User systems return errors when a system tries to use an expired certificate.

- A key escrow stores a copy of private keys used within a PKI. If the original private key is lost or inaccessible, the copy is retrieved from escrow, preventing data loss.

- Wildcard certificates use a * for child domains to reduce the administrative burden of managing certificates. Subject Alternative Name (SAN) certificates can be used for multiple domains with different domain names.

- A domain validated certificate indicates that the certificate requestor has some control over a DNS domain.

- Extended validation certificates use additional steps beyond domain validation to give users a visual indication that they are accessing the site.

- CER is a binary format for certificates and DER is an ASCII format.

- PEM is the most commonly used certificate format and can be used for just about any certificate type.

- P7B certificates are commonly used to share public keys.

- P12 and PFX certificates are commonly used to hold the private key.


--

## Chapter 11 Implementing Policies to Mitigate Risks
CompTIA Security+ objectives covered in this chapter:
2.2	Given a scenario, use appropriate software tools to assess the security posture of an organization.
Data sanitization tools
2.3	Given a scenario, troubleshoot common security issues.
Personnel issues (Policy violation, Personal email)
4.4	Given a scenario, differentiate common account management practices.
General Concepts (Onboarding/offboarding)
5.1	Explain the importance of policies, plans and procedures related to organizational security.
Standard operating procedure, Agreement types (BPA, SLA, ISA, MOU/MOA), Personnel management (Mandatory vacations, Job rotation, Separation of duties, Clean desk, Background checks, Exit interviews, Role-based awareness training [Data owner, System administrator, System owner, User, Privileged user, Executive user], NDA, Onboarding, Continuing education, Acceptable use policy/rules of behavior, Adverse actions), General security policies (Social media networks/applications, Personal email)

5.4	Given a scenario, follow incident response procedures.
Incident response plan (Documented incident types/category definitions, Roles and responsibilities, Reporting requirements/escalation, Cyber-incident response
teams, Exercise), Incident response process (Preparation, Identification, Containment, Eradication, Recovery, Lessons learned)
5.5	Summarize basic concepts of forensics.
Order of volatility, Chain of custody, Legal hold, Data acquisition (Capture system image, Network traffic and logs, Capture video, Record time offset, Take hashes, Screenshots, Witness interviews), Preservation, Recovery, Strategic intelligence/counterintelligence gathering (Active logging), Track man-hours



5.8	Given a scenario, carry out data security and privacy practices.
Data destruction and media sanitization (Burning, Shredding, Pulping, Pulverizing, Degaussing, Purging, Wiping), Data sensitivity labeling and handling (Confidential, Private, Public, Proprietary, PII, PHI), Data roles (Owner, Steward/custodian, Privacy officer), Data retention, Legal and compliance
**

Organizations often develop written security policies. These provide guiding principles to the professionals who implement security throughout the organization. These policies include personnel management policies and data protection policies. Combined with training for personnel to raise overall security awareness, they help mitigate risk and reduce security incidents. However, security incidents still occur, and incident response policies provide the direction on how to handle them.
Exploring Security Policies
Security policies are written documents that lay out a security plan within a company. They are one of many administrative controls used to reduce and manage risk. When created early enough, they help ensure that personnel consider and implement security throughout the life cycle of various systems in the company. When the policies and procedures are enforced, they help prevent incidents, data loss, and theft.
Policies include brief, high-level statements that identify goals based on an organization’s overall beliefs and principles. After creating the policy, personnel within the organization create plans and procedures to support the policies. Although the policies are often high-level statements, the plans and procedures provide details on policy implementation.
example, organizations often create standard operating procedures (SOPs) to support security policies. These typically include step- by-step instructions employees can use to perform common tasks or routine operations. Security controls such as those covered in Chapter 1, “Mastering Security Basics,” enforce the requirements of a security policy. example, a security policy may state that internal users must not use peer-to-peer (P2P)

applications. A firewall with appropriate rules to block these applications provides a technical implementation of this policy. Similarly, administrators can use port-scanning tools to detect applications running on internal systems that are violating the security policy.
A security policy can be a single large document or divided into several smaller documents, depending on the needs of the company. The following sections identify many of the common elements of a security policy.

Personnel Management Policies
Companies frequently develop policies to specifically define and clarify issues related to personnel management. This includes personnel behavior, expectations, and possible consequences. Personnel learn these policies when they are hired and as changes occur. Some of the policies directly related to personnel are acceptable use, mandatory vacations, separation of duties, job rotation, and clean desk policies. The following sections cover these and other personnel policies in more depth.





Acceptable Use Policy
An acceptable use policy (AUP) defines proper system usage or the rules of behavior for employees when using information technology (IT) systems. It often describes the purpose of computer systems and networks, how users can access them, and the responsibilities of users when they access the systems.
Many organizations monitor user activities, such as what web sites they visit and what data they send out via email. example, a proxy server typically logs all web sites that a user visits. The AUP may include statements informing users that systems are in place monitoring their activities.
In some cases, the AUP might include privacy statements informing users what computer activities they can consider private. Many users have an expectation of privacy when using an organization’s computer systems and networks that isn’t justified. The privacy policy statement helps to clarify the organization’s stance.
The AUP often includes definitions and examples of unacceptable use. example, it might prohibit employees from using company resources to access P2P sites or social media sites.
It ’s common for organizations to require users to read and sign a document indicating they understand the acceptable use policy when they’re hired and in conjunction with annual security training. Other methods, such as logon banners or periodic emails, help reinforce an acceptable use policy.
Mandatory	Vacations
Mandatory vacation policies help detect when employees are involved in malicious activity, such as fraud or embezzlement. example, employees in positions of fiscal trust, such as stock traders or bank employees, are often required to take an annual vacation of at least five consecutive workdays.

For embezzlement actions of any substantial size to succeed, an employee would need to be constantly present in order to manipulate records and respond to different inquiries. On the other hand, if an employee is forced to be absent for at least five consecutive workdays, someone else would be required to answer any queries during the employee’s absence. This increases the likelihood of discovering illegal activities by employees. It also acts as an effective deterrent.
Mandatory vacations aren’t limited to only financial institutions, though. Many organizations require similar policies for administrators. example, an administrator might be the only person required to perform sensitive activities such as reviewing certain logs. A malicious administrator can overlook or cover up certain activities revealed in the logs. However, a mandatory vacation policy would require someone else to perform these activities, which increases the chance of discovery.
Of course, mandatory vacations by themselves won’t prevent fraud. Most companies will implement the principle of defense in depth by using multiple layers of protection. Additional policies may include separation of duties and job rotation to provide as much protection as possible.






Separation of Duties
Separation of duties is a principle that prevents any single person or entity from being able to complete all the functions of a critical or sensitive process. It’s designed to prevent fraud, theft, and errors.
Accounting provides a classic example. It ’s common to divide Accounting departments into two divisions: Accounts Receivable and Accounts Payable. Personnel in the Accounts Receivable division review and validate bills. They then send the validated bills to the personnel in the Accounts Payable division, who pay the bills. Similarly, this policy would ensure personnel are not authorized to print and sign checks. Instead, a separation of duties policy separates these two functions to reduce the possibility of fraud.
If Homer were the only person doing all these functions, it would be possible for him to create and approve a bill from Homer’s Most Excellent Retirement Account. After approving the bill, Homer would then pay it. If Homer doesn’t go to jail, he may indeed retire early at the expense of the financial health of the company.
Separation of duties policies also apply to IT personnel. example, it ’s common to separate application development tasks from application deployment tasks. In other words, developers create and modify applications and then pass the compiled code to administrators. Administrators then deploy the code to live production systems. Without this policy in place, developers might be able make quick, untested changes to code, resulting in unintended outages. This provides a high level of version control and prevents potential issues created through uncontrolled changes.


As another example, a group of IT administrators may be assigned responsibility for maintaining a group of database servers. However, they would not be granted access to security logs on these servers. Instead, security administrators regularly review these logs, but these security administrators will not have access to data within the databases.
Imagine that Bart has been working as an IT administrator but recently changed jobs and is now working as a security administrator. What should happen? Based on separation of duties, Bart should now have access to the security logs, but his access to the data within the databases should be revoked. If his permissions to the data are not revoked, he will have access to more than he needs, violating the principle of least privilege. A user rights and permissions review often discovers these types of issues.




Job Rotation
Job rotation is a concept that has employees rotate through different jobs to learn the processes and procedures in each job. From a security perspective, job rotation helps to prevent or expose dangerous shortcuts or even fraudulent activity. Employees might rotate through jobs temporarily or permanently.
example, your company could have an Accounting department. As mentioned in the “Separation of Duties” section, you would separate accounting into two divisions—Accounts Receivable and Accounts Payable. Additionally, you could rotate personnel in and out of jobs in the two divisions. This would ensure more oversight over past transactions and help ensure that employees are following rules and policies.
In contrast, imagine a single person always performs the same function without any expectation of oversight. This increases the temptation to go outside the bounds of established policies.
Job rotation policies work well together with separation of duties policies. A separation of duties policy helps prevent a single person from controlling too much. However, if an organization only used a separation of duties policy, it is possible for two people to collude in a scheme to defraud the company. If a job rotation policy is also used, these two people will not be able to continue the fraudulent activity indefinitely.
Job rotation policies also apply to IT personnel. example, the policy can require administrators to swap roles on a regular basis, such as annually or quarterly. This prevents any single administrator from having too much control over a system or network.






Clean Desk Policy
A clean desk policy directs users to keep their areas organized and free of papers. The primary security goal is to reduce threats of security incidents by ensuring the protection of sensitive data. More specifically, it helps prevent the possibility of data theft or inadvertent disclosure of information.
Imagine an attacker goes into a bank and meets a loan officer. The loan officer has stacks of paper on his desk, including loan applications from various customers. If the loan officer steps out, the attacker can easily grab some of the documents, or simply take pictures of the documents with a mobile phone.
Beyond security, organizations want to present a positive image to customers and clients.
Employees with cluttered desks with piles of paper can easily turn off customers.
However, a clean desk policy doesn’t just apply to employees who meet and greet customers. It also applies to employees who don’t interact with customers. Just as dumpster divers can sort through trash to gain valuable information, anyone can sort through papers on a desk to learn information. It’s best to secure all papers to keep them away from prying eyes. Some items left on a desk that can present risks include:
Keys
Cell phones




I’ll Go to Jail Before I Give You the Passwords!
The city of San Francisco had an extreme example of the dangers of a single person with too much explicit knowledge or power. A network administrator with one of Cisco’s highest certifications—Cisco Certified Internetwork Expert (CCIE)—made changes to t h e city’s network, changing passwords so that only he knew them and ensuring that he was the only person with administrative access.
It could be that he was taking these actions to protect the network that he considered his “baby.” He was the only CCIE, and it’s possible he thought others did not have the necessary knowledge to maintain the network adequately. Over the years, fewer and fewer people had access to what he was doing, and his knowledge became more and more proprietary. Instead of being malicious in nature, he might have simply been protective, even if overly protective.
At some point, his supervisor recognized that all the proverbial information eggs were in the basket of this lone CCIE. It was just too risky. What if a bus, or one of San Francisco’s famous trolleys, hit him? What would the organization do? His supervisor asked him for some passwords and he refused, even when faced with arrest. Later, he gave law enforcement personnel passwords that didn’t work.
Law enforcement personnel charged him with four counts of tampering with a computer network and courts kept him in custody with a $5 million bail. Ultimately, a court convicted him of one felony count and sentenced him to four years in prison. This is a far fall from his reported annual salary of $127,735.
The city of San Francisco had to bring in experts from Cisco and the city
reported costs of
$900,000 to regain control of their network. Following his conviction, the court also ordered the administrator to pay $1.5 million in restitution.
What’s the lesson here? Internal security controls, such as creating and enforcing policies related to rotation of duties, separation of duties, and cross-training, might h a v e been able to avoid this situation completely. If this CCIE truly did have good intentions toward what he perceived as his network, these internal controls might have prevented

him from going over the line into overprotection and looking at the world through the bars of a jail cell.

Access cards
Sensitive papers
Logged-on computer
Printouts left in printer
Passwords on Post-it notes
File cabinets left open or unlocked
Personal items such as mail with Personally Identifiable Information (PII)

Some people want to take a clean desk policy a step further by scrubbing and sanitizing desks with antibacterial cleaners and disinfectants on a daily basis. They are free to do so, but that isn’t part of a security-related clean desk policy.




Background Check
It’s common for organizations to perform background checks on potential employees and even after employees are hired. A background check checks into a potential employee’s history with the intention of discovering anything about the person that might make him a less-than- ideal fit for a job.
A background check will vary depending on job responsibilities and the sensitivity of data that person can access. example, a background check for an associate at Walmart will be significantly less than a background check for a government employee who will handle T o p Secret Sensitive Compartmented Information.
However, background checks will typically include a query to law enforcement agencies to identify a person’s criminal history. In some cases, this is only to determine if the person is a felon. In other cases, it checks for all potential criminal activity, including a review of a person’s driving records.
Many organizations check a person’s financial history by obtaining a credit report. example, someone applying for a job in an Accounting department might not be a good fit if his credit score is 350 and he has a string of unpaid loans.
It is also common for employers to check a person’s online activity. This includes social media sites, such as Facebook, LinkedIn, and Twitter. Some people say and do things online that they would rarely do in public. One reason is a phenomenon known as the online disinhibition effect. Just as a beer or glass of wine releases inhibitions in many people, individuals are often less inhibited when posting comments online. And what they post often reflects their true feelings and beliefs. Consider a person who frequently posts hateful comments about others. A potential employer might think that this person is unlikely to work cohesively in a team environment and hire someone else.
Note that some background checks require the written permission from the potential employee. example, the Fair Credit Reporting Act (FCRA) requires organizations to obtain written permission before obtaining a credit report on a job applicant or employee. However, other background checks don’t require permission. example, anyone can look at an individual’s social media profile.

NDA
A non-disclosure agreement (NDA) is used between two entities to ensure that proprietary data is not disclosed to unauthorized entities. example, imagine BizzFad wants to collaborate with Costington’s on a project. BizzFad management realizes they need to share proprietary data with Costington’s personnel, but they want to ensure that distribution of the data is limited. The NDA is a legal document that BizzFad can use to hold Costington’s legally responsible if the proprietary data is shared.
Similarly, many organizations use an NDA to prohibit employees from sharing proprietary data either while they are employed, or after they leave the organization. It’s common to remind employees of an existing NDA during an exit interview.
Exit Interview
An exit interview is conducted with departing employees just before they leave an organization. Note that an exit interview isn’t only conducted when employees are fired from their job. They are also done when employees leave voluntarily. The overall purpose is for the employer to gain information from the departing employee. Some common questions asked



during an exit interview are:
What did you like most (and/or least) about your job here?
Do you think you had adequate training to do your job here?
Can you tell me what prompted you to leave your current position?
Can you describe the working relationship you had with your supervisor(s)?
What skills and qualification does your replacement need to excel in this position?
Exit interviews are commonly conducted by an employee in the Human Resources (HR) department. In addition to seeking feedback from the employee, departing employees are sometimes required to sign paperwork, such as a reminder about a previously signed NDA. The NDA prevents the employee from sharing proprietary information with personnel outside the organization.
From a security perspective, it’s also important to ensure other things occur during or before the exit interview. example, the user’s account should be disabled (or deleted depending on company policy). Ideally, this should occur during the interview. One way organizations do this is by informing the IT department of the time of the scheduled interview a day before. An administrator then disables the account after the interview starts. The key is that a departing employee should not have access to computing and network resources after the interview.
It’s also important to collect any equipment (such as smartphones, tablets, or laptops), security badges, or proximity cards the organization issued to the employee. This is more than just a cost issue. Equipment very likely has proprietary data on it and the company needs to take steps to protect the data. Additionally, smart cards and proximity cards can allow individuals access to protected areas.







Onboarding
Onboarding is the process of granting individuals access to an organization’s computing resources after being hired. This includes providing the employee with a user account and granting access to appropriate resources. One of the key considerations during the onboarding process is to follow the principle of least privilege. Grant the new employees access to what they need for their job, but no more.
Offboarding is the process of removing their access. When employees leave the company, it’s important to revoke their access. This is often done during the exit interview.
Policy Violations and Adverse Actions
What do you do if an employee doesn’t follow the security policy? What adverse actions should a supervisor take? Obviously, that depends on the severity of the policy violation.
Imagine that an employee sends out an email to everyone in the organization inviting them to his church. The supervisor might decide to verbally counsel the employee and make it clear




that sending out personal emails like this is unacceptable. Based on how well the conversation goes, the supervisor might choose to document this as written counseling and place the warning in the employee’s HR folder.
Some incidents require more severe responses. Imagine that an employee begins a cyberbullying campaign against another employee. He has been sending her hateful emails and posting hateful messages on social media pages. In most organizations, this bully will be looking for employment elsewhere once his activity is discovered.
Although it’s possible to document specific adverse actions within a security policy, this is rarely recommended. Actual policy violations aren’t always the same and if the policy requires a specific action in response to a policy violation, it doesn’t always allow supervisors or managers to respond appropriately to a violation.
Other General Security Policies
From a more general perspective, an organization may implement personnel management policies that affect other areas of an employee’s life. Some examples include behavior on social media networks and the use of email.
As a simple example, employees of a company should not post adverse comments about other employees or customers. Employees who engage in cyberbullying against fellow employees are typically fired. Similarly, employees who post derogatory comments about customers quickly find themselves looking for other employment.
You might think that people would know that what they post on the Internet can be seen by anyone, including their employer. However, if you do a quick Google search on “employee fired after Facebook post” or “employee fired after tweet,” you’ll find many examples where people ignored the possibility that their words would be seen by their employer.
Another consideration is personal email. Some organizations allow employees to use the organization’s IT infrastructure to send and receive personal email, while other organizations forbid it. The key here is ensuring that employees understand the policy.
Social Media Networks and Applications

Millions of people interact with each other using social media networks and applications, such as Facebook and Twitter. Facebook allows people to share their lives with friends, family, and others. Twitter allows people to tweet about events as they are happening. From a social perspective, these technologies allow people to share information about themselves with others. A user posts a comment and a wide group of people instantly see it.
However, from a security perspective, they present some significant risks, especially related to inadvertent information disclosure. Attackers can use these sites to gain information about individuals and then use that information in an attack. Organizations typically either train users about the risks or block access to the social media sites to avoid the risks.
Users often post personal information, such as birth dates, their favorite colors or books, the high school they graduated from, graduation dates, and much more. Some sites use this personal information to validate users when they forget or need to change their password. Imagine Maggie needs to reset her password for a bank account. The web site may challenge her to enter her birth date, favorite book, and graduation date for validation. This is also known as a cognitive password and, theoretically, only Maggie knows this information. However, if Maggie posts all this information on Facebook, an attacker can use it to change the password on the bank account.




example, David Kernell used Yahoo!’s cognitive password account recovery process to change former Alaska Governor Sarah Palin’s password for her email account. At the time, Yahoo! asked questions such as her high school and birth date and Kernell obtained all the information from online searches. Of course, it didn’t turn out well for him. A jury convicted him of a felony and he served more than a year in prison.
In some cases, attackers have used personal information from social networking sites to launch scams. example, attackers first identify the name of a friend or relative using the social networking site. The attackers then impersonate the friend or relative in an email, claiming to have been robbed and stuck in a foreign country. Attackers end the email with a plea for help asking the victim to send money via wire transfer.
It’s also worth considering physical security. While vacationing in Paris, Kim Kardashian West was regularly posting her status and location on social media. She also stressed that she didn’t wear fake jewelry. Thieves robbed her at gunpoint in her Paris hotel room. They bound and gagged her and took one of her rings (that is worth an estimated $4.9 million) and a jewelry box (with jewelry worth an estimated $5.6 million). After being caught and arrested, one of the thieves later admitted that it was relatively easy to track her just by watching her online activity.

Banner Ads and Malvertisements
Attackers have been delivering malware through malicious banner ads for several years now. These look like regular ads, but they contain malicious code. Many of these are Flash applets with malicious code embedded in them, but others just use code to redirect users to another server, such as one with a drive-by download waiting for anyone who clicks.
Although these malvertisements have been on many social media sites, they’ve also appeared on mainstream sites. example, attackers installed a

malvertisement on the New York Times web site where it ran for about 24 hours before webmasters discovered and disabled it.
Similarly, malvertising has appeared on the Yahoo! web site. Users who clicked on some Yahoo! ads were taken to sites hosting fake antivirus software. These sites included pop- ups indicating that users’ systems were infected with malware and encouraging the users to download and install it. Users who took the bait installed malware onto their systems. Some of these ads sent users to sites in Eastern Europe that were hosting CryptoWall, according to research by Blue Coat Systems, Inc. CryptoWall is a malicious form of ransomware that encrypts user files and demands payment to decrypt them.
Attackers have used two primary methods to get these malvertisements installed on legitimate web sites. One method is to attack a web site and insert ads onto that web site. The second method is to buy ads. They often represent an ad agency pretending to represent legitimate clients. example, one attacker convinced Gawker Media to run a series of Suzuki advertisements, which were actually malvertisements. Similarly, it’s unlikely that Yahoo! was




aware that it was hosting malvertising, but instead, these ads likely appeared as a result of attacks or by being tricked.
Social Networking and P2P
Peer-to-peer (P2P or file sharing) applications allow users to share files, such as music, video, and data, over the Internet. Instead of a single server providing the data to end users, all computers in the P2P network are peers, and any computer can act as a server to other clients.
The first widely used P2P network was Napster, an online music- sharing service that operated between 1999 and 2001. Users copied and distributed MP3 music files among each other, and these were often pirated music files. The files were stored on each user’s system, and as long as the system was accessible on the Internet, other users could access and download the files. A court order shut down Napster due to copyright issues, but it later reopened as an online music store. Other P2P software and P2P networks continue to appear and evolve.
Organizations usually restrict the use of P2P applications in networks, but this isn’t because of piracy issues. One reason is because the P2P applications can consume network bandwidth, slowing down other systems on the network. Worse, a significant risk with P2P applications is data leakage. Users are often unaware of what data they are sharing. Another risk is that users are often unaware of what data the application downloads and stores on their systems, causing them to host inappropriate data. Two examples help illustrate these data leakage risks.
Information concentrators search P2P networks for information of interest and collect it. Investigators once discovered an information concentrator in Iran with over 200 documents containing classified and secret
U.S. government data. This included classified information about Marine One, the helicopter used by the president. Although the information about Marine One made the headlines, the attackers had much more information. example, this concentrator included Iraq status reports and lists of soldiers with privacy data.
How did this happen? Investigations revealed that a defense contractor installed a P2P application on a computer. The computer had access to this data, and the P2P application shared it.

The media latched onto the news about Marine One, so this story was widely published. However, it’s widely believed that much more data is being mined via P2P networks. Most end users don’t have classified data on their systems, but they do have PII, such as banking information or tax data. When an attacker retrieves data on a user’s system and empties a bank account, it might be a catastrophe to the user, but it isn’t news.
Organizations can restrict access to P2P networks by blocking access in firewalls. Additionally, port scanners can scan open ports of remote systems to identify P2P software. Organizations often include these checks when running a port scanner as part of a vulnerability scan.





Agreement Types
Organizations often utilize different types of agreements to help identify various responsibilities. Many are used when working with other organizations, but they can often be used when working with different departments within the same organization. These include:




Interconnection security agreement (ISA). An ISA specifies technical and security requirements for planning, establishing, maintaining, and disconnecting a secure connection between two or more entities. For example, it may stipulate certain types of encryption for all data-in-transit. NIST SP 800-47, “Security Guide for Interconnecting Information Technology Systems,” includes more in-depth information on ISAs.
Service level agreement (SLA). An SLA is an agreement between a company and a vendor that stipulates performance expectations, such as minimum uptime and maximum downtime levels. Organizations use SLAs when contracting services from service providers such as Internet Service Providers (ISPs). Many SLAs include a monetary penalty if the vendor is unable to meet the agreed-upon expectations.
Memorandum of understanding (MOU) or memorandum of agreement (MOA). An MOU/MOA expresses an understanding between two or more parties indicating their intention to work together toward a common goal. An MOU/MOA is often used to support an ISA by defining the purpose of the ISA and the responsibilities of both parties. However, it doesn’t include any technical details. You can also compare an MOU/ MOA with an SLA because it defines the responsibilities of each of the parties. However, it is less formal than an SLA and does not include monetary penalties. Additionally, it doesn’t have strict guidelines in place to protect sensitive data.
•   Business partners agreement (BPA). A BPA is a written agreement that details the relationship between business partners, including their obligations toward the partnership. It typically identifies the shares of

profits or losses each partner will take, their responsibilities to each other, and what to do if a partner chooses to leave the partnership. One of the primary benefits of a BPA is that it can help settle conflicts when they arise.
Protecting Data
Every company has secrets. Keeping these secrets can often make the difference between success and failure. A company can have valuable research and development data, customer databases, proprietary information on products, and much more. If the company cannot keep private and proprietary data secret, it can directly affect its bottom line.
Data policies assist in the protection of data and help prevent data leakage. This section covers many of the different elements that may be contained in a data policy.
Information Classification
As a best practice, organizations take the time to identify, classify, and label data they use. Data classifications ensure that users understand the value of data, and the classifications help



protect sensitive data. Classifications can apply to hard data (printouts) and soft data (files).
example, the U.S. government uses classifications such as Top Secret, Secret, Confidential, and Unclassified to identify the sensitivity of data. Private companies often use terms such as Proprietary, Private, Confidential, or Public. Note that while the U.S. government has published standards for these classifications, there isn’t a published standard that all private companies use.
For comparison, the following statements identify the typical meaning of these public classifications:
Public data is available to anyone. It might be in brochures, press releases, or on web sites.
Confidential data is information that an organization intends to keep secret among a certain group of people. example, most companies consider salary data confidential. Personnel within the Accounting department and some executives have access to salary data, but they keep it secret among themselves. Many companies have specific policies in place telling people that they shouldn’t even tell anyone else their salary amount.
A proprietor is an owner and proprietary data is data that is related to ownership. Common examples are information related to patents or trade secrets.
Private data is information about an individual that should remain private. Two classic examples within IT security are Personally Identifiable Information (PII) and Personal Health Information (PHI). Both PII and PHI are covered in more depth later in this chapter.
The labels and classifications an organization uses are not as important as the fact that they use labels and classifications. Organizations take time to analyze their data, classify it, and provide training to users to ensure the users recognize the value of the data. They also include these classifications within a data policy.
Data Sensitivity Labeling and Handling
Data labeling ensures that users know what data they are handling and

processing. example, if an organization classified data as confidential, private, proprietary, and public, it would also use labeling to identify the data. These labels can be printed labels for media such as backup tapes. It’s also possible to label files using metadata, such as file properties, headers, footers, and watermarks.
Consider a company that spends millions of dollars on research and development (R&D) trying to develop or improve products. The company values this proprietary data much more than data publicly available on its web site, and needs to protect it. However, if employees have access to the R&D data and it’s not classified or labeled, they might not realize its value and might not protect it.
For example, a web content author might write an article for the company’s web site touting its achievements. If the R&D data isn’t classified and labeled, the author might include some of this R&D data in the article, inadvertently giving the company’s competitors free access to proprietary data. Although the R&D employees will easily recognize the data’s value, it’s not safe to assume that everyone does. In contrast, if the data is labeled, anyone would recognize its value and take appropriate steps to protect it.
Chapter 9,“Implementing Controls to Protect Assets,” presents information on backups. As a reminder, it’s important to protect backups with the same level of protection as the original data. Labels on backup media help personnel easily identify the value of the data on the backups.






Data Destruction and Media Sanitization
When computers reach the end of their life cycles, organizations donate them, recycle them, or sometimes just throw them away. From a security perspective, you need to ensure that the computers don’t include any data that might be useful to people outside your organization or damaging to your organization if unauthorized people receive it.
It’s common for organizations to have a checklist to ensure that personnel sanitize a system prior to disposing of it. The goal is to ensure that personnel remove all usable data from the system.
Hard drives represent the greatest risk because they hold the most information, so it’s important to take additional steps when decommissioning old hard drives. Simply deleting a file on a drive doesn’t actually delete it. Instead, it marks the file for deletion and makes the space available for use. Similarly, formatting a disk drive doesn’t erase the data. There are many recovery applications available to recover deleted data, file remnants, and data from formatted drives.
Data destruction isn’t limited to only hard drives. Organizations often have a policy related to paper containing any type of sensitive data. Shredding or incinerating these papers prevents them from falling into the wrong hands. If personnel just throw this paper away, dumpster divers can sift through the trash and gain valuable information. An organization also takes steps to destroy other types of data, such as backup tapes, and other types of devices, such as removable media.
Some common methods used to destroy data and sanitize media are:
Purging. Purging is a general sanitization term indicating that all sensitive data has been removed from a device.
File shredding. Some applications remove all remnants of a file

using a shredding technique. They do so by repeatedly overwriting the space where the file is located with 1s and 0s.
Wiping. Wiping refers to the process of completely removing all remnants of data on a disk. A disk wiping tool might use a bit-level overwrite process that writes different patterns of 1s and 0s multiple times and ensures that the data on the disk is unreadable.
Erasing and overwriting. Solid-state drives (SSDs) require a special process for sanitization. Because they use flash memory instead of magnetic storage platters, traditional drive wiping tools are not effective. Some organizations require personnel to physically destroy SSDs as the only acceptable method of sanitization.
Burning. Many organizations burn materials in an incinerator. Obviously, this can be done with printed materials, but isn’t as effective with all materials.





Paper shredding. You can physically shred papers by passing them through a shredder. When doing so, it’s best to use a cross-cut shredder that cuts the paper into fine particles. Large physical shredders can even destroy other hardware, such as disk drive platters removed from a disk drive.
Pulping. Pulping is an additional step taken after shredding paper. It reduces the shredded paper to mash or puree.
Degaussing. A degausser is a very powerful electronic magnet. Passing a disk through a degaussing field renders the data on tape and magnetic disk drives unreadable.
Pulverizing. Pulverizing is the process of physically destroying media to sanitize it, such as with a sledge hammer (and safety goggles). Optical media is often pulverized because it is immune to degaussing methods and many shredders can’t handle the size of optical media. It’s also possible to remove disk platters from disk drives and physically destroy them.
It’s also worth mentioning that hard drives and other media can be in devices besides just computers. example, many copy machines include disk drives, and they can store files of anything that employees recently copied or printed. If personnel don’t sanitize the drives before disposing of these devices, it can also result in a loss of confidentiality.

Data Retention Policies
A data retention policy identifies how long data is retained, and

sometimes specifies where it is stored. This reduces the amount of resources, such as hard drive space or backup tapes, required to retain the data. Retention policies also help reduce legal liabilities. example, imagine if a retention policy states that the company will only keep email for one year. A court order requiring all email from the company can only expect to receive email from the last year.
On the other hand, if the organization doesn’t have a retention policy, it might need to provide email from the past 10 years or longer in response to a court order. This can require an extensive amount of work by administrators to recover archives or search for specific emails. Additionally, investigations can uncover other embarrassing evidence from previous years. The retention policy helps avoid these problems.
Some laws mandate the retention of data for specific time frames, such as three years or longer. example, laws mandate the retention of all White House emails indefinitely. If a law



applies to an organization, the retention policy reflects the same requirements.
PII and PHI
Personally Identifiable Information (PII) is personal information that can be used to personally identify an individual. Personal Health Information (PHI) is PIIthat includes health information.
Some examples of PII are:
Full name
Birthday and birth place
Medical and health information
Street or email address information
Personal characteristics, such as biometric data
Any type of identification number, such as a Social Security number (SSN) or driver’s license number
In general, you need two or more pieces of information to make it PII. example, “John Smith” is not PII by itself because it can’t be traced back to a specific person. However, when you connect the name with a birth date, an address, medical information, or other data, it is PII.
When attackers gain PII, they often use it for financial gain at the expense of the individual. example, attackers steal identities, access credit cards, and empty bank accounts. Whenever possible, organizations should minimize the use, collection, and retention of PII. If it’s not kept, it can’t be compromised. On the other hand, if they collect PII and attackers compromise the data, the company is liable.
The number of security breach incidents resulting in the loss of PII continues to rise. example, a Veteran’s Affairs (VA) employee copied a database onto his laptop that contained PII on over 26 million U.S. veterans. He took the laptop home and a burglar stole it. The VA then went through the painful and expensive process of notifying all of the people who were vulnerable to identity theft, and the affected individuals spent countless hours scouring their records for identity theft incidents. Even though police later recovered the laptop, the VA paid $20 million to settle a lawsuit in the case.
This is not an isolated incident. The Identity Theft Resource Center tracks data breaches and lists them on their site (https://www.idtheftcenter.org/). Their 2015 report reported the number of

known U.S. data breaches at 780, exposing more than 177 million records containing PII and/or PHI. Some data breaches were small, affecting only a few hundred people. Others were large such as the attack on Scottrade, accessing more than 4.6 million records. Many times, the companies don’t even report how many records were accessed, so the number of data records in the hands of criminals is very likely much higher.
Each of these instances resulted in potential identity theft and the loss of goodwill and public trust of the company. Both customers and employees were negatively impacted, and the companies were forced to spend time and energy discussing the incident, and spend money trying to repair their reputations.
Protecting PII and PHI
Organizations have an obligation to protect PII. There are many laws that mandate the protection of PII, including international laws, federal laws, and local regulations. Organizations often develop policies to identify how they handle, retain, and distribute PII, and these policies help ensure they are complying with relevant regulations. When a company doesn’t use a specific



PII policy, it usually identifies methods used to protect PII in related data policies.
Many laws require a company to report data losses due to security breaches. If an attack results in the loss of customer PII data, the company is required to report it and notify affected individuals. example, Arizona enacted a security breach notification law that requires any company doing business in Arizona to notify customers of security breaches. Most states in the United States have similar laws, and similar international laws exist.
One of the common reasons data seems to fall into the wrong hands is that employees don’t understand the risks involved. They might not realize the value of the data on a laptop, or they might casually copy PII data onto a USB flash drive. As mentioned previously, data classification and labeling procedures help employees recognize the data’s value and help protect sensitive data.
Training is also important. One of the goals of security professionals is to reinforce the risks of not protecting PII. When employees understand the risks, they are less likely to risk customer and employee data to identity theft. Additionally, if employees need to transmit PII over a network, they can ensure it’s protected by using encryption. As mentioned previously in this book, encrypting data-in-transit provides strong protection against loss of
confidentiality.
Many governments have enacted laws mandating the protection of both PII and PHI. Also, there are many documents that provide guidance on how to protect it. The National Institute of Standards and Technology (NIST) created Special Publication (SP) 800-122 “Guide to Protecting the Confidentiality of Personally Identifiable Information (PII).” It identifies many specific safeguards that organizations can implement to protect PII along with steps to take in response to a data breach involving PII. You can access all the NIST publications at
https://csrc.nist. gov/publications/PubsSPs.html.


Legal and Compliance Issues
Organizations have a responsibility to follow all laws that apply to them, and ensure that they remain in compliance. Within the context of data security and privacy, the following laws are often a key concern:
Health Insurance Portability and Accountability Act of 1996 (HIPAA). HIPAA mandates that organizations protect PHI. This includes any information directly related to the health of an individual that might be held by doctors, hospitals, or any health facility. It also applies to any information held by an organization related to health plans offered to employees. Fines for not complying with the law have been as high as $4.3 million.
Gramm-Leach Bliley Act (GLBA). This is also known as the Financial Services Modernization Act and includes a Financial Privacy Rule. This rule requires financial institutions to provide consumers with a privacy notice explaining what information



they collect and how that information is used.
Sarbanes-Oxley Act (SOX). SOX was passed after several accounting scandals by major corporations, such as Enron and WorldCom. Companies were engaging in accounting fraud to make their financial condition look better than it was and prop up their stock price. example, Enron’s stock value was over $90 in 2000, but executives knew of problems and began selling their stock. As the scandal emerged, the stock crashed to
$42 a year later, and $15 in October of 2001. In December 2002, the stock was worthless at six cents a share, effectively wiping out $60 billion in investments. SOX requires that executives within an organization take individual responsibility for the accuracy of financial reports. It also includes specifics related to auditing, and identifies penalties to individuals for noncompliance.
General Data Protection Regulation (GDPR). This European Union (EU) directive supersedes the Data Protection Directive (also known as Directive 95/46/EC). Both mandate the protection of privacy data for individuals within the EU.
While this section outlined four specific laws related to data, there are others. The key is that organizations have a responsibility to know which laws apply to them and remain in compliance with the laws.
Data Roles and Responsibilities
Many people within the organization handle data. However, an organization often assigns specific roles to some people. Each of these roles has specific responsibilities as outlined in the following list:
Owner. The data owner is the individual with overall responsibility for the data. It is often a high-level position such as the chief executive officer (CEO) or a department head. The data owner is responsible for identifying the classification of the data, ensuring the data is labeled to match the classification, and ensuring security controls are implemented to protect the data.
Steward/custodian. A data steward or data custodian handles the routine tasks to protect data. example, a data custodian would ensure data is backed up in accordance with a backup policy. The

custodian would also ensure that backup tapes are properly labeled to match the classification of the data and stored in a location that provides adequate protection for the classification of the data. Data owners typically delegate tasks to the data custodian.
Privacy officer. A privacy officer is an executive position within an organization. This person is primarily responsible for ensuring that the organization is complying with relevant laws. example, if the organization handles any PHI, the privacy officer ensures the organization complies with HIPAA. If SOX applies to the organization, the privacy officer ensures that the organization is complying with SOX.




Responding to Incidents
Many organizations create incident response policies to help personnel identify and respond to incidents. A security incident is an adverse event or series of events that can negatively affect the confidentiality, integrity, or availability of data or systems within the organization, or that has the potential to do so.
Some examples include attacks, release of malware, security policy violations, unauthorized access of data, and inappropriate usage of systems. example, an attack resulting in a data breach is a security incident. Once the organization identifies a security incident, it will respond based on the incident response policy.
Organizations regularly review and update the policy. Reviews might occur on a routine schedule, such as annually, or in response to an incident after performing a lessons learned review of the incident.
example, in the early days of computers, one hacker broke into a government system and the first thing he saw was a welcome message. He started poking around, but authorities apprehended him. Later, when the judge asked him what he was doing, he replied that when he saw the welcome message, he thought it was inviting him in. The lesson learned here was that a welcome message can prevent an organization from taking legal action against an intruder. Government systems no longer have welcome messages. Instead, they have warning banners stressing that only authorized personnel should be accessing the system. It’s common to see similar warning banners when logging on to any system today.

NIST SP 800-61 Revision 2, “Computer Security Incident Handling Guide,” provides comprehensive guidance on how to respond to incidents. It is 79 pages so it’s obviously more in-depth than this section, but if you want to dig deeper into any of these topics, it’s an excellent resource. Use your favorite search engine and search for “NIST SP 800-61.”

Incident Response Plan
An incident response plan (IRP) provides more detail than the incident response policy. It provides organizations with a formal, coordinated plan personnel can use when responding to an incident. Some of the common elements included with an incident response plan include:
Definitions of incident types. This section helps employees identify the difference between an event (that might or might not be a security incident) and an actual incident. Some types of incidents include attacks from botnets, malware delivered via email, data breach, and a ransom demand after a criminal encrypts an organization’s data. The plan may group these incident types using specific category definitions, such as attacks, malware infections, and data breaches.
Cyber-incident response teams. A cyber-incident response team is composed of employees with expertise in different areas. Organizations often refer to the team as a




cyber-incident response team, a computer incident response team (CIRT), or a security incident response team. Combined, they have the knowledge and skills to respond to an incident. Due to the complex nature of incidents, the team often has extensive training. Training includes concepts, such as how to identify and validate an incident, how to collect evidence, and how to protect the collected evidence.
Roles and responsibilities. Many incident response plans identify specific roles for an incident response team along with their responsibilities. example, an incident response team might include someone from senior management with enough authority to get things done, a network administrator or engineer with the technical expertise necessary to understand the problems, a security expert who knows how to collect and analyze evidence, and a communication expert to relay information to the public if necessary.
Escalation. After identifying an incident, personnel often need to escalate it. Escalation can require a technician to inform his supervisor that he discovered a malware infection and is resolving it. If critical servers are under attack from a protracted distributed denial- of-service (DDoS) attack, escalation can require all members of the incident response team to get involved in responding to the incident.
Reporting requirements. Depending on the severity of the incident, security personnel might need to notify executives within the company of the incident. Obviously, they wouldn’t notify executives of every single incident. However, they would notify executives about serious incidents that have the potential to affect critical operations. If the incident involves a data breach, personnel need to identify the extent of the loss, and determine if outside entities are affected. example, if attackers successfully attacked a system and collected customer data such as credit information, the organization has a responsibility to notify customers of the data breach as soon as possible. The incident response plan outlines who needs to be notified and when.
Exercises. One method of preparing for incident response is to perform exercises. These can test the response of all members of the team. example, a technical exercise can test the administrator’s

ability to rebuild a server after a simulated attack. Mock interviews or press conferences can test the team’s responses to the media. NIST SP 800- 84, “Guide to Test, Training, and Exercise Programs for IT Plans and Capabilities,” provides much more in-depth information about performing exercises.
Incident Response Process
Incident response includes multiple phases. It starts with creating an incident response policy and an incident response plan. With the plan in place, personnel are trained and given the tools necessary to handle incidents. Ideally, incident response preparation will help an organization prevent an incident. However, this isn’t realistic for most organizations, but with an effective plan in place, the organization will be able to effectively handle any incidents that occur.
Some of the common phases of an incident response process are:
Preparation. This phase occurs before an incident and provides guidance to personnel on how to respond to an incident. It includes establishing and maintaining an incident response plan and incident response procedures. It also includes establishing procedures to prevent incidents. example, preparation includes implementing security controls to prevent malware infections.

Identification. All events aren’t security incidents so when a potential incident is reported, personnel take the time to verify it is an actual incident. example, intrusion detection systems (IDSs) might falsely report an intrusion, but administrators would investigate it and verify if it is a false positive or an incident. If the incident is verified, personnel might try to isolate the system based on established procedures.
Containment. After identifying an incident, security personnel attempt to isolate or contain it. This might include quarantining a device or removing it from the network. This can be as simple as unplugging the system’s network interface card to ensure it can’t communicate on the network. Similarly, you can isolate a network from the Internet by modifying access control lists on a router or a network firewall. This is similar to how you’d respond to water spilling from an overflowing sink. You wouldn’t start cleaning up the water until you first turn off the faucet. The goal of isolation is to prevent the problem from spreading to other areas or other computers in your network, or to simply stop the attack.
Eradication. After containing the incident, it’s often necessary to remove components from the attack. For example, if attackers installed malware on systems, it’s important to remove all remnants of the malware on all hosts within the organization. Similarly, an attack might have been launched from one or more compromised accounts. Eradication would include deleting or disabling these accounts.
Recovery. During the recovery process, administrators return all affected systems to normal operation and verify they are operating normally. This might include rebuilding systems from images, restoring data from backups, and installing updates. Additionally, if administrators have identified the vulnerabilities that caused the incident, they typically take steps to remove the vulnerabilities.
Lessons learned. After personnel handle an incident, security personnel perform a lessons learned review. It ’s very possible the incident provides some valuable lessons and the organization might modify  procedures  or  add  additional  controls  to  prevent	a reoccurrence of the incident. A review might indicate a need to

provide additional training to users, or indicate a need to update the incident response policy. The goal is to prevent a future reoccurrence of the incident.

Implementing Basic Forensic Procedures
A forensic evaluation helps the organization collect and analyze data as evidence it can use in the prosecution of a crime. In general, forensic evaluations proceed with the assumption that the data collected will be used as evidence in court. Because of this, forensic practices protect evidence to prevent modification and control evidence after collecting it.


Once the incident has been contained or isolated, the next step is a forensic evaluation. What do you think of when you hear forensics? Many people think about the TV program CSI (short for “crime scene investigation”) and all of its spin-offs. These shows demonstrate the phenomenal capabilities of science in crime investigations.
Computer forensics analyzes evidence from computers to determine details on computer incidents, similar to how CSI personnel analyze evidence from crime scenes. It uses a variety of different tools to gather and analyze computer evidence. Computer forensics is a growing field, and many educational institutions offer specialized degrees around the science. Although you might not be the computer forensics expert analyzing the evidence, you should know about some of the basic concepts related to gathering and preserving the evidence.
Forensic experts use a variety of forensic procedures to collect and protect data after an attack. A key part of this process is preserving the evidence during the data acquisition phase. In other words, they ensure that they don’t modify the data as they collect it, and they protect it after collection. A rookie cop wouldn’t walk through a pool of blood at a crime scene, at least not more than once. Similarly, employees shouldn’t access systems that have been attacked or power them down.
example, files have properties that show when they were last accessed. However, in many situations, accessing the file modifies this property. If the file is evidence, then accessing it has modified the evidence. This can prevent an investigation from identifying when an attacker accessed the file. Additionally, data in a system’s memory includes valuable evidence, but turning a system off deletes this data. In general, an incident response team does not attempt to analyze evidence until they have taken the time to collect and protect it.
Forensic experts have specialized tools they can use to capture data. example, many experts use EnCase Forensic by Guidance Software or Forensic Toolkit (FTK) by AccessData. These tools can capture data from memory or disks. This includes documents, images, email, webmail, Internet artifacts, web history, chat sessions, compressed files, backup files, and encrypted files. They can also capture data from smartphones and tablets.
Kali Linux includes a wide variety of forensic tools. Feel free to dig into any of them to learn more. They are available via the Applications > Forensics menu.
