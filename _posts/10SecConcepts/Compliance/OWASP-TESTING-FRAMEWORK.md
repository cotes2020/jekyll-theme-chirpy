---
title: OWASP TESTING FRAMEWORK WORK FLOW
date: 2020-07-16 11:11:11 -0400
categories: [SOC, ComplianceAndReport]
tags: [SOC, OWASP]
math: true
image:
---

[toc]

---


# OWASP TESTING FRAMEWORK WORK FLOW

[toc]

---
# Foreword by Eoin Keary
## Frontispiece
### About the OWASP Testing Guide Project
### About The Open Web Application Security Project

---

# Introduction
## The OWASP Testing Project
### Principles of Testing
### Testing Techniques Explained
### Deriving Security Test Requirements
### Security Tests Integrated in Development and Testing Workflows S3ecurity Test Data Analysis and Reporting

---

# The OWASP Testing Framework
### Overview
### Phase 1: Before Development Begins Phase 2: During Definition and Design Phase 3: During Development
### Phase 4: During Deployment
### Phase 5: Maintenance and Operations A4Typical SDLC Testing Workflow

---

# Web Application Security Testing

## Introduction and Objectives
## Testing Checklist
### Information Gathering
### Conduct `Search Engine` Discovery and Reconnaissance for Information Leakage (OTG-INFO-001) Fingerprint Web Server (OTG-INFO-002)
### Review `Webserver Metafiles` for Information Leakage (OTG-INFO-003)
### Enumerate Applications on Webserver (OTG-INFO-004)
### Review `Webpage Comments and Metadata` for Information Leakage (OTG-INFO-005)
### Identify application entry points (OTG-INFO-006)
### Map execution paths through application (OTG-INFO-007)
### Fingerprint `Web Application Framework` (OTG-INFO-008)
### Fingerprint Web Application (OTG-INFO-009)
### Map Application Architecture (OTG-INFO-010)

## Configuration and Deployment Management Testing
### Test Network/Infrastructure Configuration (OTG-CONFIG-001)
### Test Application Platform Configuration (OTG-CONFIG-002)
### Testing Guide Foreword - Table of contents
###  Test File Extensions Handling for Sensitive Information (OTG-CONFIG-003)
### Review Old, Backup and Unreferenced Files for Sensitive Information (OTG-CONFIG-004) Enumerate Infrastructure and Application Admin Interfaces (OTG-CONFIG-005)

### Test `HTTP Methods` (OTG-CONFIG-006)

```bash
1. Discover the Supported Method
   # - The OPTIONS HTTP method provides the tester with the most direct and effective way to do that.
   # RFC 2616 states that, “The OPTIONS method represents a request for information about the communication options available on the request/response chain identi- fied by the Request-URI”.

    $ nc www.victim.com 80 OPTIONS / HTTP/1.1
    Host: www.victim.com
    HTTP/1.1 200 OK
    Server: Microsoft-IIS/5.0
    Date: Tue, 31 Oct 2006 08:00:29 GMT Connection: close
    Allow: GET, HEAD, POST, TRACE, OPTIONS Content-Length: 0
    # OPTIONS provides a list of the methods that are supported by the web server
    # see that TRACE method is enabled. The danger that is posed by this method is illustrated in the following section

2. Test XST Potential
   # - the most recurring attack patterns in Cross Site Scripting is to access the document.cookie object and send it to a web server controlled by the attacker
   # - so that he or she can hijack the victim’s session.
   # - Tagging a cookie as httpOnly forbids JavaScript from accessing it, protecting it from being sent to a third party.
   # - However, the TRACE method can be used to bypass this protection and access the cookie even in this scenario.

    $ nc www.victim.com 80 TRACE / HTTP/1.1
    Host: www.victim.com
    HTTP/1.1 200 OK
    Server: Microsoft-IIS/5.0
    Date: Tue, 31 Oct 2006 08:01:48 GMT Connection: close
    Content-Type: message/http Content-Length: 39
    TRACE / HTTP/1.1 Host: www.victim.com

   # - If the tester instructs a browser to issue a TRACE request to the web server, and this browser has a cookie for that domain, the cookie will be automatically included in the request headers, and will therefore be echoed back in the resulting response.
   # - At that point, the cookie string will be accessible by JavaScript and it will be finally possible to send it to a third party even when the cookie is tagged as httpOnly.

   # to make a browser issue a TRACE request:
   # - XMLHTTP ActiveX control in Internet Explorer
   # - XM- LDOM in Mozilla and Netscape.
   # - However, for security reasons the browser is allowed to start a connection only to the domain where the hostile script resides. This is a mitigating factor, as the attacker needs to combine the TRACE method with another vulnerability in order to mount the attack.

    # An attacker has two ways to successfully launch a Cross Site Trac- ing attack:
    # • Leveraging another server-side vulnerability: the attacker injects the hostile JavaScript snippet that contains the TRACE request in the vulnerable application, as in a normal Cross Site Scripting attack
    # • Leveraging a client-side vulnerability: the attacker creates a malicious website that contains the hostile JavaScript snippet and exploits some cross-domain vulnerability of the browser of the victim, in order to make the JavaScript code successfully perform a connection to the site that supports the TRACE method and that originated the cookie that the attacker is trying to steal.
```


### Test `HSTS HTTP Strict Transport Security` (OTG-CONFIG-007)

- The `HTTP Strict Transport Security (HSTS) header`
- a mechanism that web sites have to communicate to the web browsers that all traffic exchanged with a given domain must always be sent over https
  - protect the information from being passed over unencrypted requests.
  - ensure that all the data travels encrypted from the web browser to the server.
  - The `HTTP Strict Transport Security (HSTS)` feature lets a web application to inform the browser, through the use of a special response header, that it should never establish a connection to the the specified domain servers using HTTP. Instead it should automatically establish all connection requests to access the site through HTTPS.

The HTTP strict transport security header uses two directives:
- `-max-age`:
  - the number of seconds that the browser should automatically convert all HTTP requests to HTTPS.
- `includeSubDomains`:
  - all web application’s sub-domains must use HTTPS.
  - If this parameter is set, then the HSTS policy applies to the visited domain and all the subdomains as well.
  - If not it only applies to the exact domain that the user has visited.

This header by web applications must be checked:
- `Attackers sniffing the network traffic` and accessing the information transferred through an unencrypted channel.
- `Attackers exploiting a man in the middle attack` because of the problem of accepting certificates that are not trusted.
- Users who `mistakenly entered an address` in the browser putting HTTP instead of HTTPS, or users who click on a link in a web application which mistakenly indicated the http protocol.
- user `clicks on HTTP links` or even if the user type an HTTP link.
- HSTS policy `prevents an user from accepting self-signed or abnormally signed certificates`, since remember the certification authority (CA) that signed the previous seen certificate.


```bash
$ curl --silent --head https://us.com/ | grep -i strict
strict-transport-security: max-age=7776000000;
includeSubDomains

```

### Test RIA cross domain policy (OTG-CONFIG-008)
### Identity Management Testing
### Test Role Definitions (OTG-IDENT-001)
### Test User Registration Process (OTG-IDENT-002)
### Test Account Provisioning Process (OTG-IDENT-003)
### Testing for Account Enumeration and Guessable User Account (OTG-IDENT-004)
### Testing for Weak or unenforced username policy (OTG-IDENT-005)
### Authentication Testing
### Testing for Credentials Transported over an Encrypted Channel (OTG-AUTHN-001) Testing for default credentials (OTG-AUTHN-002)
### Testing for Weak lock out mechanism (OTG-AUTHN-003)
### Testing for bypassing authentication schema (OTG-AUTHN-004)
### Test remember password functionality (OTG-AUTHN-005)
### Testing for Browser cache weakness (OTG-AUTHN-006)
### Testing for Weak password policy (OTG-AUTHN-007)
### Testing for Weak security question/answer (OTG-AUTHN-008)
### Testing for weak password change or reset functionalities (OTG-AUTHN-009)
### Testing for Weaker authentication in alternative channel (OTG-AUTHN-010) Authorization Testing
### Testing Directory traversal/file include (OTG-AUTHZ-001)
### Testing for bypassing authorization schema (OTG-AUTHZ-002)
### Testing for Privilege Escalation (OTG-AUTHZ-003)
### Testing for Insecure Direct Object References (OTG-AUTHZ-004)
### Session Management Testing
### Testing for Bypassing Session Management Schema (OTG-SESS-001)
### Testing for Cookies attributes (OTG-SESS-002)
### Testing for Session Fixation (OTG-SESS-003)
### Testing for Exposed Session Variables (OTG-SESS-004)
### Testing for Cross Site Request Forgery (CSRF) (OTG-SESS-005)
### Testing for logout functionality (OTG-SESS-006)
### Test Session Timeout (OTG-SESS-007)
### Testing for Session puzzling (OTG-SESS-008)
### Input Validation Testing
### Testing for Reflected Cross Site Scripting (OTG-INPVAL-001)
### Testing for Stored Cross Site Scripting (OTG-INPVAL-002)
### Testing for HTTP Verb Tampering (OTG-INPVAL-003)
### Testing for HTTP Parameter pollution (OTG-INPVAL-004)
### Testing for SQL Injection (OTG-INPVAL-005)
### Oracle Testing
### MySQL Testing
### SQL Server Testing
### Testing PostgreSQL (from OWASP BSP) MS Access Testing
###
### 3
### Testing Guide Foreword - Table of contents
###  Testing for NoSQL injection
### Testing for LDAP Injection (OTG-INPVAL-006) Testing for ORM Injection (OTG-INPVAL-007) Testing for XML Injection (OTG-INPVAL-008) Testing for SSI Injection (OTG-INPVAL-009) Testing for XPath Injection (OTG-INPVAL-010) IMAP/SMTP Injection (OTG-INPVAL-011) Testing for Code Injection (OTG-INPVAL-012)
### Testing for Local File Inclusion
### Testing for Remote File Inclusion
### Testing for Command Injection (OTG-INPVAL-013) Testing for Buffer overflow (OTG-INPVAL-014)
### Testing for Heap overflow
### Testing for Stack overflow
### Testing for Format string
### Testing for incubated vulnerabilities (OTG-INPVAL-015) Testing for HTTP Splitting/Smuggling (OTG-INPVAL-016)
### Testing for Error Handling
### Analysis of Error Codes (OTG-ERR-001)
### Analysis of Stack Traces (OTG-ERR-002)
### Testing for weak Cryptography
### Testing for Weak SSL/TLS Ciphers, Insufficient Transport Layer Protection (OTG-CRYPST-001) Testing for Padding Oracle (OTG-CRYPST-002)
### Testing for Sensitive information sent via unencrypted channels (OTG-CRYPST-003)
### Business Logic Testing
### Test Business Logic Data Validation (OTG-BUSLOGIC-001) Test Ability to Forge Requests (OTG-BUSLOGIC-002)
### Test Integrity Checks (OTG-BUSLOGIC-003)
### Test for Process Timing (OTG-BUSLOGIC-004)
### Test Number of Times a Function Can be Used Limits (OTG-BUSLOGIC-005) Testing for the Circumvention of Work Flows (OTG-BUSLOGIC-006)
### Test Defenses Against Application Mis-use (OTG-BUSLOGIC-007)
### Test Upload of Unexpected File Types (OTG-BUSLOGIC-008)
### Test Upload of Malicious Files (OTG-BUSLOGIC-009)
### Client Side Testing
### Testing for DOM based Cross Site Scripting (OTG-CLIENT-001) Testing for JavaScript Execution (OTG-CLIENT-002)
### Testing for HTML Injection (OTG-CLIENT-003)
### Testing for Client Side URL Redirect (OTG-CLIENT-004) Testing for CSS Injection (OTG-CLIENT-005)
### Testing for Client Side Resource Manipulation (OTG-CLIENT-006) Test Cross Origin Resource Sharing (OTG-CLIENT-007)
### Testing for Cross Site Flashing (OTG-CLIENT-008)
### Testing for Clickjacking (OTG-CLIENT-009)
### Testing WebSockets (OTG-CLIENT-010) Test Web Messaging (OTG-CLIENT-011) Test Local Storage (OTG-CLIENT-012)


---
