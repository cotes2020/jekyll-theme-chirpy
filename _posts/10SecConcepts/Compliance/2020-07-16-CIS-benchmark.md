---
title: CIS benchmarks
date: 2020-07-16 11:11:11 -0400
categories: [10SecConcept, ComplianceAndReport]
tags: [CIS]
math: true
image:
---

[toc]

---

# CIS benchmarks

## 1

```bash
1. downloads
$ cd ur_path
$ git clone link
$ cd ur_path/downloadedCIS

2. edit the host file
$ vim host
[name]
1.1.1.1

3. check server eth0 inet addr 1.1.1.1.
[server]# ifcongif -a

4.
$ ansible-playbook playbook.yml -i hosts

# check sections
$ ansible-playbook playbook.yml -i hosts --list-tags

$ ansible-playbook playbook.yml -i hosts --tags sction7, section5
# it will pass all the section configure

```


## configure account management by using group policies

```py
# show group policy
lunch - edit group policy

window setting - security setting - advanced audit policy - system audit policy - account management

select the one to config [hold control] - right klick - properties - configure the selected events to be audited - check 2 box. - close the Windows

# run CIS-CAT-lite vertion
configuration assessment tools:
- select CIS benchmarks: ...benchamrk
- profile: level 1
- next
- assessment 1-2min
- done view reports:

# now account management is all pass.








```
