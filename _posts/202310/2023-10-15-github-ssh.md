---
slug: github-ssh
title: Github SSH 설정
author: bbb1293
date: 2023-10-15T07:38:56.920Z
categories:
    - Github
    - Setup
tags:
    - github
    - ssh
pin: false
math: true
mermaid: true
lastmod: 2023-10-15T14:59:39.002Z
---

## What is SSH
---

Secure Shell Protocol(SSH)는 보호되지 않는 네트워크 환경속에서 안전하게 네트워크 서비스를 운영하기 위한 암호화 네트워크 프로토콜이다.[^ssh] 우리가 인터넷을 사용할 때 만약 민감한 정보(e.g. 비밀번호, 토큰 등)를 있는 그대로 전송했다고 하자. 중간에서 누군가가 정보를 가로채기만 하면 이를 악용할 수 있을 것이다. SSH는 이러한 문제를 해결한다.

Alice, Bob을 통한 비유로 흔히 설명되는 공개 키 암호화 방식이 사용된다. Alice와 Bob은 각자 public key와 private key를 가지고 있다. 이름 그대로 public key는 모두에게 공개되고 private key는 본인 혼자만 알고있다. Alice가 Bob에게 정보를 보낼 때 Bob의 public key로 본래의 데이터를 암호화한다. 암호화된 데이터가 Bob에게 전송되고 Bob은 본인의 private key로 이를 해독해서 본래의 데이터를 획득한다. 여기서 중간에 Eve가 데이터를 탈취하더라도 오리지널 데이터는 획득하지 못한다. 

![comment](/assets/img/202310/key.png)
_public-key encryption_

> 이게 어떻게 가능한지는 [RSA](https://en.wikipedia.org/wiki/RSA_(cryptosystem)) 문서를 참조하면 좋을 것 같다.  
{:.prompt-tip}


## Benefit to Connect to Github with SSH
---

SSH를 사용하지 않고 Github을 사용한다면 repo에 접근할 때마다 비밀번호가 필요할 수 있다. SSH를 이용하면 한번의 setup 이후에 repo에 간편하게 접근할 수 있다.

## Setup
---

[Connecting to GitHub with SSH
 문서](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)를 참조하면 된다.

프로세스는 대략적으로
1. 터미널을 연다.
2. ssh key를 생성한다. 
    1. 명령어를 입력한다.
    ```terminal
    $ ssh-keygen -t ed25519 -C "your_email@example.com" -f id_ssh_keyname
    ```
    2. 아래 prompt에서 엔터를 치면 mac 기준으로 `~/.ssh`{:.filepath} 폴더에 `id_ssh_keyname`{:.filepath} 이라는 이름의 key가 생성된다. 
    ```terminal
    > Enter a file in which to save the key (/Users/YOU/.ssh/id_ALGORITHM): [Press enter]
    ```
    3. 아래 prompt에서 ssh key 비밀번호를 입력해준다.
    ```terminal
    > Enter passphrase (empty for no passphrase): [Type a passphrase]
    > Enter same passphrase again: [Type passphrase again]
    ```
3. 생성한 SSH key를 ssh-agent에 붙여준다.
    1. 다음의 명령어로 ssh-agent를 background에 실행시킨다.
    ```terminal
    $ eval "$(ssh-agent -s)"
    > Agent pid 59566
    ```
    2. host 별 ssh-agent 세팅을 위해 `~/.ssh/config`{:.filepath} 파일에 아래 설정을 붙여넣는다. (`~/.ssh/config`{:.filepath} 파일이 없다면 하나 만들어주도록 하자)
    ```config
    Host github.com
      AddKeysToAgent yes
      UseKeychain yes
      IdentityFile ~/.ssh/id_ssh_keyname
    ```
    3. ssh-agent에 private key를 더하고 keychain에 비밀번호를 넣어주자.
    ```terminal
    $ ssh-add --apple-use-keychain ~/.ssh/id_ssh_keyname
    ```
4. 생성한 SSH key를 Github account에 저장한다.
    1. public key인 `~/.ssh/id_ssh_keyname.pub`{:.filepath} 내용을 복사한다.
    2. `Github Settings` -> `SSH and GPG keys` 로 들어간다 ([link](https://github.com/settings/keys))
    3. `New SSH key` 혹은 `Add SSH key`를 클릭한다.
    4. `title`에는 key의 용도를 입력한다 (e.g. Personal laptop)
    5. `Key type`은 `Authentication Key`로 둔다.
    6. `Key` 항목에 4-1에서 복사한 public key를 붙여넣는다.
    7. `Add SSH key`를 클릭한다.
5. 셋업이 잘 되었는지 확인한다.
    1. terminal를 연다.
    2. 아래 명령어를 입력한다.
    ```terminal
    ssh -T git@github.com
    # Attempts to ssh to GitHub
    ```
    3. 아래와 같은 output이 잘 나오는지 확인한다.
    ```terminal
    > Hi USERNAME! You've successfully authenticated, but GitHub does not
    > provide shell access.
    ```

## Setup for Multiple Keys
---

Github 계정을 여러 개 사용하고 각각 다른 키를 사용해야하는 상황이 있을 수 있다. (e.g. 회사 계정 / 개인 계정 분리)

[여기](https://stackoverflow.com/questions/3225862/multiple-github-accounts-ssh-config#answer-8483960)를 따라하면 되는데 방법은 다음과 같다. 

1. `~/.ssh/config` 파일을 열어 다음과 같이 host alias로 두 키를 구분해준다.
    ```config
    Host personal.github.com
      HostName github.com
      AddKeysToAgent yes
      UseKeychain yes
      IdentityFile ~/.ssh/personal_ssh_keyname

    Host company.github.com
      HostName github.com
      AddKeysToAgent yes
      UseKeychain yes
      IdentityFile ~/.ssh/company_ssh_keyname
    ```
2. 다음의 명령어로 테스트해본다.
    ```terminal
    $ ssh -T git@personal.github.com
    > Hi USERNAME! You've successfully authenticated, but GitHub does not
    > provide shell access.
    ```

> 여기서 주의할 점은 host allias를 repo를 clone할 때나 .git config에 입력해야한다는 점이다.
{:.prompt-danger}

예를 들어 일반적으로 로컬로 repo를 clone할 때     
```terminal
$ git clone git@github.com:bbb1293/bbb1293.github.io.git
```
라는 명령어를 사용한다면 ssh config 파일을 바꾼 이후에는 
```terminal
$ git clone git@personal.github.com:bbb1293/bbb1293.github.io.git
```
으로 host alias를 사용해주어야 한다.

이미 기존에 로컬에 repo를 가지고 있는 상황이라면 다음과 같이 진행하면 된다.
1. terminal을 켜서 repo로 이동한다.
2. git config 파일을 켠다.
```terminal
$ vim .git/config
```
3. url을 바꾸어 준다.
```config
...
[remote "origin"]
    url = git@bbb1293.github.com:bbb1293/bbb1293.github.io.git
...
```


## Reference

[^ssh]: <https://en.wikipedia.org/wiki/Secure_Shell>