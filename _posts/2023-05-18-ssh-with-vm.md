---
title: VM에 ID,PW로 접속하려면 대작전
date: 2023-05-18
categories: [troubleshooting]
tags: [gcp, cloud, deploy, ssh]
---

## 🤔 Problem

뿅뿅단어장 팀원들의 작업물을 배포하면서 혼자서 vm 을 관리하는데에 한계를 느꼈다.

그래서 부트캠프에서 배부해주는 vm 처럼 id 와 비밀번호 형식으로 팀원들이 vm 에 접속할 수 있게 해주고 싶었다.

## 🌱 Solution

<aside>
💡 해결법은 ssh 설정을 하는 것이었다.

</aside>

1. 서버 컴퓨터에 기존 개인 ssh 키로 로그인합니다.
2. 사용자 계정을 생성합니다.

   ```bash
   $ sudo useradd -m -s /bin/bash <<사용자명>>
   ```

3. 사용자의 비밀번호를 설정합니다. 다음 명령을 실행하여 비밀번호를 설정합니다. 필요한 경우 비밀번호를 변경하십시오.

   ```bash
   $ sudo passwd <<사용자명>>
   ```

4. SSH 서버 설정 파일(**`/etc/ssh/sshd_config`**)을 편집합니다. 다음 명령을 실행하여 파일을 엽니다.

   ```
   sudo vi /etc/ssh/sshd_config
   ```

5. **`sshd_config`** 파일에서 다음 라인을 찾습니다.

   ```
   # PasswordAuthentication no
   ```

6. 주석 처리된 **`PasswordAuthentication`** 옵션을 찾고 주석(**`#`**)을 제거하고 값을 **`yes`**로 변경합니다.

   ```
   PasswordAuthentication yes
   ```

7. 파일을 저장하고 에디터를 종료합니다. 저장 후에는 SSH 서비스를 재시작해야 합니다.
8. SSH 서비스를 재시작합니다. 다음 명령을 실행하여 서비스를 재시작합니다.

   ```
   $ sudo systemctl restart sshd
   ```

9. 이제 **`사용자명`** 사용자로 SSH 비밀번호 인증을 사용하여 서버에 접속할 수 있어야 합니다. 다음 명령을 실행하여 접속을 시도해 보세요.

   ```bash
   $ ssh <<사용자명>>@34.64.252.0
   ```

10. 비밀번호를 입력하고 접속이 성공하는지 확인해 보세요.

## 📎 Related articles

| 이슈명                            | 링크                                                                 |
| --------------------------------- | -------------------------------------------------------------------- |
| 우당탕탕 gcp vm 개설기            | https://www.notion.so/GCP-VM-73cb2cc466d3489ab6b4905dbdef62a5?pvs=21 |
| 우분투 가상머신에 ssh 로 접속하기 | https://www.bearpooh.com/102                                         |
