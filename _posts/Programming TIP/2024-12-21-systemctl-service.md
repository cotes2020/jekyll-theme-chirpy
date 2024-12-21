---
title: "어플리케이션 자동 실행을 위한 systemctl 서비스 등록 시 Trouble Shooting"
categories: [Programming TIP]
tags: [Linux, systemctl]
---

EC2 서버에서 특정 스케줄로 어플리케이션을 실행하거나, 서버가 재기동될 때 어플리케이션을 자동으로 실행하려면 systemctl을 활용해 서비스를 등록하는 것이 효과적입니다.

이 과정은 간단히 말해 어플리케이션 실행용 shell script를 systemctl 서비스로 등록하여 서버 기동 시 어플리케이션이 자동으로 실행되도록 설정하는 것입니다.

하지만 서비스를 설정하다 보면 예상치 못한 문제에 직면할 수 있습니다. 오늘 제가 겪었던 세 가지 주요 이슈를 공유하며, 이를 예방하기 위한 팁을 소개하겠습니다.

---

## 1. nohup 및 백그라운드 명령어를 사용하지 마세요

systemctl 서비스는 자체적으로 백그라운드에서 동작합니다.
따라서 서비스를 등록할 때 호출되는 shell script에서 nohup이나 & 같은 명령어를 제거해야 합니다.

- 예시:

```bash
# 잘못된 코드
nohup java -jar app.jar &

# 올바른 코드
java -jar app.jar
```

- 이를 무시하면 프로세스가 비정상적으로 종료되거나 서비스 상태가 불안정해질 수 있습니다.

## **2. 어플리케이션 하나 당 하나의 서비스로 분리**

하나의 shell script에서 여러 어플리케이션을 실행하도록 작성하는 경우가 종종 있습니다.

하지만 이를 systemctl 서비스로 등록하면 다음과 같은 오류가 발생할 가능성이 높습니다:

```
service start request repeated too quickly, refusing to start
```

- 원인:

어플리케이션 실행 간 **종속성 문제**가 있을 때, 특정 어플리케이션이 정상적으로 실행되지 않으면 systemctl이 재시도를 반복하다가 오류를 발생시킵니다.

- 해결책:

어플리케이션마다 독립적인 systemctl 서비스를 만들어 각자의 실행 흐름을 관리하세요.

<!-- prettier-ignore -->
> A 어플리케이션 → B 어플리케이션 실행과 같은 종속성이 있는 경우, **서비스 간 의존성 설정**(e.g., After=, Requires=)을 추가하면 됩니다.
{: .prompt-tip }

## **3. 환경 변수 설정**

로그인 시 자동으로 로드되는 환경 변수가 systemctl 서비스에서 인식되지 않을 수 있습니다.

따라서 환경 변수가 필요한 경우, 다음과 같은 방식으로 systemctl 서비스 파일에 환경 변수를 정의해야 합니다:

1. **서비스 파일에 직접 설정**

   ```ini
   [Service]
   Environment="JAVA_HOME=/usr/lib/jvm/java-11-openjdk"
   Environment="PATH=/usr/bin:/usr/local/bin"
   ExecStart=/usr/bin/java -jar /path/to/app.jar
   ```

2. **환경 변수 파일로 관리**
   별도의 파일에 환경 변수를 정의하고 이를 systemctl 서비스에서 불러오는 방법도 있습니다:

   ```ini
   [Service]
   EnvironmentFile=/path/to/envfile
   ExecStart=/usr/bin/java -jar /path/to/app.jar
   ```

## **정리**

systemctl을 활용한 서비스 등록은 효율적이지만, 다음 사항을 유의하세요:

​ 1. **nohup 및 백그라운드 명령어 사용 금지**

​ 2. **어플리케이션 별로 독립적인 서비스 구성**

​ 3. **환경 변수는 서비스 파일에 명시적으로 정의**

위 내용을 준수하면 안정적인 어플리케이션 실행 환경을 구축할 수 있습니다.

모두의 성공적인 서비스 설정을 응원합니다! 🚀
