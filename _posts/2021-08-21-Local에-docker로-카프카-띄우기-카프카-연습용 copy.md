---
title: Local에 docker로 카프카 띄우기 - 카프카 연습용
authors: Jongin Kim
date: 2021-08-21 00:00:00 +0900
categories: [kafka]
tags: [kafka]
---
Kafka 환경을 구축하기 위해 Kafka 서버를 Docker를 이용해 띄어보자

사실 상 Kafka를 다운로드 받아 그냥 로컬에 바로 띄우는게 가장 간단한 방법이겠지만, Docker를 한번 사용해봐야겠다 라는 마음으로 Docker에 설치하도록 했다.

개발 환경등 신경쓸 필요도 없고, 로컬 환경을 깔끔하게 유지한다는 장점이 있을 수 있겠다.

아래 절차는 Mac OS 환경 기준이다.

### **Docker 설치**

Mac OS의 경우에는 Docker 설치는 무척 간단하다. (링크 : [https://www.docker.com/get-started)](https://www.docker.com/get-started)

[Get Started with Docker | Docker
Learn about the complete container solution provided by Docker. Find information for developers, IT operations, and business executives.
www.docker.com](https://www.docker.com/get-started)

위 링크를 통해 들어가서 다운로드 받으면 Docker에 대한 설치는 끝난다.

다만 Docker에 간단한 가입 절차를 거쳐야되며, Github 계정을 만드는 것처럼 Docker Hub 계정을 만든다고 생각하자

### **Kafka-Docker 레포 클론**

이제 Docker를 이용해 Kafka를 띄어볼 차례인데, 사실 우리가 직접 Dockerfile 등을 작성할 필요도 없다.

이미 거의 비공식 표준으로 사용되고 있는 Kafka-Docker 레포가 존재한다. 우리는 단순히 해당 레포를 클론받아서 방금 로컬에 설치한 Docker에 띄어주기만 하면 된다.

다음과 같이 클론 받는다.

```
$ git clone https://github.com/wurstmeister/kafka-docker
```

### **docker-compose.yml 파일 수정**

엄밀히 말하면 docker가 아닌 docker-compose로 Kafka 환경을 구축할 것이다. 왜냐하면 Kafka docker뿐만 아니라 Zookeeper도 같이 띄울 것이기 때문에, 연동된 docker를 동시에 실행하기에 편리한 docker-compose를 이용한다.

또한, 여기서는 docker를 이용해 kafka를 띄우는 것에 목적을 두기 때문에, Kafka 분산 환경은 고려하지 않고 우선 한대만 띄워보자.

그러기 위해서는 레포안에 있는 파일 중 docker-compose.yml 파일을 이용하는 것이 아니라, docker-compose-single-broker.yml을 이용하자

그러면 docker를 로컬에 띄울 것이므로 docker-compose-single-broker.yml 파일을 다음과 같이 수정한다.

```
...
KAFKA_ADVERTISED_HOST_NAME: 127.0.0.1
...
```

KAFKA_ADVERTISED_HOST_NAME 만 수정해주면 된다.

### **docker-compose 실행**

```
$ docker-compose -f docker-compose-single-broker.yml up -d
```

여기서 -f 옵션은 docker-compose.yml 이 아닌 다른 이름을 가진 docker-compose를 실행시킬 경우 사용한다.

docker-compose.yml이 아닌 docker-compose.single-broker.yml을 사용할 것이기 때문에 -f 옵션 뒤에 파일 이름을 넣어주었다.

- d 옵션은 백그라운드에 띄운다는 것이다.

### **docker가 제대로 띄어졌는지 확인**

```
$ docker ps -a
```

위 명령어를 실행해보면 Kafka와 Zookeeper 두대가 잘 띄어져있음을 확인할 수 있다.

### **로컬에 Kafka 설치**

이제 Docker로 메세지를 받을 Kafka 서버는 잘 띄어져있다. 동작 확인을 위해 로컬에 Kafka를 설치해서 실제 메세지를 날려보고 받아보자.

우선 Kafka를 설치하기 위해 우리가 방금 띄운 Kafka의 버전을 확인해보고 같은 버전을 설치해주는 것이 좋다.

위에서 클론 받은 레포에 들어가 Dockerfile 파일을 확인해보자.

```
ARG kafka_version=2.4.1
ARG scala_version=2.12
```

이 부분만 보면 된다. Kafka 버전과 Scala 버전을 확인했다면 이에 맞게 Kafka를 다음과 같이 설치한다.

```
$ wget http://mirror.navercorp.com/apache/kafka/2.4.1/kafka_2.12-2.4.1.tgz
$ tar xzvf kafka_2.12-2.4.1.tgz
```

### **TOPIC 생성**

```
$ cd kafka_2.12-2.4.1
$ bin/kafka-topics.sh --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic test_topic
```

### **Producer 실행**

```
$ bin/kafka-console-producer.sh --topic test_topic --broker-list localhost:9092
```

위와 같이 실행시키고 나면 메세지 전송 대기 상태로 변경된다.

메세지를 타이핑하기 전에 Consumer를 띄워보자

### **Consumer 실행**

Consumer는 터미널창을 하나 새로 띄어 아래와 같이 실행시켜준다.

```
$ bin/kafka-console-consumer.sh --topic test_topic --bootstrap-server localhost:9092 --from-beginning
```

### **Producer에서 메세지 전송 후 Consumer에서 확인**

다시 Producer를 실행시킨 터미널 창으로 돌아가서 아무거나 타이핑 해본다.

그러면 Consumer 터미널 창에서 해당 메세지를 받은 것을 확인할 수 있을 것이다.

Kafka 환경을 구축하기 위해 Kafka 서버를 Docker를 이용해 띄어보자

사실 상 Kafka를 다운로드 받아 그냥 로컬에 바로 띄우는게 가장 간단한 방법이겠지만, Docker를 한번 사용해봐야겠다 라는 마음으로 Docker에 설치하도록 했다.

개발 환경등 신경쓸 필요도 없고, 로컬 환경을 깔끔하게 유지한다는 장점이 있을 수 있겠다.

아래 절차는 Mac OS 환경 기준이다.

### **Docker 설치**

Mac OS의 경우에는 Docker 설치는 무척 간단하다. (링크 : [https://www.docker.com/get-started)](https://www.docker.com/get-started)

[Get Started with Docker | Docker
Learn about the complete container solution provided by Docker. Find information for developers, IT operations, and business executives.
www.docker.com](https://www.docker.com/get-started)

위 링크를 통해 들어가서 다운로드 받으면 Docker에 대한 설치는 끝난다.

다만 Docker에 간단한 가입 절차를 거쳐야되며, Github 계정을 만드는 것처럼 Docker Hub 계정을 만든다고 생각하자

### **Kafka-Docker 레포 클론**

이제 Docker를 이용해 Kafka를 띄어볼 차례인데, 사실 우리가 직접 Dockerfile 등을 작성할 필요도 없다.

이미 거의 비공식 표준으로 사용되고 있는 Kafka-Docker 레포가 존재한다. 우리는 단순히 해당 레포를 클론받아서 방금 로컬에 설치한 Docker에 띄어주기만 하면 된다.

다음과 같이 클론 받는다.

```
$ git clone https://github.com/wurstmeister/kafka-docker
```

### **docker-compose.yml 파일 수정**

엄밀히 말하면 docker가 아닌 docker-compose로 Kafka 환경을 구축할 것이다. 왜냐하면 Kafka docker뿐만 아니라 Zookeeper도 같이 띄울 것이기 때문에, 연동된 docker를 동시에 실행하기에 편리한 docker-compose를 이용한다.

또한, 여기서는 docker를 이용해 kafka를 띄우는 것에 목적을 두기 때문에, Kafka 분산 환경은 고려하지 않고 우선 한대만 띄워보자.

그러기 위해서는 레포안에 있는 파일 중 docker-compose.yml 파일을 이용하는 것이 아니라, docker-compose-single-broker.yml을 이용하자

그러면 docker를 로컬에 띄울 것이므로 docker-compose-single-broker.yml 파일을 다음과 같이 수정한다.

```
...
KAFKA_ADVERTISED_HOST_NAME: 127.0.0.1
...
```

KAFKA_ADVERTISED_HOST_NAME 만 수정해주면 된다.

### **docker-compose 실행**

```
$ docker-compose -f docker-compose-single-broker.yml up -d
```

여기서 -f 옵션은 docker-compose.yml 이 아닌 다른 이름을 가진 docker-compose를 실행시킬 경우 사용한다.

docker-compose.yml이 아닌 docker-compose.single-broker.yml을 사용할 것이기 때문에 -f 옵션 뒤에 파일 이름을 넣어주었다.

- d 옵션은 백그라운드에 띄운다는 것이다.

### **docker가 제대로 띄어졌는지 확인**

```
$ docker ps -a
```

위 명령어를 실행해보면 Kafka와 Zookeeper 두대가 잘 띄어져있음을 확인할 수 있다.

### **로컬에 Kafka 설치**

이제 Docker로 메세지를 받을 Kafka 서버는 잘 띄어져있다. 동작 확인을 위해 로컬에 Kafka를 설치해서 실제 메세지를 날려보고 받아보자.

우선 Kafka를 설치하기 위해 우리가 방금 띄운 Kafka의 버전을 확인해보고 같은 버전을 설치해주는 것이 좋다.

위에서 클론 받은 레포에 들어가 Dockerfile 파일을 확인해보자.

```
ARG kafka_version=2.4.1
ARG scala_version=2.12
```

이 부분만 보면 된다. Kafka 버전과 Scala 버전을 확인했다면 이에 맞게 Kafka를 다음과 같이 설치한다.

```
$ wget http://mirror.navercorp.com/apache/kafka/2.4.1/kafka_2.12-2.4.1.tgz
$ tar xzvf kafka_2.12-2.4.1.tgz
```

### **TOPIC 생성**

```
$ cd kafka_2.12-2.4.1
$ bin/kafka-topics.sh --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic test_topic
```

### **Producer 실행**

```
$ bin/kafka-console-producer.sh --topic test_topic --broker-list localhost:9092
```

위와 같이 실행시키고 나면 메세지 전송 대기 상태로 변경된다.

메세지를 타이핑하기 전에 Consumer를 띄워보자

### **Consumer 실행**

Consumer는 터미널창을 하나 새로 띄어 아래와 같이 실행시켜준다.

```
$ bin/kafka-console-consumer.sh --topic test_topic --bootstrap-server localhost:9092 --from-beginning
```

### **Producer에서 메세지 전송 후 Consumer에서 확인**

다시 Producer를 실행시킨 터미널 창으로 돌아가서 아무거나 타이핑 해본다.

그러면 Consumer 터미널 창에서 해당 메세지를 받은 것을 확인할 수 있을 것이다.

[https://www.youtube.com/watch?v=HbbI6G24LZs&list=PL3Re5Ri5rZmkY46j6WcJXQYRlDRZSUQ1j&index=7](https://www.youtube.com/watch?v=HbbI6G24LZs&list=PL3Re5Ri5rZmkY46j6WcJXQYRlDRZSUQ1j&index=7)

Kafka 환경을 구축하기 위해 Kafka 서버를 Docker를 이용해 띄어보자

사실 상 Kafka를 다운로드 받아 그냥 로컬에 바로 띄우는게 가장 간단한 방법이겠지만, Docker를 한번 사용해봐야겠다 라는 마음으로 Docker에 설치하도록 했다.

개발 환경등 신경쓸 필요도 없고, 로컬 환경을 깔끔하게 유지한다는 장점이 있을 수 있겠다.

아래 절차는 Mac OS 환경 기준이다.

### **Docker 설치**

Mac OS의 경우에는 Docker 설치는 무척 간단하다. (링크 : [https://www.docker.com/get-started)](https://www.docker.com/get-started)

[Get Started with Docker | Docker
Learn about the complete container solution provided by Docker. Find information for developers, IT operations, and business executives.
www.docker.com](https://www.docker.com/get-started)

위 링크를 통해 들어가서 다운로드 받으면 Docker에 대한 설치는 끝난다.

다만 Docker에 간단한 가입 절차를 거쳐야되며, Github 계정을 만드는 것처럼 Docker Hub 계정을 만든다고 생각하자

### **Kafka-Docker 레포 클론**

이제 Docker를 이용해 Kafka를 띄어볼 차례인데, 사실 우리가 직접 Dockerfile 등을 작성할 필요도 없다.

이미 거의 비공식 표준으로 사용되고 있는 Kafka-Docker 레포가 존재한다. 우리는 단순히 해당 레포를 클론받아서 방금 로컬에 설치한 Docker에 띄어주기만 하면 된다.

다음과 같이 클론 받는다.

```
$ git clone https://github.com/wurstmeister/kafka-docker
```

### **docker-compose.yml 파일 수정**

엄밀히 말하면 docker가 아닌 docker-compose로 Kafka 환경을 구축할 것이다. 왜냐하면 Kafka docker뿐만 아니라 Zookeeper도 같이 띄울 것이기 때문에, 연동된 docker를 동시에 실행하기에 편리한 docker-compose를 이용한다.

또한, 여기서는 docker를 이용해 kafka를 띄우는 것에 목적을 두기 때문에, Kafka 분산 환경은 고려하지 않고 우선 한대만 띄워보자.

그러기 위해서는 레포안에 있는 파일 중 docker-compose.yml 파일을 이용하는 것이 아니라, docker-compose-single-broker.yml을 이용하자

그러면 docker를 로컬에 띄울 것이므로 docker-compose-single-broker.yml 파일을 다음과 같이 수정한다.

```
...
KAFKA_ADVERTISED_HOST_NAME: 127.0.0.1
...
```

KAFKA_ADVERTISED_HOST_NAME 만 수정해주면 된다.

### **docker-compose 실행**

```
$ docker-compose -f docker-compose-single-broker.yml up -d
```

여기서 -f 옵션은 docker-compose.yml 이 아닌 다른 이름을 가진 docker-compose를 실행시킬 경우 사용한다.

docker-compose.yml이 아닌 docker-compose.single-broker.yml을 사용할 것이기 때문에 -f 옵션 뒤에 파일 이름을 넣어주었다.

- d 옵션은 백그라운드에 띄운다는 것이다.

### **docker가 제대로 띄어졌는지 확인**

```
$ docker ps -a
```

위 명령어를 실행해보면 Kafka와 Zookeeper 두대가 잘 띄어져있음을 확인할 수 있다.

### **로컬에 Kafka 설치**

이제 Docker로 메세지를 받을 Kafka 서버는 잘 띄어져있다. 동작 확인을 위해 로컬에 Kafka를 설치해서 실제 메세지를 날려보고 받아보자.

우선 Kafka를 설치하기 위해 우리가 방금 띄운 Kafka의 버전을 확인해보고 같은 버전을 설치해주는 것이 좋다.

위에서 클론 받은 레포에 들어가 Dockerfile 파일을 확인해보자.

```
ARG kafka_version=2.7.0
ARG scala_version=2.13
```

이 부분만 보면 된다. Kafka 버전과 Scala 버전을 확인했다면 이에 맞게 Kafka를 다음과 같이 설치한다.

```
$ wget http://mirror.navercorp.com/apache/kafka/2.7.0/kafka_2.13-2.7.0.tgz
$ tar xzvf kafka_2.13-2.7.0.tgz
```

### **TOPIC 생성**

```
$ cd kafka_2.13-2.7.0
$ bin/kafka-topics.sh --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic test_topic
```

### **Producer 실행**

```
$ bin/kafka-console-producer.sh --topic test_topic --broker-list localhost:9092
```

위와 같이 실행시키고 나면 메세지 전송 대기 상태로 변경된다.

메세지를 타이핑하기 전에 Consumer를 띄워보자

### **Consumer 실행**

Consumer는 터미널창을 하나 새로 띄어 아래와 같이 실행시켜준다.

```
$ bin/kafka-console-consumer.sh --topic test_topic --bootstrap-server localhost:9092 --from-beginning
```

### **Producer에서 메세지 전송 후 Consumer에서 확인**

다시 Producer를 실행시킨 터미널 창으로 돌아가서 아무거나 타이핑 해본다.

그러면 Consumer 터미널 창에서 해당 메세지를 받은 것을 확인할 수 있을 것이다.

## 카프카 매니저 설치하기

---

[https://www.notion.so/jonginkim/13557ea0982b46b49c654a1e0ae9f98f#7782593e483e4c56a530f980389c347d](https://www.notion.so/13557ea0982b46b49c654a1e0ae9f98f)

여기 도커 컴포즈 파일 맨 밑에 아래 내용을 추가한다

```yaml
version: '2'
version: '2'
services:
  zookeeper:
    image: wurstmeister/zookeeper
    ports:
      - "2181:2181"
  kafka:
    build: .
    ports:
      - "9092:9092"
    environment:
      KAFKA_ADVERTISED_HOST_NAME: 127.0.0.1
      KAFKA_CREATE_TOPICS: "test:1:1"
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
  cmak:
    image: hlebalbau/kafka-manager:stable
    restart: always
    ports:
      - "9000:9000"
    environment:
      ZK_HOSTS: "zookeeper:2181"
```

```yaml
$ docker-compose -f docker-compose-single-broker.yml up -d
```

```jsx
$ docker exec -it kafka-docker_zookeeper_1 bash
$ ./bin/zkCli.sh
$ create /kafka-manager/mutex ""
$ create /kafka-manager/mutex/locks ""
$ create /kafka-manager/mutex/leases ""
```

- [localhost:9000](http://localhost:9000) 접속
![](/assets/img/posts/15.png)
    
> 참고 https://www.youtube.com/watch?v=HbbI6G24LZs&list=PL3Re5Ri5rZmkY46j6WcJXQYRlDRZSUQ1j&index=7