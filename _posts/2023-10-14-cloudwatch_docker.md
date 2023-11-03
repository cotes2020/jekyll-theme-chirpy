---
title: AWS EC2 & CloudWatch & Docker-compose 로 로그 기록하기
date: 2023-10-14
categories: [troubleshooting]
tags: [logging, cloudwatch, aws, ec2]
---
## 🤔 Problem

ec2 에서 실행중인 인스턴스의 로그를 지켜보고 싶었으나 docker compose logs -f 로는 부족함을 느꼈다. 
새로 컨테이너를 올려도 계속해서 이전 로그를 보고 싶기도 해서 방법을 찾아보았다.

데이터독도 로고가 귀여워서 끌렸지만 접근성이 더 좋은 Watch Tower 를 먼저 사용해보기로 했다.

## 🌱 Solution

### 1. IAM 에서 역할만들기

IAM 역할을 생성해준다.

<div markdown="block" style="width: 85%;">
![https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-10-14-cloudwatch_docker/01.png](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-10-14-cloudwatch_docker/01.png)
</div>

AWS 서비스, EC2 를 선택해준다.

<div markdown="block" style="width: 85%;">
![https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-10-14-cloudwatch_docker/02.png](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-10-14-cloudwatch_docker/02.png)
</div>


CreateLogStream, PutLogEvents 을 선택하라고 나와있으나 내 검색창엔 보이지 않아서 EC2 full access 를 선택해줬다.

<div markdown="block" style="width: 85%;">
![https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-10-14-cloudwatch_docker/03.png](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-10-14-cloudwatch_docker/03.png)
</div>

역할명을 원하는 이름으로 변경해주고 생성해주면 완료

<div markdown="block" style="width: 85%;">
![https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-10-14-cloudwatch_docker/04.png](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-10-14-cloudwatch_docker/04.png)
</div>

 
 
### 2. EC2 에 IAM 연결하기

EC2 인스턴스를 체크하고 작업 > 보안 > IAM 역할 수정을 클릭한다.

<div markdown="block" style="width: 85%;">
![https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-10-14-cloudwatch_docker/05.png](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-10-14-cloudwatch_docker/05.png)
</div>

방금 생성한 IAM 역할을 선택해준다.

<div markdown="block" style="width: 85%;">
![https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-10-14-cloudwatch_docker/06.png](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-10-14-cloudwatch_docker/06.png)
</div>


 
### 3. Cloud Watch 에서 로그 그룹  & 로그 스트림 만들기

클라우드 워치로 들어간 뒤 로그 그룹을 생성한다.

<div markdown="block" style="width: 85%;">
![https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-10-14-cloudwatch_docker/07.png](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-10-14-cloudwatch_docker/07.png)
</div>

로그 그룹명을 생성하고

<div markdown="block" style="width: 85%;">
![https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-10-14-cloudwatch_docker/08.png](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-10-14-cloudwatch_docker/08.png)
</div>

로그 스트림 생성을 누른 뒤 로그스트림 명을 적어준다. 나는 확인하기 편하게 컨테이너 명과 동일하게 적어 줬다.

 <div markdown="block" style="width: 85%;">
![https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-10-14-cloudwatch_docker/09.png](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-10-14-cloudwatch_docker/10.png)
</div>

<div markdown="block" style="width: 85%;">
![https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-10-14-cloudwatch_docker/10.png](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-10-14-cloudwatch_docker/09.png)
</div>
 
### 4. docker-compose.yml 에 로깅 부분 추가하기

인스턴스의 docker-compose.yml 에 logging 설정을 적어준다. 로깅을 원하는 컨테이너에 추가해주면 된다.

```yml
    logging:
      driver: awslogs
      options:
        awslogs-group: "로그그룹이름"
        awslogs-region: "ap-northeast-2"
        awslogs-stream: "로그스트림명"
```


### 5. 로그 출력 확인


![https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-10-14-cloudwatch_docker/11.png](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-10-14-cloudwatch_docker/11.png)
</div>

docker compose up -d 를 해보면 해당 컨테이너가 다시 생성되고,

클라우드 워치의 스트림에서 정상적으로 로그가 출력된다.


### 📎 Related articles

| 이슈명                                      | 링크                                                        |
| ------------------------------------------- | ----------------------------------------------------------- |
| aws ec2 docker log cloudwatch 에서 받아보기 | [ https://devnm.tistory.com/8](https://devnm.tistory.com/8) |
