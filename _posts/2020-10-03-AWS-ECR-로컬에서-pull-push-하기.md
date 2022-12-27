---
title: ECR 로컬에서 pull push 하기
authors: jongin_kim
date: 2020-10-03 00:00:00 +0900
categories: [aws]
tags: [aws, docker]
---
### aws ecr get-login
```bash
aws ecr get-login --region ap-northeast-2
```

### login
위 커맨드로 나온 결과 그대로 복사 후 실행 ( `-e none 부분 제거` )
```bash
docker login -u AWS -p ??????????? https://xxxxxxxxxx.dkr.ecr.ap-northeast-2.amazonaws.com
```

### 태그 수정
기존 docker hub로 되어있는 태그들 변경
```bash
docker tag jongin/test-app:v0.0.1 xxxxxxxxxx.dkr.ecr.ap-northeast-2.amazonaws.com/test-app:v0.0.1
```

### push
```bash
docker push xxxxxxxxxx.dkr.ecr.ap-northeast-2.amazonaws.com/test-app:v0.0.1
```