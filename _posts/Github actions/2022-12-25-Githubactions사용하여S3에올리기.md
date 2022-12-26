---
title: Github actions 사용하여 S3에 올리기
author: jimin
date: 2022-@-@ 00:00:00 +0900
categories: [Github_actions]
tags: [Github_actions, AWS, CI/CD]
pin: false
---

# 순서

 1. AWS IAM 사용자 등록
 2. AWS S3 버킷 생성
 3. github에 secret 생성하기
 4. .github/workflows 디렉토리 생성 후 @@.yml 파일 생성
 5. 코드 작성


 # 1. AWS IAM 사용자 등록
 - github actions에서 버킷에 파일을 올리기 위해서는 권한이 부여된 IAM 사용자가 필요합니다.
 - IAM 사용자를 추가하고 AWS 자격 증명 유형은 엑세스 키 방식으로 생성해줍니다.
 - S3에만 올릴 것이므로 다른 권한은 추가하지 않고 AmazonS3FullAccess만 추가해줍니다.
 - 완료 후 .csv 다운로드하여 계정 관리

 # 2. AWS S3 버킷 생성
 - AWS S3로 접속
 - 버킷 만들기 -> 이름 작성(대문자 사용 안됨) -> 퍼블릭 엑세스 차단 설정 해제 -> 버킷 만들기

 # 3. github에 secret 생성하기
 - 1에서 만들었던 IAM 사용자의 ID,PASSWORD를 저장해야한다.
 - repository의 Settings -> Secrets -> Actions -> New repository secret
 - AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY를 생성하고 그에 맞는 ID,PASSWORD를 작성 후 저장
 - 추가적으로 .gitignore 파일로 인해 github에서 제외된 파일들도 여기서 관리한다.
 - APPLICATION_DBCONFIG_PROPERTIES라는 secret을 만들어 파일 내용을 저장해준다.


 # 4. .github/workflows 디렉토리 생성 후 @@.yml 파일 생성
 - github actions을 사용하기 위해 디렉토리 최상단에 .github/workflows/uploadS3.yml 파일을 생성해주었다.

 # 5. 코드 작성

```java
on:
  pull_request:
    types:
      - closed // pr이 close 될 때
    branches: [ main ]

jobs:

  if_merged:
    if: github.event.pull_request.merged == true // merge가 성공하면
    runs-on: ubuntu-20.04
    steps:
      - name: Checkout
        uses: actions/checkout@v2

      - name: Setup Java JDK 11
        uses: actions/setup-java@v1
        with:
          java-version: 11

      - name: make application-dbconfig.properties
        run: |
          cd ./src/main/resources
          touch ./application-dbconfig.properties
          echo "${{ secrets.APPLICATION_DBCONFIG_PROPERTIES }}" > ./application-dbconfig.properties
        shell: bash

      - name: Grant execute permission for gradlew
        run: chmod +x ./gradlew
        shell: bash

      - name: Build with Gradle
        run: ./gradlew build
        shell: bash
        
      - name: Make Zip File
        run: zip -qq -r ./build.zip ./build/libs/Awesomely_Delicious-0.0.1-SNAPSHOT.jar
        shell: bash

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v1
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ap-northeast-2

      - name: Upload to S3
        run: aws s3 cp --region ap-northeast-2 ./build.zip s3://awesomelydelicious/build/build.zip
```

 # 결과
 - pr 후 merge에 성공하면 Actions에서 진행상황을 알 수 있음

# 참고


 - [https://velog.io/@tigger/Github-Action%EA%B3%BC-AWS-S3-%EC%97%B0%EB%8F%99%ED%95%98%EA%B8%B0](https://velog.io/@tigger/Github-Action%EA%B3%BC-AWS-S3-%EC%97%B0%EB%8F%99%ED%95%98%EA%B8%B0)

 - [https://bcp0109.tistory.com/363](https://bcp0109.tistory.com/363)

 - [https://zzsza.github.io/development/2020/06/06/github-action/](https://zzsza.github.io/development/2020/06/06/github-action/)