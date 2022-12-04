---
title: AWS Infra / Part11. Amazon S3
date: 2022-12-04 19:00:00 +0900
categories: [AWS, 2Tier 아키텍쳐 구축]
tags: [AWS, S3]     # TAG names should always be lowercase
typora-root-url: ../
---
# Intro

[지난 파트](/posts/AWS-Part10-AmazonEFS/)에서는 동적 파일을 담기 위한 Amazon EFS를 생성하여 배치 했습니다.

이번 파트에서는 정적 파일을 담기 위한 Amazon S3을 생성하여 배치 하겠습니다.

<br>

# Amazon S3?

> *Amazon Simple Storage Service(Amazon S3)*는 업계 최고 수준의 확장성, 데이터 가용성, 보안 및 성능을 제공하는 객체 스토리지 서비스입니다. 고객은 규모와 업종에 관계없이 원히는 양의 데이터를 저장하고 보호하여 데이터 레이크, 클라우드 네이티브 애플리케이션 및 모바일 앱과 같은 거의 모든 사용 사례를 지원할 수 있습니다. 비용 효율적인 스토리지 클래스와 사용이 쉬운 관리 기능을 통해 비용을 최적화하고, 데이터를 정리하고, 세분화된 액세스 제어를 구성하여 특정 비즈니스, 조직 및 규정 준수 요구 사항을 충족할 수 있습니다.  \- *AWS Document*

S3도 EFS와 마찬가지로 저장소 역할을 수행 합니다.

EFS와는 다르게 정적인 파일을 담아서 서비스 시킬것 입니다.

S3는 한번 데이터가 쓰이고 나서 읽기만 되는 환경에 유리하고,

EFS는 빈번하게 입출력이 진행되는 환경에 유리 합니다.

<br>

# 실전

![01](/assets/post/2022-12-04-AWS-Part11-AmazonS3/01.png)

S3를 만들기 전에, Private 서브넷에 있는 인스턴스들이 S3에 접근 가능하게 하려면

엔드포인트를 생성 해야 합니다. 엔드포인트는 일종의 통로 라고 생각하시면 될 것 같습니다.

VPC 대시보드에서 **엔드포인트** 메뉴로 접근 합니다.

<br>

![02](/assets/post/2022-12-04-AWS-Part11-AmazonS3/02.png)

엔드포인트 생성을 위해 **엔드포인트 생성** 버튼을 누릅니다.

<br>

![03](/assets/post/2022-12-04-AWS-Part11-AmazonS3/03.png)

이름을 지정 하고, 서비스 범주는 AWS 서비스를 선택 합니다.

그 후 아래의 서비스란에서 S3를 검색하여, **유형이 Gateway인 s3 항목**을 선택 합니다.

<br>

![04](/assets/post/2022-12-04-AWS-Part11-AmazonS3/04.png)

VPC는 연결할 VPC를 선택 합니다.

<br>

![05](/assets/post/2022-12-04-AWS-Part11-AmazonS3/05.png)

VPC를 선택하면 연결할 라우팅 테이블을 선택 할 수 있습니다.

각각의 Private 서브넷은 PrivateSNa, c의 라우팅 테이블에 연동되어 있으므로

해당 라우팅 테이블을 선택 합니다.

<br>

![06](/assets/post/2022-12-04-AWS-Part11-AmazonS3/06.png)

정책은 전체 엑세스를 선택하고, 스크롤을 내려 **엔드포인트 생성**을 클릭 합니다.

<br>

![07](/assets/post/2022-12-04-AWS-Part11-AmazonS3/07.png)

엔드포인트가 생성이 되었습니다.

생성된 **VPC 엔드포인트 ID를 메모**해 둡니다.

<br>

![08](/assets/post/2022-12-04-AWS-Part11-AmazonS3/08.png)

이번엔 S3를 생성 할 차례 입니다.

상단 검색바에서 **S3을 검색**하여 접근 합니다.

<br>

![09](/assets/post/2022-12-04-AWS-Part11-AmazonS3/09.png)

**버킷 만들기를 클릭** 합니다.

<br>

![10](/assets/post/2022-12-04-AWS-Part11-AmazonS3/10.png)

버킷 이름(실제로는 소문자만 지정 가능 합니다.)을 지정 합니다. 리전도 제대로 선택되어 있는지 확인 합니다.

<br>

![11](/assets/post/2022-12-04-AWS-Part11-AmazonS3/11.png)

Private 서브넷에 있는 인스턴스만 접근을 가능하게 할 것이기 때문에,

모든 퍼블릭 액세스 차단을 선택한 상태에서 스크롤을 내려 **버킷 만들기**를 클릭 합니다.

<br>

![12](/assets/post/2022-12-04-AWS-Part11-AmazonS3/12.png)

버킷이 위와 같이 생성 되었습니다.

이름의 링크를 클릭 하여 상세정보란으로 접근 합니다.

<br>

![13](/assets/post/2022-12-04-AWS-Part11-AmazonS3/13.png)

상세정보 탭 중 **권한 탭을 클릭** 합니다.

<br>

![14](/assets/post/2022-12-04-AWS-Part11-AmazonS3/14.png)

퍼블릭 액세스 차단이 활성화 되어 있습니다.

이대로는 접근이 안되기 때문에 버킷 정책에 있는 **편집 버튼**을 누릅니다.

<br>

![15](/assets/post/2022-12-04-AWS-Part11-AmazonS3/15.png)

<br>

```json
{
    "Version": "2012-10-17",
    "Id": "PolicyForS3ConnectToEndpoint",
    "Statement": [
        {
            "Sid": "Access-to-specific-VPCE-only",
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": "[버킷 ARN]/*",
            "Condition": {
                "StringEquals": {
                    "aws:sourceVpce": "[VPC Endpoint ID]"
                }
            }
        }
    ]
}
```

버킷 정책란에 위의 코드를 복사하여 붙여 넣고, [버킷 ARN] 부분은 정책 위에 있는 ARN을 넣고,

아래의 [VPC Endpoint ID]는 아까 복사해둔 VPC 엔드포인트 ID를 삽입 합니다.

그후 **변경 사항 저장**을 클릭 합니다.

<br>

# 결과 확인

![16](/assets/post/2022-12-04-AWS-Part11-AmazonS3/16.png)

이제 위와 같이 S3 버킷도 배치 되었습니다.

<br>

# Outro

이번 파트 에서는 Amazon S3 버킷을 생성하고 설정 하였습니다.

다음 파트는 실제 인스턴스에 적용할 수 있게 인스턴스 세부설정 파트를 진행 하겠습니다.
