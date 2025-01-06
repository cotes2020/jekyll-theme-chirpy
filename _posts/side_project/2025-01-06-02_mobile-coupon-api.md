---
title: "[OMP_#002] 모바일쿠폰 정보 획득 API 탐색! 그리고.."
categories: [Side Project]
tags: [barcodeAPI, pyzbar]
---

kick-off 글에서 살짝 언급했던 **모바일 쿠폰 이미지의 바코드를 인식해서 정보를 얻어오는** 작업에 대해 간단히 테스트를 해볼 예정이다!

아래의 순서로 진행될 에정!

1. 모바일 쿠폰 이미지의 바코드 읽기
2. 바코드로 읽어온 모바일쿠폰 번호를 통해 정보 획득하기

## 1. 모바일 쿠폰 이미지의 바코드를 읽어 정보 획득

- Pillow 라이브러리를 통해 Image를 읽어오고
- pyzbar 라이브러리를 통해 바코드의 정보를 획득한다.

```python
from pyzbar.pyzbar import decode
from PIL import Image

# 바코드가 포함된 이미지 파일 경로
image_path = "IMG_PATH"

# 이미지 로드
image = Image.open(image_path)

# 바코드 읽기
decoded_objects = decode(image)

# 결과 출력
for obj in decoded_objects:
    print("Type:", obj.type)  # 바코드 유형 (Code128, QR Code 등)
    print("Data:", obj.data.decode('utf-8'))  # 바코드 데이터
```

### Case1) wifi QR 코드

![Image]({{"/assets/img/posts/2025-01-06-21-20-49.png" | relative_url }}){: w="200"}

결과

```
Type: QRCODE
Data: WIFI:S:SB1Guest;P:12345;T:WEP;;
```

`Data`안에 정보가 직렬화되어 오는데, Wifi id, password, type 등이 전달되어 온다.

### Case2) Gifticon 모바일 쿠폰

![Image]({{"/assets/img/posts/2025-01-06-21-23-21.png" | relative_url }}){: w="200"}

결과

```
Type: CODE128
Data: $$$쿠폰번호$$$
```

바코드 타입과 쿠폰번호만 온다. 나는 유효일자, 사용여부, 사용처 등이 궁금했는데..

### Case3) 카카오톡 모바일 쿠폰

![Image]({{"/assets/img/posts/2025-01-06-21-26-29.png" | relative_url }}){: w="200"}

결과

```
Type: CODE128
Data: $$$쿠폰번호$$$
```

역시 바코드 타입과 쿠폰 번호만 온다...
쿠폰 번호를 통해 정보를 얻어올 수 있는 API를 찾아보자!

## API를 통해 쿠폰번호로 정보 획득

### gifticon

gifticon은 모바일 쿠폰 번호를 통해 웹사이트에서 조회 가능한 곳이 있긴 하나, 보낸 사람 휴대전화 번호를 넣어야 하는데 나는 주로 공공기관으로부터 문자로 받아서 홈페이지를 통한 조회도 잘 안됐다..
API도 딱히 제공하고 있는 것 같지 않다.

### 기프티쇼

기프티쇼도 api를 제공한다고 하는데..

![Image]({{"/assets/img/posts/2025-01-06-21-45-36.png" | relative_url }})

사이드프로젝트에 도입하기엔 돈이 많이 들어보인다..ㅎㅎ

### 카카오 모바일쿠폰

![Image]({{"/assets/img/posts/2025-01-06-21-56-48.png" | relative_url }})

맙소사.. 왜 아직 모바일쿠폰 사용여부 조회 어플이 안만들어졌나 했더니,
정보를 안주기 때문이었구나..?

프로젝트 주제를 바꿔야 할 수도.....ㅎㅎㅎㅎㅎㅎ끄엑 ㅠㅠㅠ
