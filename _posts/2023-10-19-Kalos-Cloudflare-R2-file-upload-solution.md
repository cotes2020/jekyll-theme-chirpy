---
title: Kalos, Cloudflare R2 file upload solution # 파일 명은 영어만 써야하지만, 여기는 한글이 가능
author: chungjung # \_data/authors.yml 에 있는 author id (여러명인경우 authors: [id1, id2, ...])
date: 2023-10-19 10:33:00 +0900 # +0900은 한국의 타임존  (날짜가 미래인경우 빌드시 스킵함.)
categories: [DevOps, cloud] # 카테고리는 메인, 서브 2개만 설정 가능 (띄어쓰기 가능)
tags: [cloudflare R2, kalos, image upload, cloudflare, s3, R2] # 태그는 개수제한 X (띄어쓰기 가능)
---

현재 클라우드 파일스토리지라고 하면 가장 먼저 떠오르는 솔루션은 단연코 AWS의 S3라고 말할 수 있을 것입니다. 그만큼 수많은 사람들이 AWS S3를 사용하고 있으며, S3는 그에 걸맞게 탄탄한 기능과 일정 수준의 보안을 제공합니다. 하지만 비싸다는 단점을 가지고 있습니다. 정확히 따지면 AWS S3에는 Egress fee라는 요금이 존재하는데, 이는 S3에 존재하는 파일을 요청해서 다른 사람이 파일을 요청할 때, 다운로드 용량에 따른 요금을 부과하는 방식입니다.

이 Egress fee가 바로 S3에서 AWS에서 벌어들이는 소득의 대부분입니다.

![s3 fee](/assets/img/2023-10-19-Kalos-Cloudflare-R2-file-upload-solution/S3%20fee.png)

위의 모식도는 2023/10/07 기준 Cloudflare 사이트에 있는 AWS S3와의 비교한 자료를 바탕으로 만들었습니다. 다이어그램을 보게되면, upload file을 할 때 AWS network 망을 사용하는데에 비용이 들지 않고 대신에 1GB당 storage를 사용하는 대가로 $0.0023의 비용이 발생하는 것을 볼 수 있습니다.

반면에 download file을 하게 된다면1GB당 AWS netowork를 사용하는 대가로 $0.09 달러를 지불해야 함을 알 수 있습니다. 이 말은 즉 1GB의 이미지가 한달에 100만번 다운로드 된다고 했을 때 90000달러를 지불햐애 함을 알 수 있죠. 물론 이는 단순 계산일 뿐, 이런 저런 할인이나 정책을 잘 사용하면 이 가격의 반값, 혹은 그 이상까지 깎을 수 있을 것입니다. 그렇다 하더라도 instargram, facebook같이 엄청난 양의 파일 트래픽을 감당해야하는 서비스들에게는 천문학적인 금액이 청구 될 것입니다.

## S3의 대항마 R2

이러한 문제를 해결하고자 Cloudflare에서는 S3와 호환되는 인터페이스를 가지는 R2라는 서비스를 만들었습니다. R2는 기존 S3의 개발 툴을 거의 그대로 쓸 수 있어서 S3를 쓰던 기업들이 쉽게 R2로 마이그레이션 할 수 있게 만들어졌으며, 가격은 훨씬 저렴합니다.

![R2 fee](/assets/img/2023-10-19-Kalos-Cloudflare-R2-file-upload-solution/R2%20fee.png)

## 💫Kalos system

Kalos system은 저희 술주정 사이드 프로젝트 팀에서, 위와 같은 이유로 R2를 사용하다가, 사용자에게 이미지 파일을 업로드 하는데 있어서, R2 upload key노출을 없에는 것과 동시에 백엔드에 용량이 큰 파일을 보내지 않고 유저가 파일을 업로드하는 방법을 고민하다가 직접 만들게 된 R2용 파일 업로드 시스템입니다.

Kalos system을 설명하기에 앞어서 통상적으로 주로 이미지를 업로드 하는 방법은 어떤 방법이 있는지 먼저 알아보겠습니다.

## How to Upload file

잠깐 설명하기에 앞서서, 이 파트에서는 multipart업로드나 binary upload에 대한 내용은 다루지 않을 예정입니다. 단순히 어떤 흐름으로 파일이 업로드 될 수 있는가에 대해서만 설명합니다.

### Client saves files to R2

![client saves files to R2](/assets/img/2023-10-19-Kalos-Cloudflare-R2-file-upload-solution/save%20file%20on%20client.png)
첫 번째는 file을 직접 클라이언트가 R2에 저장하고, 저장한 URL을 백엔드로 보내서 백엔드가 DB에 파일의 url과 연관되는 데이터를 저장하는 방식입니다. 사실 이 방법은 절대로 쓰면 안되는 방법입니다. Frontend가 R2에 access할 수 있는 secret을 보유하고 있기 때문에, 이대로 사용하게 된다면 R2 Access token이 탈취당해서 마음만 먹으면 공격이 가능하기 때문입니다.

### Backend saves files to R2

![save file on backend.png](/assets/img/2023-10-19-Kalos-Cloudflare-R2-file-upload-solution/save%20file%20on%20backend.png)
두 번째 방법은 backend server에 클라이언트가 저장하고자 하는 파일을 보내면, Backend가 직접 cloudflare R2에 파일을 올리고, 올린 파일 url을 클라이언트에 다시 리턴해주는 방식입니다. 클라우드 스토리지에 접근할수 있는 키 값을 외부에 노출하지 않기 때문에 보안성이 뛰어나, 현재는 이와 같은 방식이 가장 보편적으로 사용되고 있습니다. 다만 단점도 있는데, 파일의 용량이 커질경우, 예를 들어서 이미지 고해상도 파일같은 경우 백엔드 서버를 한번 더 거쳐야 하기 때문에 처리해야하는 용량이 커지고, 속도가 느려지는 문제가 발생할 수 있습니다.

### Direct Creator Upload

![diract creater upload.png](/assets/img/2023-10-19-Kalos-Cloudflare-R2-file-upload-solution/diract%20creater%20upload.png)

세 번째 방법은 Direct Creator Upload라는 방법으로 cloudflare images 솔루션에서만 제공하는 방법입니다.

Direct Creator Upload는 클라이언트에서 백엔드로 이미지를 업로드하고 싶다는 요청을 먼저 보내면, 백엔드에서 cloudflare images에 접근이 가능한 키를 이용해서 일회성 upload url을 달라는 요청을 보냅니다. 그렇게 받은 일회성 upload url을 클라이언트에 전달하고, 이를 이용해서 클라이언트는 직접 이미지를 저장하고 이미지 url을 받아서 백엔드에 전달해 저장합니다.

## Why kalos?

첫 번째, 두 번재 방법은 각각 장단점이 있다고 치고, 왜 세 번째 방법을 그대로 쓰지 않았냐는 의문이 있을 수 있습니다.

몇 가지 이유가 있는데, 첫 번째로 사이드프로젝트로 만든 앱인 만큼 가능하면 가격을 절감하고 싶었습니다. 위에서 말한 Direct Creator Upload는 cloudflare images에만 제공되는 기능으로 한달에 $5를 청구애햐 합니다. 두번 째 이유는 현재 프로젝트에서 Terrafrom을 사용해서 클라우드 서비스들의 리소스들을 한 군데 모아서 관리하고 있는데, R2는 이러한 Terrafrom라이브러리가 존재하지만 images는 아직 존재하지 않습니다. 그래서 이러한 점을 보완하고자, R2를 사용하되 cloudlfare images에서만 제공하는 Direct creator upload방식을 사용하고자 Kalos system을 만들게 되었습니다.

그러면 이제부터는 칼로스 시스템이 어떻게 구성되어 있고 어떤 흐름으로 이미지를 저장하는지에 대해서 작성하겠습니다.

## Cloudflare Worker

칼로스 시스템에 대해서 설명하기에 앞서서 먼저 Cloudflare worker에 대해서 설명해야 합니다. Cloudflare worker는 한마디로 표현하자면 CDN네트워크에 배포할수 있는 serverless 어플리케이션입니다. 즉 worker가 인식할 수 있는 스크립트를 배포하면 전 세계에 클라우드 플레어가 가지고 있는 CDN에 스크립트가 배포되고, 이들은 마치 AWS lambda처럼 작동하게 됩니다 .

Cloudflare worker의 가장 큰 특징은 다른 cloudflare solution들과 연동이 가능하다는 겁니다. 특히 R2와의 연결성도 뛰어나서, 아래 코드와 같이 Terraform 기준 별도의 API key없이 R2와 연결할 수 있습니다.

```hcl
resource "cloudflare_worker_script" "kalos_r2_script" {
  provider = cloudflare.worker

  name = "kalos-r2-script"
  account_id = var.cloudflare_account_id
  content = file("file path")

  r2_bucket_binding {
    name = "SULJUJEONG_IMAGES_BUCKET"
    bucket_name = cloudflare_r2_bucket.suljujeong_images_bucket.name
  }
}
```

이 뿐만 아니라 기존의 cloudflare zone과도 연계가 가능해서 아래처럼 worker에 접근할 수 있는 커스텀 url을 직접 설정할 수 있습니다.

```hcl
resource "cloudflare_worker_domain" "kalos_r2_domain" {
  provider = cloudflare.worker

  zone_id = var.cloudflare_suljujeong_zone_id
  account_id = var.cloudflare_account_id
  hostname = "your-sub-domain.example.com"
  service = cloudflare_worker_script.kalos_r2_script.name
}
```

## Cloudflare worker in Kalos

칼로스 시스템은 kalos access, 나머지 하나는 kalos r2라는 두 개의 cloudflare worker를 사용해서 구현되었습니다. 그렇다면 각각 무슨 역할을 하는지에 대해서 간략하게 설명해보겠습니다.

### Kalos Access

![Kaloss access.png](/assets/img/2023-10-19-Kalos-Cloudflare-R2-file-upload-solution/Kaloss%20access.png)

Kalos Access는 업로드 url을 제공하는 역할을 합니다.

Backend에서 Body로 user , action, image_id를 Body로 제공하고 Header에 Kalos와 공유하는 HMAC 시크릿으로 서명을 생성해서 header에 제공합니다. Kalos Access 서버는 HMAC서명이 올바른지 검증하고, 올바르다고 판단이 되면 Body로 온 정보를 이용해서 2~3분짜리의 JWT토큰을 만들고 이 토큰을 url 쿼리에 붙혀서 업로드 url를 생성하고 response Body에 실어서 보냅니다.

### Kalos R2

![Kaloss r2.png](/assets/img/2023-10-19-Kalos-Cloudflare-R2-file-upload-solution/Kaloss%20r2.png)

Kalos Access에서 생성된 URL로 요청하게 되면 Kalos R2로 요청이 들어가게 됩니다. Kalos R2는 다음과 같은 역할을 합니다.

1. JWT Token 검증
2. Action 검증 (UPLOAD와 DELETE가 있으며 각각에 따라 동작방식과 요청해야하는 HTTP method가 다릅니다.)
3. R2에 실제로 접근해서 작업 수행

위와 같은 역할을 통해서 이미지를 upload혹은 삭제하고, 각 작업이 수행된 이미지의 url을 리턴하게 됩니다.

## Kalos Flow

그렇다면 이제 전체적으로 Kalos system이 어떻게 동작하는지에 대한 flow를 보여드리겠습니다.

![Kalos workflow .png](/assets/img/2023-10-19-Kalos-Cloudflare-R2-file-upload-solution/Kalos%20workflow%20.png)

1. Client에서 Backend에 이미지를 생성(혹은 삭제)하고 싶다는 요청을 보냅니다.
2. 요청을 받은 백엔드는 유저의 권한을 검증합니다.
3. 권한 검증이 통과되면, Kalos access에 이미지 생성에 관한 payload와 HMAC 키를 이용해서 보낸 백엔드가 인증된 백엔드인지, 위변조 되지 않은 값인지 검사합니다.
4. 검증이 통과되면 Kalos R2와 공유하는 JWT secret을 이용해서 Kalos R2서버에서 행동 하기 위해 필요한 데이터를 포함한 2~3분의 유효기간을 가지는 JWT Token이 포함되어 있는 URL을 리턴합니다.
5. 백엔드는 4에서 받은 URL을 다시 client로 전달합니다.
6. client는 이 url과, body에 image를 바이너리 형식으로 Kalos R2에 요청합니다.
7. Kalos R2는 받은 JWT의 변조 여부와, 유효기간등의 정보를 확인하고, 올바른 형태의 파일 형태인지 등을 검증한 뒤에 통과하면 R2에 저장합니다.
8. 최종 저장된 image의 url을 클라이언트에 리턴합니다.

이렇게 생성된 url을 클라이언트는 다시 백엔드에 보냄으로써 백엔드는 직접 이미지를 업로드 하지 않아도, 이미지를 저장할 수 있습니다.

이와 같은 방식을 이용한다면 직접 API키를 노출하지 않으면서도 클라이언트가 파일을 직접 업로드 할 수 있게 됩니다.

## Limitation

현재 이와 같은 방식을 사용해서 구현한 칼로스 시스템에서는 아직 해결되지 않은 취약점과 한계가 있습니다.

1. 3번에서 보내는 요청이 탈취당해서 Reply attack을 당할 수 있습니다. 이에 대한 부분은 추후 Timestamp를 payload에 포함하고, cloudflare worker자체에서 지원하는 KV 스토어에 각 요청에 대한 HMAC 서명을 저장해서 재전송 시 요청을 거부하는 방법으로 해결 할 생각입니다.
2. 4번에서 아주 짧은 유효기간을 가지는 JWT를 사용하지만, 완벽한 일회성은 아닙니다. 이 때문에 스크립트를 이용해서 짧은시간동안 매우 많은 요청이 들어오는 식으로 작동할 우려가 있습니다. 이 문제도 KV 스토어를 이용해서 중복을 방지하려고 합니다.

## 💫Deploy Kalos

현재 칼로스 배포는 완벽히 자동화되어 있지는 않지만, 선언적으로 관리되고 있어서 같이 소개해 보려고 합니다. 다만 코드를 첨부할 때에는 코드가 완전히 공개되는 것을 막기 위해서 변수명이나 민감한 정보에 대해서는 수정을 가했습니다.

![Provisoning workflow .png](/assets/img/2023-10-19-Kalos-Cloudflare-R2-file-upload-solution/Provisoning%20workflow%20.png)

우선 현재 칼로스 시스템은 Terraform을 이용해서 프로비저닝 되고 있으며, 또한 cloudflare worker에 올라가 있는 script를 배포하는 것도 Terraform을 이용해서 진행되고 있습니다.

## Defining cloudflare resource

먼저 각 cloudflare 리소스를 정의하고 프로비저닝 하는 코드부터 보겠습니다.

```hcl
provider "cloudflare" {
  alias = "r2"
  api_token = var.cloudflare_r2_access_token
}

resource "cloudflare_r2_bucket" "my_bucket" {

  provider = cloudflare.r2

  account_id = var.cloudflare_account_id
  name       = "your bucket name"
  location   = "APAC"
}
```

위에 코드는 R2를 프로비저닝 하는 코드 입니다. 원래 R2를 프로비저닝 하는 방법은 현재 기준 크게 2가지가 있습니다. S3 호환 API를 이용하는 방법, 그리고 cloudflare에서 자체 제공하는 provider를 이용하는 방법이 있으나 저희는 후자를 택했습니다.

실제로 지원하는 기능은 AWS S3 provider를 이용하는 것이 더 많지만, 우선은 상세한 기능이 필요하지 않아서, 시범적으로 후자를 선택했으나, 이는 추후 마이그레이션 할 예정입니다.

```hcl
provider "cloudflare" {
  alias = "worker"
  api_token = var.cloudflare_worker_api_token
}

resource "cloudflare_worker_domain" "my_domain" {
  provider = cloudflare.worker

  zone_id = var.cloudflare_suljujeong_zone_id
  account_id = var.cloudflare_account_id
  hostname = "access my domain"
  service = cloudflare_worker_script.my_script.name
}

resource "cloudflare_worker_script" "my_script" {
  provider = cloudflare.worker

  name = "my-script"
  account_id = var.cloudflare_account_id
  content = file("worker script file path")

  secret_text_binding {
    name = "JWT_SECRET"
    text = var.jwt_secret
  }

}
```

두 번째로, cloudflare worker에 대한 부분 입니다. 크게 두 파트로 나누어 지는데 **Domain이랑 binding하는 부분, script와 env를 worker에 binding** 하는 부분입니다.

```hcl
resource "cloudflare_worker_domain" "my_domain" {
  provider = cloudflare.worker

  zone_id = var.cloudflare_suljujeong_zone_id
  account_id = var.cloudflare_account_id
  hostname = "access my domain"
  service = cloudflare_worker_script.my_script.name
}
```

이 부분이 cloudflare worker에 subdomain을 할당하는 부분입니다. hostname에 원하는 subdomain을 넣으면 binding 됩니다.

```hcl
resource "cloudflare_worker_script" "my_script" {
  provider = cloudflare.worker

  name = "my-script"
  account_id = var.cloudflare_account_id
  content = file("worker script file path")

  secret_text_binding {
    name = "JWT_SECRET"
    text = var.jwt_secret
  }

}
```

이 부분은 worker에 대한 script를 지정하고, 그 script내에서 사용할 secert을 주입하는 역할을 합니다. 이 파트에서 R2 bucket binding도 수행할 수 있습니다.

## Deploy worker script

앞의 스크립트에서 `content = file("worker script file path")` 부분이 바로 직접적으로 worker에 올라갈 스크립트를 지정하는 부분입니다. 다만 이 치명적인 문제점이 하나 있는데, 여러개의 파일은 인식하지 못하며, 심지어 외부 라이브러리도 인식할 수 없고, worker script가 실행되는 환경은 nodejs runtime 환경도 아니라는 점입니다. 이러한 부분이 야기하는 문제점은 다음과 같습니다.

1. nodejs 기반의 api를 사용할 수 없다.
2. 외부 라이브러리를 일절 사용할 수 없다.
3. TS를 사용하기가 까다롭다.

일단 2번과 3번은 그나마 해결하기가 쉽습니다. JS에는 Bundler라는 개념이 있습니다. 대표적인 Bundler로는 webpack과 rollup이 있습니다. 현재 기준 webpack을 이용하는 방법만 나와 있지만, 어짜피 원리는 같기 때문에 kalos에서는 rollup.js 번들러를 사용했습니다.

```jsx
import typescript from "@rollup/plugin-typescript";
import { nodeResolve } from "@rollup/plugin-node-resolve";
import commonjs from "@rollup/plugin-commonjs";
import nodePolyfills from "rollup-plugin-polyfill-node";
import globals from "rollup-plugin-node-globals";

export default {
  input: "src/main.ts",
  output: {
    dir: "dist/",
    format: "iife"
  },
  plugins: [
    typescript(),
    nodeResolve(),
    commonjs(),
    nodePolyfills({ crypto: false }),
    globals()
  ]
};
```

아래는 Rollup.js 파일의 설정입니다.

여기서 주목해야 할 점은 format과 plugins파트입니다.

### iife format

format에서 iife로 설정했다는 것은 즉시 실행함수를 의미합니다. 원래는 rollup 같은 번들러에서는 번들링을 할떄 cjs나 다른 형태로 번들링 format을 지정하면 라이브러리 부분은 require()같은 형태로 불러오게 나타납니다. 이는 외부 라이브러리를 불러올 수 있는 환경에서는 문제가 되지 않으나, cloudflare worker에서는 아예 실행조차 되지 않는 문제를 야기합니다 .

반면에 iife형태로 설정했을 경우, 라이브러리에 있는 모든 코드들까지 한번에 눌러 담아서 하나의 js파일로 만들어줍니다. 즉 dist/main.js 파일 하나만 있으면 외부 라이브러리의 주입없이 실행을 시킬 수 있습니다.

### plugins

플러그인은 rollup을 번들링할 때 필요한 부분들을 주입해줍니다.

`plugins: [typescript(), nodeResolve(), commonjs(), nodePolyfills({ crypto: false }), globals()]` 이 부분에서 많은 문제들이 해결됩니다. typescript를 번들링 하면서 동시에 javascript로 컴파일 해주며, nodePolyfills같은 플러그인들이 nodejs환경이 아닌 환경에 대해서도 nodejs api를 사용할 수 있도록 설정해줍니다.

그러나 여기서도 문제가 하나 있습니다. `nodePolyfills` 플러그인, 그리고 그와 유사한 기능을 하는 나머지 플러그인도 HMAC을 이용하는 부분에서 사용하는 crypto api에 대한 구현이 제대로 되어 있지 않습니다. 실제로 이 문제는 아직도 해결되지 않고 github issue에 등록되어 있습니다.

### nodejs free lib

그래서 이를 해결하기 위해서 그냥 nodejs에 의존하지 않는 crypto 라이브러리를 사용했습니다. 실제로 이를 이용하니까 crypto api쪽 관련된 문제가 해결된 것을 확인 할 수 있었습니다.

이제 마지막 단계만 남았습니다. rollup을 이용해서 /src 폴더에 있는 파일들을 한번에 번들링해서 /dist/main.js에 넣고 이 경로를 `content = file("worker script file path")` 부분에 등록해주면, terraform apply를 할 때마다 만약에 파일이 바뀌었으면 배포를 하게 됩니다.

## Outro

지금까지 파일 업로드 시스템 Kalos system에 대해서 간단하게 설명해보았습니다. 사이드 프로젝트에 붙혀서 만드는 것 치고는 규모가 많이 커져버린 바람에 보완할 점들이 많지만 이러한 방법을 시도해 볼 수 있다, 정도의 지침으로 봐주셨으면 좋겠습니다.
