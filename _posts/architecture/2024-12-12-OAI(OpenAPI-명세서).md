---
title: "OAS(OpenAPI 명세서)에 대해 쉽게 알아보자"
categories: [Architecture]
tags: [API, Architecture, OAS]
---

REST API 구조는 우리가 개발하면서 흔히 많이 보는 구조이죠?

이러한 REST API 구조를 document로 만들어서, 아주 다양하게 활용할 수 있도록 해주는 표준이 존재합니다! 그것이 바로 openAPI 명세서인데요, 이 표준에 따라 정리해두기만 해도 아주 다양한 방식으로 바로 적용해서 활용할 수 있기 때문에 이 표준을 따른 것이 좋습니다.
예를 들어 openAPI 명세서 기준에 따라서 정리해둔다면, API와 LLM과 연동하는 모듈에 openAPI 명세서를 로딩함으로써 즉각적으로 연동할 수 있죠. 그 외에도 OpenAPI 명세서를 통해 실제 구현 코드를 생성할 수도 있고, 설계서로 바로 생성이 가능합니다.

이렇게 많은 곳에 활용할 수 있는 OpenAPI 명세서에 대해 알아볼까요?

---

### 1. OAS 란?

> OAS는 OpenAPI Specification의 약자로,
> "OpenAPI 명세서" 라는 의미이다.

- 여기서 사용되는 openAPI는 기존의 "개방된 API"(ex.지도API, 날씨조회API 등..)의 개념과는 조금 다르다.
- `OpenAPI` 또는 `OAS(OpenAPI Specification)`로 불리며,
- RESTful API를 **정해진 표준 규칙**에 따라 **API Spec**을 `json` 혹은 `yaml` 로 표현하는 방식을 의미한다.
- ==> **"Restful API 디자인의 정의 방법의 표준"** 으로 정리할 수 있다.
- aka. `OpenApi 3.0 Specification`

### 2. OpenAPI vs Swagger

![Image]({{"/assets/img/posts/2024-12-12-13-23-17-i23pmg4.png" | relative_url }})

- Swagger라는 제품을 SmartBear라는 회사에서 구매했고,
- SmartBear가 OpenAPI Initiative라는 회사에 Swagger라는 제품을 기부하며
- Swagger라는 제품(개념)이 OpenAPI Specification으로 이름이 변경되었다.
- ☝ 하지만 swagger는 여전히 사용되는 용어이며, openAPI와는 다른 의미로 사용된다!
- **OpenAPI**
  - "개념"
  - RESTful API 디자인에 대한 명세 정의의 표준(specification)!
- **Swagger**
  - "제품"
  - openAPI를 실행하기 위한 도구 (SmartBear사의 tool)
  - Swagger는 API들이 갖고 있는 명세을 정의할 수 있는 툴들 중 하나
- ⛏ Swagger의 여러 도구
  - `Swagger Editor` : 브라우저 기반의 편집기, OpenAPI Spec을 쉽게 작성할 수 있게 도와줌
  - `Swagger UI` : OpenAPI spec문서를 브라우저에서 확인할 수 있게 해줌, API Test도 가능
  - `Swagger Codegen` : OpenAPI spec에 맞게 Server나 Client의 stub code생성

### 3. Open API 구조 및 튜토리얼

- **전체 구조**

  ![Image]({{"/assets/img/posts/2024-12-12-13-24-12-i24pmg4.png" | relative_url }})

- **튜토리얼** : YAML 방식(swagger 공식문서 추천방식)
  - [OpenAPI 3.0 Tutorial | SwaggerHub Documentation](https://support.smartbear.com/swaggerhub/docs/tutorials/openapi-3-tutorial.html)

### 4. 예시

```yaml
openapi: 3.0.0 # openAPI 3.0 버전임을 명시
info: # API 버전 및 간략 설명
  version: 1.0.0 # required
  title: Sample API # required
  description: A sample API to illustrate OpenAPI concepts # optional

servers: # 목록으로 여러개의 url 사용 가능
- url: https://example.io/v1 # API endpoint 경로 정의

paths: # API 경로 및 요청, 응답 파라미터 정의
# [GET https://example.io/v1/list?id=1&isExist=true] 을 표현
  /list:
    get: # method 정의 (get, post ..)
      description: Returns a list of stuff # optional

      # 쿼리 스트링 방식
      parameters: # query paramters는 optional
        - name: id # 파라미터 명
          in: query # query string에 들어가는 요소임을 기재
          schema:
            type: integer # 데이터 타입
        - name: isExist
          in: query
          description: tell item is exist or isn't exist # optional
          schema:
            type: boolean

      responses: # reponses는 필수
        '200': # 성공응답 케이스
          description: Successful response
          content:
            application/json: # 응답 콘텐트 타입
              schema:
                type: array # 응답 데이터 타입
                items: # array의 각 요소
                  type: object
                  required: # 필수 요소 정의
                    - id
                    - name
                  properties: # object의 각 파라미터의 이름과 타입 정의
                    id:
                      type: integer
                    name:
                      type: string
                    isExist:
                      type: boolean
        '400': # 오류 응답 케이스 (예시 - 400)
          description: Invalid request
          content:
            application/json:
              schema:
                type: object
                properties:
                  message: 'unsuccessful request.' # 에러 메시지
                    type: string


# [POST https://example.io/v1/list] 을 표현
    post: # method 정의
      description: Lets add a new stuff
      requestBody: # POST, PUT and PATCH 과 같은 메서드에서의 Request Body 방식
        required: true
        content:
          application/json:
            schema:
              type: object
              required:
                - id
                - name
              properties:
                id:
                  type: integer
                name:
                  type: string
                isExist:
                  type: boolean

      responses:
        '200':
          description: Successfully created a new stuff

        '400':
          description: Invalid request
          content:
            application/json:
              schema:
                type: object
                properties:
                  message: 'FAIL!'
                    type: string

# [GET https://example.io/v1/list/{id}] 을 표현
  /list/{username}:
    get:
      description: Obtain information about an list from a unique id
      parameters:
        - name: id
          in: path # Path 파라미터 방식
          required: true
          schema:
            type: integer

      responses:
        '200':
          description: Successfully returned an stuff
          content:
            application/json:
              schema:
                type: object
                properties:
                  id:
                    type: integer
                  name:
                    type: string
                  isExist:
                    type: boolean

        '400':
          description: Invalid request
          content:
            application/json:
              schema:
                type: object
                properties:
                  message:
                    type: string


#/components/schemas/Stuff

  #  ----- Added lines  ----------------------------------------
  schemas:
    Stuff:
      type: object
      required:
        - id
        - name
      properties:
        id:
          type: integer
        name:
          type: string
        isExist:
          type: boolean
  #  ---- /Added lines  ----------------------------------------
```

- 만약 반복적으로 사용되는 되어 공통으로 사용할 수 있는 부분은 `component`에 정의하여 사용 가능하다.
- 예를 들어 위 소스에서 `schemas` 부분에 선언된 부분은 아래 세 곳에서 공통으로 사용된다.
  - `[GET https://example.io/v1/list?id=1&isExist=true]` 의 `responses > '200' > 'content' > 'application/json' > 'schema' > 'items'`부분
  - `[POST https://example.io/v1/list]`의 `requestBody > 'content' > 'application/json'` 부분
- 아래 코드와 같이 공통의 부분을 component로 선언한다.

```yaml
components: # component 생성 선언
  schemas: # 중간 분류명 정의 (무엇이든 가능)
    Stuff: # component 명 정의
      # 반복적으로 사용되는 공통 부분 정의
      type: object
      required:
        - id
        - name
      properties:
        id:
          type: integer
        name:
          type: string
        isExist:
          type: boolean
```

- 사용은 필요한 부분에 아래와 같이 선언하여 사용할 수 있다.

```yaml
$ref: "#/components/{중간분류명}/{컴퍼넌트}"
```

````yaml
# [GET https://example.io/v1/list?id=1&isExist=true] 을 표현
/list:
  get:
    # .. 중략 ..
    responses:
      "200":
        description: Successful response
        content:
          application/json:
            schema:
              type: array
              items:
                #  ----- use component  ----------------------------------------
                $ref: "#/components/schemas/Stuff"
                #  ----- use component  ----------------------------------------
    # .. 중략 ..

  # [POST https://example.io/v1/list] 을 표현
  post:
    description: Lets add a new stuff
    requestBody:
      required: true
      content:
        application/json:
          schema:
            #  ----- use component  ----------------------------------------
            $ref: "#/components/schemas/Stuff"
            #  ----- use component  ----------------------------------------
    # .. 중략 ..```
````

- **참고**

> - [YAML full source](https://github.com/OAI/OpenAPI-Specification/blob/main/examples/v3.0/api-with-examples.yaml)
> - [JSON full source](https://github.com/OAI/OpenAPI-Specification/blob/%E3%85%87main/examples/v3.0/api-with-examples.json)
