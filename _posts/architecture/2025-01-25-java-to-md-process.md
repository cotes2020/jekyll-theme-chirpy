---
title: "Java(POJO) API 문서 자동화 구축기: 지속 가능한 문서화 프로세스"
categories: [Architecture]
tags: [document engineering, API명세서, API Documentation, Automation]
---

제품의 API 명세서를 작성하는 작업을 맡게 되었습니다!

단순히 API 명세서 형식을 잡아 바로 명세서를 작성하는 방법도 있지만, 늘 **문서 현행화의 어려움**에 대해 고통을 받았던 터라 이 문제를 해결하고 싶었습니다!

**java code내에 API 설명(javadoc)을 작성하면, 원하는 형태의 API 명세서 웹사이트에 반영이 바로 되는 <mark>API 명세서 자동화 프로세스</mark>**를 구축하게 되었습니다.

이번 글에서는 이러한 자동화 과정을 통해 최신 상태로 문서를 유지하는 프로세스를 구축한 경험을 공유하고자 합니다! 😀

## 배경: API 명세서 작성의 도전

일반적으로 REST API는 OpenAPI 명세서 스펙을 활용하여 Swagger와 같은 도구를 통해 문서화합니다. 하지만 저는 Java POJO 인터페이스로 구현된 메서드 명세서를 작성하는 작업이었기에, 이와 같은 표준화된 방식이 부족했습니다.

그래서 전 **"두번의 작업 없이 코드에서 변경사항이 생기면 API 명세서에 자동으로 반영되어 <mark>늘 간편하게 현행화를 유지할 수 있는 것</mark>"** 을 최우선 목표로 설정하고, 다음 세 가지 방안을 고민했습니다.

1. Docusaurus를 사용하여 API 명세서 작성
   - 두 번의 작업(코드 변경, 변경된 내용으로 API 명세서 재작성)이 필요해서 탈락!
2. 코드 내 Javadoc을 작성하여 HTML 형태로 제공
   - 예쁘지 않고, 가독성이 좋지 않아 탈락! (API 명세서에 기존 javadoc으로 구현할 수 있는 것 외 추가하고 싶은 내용이 더 많았습니다.)
3. **코드 내 Javadoc을 작성하고 이를 변환 프로그램을 통해 Docusaurus Markdown 문서로 생성**
   - 당첨!

코드 기반의 Javadoc과 Docusaurus를 연결함으로써 문서의 최신화를 자동화하고, 개발자 경험을 개선할 수 있을 것이라 기대했습니다.

🤭 개인적으로는 평소 Docusaurus가 꽤 이쁘고, 오픈소스여서 사용도 편리하고, 확장성도 좋아 호감이 있었습니다. 또 저는 주로 문서 정리는 markdown으로 하는데, 도큐사우르스는 markdown 기반이고, 또 react기반이라 제가 이용하기에 진입장벽도 낮다고 판단했습니다.

## 단계 1: Docusaurus API 명세서 포맷 정의

우선 가장 먼저는 Docusaurus API 명세서 웹사이트에서 **최종적으로 어떤 포맷으로, 어떤 정보들을 제공할 지**에 대해 정의했습니다.

> 이를 위해 로컬에 Docusaurus 프로젝트를 만들어서, 전체적인 메인 페이지 구조와 인터페이스 상세 페이지 요소를 정리했습니다.

**[제공 포맷]**

- 각 인터페이스마다 페이지를 분리
- 인터페이스 페이지에 method 단위로 api 명세서 작성
- API의 파라미터가 Object인 경우, 해당 Object를 구성하는 VO의 필드 정보를 제공. -> 이를 위해 VO 마다 페이지 생성, 파라미터에 링크를 주어 VO 페이지로 이동
- API의 파라미터 필드가 코드값인 경우, 해당 코드값의 상세 정보를 제공.
  API 명세서에 들어가야 할 요소 -> 이를 위해 ENUM마다 페이지 생성, 파라미터에 링크를 주어 ENUM 페이지로 이동

<details>
<summary>도큐사우르스 명세서 미리보기</summary>

- 인터페이스 페이지

<img src="/assets/img/posts/2025-01-25-15-56-37.png" alt="이미지">

- VO 페이지

<img src="/assets/img/posts/2025-01-25-15-57-41.png" alt="이미지">

- Enum 페이지

<img src="/assets/img/posts/2025-01-25-15-59-54.png" alt="이미지">

</details>

## 단계 2: API 명세서 요소 정의 및 Javadoc 표준 가이드 작성

### 2-1. 상세 요소 정의

API 명세서 포맷이 정해졌으니, API 명세서에 들어가야 하는 상세 요소들에 대한 정의를 진행했습니다.

<summary>상세 요소 보기</summary>

<strong>[Interface 페이지]</strong>

<ul>
  <li>interface 단위
    <ul>
      <li>인터페이스 설명</li>
    </ul>
  </li>
  <li>method 단위
    <ul>
      <li>Category : 메서드 기능 분류 용도</li>
      <li>메서드 설명</li>
      <li>Input Parameters</li>
      <li>Output Parameters</li>
      <li>Usage Sample</li>
      <li>관련 테이블 목록</li>
    </ul>
  </li>
</ul>

<strong>[VO 페이지]</strong>

<ul>
  <li>VO 설명</li>
  <li>관련 테이블 목록</li>
  <li>각 필드 별 설명</li>
</ul>

<strong>[ENUM 페이지]</strong>

<ul>
  <li>ENUM 설명</li>
  <li>각 코드값, 코드키, 코드 설명</li>
</ul>

</details>

### 2-2. Javadoc 표준 가이드 작성

위 상세 요소들을 각 파일 별로 작성하기 위한 Javadoc 표준 가이드를 작성했습니다.

- Interface javadoc 표준

```java
/**
 * <pre>
 * interface 설명
 * </pre>
 *
 * @note
 * <pre>
 * interface 추가 정보
 * </pre>
 */
```

- method javadoc 표준

```java
/**
 * 메서드 설명
 *
 * @category1 Category Level 1
 * @category2 Category Level 2
 *
 * @param	변수명	{@link 데이터타입}	필드설명	(Required)
 * @return	변수명	{@link 데이터타입}	필드설명
 *
 * @related_tables 테이블 목록
 *
 * @exception ErrorName ErrorCode
 *
 * @example
 * <pre>
 * 사용 예제 코드
 * </pre>
 */
```

- VO javadoc 표준

```java
/**
 * VO 설명
 *
 * @related_tables VO와 연관된 Table 목록
 */
public class VO {
	/**
	 * 필드명	(Required)
	 * @see 참조Enum명
	 */
	private String field1;

	// Getter, Setter method..
}

```

- ENUM javadoc 표준

```java
/**
* Enum에 대한 설명
*/
public enum ENUM클래스명 {

	ENUM_NAME(ENUM_VALUE), // 설명
	;

	//생성자 및 method 생략
}
```

## 단계 3: Javadoc -> Docusaurus Markdown 변환 프로그램 개발

Javadoc에서 Docusaurus 형식의 Markdown 문서를 자동으로 생성하기 위해 Java 프로그램을 개발했습니다.

> - 처음에는 빠른 개발을 위해 python 코드로 작성을 계획했었는데요, python에서는 javadoc을 파싱해주는 라이브러리가 별로 성능이 좋지 않았습니다.
> - 그래서 java 코드를 종류 별 인지를 잘 해서 파싱해주고, javadoc도 태그별로 잘 파싱해주는 `javaparser` 라이브러리를 사용하기 위해 java로 코드를 작성하였습니다.

프로그램은 다음 과정을 수행합니다:

1. Javadoc 분석: 코드 내 주석 정보를 파싱하여 필요한 데이터를 추출 (`javaparser` library 활용)
2. Markdown 생성: 추출된 데이터를 Docusaurus 문서 구조에 맞게 변환
3. 파일 저장: 생성된 Markdown 파일을 Docusaurus 문서 디렉토리에 저장

> Interface, VO, Enum class 코드의 javadoc과 markdown 형식이 다르기 때문에 각각의 프로그램으로 분리하였습니다.

나중에 **cicd를 통해 자동으로 수행되는 구조**로 수행되어야 한다는 점을 고려하여 개발에 반영하였습니다. 이를 위해 input-output 값을 CLI를 통해 argument를 받아 수행될 수 있도록 개발을 했습니다.

argument로 `변경대상 파일경로`와 `markdown 문서 저장 경로`를 받도록 하였습니다.

- Enum 변환 프로그램 예시

```java
public static void main(String[] args) {
  if (args.length < 2) {
          System.out.println("Usage: java EnumToMdConverter <inputFilePath> <outputDirectory>");
          return;
      }
      String inputFilePath = args[0]; // ENUM 파일 경로
      String outputFilePath = Paths.get(args[1], "code-mapping").toString(); // md파일 생성 경로

      convertEnumToMarkdown(inputFilePath, outputFilePath);
   // 생략
}
```

이 프로그램을 통해 가이드에 맞게 Javadoc만 작성하면 추가적인 수작업 없이 Docusaurus 문서가 자동으로 생성될 수 있도록 구현했습니다.

저는 여기까지 진행 한 후에 javadoc 가이드에 따라 javadoc을 작성하고, 변환 프로그램을 이용해서 변환이 잘 되는지, 추가/수정해야 할 사항에 대해 보완해가며 진행하였습니다.

## 단계 4: CI/CD 기반 자동화 프로세스 구축

이제 Javadoc까지 모두 작성한 후에 최종 작업이 남았습니다!

바로 문서의 최신화를 자동화 하기 위한 CI/CD 파이프라인 구축입니다!

프로세스는 다음과 같습니다:

![Image]({{"/assets/img/posts/2025-01-25-16-00-32.png" | relative_url }})

- Javadoc 작성 및 코드 배포: 개발자가 코드와 함께 Javadoc을 작성하여 저장소에 커밋
- 변환 프로그램 실행: CI/CD 파이프라인에서 Javadoc -> Markdown 변환 프로그램 실행
- Docusaurus에 반영: 변환된 Markdown 파일을 Docusaurus 기반 API 명세서 저장소에 반영
- 배포 자동화: Docusaurus 문서를 웹사이트 서버에 배포

이 과정을 통해 새로운 API 변경 사항이 문서에 실시간으로 반영되며, 최신화된 상태를 유지할 수 있었습니다.

## 느낀점 및 결론

이번 작업은 단순히 API 명세서를 작성하는 것을 넘어, 문서의 지속 가능한 관리와 최신화를 위한 기반을 마련한 점에서 큰 의미가 있었습니다. 특히, 자동화 프로세스를 통해 문서와 코드 간 불일치를 최소화하고, 개발자들에게 신뢰할 수 있는 문서를 제공할 수 있으리라는 기대를 했습니다.

사실 요즘에는 REST API 기반으로 하여, OpenAPI 명세서를 활용할 수 있는 점이 많습니다.
특히 FastAPI를 사용한다면 바로 swaggerUI를 제공받을 수 있고, 그렇지 않더라도 Controller code와 작성한 javadoc을 통해 OpenAPI 명세서를 생성하여 OpenAPI spec을 통해 Swagger연동, 그 외 다양한 툴을 통해 웹사이트로 구현하는 작업이 용이하기도 합니다.

하지만 꼭 REST API 기반이 아니더라도 코드 기반의 API명세서 자동화 프로그램을 만들어 볼 수 있음을 진행해볼 수 있어 재밌는 작업이었습니다 ㅎㅎ
Java 기반 프로젝트에서 API 명세서를 작성하고 관리하는 데 고민 중인 분들에게 유용한 참고자료가 되길 바랍니다!
