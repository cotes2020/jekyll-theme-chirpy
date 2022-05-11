---
title: Firestore 내용 검색 가능하게 만들기 (Feat. Typesense)
author: Bean
date: 2022-04-09 16:32:00 +0800
categories: [Web frontend, React]
tags: [Catty, Typesense]
layout: post
current: post
class: post-template
subclass: 'post'
navigation: True
cover: assets/img/post_images/catty_cover2.png
---

Firestore는 전체 텍스트 검색이 되지 않는다. [Firebase 공식 문서](https://firebase.google.com/docs/firestore/solutions/search?provider=typesense)에서도 Cloud Firestore은 문서 텍스트 필드 검색이 지원되지 않으며 타사 전용 서비스를 사용하라고 추천하고 있다. 여기서 추천하는 서비스는 `Elastic`, `Algolia`, `Typesense` 3가지 인데 그 중에서 Catty 서비스에는 [Typesense](https://typesense.org/)를 사용하였다. 원하는 기능이 다 있으면서 가격이 가장 합리적이기 때문이다. 뿐만 아니라 사실 돈을 지불하는 부분은 클라우드 서비스이고 도커에 별도로 시스템을 구축한다면 그마저도 내지 않아도 괜찮다. 그렇지만 지금은 별도의 시스템을 만들 여력이 없기 때문에 그냥 돈내고 클라우드 서비스를 사용했다.

## Firestore 대신 Typesense
---

Typesense를 사용하려고 맘먹고 보니 Firestore랑 Typesense를 이중으로 사용할 필요는 없어서 Typesense를 메인으로 사용하고, Firestore은 보조 역할로만 남겨두었다. 좀 더 구체적으로는 메인 DB를 Typesense로 하되 Typesense에 저장이 안되는 data type(object)만 Firestore에 별도로 저장했다. 또한, 크롬 익스텐션과 모바일에서는 Firestore에 리소스를 저장하고 Firebase cloud function을 통해 서버에서 firestore에 들어온 데이터를 typesensedp 넘기도록 구현하였다. 굳이 이렇게 비효율적으로 하는 이유는 크롬 익스텐션과 모바일에서 typesense를 사용할 최적의 방법을 못찾았기 때문인데(..) 사실 좀만 더 서치하면 찾을 수 있을 것도 같으나 우선 별 문제없이 잘 돌아가고 있어 다른 이슈에 비해 우선순위가 많이 밀려있긴 하다.

Typesense의 단점이라면 다른 서비스에 비해 참고 문서를 찾기가 어렵다. 물론 공식 문서가 잘되어 있긴 하나 헷갈리는 부분이 있을 때 다른 서비스에 비해 관련 리소스를 찾기 조금 어려운 편이었다. 그래서 이번 글에서 Typesense를 이용하면서 알게된 내용들을 조금 정리해보려고 한다.

## Typesense 환경 설정
---

우선 Typesense를 사용하기 위해서는 [https://cloud.typesense.org/clusters](https://cloud.typesense.org/clusters) Typesense cloud에 들어가서 collection을 생성해야 한다. collection 이름과 각 field의 이름과 type을 적어둔 다음에 create를 누르면 collection이 생성된다.

<div style="text-align: left">
   <img src="/assets/img/post_images/typesense1.png" width="100%"/>
</div>

다음에 웹앱을 개발하는 코드로 돌아가서 다음의 initialization 코드를 추가한다.

``` javascript
import Typesense from 'typesense';
const TYPESENSE_HOST = process.env.NEXT_PUBLIC_TYPESENSE_HOST;
const TYPESENSE_ADMIN_API_KEY = process.env.NEXT_PUBLIC_TYPESENSE_ADMIN_API_KEY;
const TYPESENSE_SEARCH_API_KEY = process.env.NEXT_PUBLIC_TYPESENSE_SEARCH_API_KEY;
const client = new Typesense.Client({
  'nodes': [{
    'host': TYPESENSE_HOST,
    'port': '443',
    'protocol': 'https',
  }],
  'apiKey': TYPESENSE_ADMIN_API_KEY,
  'connectionTimeoutSeconds': 2,
});
```
이렇게 하면 Typesense를 사용할 기초 작업이 모두 끝났다.

## document 추가/수정/삭제
---

Typesense collection에 데이터가 추가되는 단위는 document이다. 이 collection, document 개념은 firestore와 동일하다.

document를 typesense에 추가/수정/삭제하는 방법은 매우매우 간단하다. 추가와 수정의 경우에는 각각 추가/수정할 내용을 object type으로 만들어서 넣어주면 끝난다.
또한 document 삭제의 경우에도 해당 document id로 한 줄의 코드를 실행하면 끝이다.

* document 추가
  ```javascript
  const resource = {} <-- resource object
  client.collections('resources').documents().upsert(resource)
  ```

* document 수정
  ```javascript
  const resource = {} <-- resource object
  client.collections('resources').documents().update(resource)
  ```

* document 삭제
  ```javascript
  const resource = {} <-- resource object
  client.collections('resources').documents(id).delete()
  ```

## document 검색
---

Typesense의 강점이자 조금 복잡한 부분은 이 검색 부분이다. 아래 짧은 코드로 Collection에서 다양한 조건으로 document를 검색할 수 있다. 요소들을 하나씩 뜯어 살펴보자.

```javascript
const searchField = {
  q,
  drop_tokens_threshold,
  query_by,
  page,
  per_page,
  filter_by,
  sort_by : 'inserted_date:desc',
}
const searchResults = await client
  .collections("resources")
  .documents()
  .search(searchField)
const hits = searchResults.hits
return hits
```

### q와 drop_tokens_threshold
q에 검색하고 싶은 단어를 명시하면 그 단어가 포함된 document들을 가져온다. 이 때 'hello world'처럼 띄어쓰기로 구분해서 검색하고 싶은 단어를 여러개 지정할 수 있다. 이러면 hello와 world를 모두 포함하는 document를 검색하게 된다. 그렇지만 이렇게 띄어쓰기로 구분된 단어들을 모두 포함하는 document를 찾으면 충분한 양의 검색결과가 없을 경우도 많이 생길 것이다. 이 때, 더 세부적으로 들어가서 `drop_tokens_threshold`을 설정하면 내가 원하는 양의 검색 결과가 나올 때 까지 단어들을 빼면서 검색할 수 있다. 즉, hello와 world를 모두 포함하는 document가 많이 없으면 hello를 빼고 world로만으로 검색해보고 반대로 hello로만으로도 검색을 해보고 할 수 있다. 참고로 띄어쓰기로 구분된 단어들을 AND로 연결하는 AND 검색은 지원하지만 OR 검색(hello 또는 world를 포함하는 document)은 지원하지 않는다.

### query_by
query_by는 collection field 중 어떤 field를 기반으로 검색할 것인지를 명시할 수 있는 부분이다. 예로 collection의 document가 아래와 같은 object 일 때,

```javascript
const document = {
  name,
  description,
  memo,
  tags,
  inserted_date
}
```
query_by를 memo로 설정하면 메모에 q의 단어가 포함되어 있는 document들을 검색한다.

### page
page는 우리가 일반적으로 게시판앱에서 볼 수 있는 그 페이지를 생각하면 된다. collection에서 몇번 째 페이지의 document들을 가져올 지 결정한다.

### per_page
per_page는 페이지 당 몇개의 document를 리스트할 지를 결정한다. per_page가 20이면 한 페이지당 20개의 document를 가져오게 된다.

### filter_by
filter_by는 다양한 필터를 설정할 수 있는 term이다. 예로 특정 날짜 범위의 document를 검색하거나, 특정 tag를 포함하고 있는 document만 검색할 수 있다. tag는 string 타입으로 작성해야 한다. 여러개 필드의 태그는 `&&`로 구분한다. 또한 sampleTag1 또는 sampleTag2를 포함하는 tag로 필터링을 하려면 `tags:[sampleTag1, sampleTag2]`와 같이 쓰면 된다.

필터 string 예시는 다음과 같다.
```javascript
  const filter_string = `tags:[sampleTag1, sampleTag2] && name: sampleName`
 ```

### sort_by
sort_by는 이름에서 직관적으로 알 수 있듯이 정렬 조건을 설정하는 부분이다. 정렬 기준이 되는 field명과 오름차순인지, 내림차순인지만 명시하면 된다. 조금 주의할 것은 아직은 string sorting이 안되고 숫자로만 sorting이 가능하다.
