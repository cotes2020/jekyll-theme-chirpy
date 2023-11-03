---
title: Mongodb atlas로 통합검색 만들기 대작전
date: 2023-06-21
categories: [troubleshooting]
tags: [mongodb, mongodbatlas, python, fastapi, database, index]
---

## 🤔 Problem

맛이슈 검색기능을 제작하면서 text index 만으로는 사용자 친화적인 검색기능을 만드는데 부족함을 느꼈다.

띄어쓰기나 철자가 정확하지 않아도 원하는 레시피를 검색할 수 있게끔 데이터베이스 검색 기능을 최적화해야 했다.

개발자로서 통합검색 쯤은 한번 쯤 만들어보고 싶다는 로망도 있었기에 개발에 착수했다.


## 🌱 Solution

### Search Index 생성하기

몽고디비 Atlas 에서 제공하는 search index 기능을 사용하기로 했다.

1. **몽고디비 아틀라스에 접속한 후 데이터베이스명을 클릭한다.**

   <div markdown="block" style="width: 80%;">
   ![https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-06-21-mongo-db-search-index/1.png](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-06-21-mongo-db-search-index/1.png)
   </div>

2. **Search 탭으로 이동한 뒤, "create index" 를 클릭한다.**

    <div markdown="block" style="width: 80%;">
    ![https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-06-21-mongo-db-search-index/2.png](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-06-21-mongo-db-search-index/2.png)
    </div>

3. **Visual Editor 를 선택하고 Next 로 넘어간다.**

    <div markdown="block" style="width: 80%;">
    ![https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-06-21-mongo-db-search-index/3.png](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-06-21-mongo-db-search-index/3.png)
    </div>

4. **기본으로 지정 되어있는 설정으론 부족하다. "Refine Your Index" 를 클릭한다.**

    <div markdown="block" style="width: 80%;">
    ![https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-06-21-mongo-db-search-index/4.png](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-06-21-mongo-db-search-index/4.png)
    </div>

5. **Analyzer 를 lucene.korean 으로 바꿔주고 Dynamic Mapping 을 On 해둔다.**

    <div markdown="block" style="width: 80%;">
    ![https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-06-21-mongo-db-search-index/5.png](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-06-21-mongo-db-search-index/5.png)
    </div>

    - [lucene](https://ko.wikipedia.org/wiki/%EC%95%84%ED%8C%8C%EC%B9%98_%EB%A3%A8%EC%94%AC)이란

        자바 언어로 만들어진 오픈소스 텍스트 형태소분석기를 말합니다.
        입력된 문자열의 형태소 분석을 수행하고 조사를 제외한 단어만을 남겨놓게 됩니다.

        > 글쓰기**를** <br/>
        > 글쓰기**입니다** <br/>
        > 글쓰기**지만** <br/>
        > 글쓰기**라도** <br/>


    - Dynamic Mapping

        MongoDB나 Elastic Search 같은 NoSql 데이터베이스들이 기본적으로 채택하는 방식입니다.
        
        Dynamic Mapping을 활성화하면 새로운 문서(document)가 데이터베이스에 추가될 때 동적으로 필드와 데이터 유형이 감지됩니다. 

        Dynamic Mapping을 비활성화 하면 새로운 문서의 필드와 데이터 유형을 자동으로 감지하지 않습니다. 대신, 미리 정의된 스키마에 따라 데이터를 저장합니다. (Sql 처럼 작동)

6. **인덱스를 생성한다.**

    나머지는 건들지 않고 인덱스를 생성하면 귀여운 로봇 팔이 index 를 생성해준다. 조금 시간이 걸리니 기다려 주면 된다.

    <div markdown="block" style="width: 80%;">
    ![https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-06-21-mongo-db-search-index/6.png](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-06-21-mongo-db-search-index/6.png)
    </div>

### FastApi 에서 사용하기

몽고디비 프레임워크가 제공하는 aggregete 라는 메서드를 사용하면 여러 검색 조건을 한번에 이용할 수 있다.

우리가 몽고디비 Atlas 에서 Search Index 를 생성해 두었다면, 해당 인덱스를 참조하여 검색결과를 가져다 줄 것이다.

```python
async def search_recipes_with_comments(self, value: str, skip: int = 0, limit: int = 160):
    pipeline = [
        {
            "$match": {
                "$or": [
                    {"recipe_title": {"$regex": value, "$options": "i"}},
                    {"recipe_category": {"$regex": value, "$options": "i"}},
                    {"recipe_description": {"$regex": value, "$options": "i"}},
                    {"recipe_info": {"$regex": value, "$options": "i"}},
                    {"recipe_ingredients.name": {"$regex": value, "$options": "i"}},
                ]
            }
        },
        {"$sort": {"created_at": -1}},
        {"$skip": skip},
        {"$limit": limit},
        {
            "$lookup": {
                "from": "comments",
                "localField": "recipe_id",
                "foreignField": "comment_parent",
                "as": "comments"
            }
        }
    ]
    result = await self.collection.aggregate(pipeline).to_list(length=None)
    return result
```

[https://github.com/YubinShin/matissue-BE/blob/dev/dao/recipe_dao.py](https://github.com/YubinShin/matissue-BE/blob/dev/dao/recipe_dao.py)



### 📎 Related articles

| 이슈명                                                          | 링크                                                                                                                       |
| --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| 검색기능 만들기 3 : 네이버같은 검색기능 만들려면 (Search index) | [https://codingapple.com/course/node-express-mongodb-server/](https://codingapple.com/course/node-express-mongodb-server/) |
