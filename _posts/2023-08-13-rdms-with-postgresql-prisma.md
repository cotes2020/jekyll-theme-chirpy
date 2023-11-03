---
title: Postgresql 과 prisma 로 배우는 관계형 데이터 베이스
date: 2023-08-13
categories: [blog]
tags: [postgresql, prisma, rdms, orm]
---

## 🤔 개론

### PostgreSQL 을 선택한 이유

전장의 안개라는 지도 사용 프로젝트를 하면서 **지리정보 저장에 용이**하다는 PostgreSQL 을 사용해보기로 했다. MySQL 보다 복잡한 **대규모 분석 프로세스에 적합**하다는 부분도 추후 서비스 고도화 시 사용자들의 활동 기록을 분석할 때에 강한 이점이 될 것 같았다. 엘리스에서 수정코치님께 배웠던 내용처럼 **JSON 형식의 컬럼**을 지정할 수 있다는 점도 매력적이었다. FastAPI 강의를 들었던 Amigoscode 선생님의 유튜브 강의로 차근차근 명령어를 학습하며 60 강까지 수강 완료했다.

### Prisma 를 선택한 이유

SQL 문법에 익숙해질 겸 클라이언트 코드를 직접 작성해보는 것이 좋을지, ORM 을 쓰는 것이 좋을지 고민이 많았다.
그러다가 최신 Nest.js 강의를 들어보면서 사용해본 Prisma 의 사용 경험이 압도적으로 좋았기 때문에 스키마만 정해주면 프리즈마가 Postgresql 문법을 자동으로 생성해주기도 하고, 마이그레이션도 압도적으로 편리하고, 데이터를 보고 싶을 때 별도의 프로그램 설치 없이 명령어 한 번에 GUI 로 편하게 확인할 수 있는 점이 관계형 DB 입문자인 나에게 심리적 안정을 주었다.
바퀴를 굳이 다시 발명하지 말라는 말도 있듯이, 돌아가는 원리만 알면 적재적소에 좋은 기술을 쓰는게 바람직하겠단 결론을 내리고 프로젝트에도 사용하기로 했다. ^^

## 💻 사용

Relation, Pk, Fk 등 들어만 보았던 개념들을 직접 부딪히며 사용해본 것이 굉장히 의미 있었다.
부트캠프 기간 동안 MongoDB만 사용해봤던 입장으로서, 이번 프로젝트의 **데이터베이스 설계 (스키마 작성)**에서 어려웠던 점은 관계를 설정해주는 부분이었다.

ORM 사용이 꿀 발린 독일까 걱정했었는데 실제로 사용해보니 잘한 선택이었다.
관계형 데이터베이스에 대한 이해를 완벽하기 하기 전이라 스키마를 짜며 실수할 때가 잦았다.

그때마다 prisma 가 과외선생님처럼 바로 어딜 고쳐야 한다고 알려주니 틀린 이유에 대해서도 바로바로 찾아볼 수 있었다. 그러면서 관계 짓는 것에 대한 두려움이 많이 줄어들었달까.

그래서 나도 다른 사람들이 RDMS 입문 시에 ORM을 쓸지 말지 고민한다면 써보는 것도 좋은 경험이라고 이야기 해줄거 같다.

### 1:N 관계

- 스키마 코드
  [https://github.com/fog-of-war/dev-be/blob/dev/prisma/schema.prisma](https://github.com/fog-of-war/dev-be/blob/dev/prisma/schema.prisma)

  ```tsx

  /**
  Annotation 으로 관계를 설정해주니 생산성에 매우 도움이 되었다.

  @relation(fields:[], references : [])
  @@map(name:)
  */

  model Place {
  place_id          Int          @id @default(autoincrement())
  place_created_at  DateTime     @default(now())
  place_updated_at  DateTime     @updatedAt
  place_name        String
  place_star_rating Float?
  place_point       Int?
  place_address     String?
  place_latitude    Float?
  place_longitude   Float?
  place_visited_by  PlaceVisit[]
  place_category    Category?    @relation(fields: [place_category_id], references: [category_id])
  place_category_id Int
  place_posts       Post[] // 한 장소에 여러 개의 포스트가 연결됩니다.
  }
  ```

- 서비스 코드
  [https://github.com/fog-of-war/dev-be/blob/dev/src/posts/posts.service.ts](https://github.com/fog-of-war/dev-be/blob/dev/src/posts/posts.service.ts)

  ```tsx

  /**
  connect 를 사용해 관계 지어진 필드를 사용할 수 있었다.
  */

  private async createPostWithExistingPlace(
      existingPlace: any,
      userId: number,
      dto: CreatePostDto
    ) {
      return this.prisma.post.create({
        data: {
          post_star_rating: dto.post_star_rating,
          post_description: dto.post_description,
          post_image_url: dto.post_image_url,
          post_place: {
            connect: {
              place_id: existingPlace.place_id,
            },
          },
          post_author: {
            connect: { user_id: userId },
          },
        },
      });
    }
  ```

### M:N 관계

- 스키마

  ```tsx
  model Place {
    place_id           Int                @id @default(autoincrement())
    place_created_at   DateTime           @default(now())
    place_updated_at   DateTime           @updatedAt
    place_name         String             @unique
    place_star_rating  Float?
    place_point        Int?
    place_address      String?
    place_latitude     Float?
    place_longitude    Float?
    place_visited_by   PlaceVisit[]
    place_posts        Post[]
    place_category_map MapPlaceCategory[]
  }

  model Category {
    category_id           Int                @id @default(autoincrement())
    category_name         String             @unique
    category_score        Int?
    category_created_at   DateTime           @default(now())
    category_updated_at   DateTime           @updatedAt
    category_badges       Badge[]
    category_map_category MapPlaceCategory[]
  }

  model MapPlaceCategory {
    place      Place    @relation(fields: [placeId], references: [place_id])
    placeId    Int
    category   Category @relation(fields: [categoryId], references: [category_id])
    categoryId Int

    @@id([placeId, categoryId]) // 복합 기본 키 조합이 pk 대신 고유한 식별자로 작용합니다.
  }
  ```

- 쿼리

  ```tsx
  async insertPlaces() {
      for (const placeData of placesData) {
        const existingPlace = await this.place.findFirst({
          where: { place_name: placeData.place_name },
        });

        if (!existingPlace) {
          const categories = await this.category.findMany({
            where: { category_id: { in: placeData.place_category_ids } },
          });

          const createdPlace = await this.place.create({
            data: {
              place_name: placeData.place_name,
              place_address: placeData.place_address,
              place_latitude: placeData.place_latitude,
              place_longitude: placeData.place_longitude,
              place_category_map: {
                create: categories.map((category) => ({
                  category: { connect: { category_id: category.category_id } },
                })),
              },
            },
          });

          console.log("Created place:", createdPlace);
          //  커밋용
        }
      }
    }
  ```

### Github actions 에서 cicd

[https://www.prisma.io/docs/guides/deployment/deploy-database-changes-with-prisma-migrate](https://www.prisma.io/docs/guides/deployment/deploy-database-changes-with-prisma-migrate)

```yaml
npx prisma migrate deploy
```

## 🌱 강의

<iframe width="560" height="315" src="https://www.youtube.com/embed/XQ_6G0iCyMQ?list=PLwvrYc43l1MxAEOI_KwGe8l42uJxMoKeS" title="PostgreSQL: What is a Database | Course | 2019" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/RebA5J-rlwg" title="Learn Prisma In 60 Minutes" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

## 📎 Related articles

| 이슈명                  | 링크                                                                                                                                                                       |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PosgresSQL Course       | [https://www.youtube.com/watch?v=XQ_6G0iCyMQ&list=PLwvrYc43l1MxAEOI_KwGe8l42uJxMoKeS](https://www.youtube.com/watch?v=XQ_6G0iCyMQ&list=PLwvrYc43l1MxAEOI_KwGe8l42uJxMoKeS) |
| 60분 안에 Prisma 배우기 | [https://www.youtube.com/watch?v=RebA5J-rlwg](https://www.youtube.com/watch?v=RebA5J-rlwg)                                                                                 |
| Prisma 처음 시작하기    | [https://www.daleseo.com/prisma](https://www.daleseo.com/prisma)                                                                                                           |
