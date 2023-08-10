---
layout: post
title:  "Master Password - ffpass] "
date:   22022-06-28 16:44:49
categories: [Forensic, 암호 알고리즘]
---

# Master Password - ffpass

다음 모듈의 작업 트리는 다음과 같다. 

* Window 에서의 작업 트리 구조 확인 명령어

``` tree ``` 

```
│  password.csv
│  README.md
│  setup.py
├─.github
│  └─workflows
│          testing.yaml
├─ffpass
│      __init__.py
└─tests
    │  test_key.py
    │  test_run.py
    ├─firefox-70
    │      key4.db
    │      logins.json
    ├─firefox-84
    │      key4.db
    │      logins.json
    ├─firefox-mp-70
    │      key4.db
    │      logins.json
    └─firefox-mp-84
            key4.db
            logins.json
```

* 실행 명령어

```python 
python ffpass/ffpass.py export -f password.csv
```

### import 필수 모듈 


```python
from base64 import b64decode, b64encode
from hashlib import sha1, pbkdf2_hmac
from Crypto.Cipher import AES, DES3
```

# Getkey 모듈 실행 절차 

`key4.db`의 내용은 다음과 같습니다. 

![image](https://user-images.githubusercontent.com/46625602/176129409-3af98ea2-665b-459a-a602-f53ec8adbc90.png)

### 1. password checked

item1에 담겨 있는 내용은 `globalSalt`로 사용됩니다. 
item2에 담겨 있는 내용은 `der_decode`로 디코딩 된다음 3DES의 방식으로 복호화를 시도합니다.

```entrySalt = decodedItem2[0][1][0].asOctets()``` 의 내용은 entrySalt 로 사용하고, 
```decodedItem2[1].asOctets()``` 의 내용은 ciperT 라는 내용으로 사용합니다. 

``` clearText = decrypt3DES(globalSalt, masterPassword, entrySalt, cipherT)``` 
로 복호화를 시도 한 후, 에러가 발생할 경우 AES로 다시 복호화를 시도합니다. 

```clearText = decrypt_aes(decodedItem2, masterPassword, globalSalt)```

만약 복호화 된 ClearText 가 

```python
if clearText != b"password-check\x02\x02":
        raise WrongPassword()
```

의 조건문을 만족하지 못한다면, 정상 master key 가 아니라는 것을 확인할 수 있습니다. 
만약 해당 테스트를 통과한다면, 


```python
def getKey(directory: Path, masterPassword=""):
    dbfile: Path = directory / "key4.db"

    if not dbfile.exists():
        raise NoDatabase()

    conn = sqlite3.connect(dbfile.as_posix())
    c = conn.cursor()
    c.execute("""
        SELECT item1, item2
        FROM metadata
        WHERE id = 'password';
    """)
    row = next(c)
    globalSalt, item2 = row

    try:
        decodedItem2, _ = der_decode(item2)
        encryption_method = '3DES'
        entrySalt = decodedItem2[0][1][0].asOctets()
        cipherT = decodedItem2[1].asOctets()
        clearText = decrypt3DES(
            globalSalt, masterPassword, entrySalt, cipherT
        )  # usual Mozilla PBE
    except AttributeError:
        encryption_method = 'AES'
        decodedItem2 = der_decode(item2)
        clearText = decrypt_aes(decodedItem2, masterPassword, globalSalt)

    if clearText != b"password-check\x02\x02":
        raise WrongPassword()

    logging.info("password checked")
```

### 2. find Password 

a11 의 값만 필요 하지만 우선 a102의 값이 MAGIC1(b"\xf8\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01")
이 맞는지 확인한다. 

만약 이 값이 다르다면, FireFox database가 손상되었다고 식별한다. 

a11의 값을 3DES나 AES로 복호화 하여 확인한다. 

![image](https://user-images.githubusercontent.com/46625602/176130953-0696aa3b-4af4-4e2c-8c70-02151050d810.png)

```python 
    # decrypt 3des key to decrypt "logins.json" content
    
#MAGIC1 = b"\xf8\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01"

    c.execute("""
        SELECT a11, a102
        FROM nssPrivate
        WHERE a102 = ?;
    """, (MAGIC1,))
    try:
        row = next(c)
        a11, a102 = row  # CKA_ID
    except StopIteration:
        raise Exception(
            "The Firefox database appears to be broken. Try to add a password to rebuild it."
        )  # CKA_ID

    if encryption_method == 'AES':
        decodedA11 = der_decode(a11)
        key = decrypt_aes(decodedA11, masterPassword, globalSalt)
    elif encryption_method == '3DES':
        decodedA11, _ = der_decode(a11)
        oid = decodedA11[0][0].asTuple()
        assert oid == MAGIC3, f"The key is encoded with an unknown format {oid}"
        entrySalt = decodedA11[0][1][0].asOctets()
        cipherT = decodedA11[1].asOctets()
        key = decrypt3DES(globalSalt, masterPassword, entrySalt, cipherT)

    logging.info("{}: {}".format(encryption_method, key.hex()))
    return key[:24]

```




---

**[Reference]**

* 