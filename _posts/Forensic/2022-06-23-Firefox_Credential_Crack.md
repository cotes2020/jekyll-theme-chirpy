---
layout: post
title:  "Password Crack - Firefox Password "
date:   2022-06-28 14:08:53
categories: [Forensic, 암호 알고리즘]
---

# Firefox Password Cracking 

Firefox에는 credential key 를 자동저장하는 기능과 이 Key 들을 암호화하는 master password 가 있다. <br/>
이를 Cracking 하여 FireFox에 저장된 Password를 Cracking 하는 Password를 확인하기 위해 <br/>
Windows의 Google Chrome 및 Mozilla Firefox에 로컬로 저장된 자격 증명을 추출 하는 [HarvestBrowserPasswords.exe](https://github.com/Apr4h/HarvestBrowserPasswords)가 있습니다 <br/>

FireFox가 제공하는 암호 기반 다이어 그램을 확인할 수 있는 그림은 [lClevy](https://github.com/lclevy/firepwd/blob/master/mozilla_pbe.pdf)에서 확인할 수 있습니다 <br/><br/>

해당연구는 [firepwd](https://github.com/lclevy/firepwd)의 개발자가 수행한 연구가 잘 정리 되어 있습니다. Mozilla가 자격증명의 저장 및 암호화를 처리하는 방법을 이해하는데 큰 도움이 됩니다. 
<br/>

FireFox는 CBC 모드에서 3DES를 사용하여 로그인을 암호화 합니다. 다이어그램은 KEY3.DB(Berkey DB) 형식에 저장된 Mozila의 마스터 복호화 키와 Signons.sqlite에 저장된 암호화된 로그인을 보여줍니다. 이것은 이전 버전의 FireFox에서 사용되었으며, 버전 58의 로그인은 이제 Key4.db(SQLite)에 저장되고 암호화된 로그인은 logins.json에 저장됩니다.

![image](https://user-images.githubusercontent.com/46625602/175192325-d368f036-244c-4538-b84d-0440c4361fc6.png)

<br/>

Mozilla 는 특히 데이터 직렬화에 ASN.1을 사용하기 대문에 나중에 중요해지는 NSS(네트워크 보안 서비스)라는 자체 암호화 라이브러리를 유지 관리 합니다. Chrome과 FireFox의 또 다른 큰 차이점은 FireFox를  사용하면 사용자가 저장된 모든 로그인을 암호화 하기 위해 `마스터 암호`를 제공할 수 있다는 것입니다. 
`HarvestBrowserPasswords` 라는 프로그램은 `마스터 암호`를 명령줄 인수로 가져와 암호 해독에 사용할 수 있습니다(암호를 알고 있다고 가정). 사용자가 마스터 암호를 제공하지 않은 경우 사용자 프로필 디렉터리의 SQLite 데이터 베이스에서 Password key 추출할 수 있습니다. 

<br/>

Firefox 가 저장된 Password를 암호화 하는 과정은 다음과 같습니다. 

![image](https://user-images.githubusercontent.com/46625602/175479052-8e6e8078-335e-4b91-b592-0a6159590ee8.png)

## Master Password 암호화는 낮은 SHA1 반복 횟수를 사용합니다. 

마스터 암호 암호화는 낮은 SHA1 반복 횟수를 사용합니다.
"저는 소스 코드를 조사했습니다."라고 Palant는 말합니다. "나는 SHA-1 해싱을 임의의 솔트로 구성된 문자열에 적용하여 [웹사이트] 암호를 암호화 키로 변환하는 `sftkdb_passwordToKey()` 함수를 결국 찾았습니다. 실제 마스터 비밀번호입니다."


### Credential Key 저장 경로 

<br/>

사용자의 Firefox 프로필은 각각 의 자체 디렉토리에 저장됩니다 C:\Users\[User Name]\Roaming\Mozilla\Firefox\Profiles\<random text>.default\. 최신 버전의 Firefox에는 저장된 자격 증명의 암호 해독에 필요한 두 가지 관련 아티팩트가 있습니다.

* C:\Users\[User Name]\Roaming\Mozilla\Firefox\Profiles\<random text>.default\key4.db
* C:\Users\[User Name]\Roaming\Mozilla\Firefox\Profiles\<random text>.default\logins.json


### 저장 방식 

logins.json URL, 사용자 이름, 비밀번호 및 기타 메타데이터를 포함한 모든 사용자 로그인을 JSON으로 저장합니다. 이 파일의 사용자 이름과 암호는 모두 3DES로 암호화 된 다음 ASN.1로 인코딩 되고 마지막으로 base64로 인코딩 된 파일에 기록된다는 점은 주목할 가치가 있습니다. 

<br/>

```
{
"nextId":9,
"logins":[
{
"id":1,
"hostname":"http://thestock6nonb74owd6utzh4vld3xsf2n2fwxpwywjgq7maj47mvwmid.onion",
"httpRealm":null,
"formSubmitURL":"http://thestock6nonb74owd6utzh4vld3xsf2n2fwxpwywjgq7maj47mvwmid.onion",
"usernameField":"username",
"passwordField":"password",
"encryptedUsername":"MDoEEPgAAAAAAAAAAAAAAAAAAAEwFAYIKoZIhvcNAwcECF6123123123132123321213132131213132",
"encryptedPassword":"MDoEEPgAAAAAAAAAAAAAAAAAAAEwFAYIKoZIhvcNAwcECF6321321321213121213213121232123222",
"guid":"{ce61ac9d-f640-453d-91c9-402f6f434776}",
"encType":1,
"timeCreated":1654836416726,
"timeLastUsed":1654836416726,
"timePasswordChanged":1654836416726,
"timesUsed":1
}],
"potentiallyVulnerablePasswords":[
],
"dismissedBreachAlertsByLoginGUID":{
},
"version":3
}
```

`key4.db`에 저장된 모든 암호의 3DES 암호 해독을 위한 마스터 키를 마스터 키의 암호 `logins.json` 해독을 확인하는 데 사용되는 "Password" 값과 함께 저장합니다. "password-check" 값은 다음 위치에 있습니다.


##  로그인 암호 해독

이 정보를 기반으로 로그인 암호를 해독하는 단계는 다음과 같습니다.

1. 사용자 프로필을 찾은 다음 인코딩된 + 암호화된 "암호 확인" 데이터를 `key4.db`에서 추출합니다.
2. ASN.1 디코딩 후 3DES는 "암호 확인" 데이터를 해독합니다.
    * 이는 제공된 마스터 암호가 정확하거나 암호가 제공되지 않았는지 확인하기 위해 수행됩니다.
3. 인코딩된 + 암호화된 `key4.db`에서 마스터 키 추출
4. ASN.1 디코딩 후 3DES가 ​​마스터 키를 해독합니다.
5. 암호화된 로그인 읽기 및 `logins.json` JSON 역직렬화
6. ASN.1 디코딩 후 3DES는 마스터 키를 사용하여 로그인 데이터를 해독합니다.

### 1단계 - 프로필 찾기 및 "비밀번호 확인" 데이터 추출

`HarvestBrowserPasswords`는 Firefox 프로필 디렉토리/파일을 찾고 Chrome에서와 동일하게 SQLite 데이터베이스를 쿼리합니다.

<br/>

아래 이미지는 'password-check' 값을 포함하는 ASN.1 DER(Distinguished Encoding Rules) 인코딩 데이터의 위치를 ​​보여줍니다. 'password' 행의 `item1` 값에는 암호화 중에 사용되는 `전역 솔트` 값이 포함됩니다. `item2`에는 암호화된 값 `password-check\x02\x02`와 암호화에 사용되는 항목 `솔트`를 포함하는 ASN.1 인코딩된 BLOB가 포함되어 있습니다.

* key4.db
![image](https://user-images.githubusercontent.com/46625602/175475191-1a922942-a724-4766-8a20-c5ee22124ab9.png)

암호 추출 프로세스 전체에서 사용되는 ASN.1 인코딩 데이터를 반복적으로 구문 분석하고 저장하기 위해 클래스를 함께 분석했다. 

ASN.1은 TLV(Type, Length, Value) 데이터 형식을 사용하고 문제의 데이터는 작업을 더 쉽게 만든 DER 데이터 유형 중 일부만 사용합니다. 

다음 스니펫은 인코딩된 BLOB에 있는 각 TLV의 DER 데이터 유형을 확인하기 위해 파서 클래스에서 사용되는 enum 입니다. 암호 기반 암호화를 위해 Mozilla가 사용하는 데이터 유형과 각 'Type' 바이트의 해당 값을 보여줍니다. TLV 시퀀스.

```
enum ASN1Types
{
    SEQUENCE = 0x30,
    OCTETSTRING = 4,
    OBJECTIDENTIFIER = 6,
    INTEGER = 2,
    NULL = 5
}
```

다음은 `password-check` 값에 대해 구문 분석된 ASN.1 데이터의 예입니다(firepwd에서).

```
SEQUENCE {
    SEQUENCE {
        OBJECTIDENTIFIER 1.2.840.113549.1.12.5.1.3
        SEQUENCE {
        OCTETSTRING entry_salt_for_passwd_check
        INTEGER 01
        }
    }
    OCTETSTRING encrypted_password_check
    }
```

### 2단계 - ASN.1 '비밀번호 확인' 복호화 및 복호화

ASN.1 파서 클래스는 ASN.1 인코딩된 BLOB를 객체로 재귀적으로 구문 분석합니다. 각 개체에는 Sequence위의 구조를 나타내고 인코딩된 로그인 데이터 및 마스터 키에 필요한 다른 경우를 처리하기 위한 개체 목록이 포함되어 있습니다.

```
GlobalSalt = (byte[])dataReader[0]; //item1 from key4.db

byte[] item2Bytes = (byte[])dataReader[1]; //item2 from key4.db

ASN1 passwordCheckASN1 = new ASN1(item2Bytes);

EntrySaltPasswordCheck = passwordCheckASN1.RootSequence.Sequences[0].Sequences[0].Sequences[0].OctetStrings[0];
CipherTextPasswordCheck = passwordCheckASN1.RootSequence.Sequences[0].Sequences[0].OctetStrings[1];
```

데이터가 시퀀스로 ASN1 개체이면 암호 해독에 필요한 값을 추출할 수 있으며 결과를 하드 코딩된 값과 비교 password-check\x02\x02하여 올바른 암호/값이 사용되었는지 확인할 수 있습니다. MasterPassword암호가 명령줄 인수로 제공되지 않은 경우 빈 문자열입니다 .

```
DecryptedPasswordCheck = Decrypt3DES(GlobalSalt, EntrySaltPasswordCheck, CipherTextPasswordCheck, MasterPassword);
```

### Mozilla 암호 기반 암호화

이 Decrypt3DES()함수는 위의 Mozilla PBE 다이어그램의 빨간색 상자에 지정된 형식을 따릅니다. "비밀번호 확인" 및 마스터 키 복호화에 대해 정확히 동일한 프로세스를 따릅니다.

먼저 마스터 암호는 Decrypt3DES()함수에 전달된 매개변수를 사용하여 해시됩니다.

```
byte[] hashedPassword = new byte[globalSalt.Length + password.Length];
Buffer.BlockCopy(globalSalt, 0, hashedPassword, 0, globalSalt.Length);
Buffer.BlockCopy(password, 0, hashedPassword, globalSalt.Length, password.Length);

using (SHA1 sha1 = new SHA1CryptoServiceProvider())
{
	hashedPassword = sha1.ComputeHash(hashedPassword);
}
```

그런 다음 해시된 암호는 항목 솔트와 결합되고 해시됩니다.

```
byte[] combinedHashedPassword = new byte[hashedPassword.Length + entrySalt.Length];
Buffer.BlockCopy(hashedPassword, 0, combinedHashedPassword, 0, hashedPassword.Length);
Buffer.BlockCopy(entrySalt, 0, combinedHashedPassword, hashedPassword.Length, entrySalt.Length);

using (SHA1 sha1 = new SHA1CryptoServiceProvider())
{   
	combinedHashedPassword = sha1.ComputeHash(combinedHashedPassword);
}
```

그런 다음 이전에 생성된 값을 사용하여 계산된 두 개의 HMAC-SHA1 값을 결합하여 암호 해독 키(및 초기화 벡터/nonce)를 생성합니다. 키는 처음 24바이트에서 가져오고 IV는 마지막 8바이트에서 가져옵니다.

```
byte[] edeKey;

using (HMACSHA1 hmac = new HMACSHA1(combinedHashedPassword))
{
    //First half of EDE Key = HMAC-SHA1( key=combinedHashedPassword, msg=paddedEntrySalt+entrySalt)
    byte[] firstHalf = new byte[paddedEntrySalt.Length + entrySalt.Length];
    Buffer.BlockCopy(paddedEntrySalt, 0, firstHalf, 0, paddedEntrySalt.Length);
    Buffer.BlockCopy(entrySalt, 0, firstHalf, paddedEntrySalt.Length, entrySalt.Length);

    //Create TK = HMAC-SHA1(combinedHashedPassword, paddedEntrySalt)
    keyFirstHalf = hmac.ComputeHash(firstHalf);
    byte[] tk = hmac.ComputeHash(paddedEntrySalt);

    //Second half of EDE key = HMAC-SHA1(combinedHashedPassword, tk + entrySalt)
    byte[] secondHalf = new byte[tk.Length + entrySalt.Length];
    Buffer.BlockCopy(tk, 0, secondHalf, 0, entrySalt.Length);
    Buffer.BlockCopy(entrySalt, 0, secondHalf, tk.Length, entrySalt.Length);

    keySecondHalf = hmac.ComputeHash(secondHalf);

    //Join first and second halves of EDE key
    byte[] tempKey = new byte[keyFirstHalf.Length + keySecondHalf.Length];
    Buffer.BlockCopy(keyFirstHalf, 0, tempKey, 0, keyFirstHalf.Length);
    Buffer.BlockCopy(keySecondHalf, 0, tempKey, keyFirstHalf.Length, keySecondHalf.Length);

    edeKey = tempKey;
}

byte[] key = new byte[24];
byte[] iv = new byte[8];

//Extract 3DES encryption key from first 24 bytes of EDE key
Buffer.BlockCopy(edeKey, 0, key, 0, 24);

//Extract initialization vector from last 8 bytes of EDE key
Buffer.BlockCopy(edeKey, (edeKey.Length - 8), iv, 0, 8);
```

이제 남은 것은 키와 IV를 사용하여 3DES 암호 해독을 수행하는 것입니다.

```
using (TripleDESCryptoServiceProvider tripleDES = new TripleDESCryptoServiceProvider
{
    Key = key,
    IV = iv,
    Mode = CipherMode.CBC,
    Padding = PaddingMode.None
})
{
    ICryptoTransform cryptoTransform = tripleDES.CreateDecryptor();
    decryptedResult = cryptoTransform.TransformFinalBlock(cipherText, 0, cipherText.Length);
}
```

함수 에서 decryptedResult반환되고 'password-check\x02\x02' 값에 대해 검사됩니다. 성공하면 암호 해독에 올바른 암호가 사용되었으며 이 프로세스를 반복하여 사용자 로그인 데이터에 대한 마스터 3DES 암호화 키를 해독할 수 있습니다.

### 3단계 - 마스터 암호 해독 키 추출

SQLiteDatabaseConnection암호 확인을 위해 데이터를 쿼리하는 데 사용 된 것과 동일한 항목이 nssPrivate 테이블에서 항목 솔트 및 암호화된 3DES 키를 쿼리하는 데 다시 사용됩니다. 이 값은 이 테이블에서 유일한 행 의 a11 열에 저장됩니다.

![image](https://user-images.githubusercontent.com/46625602/175476937-31907738-0e0b-43f4-a252-e4fa41c7f5e3.png)

```
SQLiteCommand commandNSSPrivateQuery = connection.CreateCommand();
commandNSSPrivateQuery.CommandText = "SELECT a11 FROM nssPrivate";
dataReader = commandNSSPrivateQuery.ExecuteReader();
```

### 4단계 - 마스터 키 디코딩 및 암호 해독

이 단계에는 새로운 것이 필요하지 않습니다. 먼저 엔트리 솔트와 암호화된 마스터 키를 추출하기 위해 ASN.1로 인코딩된 BLOB가 구문 분석됩니다.

```
byte[] a11 = (byte[])dataReader[0];

ASN1 masterKeyASN1 = new ASN1(a11);

EntrySalt3DESKey = masterKeyASN1.RootSequence.Sequences[0].Sequences[0].Sequences[0].OctetStrings[0];
CipherText3DESKey = masterKeyASN1.RootSequence.Sequences[0].Sequences[0].OctetStrings[1];
```

Decrypt3DES()그런 다음 이러한 값은 동일한 암호 값 을 사용 하여 함수에 전달됩니다 . 해독된 마스터 키는 PKCS#7 패딩됨

### 5단계 - 암호화된 자격 증명 읽기 및 역직렬화

Newtonsoft의 Json.NET 을 사용 하여 .NET에서 로그인 데이터를 역직렬화했습니다 `logins.json`. Visual Studio를 사용하면 JSON 데이터를 매우 쉽게 처리할 수 있으며 JSON 데이터를 새 클래스 파일에 복사/붙여넣기만 하면 클래스를 생성할 수 있습니다. 이 특별한 경우에 "Rootobject"는 각 개별 로그인에 대한 모든 데이터를 저장하는 "Login" 중첩 클래스의 배열을 포함합니다. 거기에서 모든 로그인 데이터를 쉽게 추출할 수 있습니다.

```
public FirefoxLoginsJSON.Rootobject GetJSONLogins(string profileDir)
{
    string file = File.ReadAllText(profileDir + @"\logins.json");
    FirefoxLoginsJSON.Rootobject JSONLogins = JsonConvert.DeserializeObject<FirefoxLoginsJSON.Rootobject>(file);

    return JSONLogins;
}
```

### 6단계 - 신용을 해독하십시오!

이제 로그인 데이터 모음을 로 사용할 수 있습니다 JSON Logins. 각각 JSON Logins.Login.EncryptedUsername은 JSONLogins.Login.EncryptedPAssword 여전히 ​​ASN.1입니다. ASN.1 데이터 구조는 아래에서 볼 수 있듯이 마스터 키 및 암호 확인에 사용된 것과 약간 다릅니다. 이제 남은 작업은 다음과 같습니다.

1. Login각 객체에 대해 반복JSONLogins
2. ASN.1은 각 사용자 이름과 암호를 해독합니다.
3. 3DES는 마스터 키를 사용하여 각 사용자 이름과 비밀번호를 해독합니다.
4. 해독된 각 사용자 이름과 비밀번호를 해당 URL과 함께 컬렉션에 추가합니다.

```
foreach (FirefoxLoginsJSON.Login login in JSONLogins.Logins)
{                 
 if (string.IsNullOrWhiteSpace(login.FormSubmitURL))
 {
 byte[] usernameBytes = Convert.FromBase64String(login.EncryptedUsername);
 byte[] passwordBytes = Convert.FromBase64String(login.EncryptedPassword);

 ASN1 usernameASN1 = new ASN1(usernameBytes);

 byte[] usernameIV = usernameASN1.RootSequence.Sequences[0].Sequences[0].OctetStrings[0];
 byte[] usernameEncrypted = usernameASN1.RootSequence.Sequences[0].Sequences[0].OctetStrings[1];

 ASN1 passwordASN1 = new ASN1(passwordBytes);

 byte[] passwordIV = passwordASN1.RootSequence.Sequences[0].Sequences[0].OctetStrings[0];
 byte[] passwordEncrypted = passwordASN1.RootSequence.Sequences[0].Sequences[0].OctetStrings[1];

 string decryptedUsername = Encoding.UTF8.GetString(Unpad(Decrypt3DESLogins(usernameEncrypted, usernameIV, Decrypted3DESKey)));
 string decryptedPassword = Encoding.UTF8.GetString(Unpad(Decrypt3DESLogins(passwordEncrypted, passwordIV, Decrypted3DESKey)));

 BrowserLoginData loginData = new BrowserLoginData(login.FormSubmitURL, decryptedUsername, decryptedPassword, "Firefox");
 FirefoxLoginDataList.Add(loginData);
 }
}
```

![image](https://user-images.githubusercontent.com/46625602/175477369-c32e1b6c-61f9-4d59-9098-b2b2a885ca26.png)

동일한 `HarvestBrowserPasswords`의 내용을 복호화 하기 위해 디버깅을 수행했지만, ANSI.cs 에 대한 에러와, 처리되지 않은 예의 의 개체 참조가 개체의 인스턴스로 설정되지 않았다는 `Decrypt3DESMasterKey(Byte[] globalSalt, Byte[] entrySalt, Byte[] cipherText, String masterPassword)` 코드 상에서의 에러가 존재하여 실행해보지는 못했다. 

![image](https://user-images.githubusercontent.com/46625602/175477506-9fb39ff1-38fa-446d-8609-89c032c7c869.png)


다만, 동일하게 Firefox에 저장된 Password와 URl, UserId를 복호화 해주는 도구가 존재하여 사용해 봤다. 
<br/><br/>

[ffpass](https://github.com/louisabraham/ffpass)에 존재하는 모듈이며, 

* C:\Users\User\AppData\Local\Mozilla\Firefox\Profiles\profile.default

해당 경로에 

* `cert9.db`, `key4.db`, `logins.json`, `login-backup.json`을 가져다 둠 - mastor password 설정 안 되어 있는 파일 

![Untitled (1)](https://user-images.githubusercontent.com/46625602/175478210-544e67a8-1bb0-4bdc-9f6d-1fc6d8b2a17d.png)
)

1. ffpass 모듈을 설치한다 
    명령어 : ```pip install ffpass```
2. ```ffpass export --file passwords.csv``` 로 결과 URL 을 csv file 로 내보낸다. 


### ETC...

FireFox 에 저장된 암호들을 Password Cracking 하는 도구 들입니다. 

[https://www.raymond.cc/blog/how-to-find-hidden-passwords-in-firefox/](https://www.raymond.cc/blog/how-to-find-hidden-passwords-in-firefox/)




---

**[Reference]**

* [https://apr4h.github.io/2019-12-20-Harvesting-Browser-Credentials/](https://apr4h.github.io/2019-12-20-Harvesting-Browser-Credentials/)
* [https://github.com/Apr4h/HarvestBrowserPasswords](https://github.com/Apr4h/HarvestBrowserPasswords)
* [https://github.com/louisabraham/ffpass](https://github.com/louisabraham/ffpass)
* [https://github.com/lclevy/firepwd](https://github.com/lclevy/firepwd)
* [https://security.stackexchange.com/questions/215881/how-are-mozilla-firefox-passwords-encrypted](https://security.stackexchange.com/questions/215881/how-are-mozilla-firefox-passwords-encrypted)
* [https://github.com/AlessandroZ/LaZagne](https://github.com/AlessandroZ/LaZagne)