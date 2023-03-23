
# use pytho for network work.

[toc]

---

# make a socket, a connection

```py
import socket
mysock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
mysock.connect( ('data.pr4e.org', 80))   # host and port
```

---

# HTTP
http:// www.goo.com/ index.html
protocol+host+document

1. usually
```py
# install telnet
brew install telnet

# get a connection and ask for files
$ telnet data.pr4e.org 80
Trying 192.241.136.170...
Connected to data.pr4e.org.
Escape character is '^]'.
GET http://data.pr4e.org/page1.htm HTTP/1.0   # send a GET request

HTTP/1.1 200 OK                               # the metadata of the file you ask
Date: Fri, 15 May 2020 00:39:30 GMT
Server: Apache/2.4.18 (Ubuntu)
Last-Modified: Mon, 15 May 2017 11:11:47 GMT
ETag: "80-54f8e1f004857"
Accept-Ranges: bytes
Content-Length: 128
Cache-Control: max-age=0, no-cache, no-store, must-revalidate
Pragma: no-cache
Expires: Wed, 11 Jan 1984 05:00:00 GMT
Connection: close
Content-Type: text/html

<h1>The First Page</h1>                        # the file you ask
<p>
If you like, you can switch to the
<a href="http://data.pr4e.org/page2.htm">
Second Page</a>.
</p>
Connection closed by foreign host.
J:~ luo$
```

2. in python

send request to the server and retrive a document

```py
py.file
import socket
mysock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # make the door open
mysock.connect( ('data.pr4e.org', 80))    # connect to a host and port
# encode it from unicode to UTF-8
cmd = 'GET http://data.pr4e.org/romeo.txt HTTP/1.0\n\n'.encode() # makeup request, encode to byte from unicode
mysock.send(cmd) # send to the server

while True:
    data = mysock.recv(512)   # recieve up to 512 character
    if (len(data) < 1):       # if no data receive
        break
    print(data.decode())  # from byte to UTF-8 data
mysock.close()

python3 py.file
## result: http://data.pr4e.org/romeo.txt
# meta data
HTTP/1.1 200 OK
Date: Fri, 15 May 2020 00:47:57 GMT
Server: Apache/2.4.18 (Ubuntu)
Last-Modified: Sat, 13 May 2017 11:22:22 GMT
ETag: "a7-54f6609245537"
Accept-Ranges: bytes
Content-Length: 167
Cache-Control: max-age=0, no-cache, no-store, must-revalidate
Pragma: no-cache
Expires: Wed, 11 Jan 1984 05:00:00 GMT
Connection: close
Content-Type: text/plain

# file content
But soft what light through yonder window breaks
It is the east and Juliet is the sun
Arise fair sun and kill the envious moon
Who is already s
ick and pale with grief
```

3. use urllib

```py
import urllib.request, urllib.parse, urllib.error
# open the file
fhand = urllib.request.urlopen('http://data.pr4e.org/romeo.txt')

for line in fhand:
    print(line.decode().strip()) # string

# result
But soft what light through yonder window breaks
It is the east and Juliet is the sun
Arise fair sun and kill the envious moon
Who is already sick and pale with grief
```


---

## Unicode Characters and Strings

![Screen Shot 2020-05-14 at 21.36.13](https://i.imgur.com/FoPFM3q.png)

- each character is represented by a number between 0 and 256 stored in 8 bits of memory.
- ACSII: American Standard Code
- unicode: universal code
    - UTF-16
    - UTF-32: 4 byte
    - UTF-8: 1-4 byte


```py
# the numeric value of the ASCII character
print(ord('h'))
104
```

---

## network scrape

beautifulsoup webcrawler
1. run it install in python: `pip install beautifulsoup4`
2. download and unzip it in the same file with py file.

```py

import urllib.request, urllib.parse, urllib.error
import ssl
from bs4 import BeautifulSoup

ctx = ssl.create_default_context()   # for https
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = input('Enter - ')  #http://www.dr-chuck.com/
html = urllib.request.urlopen(url, context=ctx).read() # return entire website in a single string
soup = BeautifulSoup(html, 'html.parser')

# retrieve all of the anchor tags
tags = soup('a')
for tag in tags:
    print(tag.get('href', None))

# result
Enter - http://www.dr-chuck.com/
https://www.dr-chuck.com/csev-blog/
https://www.si.umich.edu/
https://www.ratemyprofessors.com/ShowRatings.jsp?tid=1159280
https://www.dr-chuck.com/csev-blog/
https://www.twitter.com/drchuck/
https://www.dr-chuck.com/dr-chuck/resume/speaking.htm
https://www.slideshare.net/csev
/dr-chuck/resume/index.htm
https://amzn.to/1K5Q81K
```

---

## data on the web

`python dictionary` ->serialize-> `XML` <-De-Serialize<- `JavaHashMap`

`XML`: extensible markup language
- element / nodes:
    - attribute node
    - text area node
- XML as Path:
    - /a/b X
    - /a/c Y

XML Validation
- XML documnet
- XML schema contract

XML file:

```xml
<people>                     // complex element
    <person>                 // complex element
        <name>Chuch</name>   // simple element
        <phone>1122</phone>
    </person>
    <person>
        <name>beh</name>
        <phone>3344</phone>
        <email hide='yes'/>
    </person>
</people>

```

XML schema contract:
```xml
<xs:complexType name="person">
    <xs:sequence>
        <xs:element name="name" type="xs:string"/>
        <xs:element name="phone" type="xs:integar"/>
 </xs:sequence>
</xs:complexType>
```

use python

```py
import xml.etree.ElementTree as ET

data = '''
<person>
    <name>Chusck</name>
    <Phone>1122</phone>
    <email hide='yes'/>

    <users>
        <user x="2">
            <id>1</id>
            <name>hhed</name>
        </user>
        <user x="7">
            <id>2</id>
            <name>rree</name>
        </user>
    </users>

</person>'''

# make the tree
tree = ET.fromstring(data)
print('Name:', tree.find('name').text)  # .text item under the name
print('Attr:', tree.find('email').get('hide'))  # Yes

lst=tree.findall('users/user')  # result is a list
print(lst)
print('User account:', len(lst))
for item in lst:
    print('Name:', item.find('name').text)

# result
# [<Element 'user' at 0x1035b1fb0>, <Element 'user' at 0x1035ce170>]
# User account: 2
# Name: hhed
# Name: rree
```
