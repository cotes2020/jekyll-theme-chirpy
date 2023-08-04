---
title : "Wechall - Limited Access Too [Not Solved]"
categories : [Wargame, Wechall]
tags : [".htaccess", Apache Authentication, Javascript XMLHttpRequest, Incomplete]
---

## Limited Access Too
<hr style="border-top: 1px solid;"><br>

```
Haha, thank you so much for your feedback from the first challenge.
Especially thanks to a special person 
who sent in a fixed .htaccess to secure my pages.

The protected/protected.php is now secured :)

To prove me wrong, please access protected/protected.php again.
```

<br>

**GeSHi`ed Plaintext code for .htaccess** 

```console
AuthUserFile .htpasswd
AuthGroupFile /dev/null
AuthName "Authorization Required for the Limited Access Too Challenge"
AuthType Basic
<Limit GET POST HEAD PUT DELETE CONNECT OPTIONS PATCH>
require valid-user
</Limit>
# TRACE is not allowed in Limit if TraceEnable is off, so disallow it completely
# to support both TraceEnable being on and off
RewriteEngine On
RewriteCond %{REQUEST_METHOD} ^TRACE
RewriteRule ^ - [F]
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Solution
<hr style="border-top: 1px solid;"><br>

Link 
: <a href="https://httpd.apache.org/docs/2.2/ko/mod/mod_rewrite.html" target="_blank">httpd.apache.org/docs/2.2/ko/mod/mod_rewrite.html</a>  

<br>

RewriteRule 
: <a href="https://blog.munilive.com/posts/how-to-use-htaccess-rewrite-rule.html" target="_blank">blog.munilive.com/posts/how-to-use-htaccess-rewrite-rule.html</a>   
: <a href="https://gmate.tistory.com/317" target="_blank">gmate.tistory.com/317</a>

<br><br>
<hr style="border: 2px solid;">
<br><br>
