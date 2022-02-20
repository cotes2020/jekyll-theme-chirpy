---
title: Basic - Web Authentication - HTTP Authentication
# author: Grace JyL
date: 2020-09-23 11:11:11 -0400
description:
excerpt_separator:
categories: [01Basic, Authentication]
tags: [Basic, Authentication]
math: true
# pin: true
toc: true
# image: /assets/img/sample/devices-mockup.png
---

- [Authentication](#authentication)
  - [HTTP authentication](#http-authentication)
    - [basic](#basic)
    - [Authentication Type/schemes](#authentication-typeschemes)
    - [`Basic authentication scheme`](#basic-authentication-scheme)
    - [step:](#step)
    - [drawbacks](#drawbacks)
    - [Security of basic authentication](#security-of-basic-authentication)
    - [Proxy authentication](#proxy-authentication)
    - [Authentication of cross-origin images](#authentication-of-cross-origin-images)
  - [session-based authentication](#session-based-authentication)
    - [step](#step-1)
  - [Token-based authentication](#token-based-authentication)
  - [characteristics of token](#characteristics-of-token)
  - [JWT, JSON web tokens](#jwt-json-web-tokens)
  - [OAuth - Open Authorization](#oauth---open-authorization)
- [compare](#compare)



---

# Authentication

1. HTTP authentication: `username and passwd` are sent in each request
2. Session based authentication: `session id` are sent in each request
3. Token based authentication: `token` are sent in each request


---


## HTTP authentication

### basic

server:
- `WWW-Authenticate` and `Proxy-Authenticate` headers
- The `WWW-Authenticate` and `Proxy-Authenticate` response headers **define the authentication method that should be used** to gain access to a resource.
- must specify which authentication scheme is used, so the client knows how to provide the credentials.

The syntax for these headers:

```html
WWW-Authenticate: <type> realm=<realm>
Proxy-Authenticate: <type> realm=<realm>
```

- `<type>` is the **authentication scheme**
  - `Basic` : the most common scheme and introduced below

- `realm` : describe the protected area or to indicate the scope of protection.
  - a message like "Access to the staging site" or similar
  - so that the user knows to which space they are trying to get access to.


client:
- `Authorization` and `Proxy-Authorization` equest headers
- contain the credentials to authenticate a user agent with a (proxy) server
- `<type>` is needed again
- `credentials`: be encoded or encrypted depending on which authentication scheme is used.

```html
Authorization: <type> 12345678
Proxy-Authorization: <type> 123456
```

---

### Authentication Type/schemes
The general HTTP authentication framework is used by several authentication schemes.
- Schemes can differ in security strength and in their availability in client or server software.
- there are other schemes offered by host services, such as Amazon AWS.

Schemes | Note
---|---
`Basic` | RFC 7617, **base64-encoded** credentials.
`Bearer` | See RFC 6750, bearer tokens to access OAuth 2.0-protected resources
`Digest` | See RFC 7616, only md5 hashing is supported in Firefox, see bug 472823 for SHA encryption support
`HOBA` | See RFC 7486, Section 3, HTTP Origin-Bound Authentication, digital-signature-based
`Mutual` | See RFC 8120
`AWS4-HMAC-SHA256` | See AWS docs

---


### `Basic authentication scheme`

![Basic authentication](https://i.imgur.com/W4lnwIN.png)


- transmits credentials as user ID/password pairs, encoded using `base64`.

- the simplest possible way to enforce access control
  - as it `doesn't require cookies, sessions or anything else`.


- the client
  - provide username and password when making a request.
  - has to send the `Authorization` header along with every request it makes.
- the exchange must happen over an **HTTPS (TLS) connection** to be secure.

- The username and password are not encrypted, but constructed this way:
  1. username and password are concatenated into a single string: `username:password`
  2. this string is encoded with `Base64`
  3. the Basic keyword is put before this encoded value

---

### step:

![HTTPAuth](https://i.imgur.com/HMO7vyi.png)

1. client access come protected URL: `https://some.url`
2. server check the request has `Authorization header` with `valid usrname and passwd`

    ```
    HTTTP/1.1 401 Unauthorized
    Date: Sat, 16 May 2020 16:50:53 GMT
    WWW-Authenticate: Basic realm="MyApp"
    ```

   - The server responds to a client with a `401 (Unauthorized) response status` and provides information on how to authorize with a WWW-Authenticate response header containing at least one challenge.
     - `401 Unauthorized`: invalid, authentication is impossible for this user.
     - `200 OK`: exist and vaild
     - `403 forbidden`: valid credentials that are inadequate to access a given resource
     - `407 (Proxy Authentication Required)`: authentication is impossible for this user.
   - realm:
     - protection space:
     - group of pages use the same credential.
     - browser can cache the calid credentials for given realm and use them in future
   - "free text":
     - server is responsible for defininf realms and do the authentication


3. browser notice the `WWW-Authenticate` header in response:
   - show the windoW
   - presents the alert for credentials


4. user submit username and passwd.


5. browser encode it with  `base64` and sends in the next request
   - Browsers use `utf-8` encoding for usernames and passwords.
   - `base64("username:passwd")`
   - send a `Authorization` request header with the credentials.


6. server do step 2 again


Example

```
curl --header "Authorization: Basic am9objpzZWNyZXQ=" my-website.com
```

The same can be observed in Chrome

![google_chrome_basic_web_authentication_method-1448359567226](https://i.imgur.com/V1x3yw7.png)

Implementing in `Node.js`

```js
import basicAuth from 'basic-auth';

export default function auth(req, res, next) {
  const {name, pass} = basicAuth(req) || {};

  if (!name || !pass) return unauthorized(res);
  if (name === 'john' && pass === 'secret') return next();
  return unauthorized(res);
};

function unauthorized(res) {  
  res.set('WWW-Authenticate', 'Basic realm=Authorization Required');
  return res.send(401);
};
```

---

### drawbacks
1. the **username and password are sent with every request**
   - not secure unless used with TLS/HTTPS.
     - anyone can eavesdrop and decode the credentials.
   - potentially exposing them
   - even sent via a secure connection connected to SSL/TLS, if a website uses weak encryption, or an attacker can break it, the usernames and passwords will be exposed immediately
2. **no way to log out** the user using Basic auth
3. expiration of credentials is not trivial
   - have to ask the user to change password to do so

---

### Security of basic authentication

the user ID and password are passed over the network as clear text
- base64 encoded, but is a reversible encoding
- the `basic authentication scheme` is not secure.
- HTTPS/TLS should be used for basic authentication.
  - Without additional security enhancements
  - basic authentication should not be used to protect sensitive or valuable information.

1. Restricting access with `Apache` and basic authentication

   - To password-protect a directory on an Apache server
     - need a `.htaccess` and a `.htpasswd` file.
   - `.htpasswd file`:
     - each line consists of a username and a password
     - separated by a colon (:).
     - the passwords are hashed (MD5-based hashing)
     - can name the `.htpasswd file` differently
       - but keep in mind this file shouldn't be accessible to anyone.
       - Apache usually configured to prevent access to `.ht* files`


      ```
      .htaccess file:

      AuthType Basic
      AuthName "Access to the staging site"
      AuthUserFile /path/to/.htpasswd
      Require valid-user
      ```

      ```
      .htpasswd file :

      aladdin:$apr1$ZjTqBB3f$IF9gdYAGlMrs2fuINjHsz.
      user2:$apr1$O04r.y2H$/vEkesPhVInBByJUkXitA/
      ```

2. Restricting access with `nginx` and basic authentication
   1. a `location` going to protect
   2. the `auth_basic directive`: provides the name to the password-protected area.
   3. The `auth_basic_user_file directive` : points to a `.htpasswd file` containing the encrypted user credentials, just like Apache

      ```
      location /status {                                       
          auth_basic           "Access to the staging site";
          auth_basic_user_file /etc/apache2/.htpasswd;
      }
      ```

3. Avoid access using `credentials in the URL`
   - Many clients can avoid the login prompt by using an `encoded URL` containing the credentials
   - `https://username:password@www.example.com/`
   - The use of these URLs is deprecated.
     - In Chrome, the `username:password@` part in URLs is even stripped out for security reasons.
     - In Firefox, it is checked if the site actually requires authentication and if not,
       - Firefox will warn the user with a prompt
       - "You are about to log in to the site “www.example.com” with the username “username”, but the website does not require authentication. This may be an attempt to trick you."



---

### Proxy authentication

The same challenge and response mechanism can be used for proxy authentication.

- As both resource authentication and proxy authentication can coexist
- but different set of `headers` and `status codes` is needed.

1. the challenging status code is `407 (Proxy Authentication Required)`
   - the `Proxy-Authenticate` response header
     - contains at least one challenge applicable to the proxy
     - used for providing the credentials to the proxy server.


### Authentication of cross-origin images
security hole recently been fixed by browsers is `authentication of cross-site images`.
- From Firefox 59
- **image resources loaded from different origins to the current document** are no longer able to trigger `HTTP authentication dialogs` (bug 1423146),
- preventing user credentials being stolen if attackers were able to embed an arbitrary image into a third-party page.


---

## session-based authentication

![session-base](https://i.imgur.com/9PNfrgQ.png)

client sends the **seesion id** in all the request, and server uses it to identify the user.

- a stateful authentication method
- as the server need it to be implemented

---

### step

1. client: sent the login request

2. server: `creates and stores the session data` in the **server memory** after the user logs in
   - some random unique identifier to identify the user

3. client: the `session id` also stores in a **cookie on the user browser**.
   - store in cookie if cokkies enabled
   - or somewhere else, e.g. in `local/session storage`
   - The `session Id` is sent on subsequent requests to the server

4. server: compares it with the `stored session data` and proceeds to process the requested action.

5. server: 
   - when user logout, the session is destroyed
   - cookie removed and session removed from the server
   - same session ID can not be reused

```
1. client
http://someurl/login (username:passwd)

2. server
session ID: 123456

3. client stored it

4. slient send request with session id
http://someurl/login (session id:123456)

5. server check the session id
- 200 ok
- 401 unauthorized
```

![Screen Shot 2020-09-24 at 00.36.44](https://i.imgur.com/RWalJf8.png)

![Screen Shot 2020-09-24 at 00.36.54](https://i.imgur.com/3nF9a8H.png)

![Screen Shot 2020-09-24 at 00.37.27](https://i.imgur.com/NBesRsm.png)

![Screen Shot 2020-09-24 at 00.37.40](https://i.imgur.com/k4Ajzfo.png)

![1_Hg1gUTXN5E3Nrku0jWCRow](https://i.imgur.com/8w0haoq.png)


---


## Token-based authentication

![token-based](https://i.imgur.com/HKxHiDr.png)

token

- the `user state is stored inside the token` **on the client side**.

- token is a normal URL-safe string
  - can be pass in header, body, or URL
  - most of the time sent by HTTP headers, not cookies
    - not only for browser, but also app.
  - the preferred mode of authentication for `RESTful APIs`.

- token is `self-contained`
  - containes session infor and user info as well
  - carries the data

- anyone can view the contained

- as a standard

- use when interact between multible untrsuted parties:
  - bank - app - user

- have lifetime

- can grant access to only a subset of data
  - one `token` for one `API`
  - just give specific right: make transaction, just search ....


token has 3 part: `header.payload.signature`
- `header`:
  - string generated using `base64(tokenMetadata)`
  - token's Metadata
    ```
    {
      "typ":"jwt",
      "alg":"HS256",
    }
    ```
- `payload`:
  - string generated using `base64(ourDatas)`
  - ourDatas: data that want to embed in the token (aka JWT Claims)
  - these are called `claims`
    - 3 type of `claims`
    - Registered claims
    - ![Registered claims](https://i.imgur.com/KHBHULL.png)
    - Public claims
      - claims to defines and use for our own data
      - e.g. userid, email...
    - Private claims
      - names withour meaning except the consumer and producer of tokens
    ```
    {
      "userid":"123",
      "email":"234",
    }
    ```
- `signature`:
  - string generated by hashing the header+payload with a secret
  - `HMACSHA256(header + ',' + payload, 'secret')`
  - `secret`: held at server and used to generate and verify tokens




1. client send credentials to generate a token
2. server validate the credentials:
   - `422`: unprocessable entity
   - `200`: ok + `token in body or header`
     - the user data is encrypted into a `JWT (JSON Web Token)` with a `secret`
     - and then sent back to the client.
3. The `JWT` stored on the client-side in `localStorage/cookie` and sent as a `header` for every subsequent request.
4. The server validates the `JWT` before proceeding to send a response to the client.
   - `401`: unauthorized
   - `200`: ok + `token in body or header`

```
headers:{
"Authorization": "Bearer ${JWT_TOKEN}"
}
```

![Screen Shot 2020-09-24 at 00.42.22](https://i.imgur.com/c4WAwrS.png)

![1_PDry-Wb8JRquwnikIbJOJQ](https://i.imgur.com/jtgGEVO.png)


## characteristics of token
1. random string
2. server does not store it (stateless)
3. has an expiry, then token is useless
4. normally sighed with a secret so to identify any tampering and thus can be trusted by the server
5. normally sent in the authorization header
6. can be `Opaque` or `Self-contained`
   - `Opaque`
     - random string, no meaning
     - can only be verfied by the autorization server
     - just like session ids
   - `Self-contained`
     - token has the data and can be reviewd by the clients
     - e.g. JWT tokens


emaple of token based authentication strategies
- SWT, simple web tokenss
- JWT, JSON web tokens
- OAuth, open authorization
- SAML, security assertions markup language
- OpenID


## JWT, JSON web tokens
- form of token based authentication
- based on an Open Standard


## OAuth - Open Authorization
- allow users to share their pricate resources to a third party
  - allow some app log you in using twitter
    - expose your twitter info to an external app using `twitter's OAuth server`
  - authorizing you app's frontend from your API
    - your `custom OAuth server`: where user of your website get authorized using `OAuth`
- 2 version: OAuth1.0, 2.0(active, not backward compatible), 2.1

grant types
- token response in all grant types is normally accompanied by an `expiry date` and a `refresh token` (to refresh the token when expired)

![Screen Shot 2020-09-24 at 01.37.09](https://i.imgur.com/kBixqIO.png)

![Screen Shot 2020-09-24 at 01.39.33](https://i.imgur.com/RhXzwxG.png)

---

# compare

usually seesion-based for web browser, token-based for app

Scalability
1. Session based authentication:
   - Because the sessions are stored in the server’s memory
   - scaling becomes an issue when there is a huge number of users using the system at once.
2. Token based authentication:
   - no issue with scaling
   - because token is stored on the client side.


Multiple Device
1. Session based authentication:
   - Cookies normally work on a single domain or subdomains and they are normally disabled by browser if they work cross-domain (3rd party cookies).
   - It poses issues when APIs are served from a different domain to mobile and web devices.
2. Token based authentication:
   - no issue with cookies as the JWT is included in the request header.



- JWT
  - the size is much bigger comparing with the session id stored in cookie
  - because JWT contains more user information.
- Care must be taken to ensure only the necessary information is included in JWT
- and sensitive information should be omitted to prevent XSS security attacks.






ref:
- [Session vs Token Based Authentication](https://medium.com/@sherryhsu/session-vs-token-based-authentication-11a6c5ac45e4)
- [HTTP authentication](https://developer.mozilla.org/en-US/docs/Web/HTTP/Authentication)
- [Session vs Token-Based Authentication](https://medium.com/@allwinraju/session-vs-token-based-authentication-b1f862dd7ed8)
- [Difference between cookies, session and tokens](https://www.youtube.com/watch?v=44c1t_cKylo&ab_channel=ValentinDespa)
- [Authentication Types Ethical Hackers Academy || Cyber Security News](https://www.linkedin.com/posts/ethical-hackers-academy_authentication-types-ethical-hackers-academy-activity-6710268783136796672-g5Fl)



.
