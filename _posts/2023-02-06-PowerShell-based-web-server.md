---
layout: post
title: PowerShell based web server
date: 2023-02-06
category: 
- Instructions
author: ebmarquez
tags:
- API
- Web Server
- Container
- docker
- dotnet
- PowerShell
summary: Using powershell to run a web server
---
* Table of Contents
{:toc}

Recently, I needed to write a web server using PowerShell. PowerShell is based on the dotnet framework. It can use dotnet's libraries to build tools and different services. In this case I needed to utilize the [System.Net.HttpListener][httplistener]{:target="_blank"} library.

Setting up a web server with PowerShell is very simple.  In this example, the service will operate  within a container. The container I'm using is based on dotnet6 with PowerShell 7 running on a debian based container.

```powershell
$urlPrefix = '+' # the plus is special, this tells it to listen on any IP.
                 # A specific IP can be used where the + is, but in a container the + is required.
$port = 8080     # port 80 is typically reserved, I found I had conflicts with dotnet in a container.
$path = 'test'
$listener = New-Object System.Net.HttpListener
$listener.Prefixes.Add(("http://{0}:{1}/{2}" -f $UrlPrefix, $Port, ($path + '/') ))
$listener.Start()  # starting the listener service.

# This will method will wait until a request is recieved.
$context = $listener.GetContext()
```

In the code, I'm using the URLPrefix with the magic `+` in the place of the IP.  The `+` in dotnet is the equivalent of `0.0.0.0` which will be important for a container.  Port `8080` was used instead of 80.  I initially used 80 but at some point, I encountered a port conflict.  I think this has something to do with how dotnet starts the listener (not really sure). To get around this I selected `8080` instead. There might be a special switch that can be disable the port 80 conflict, but I didn't find it.

The URL path is set to`/test`. The web service property GetContext will respond to any request. In this example the code will only respond to the my specific path. `New-Object` was used to call the dotnet library and assign it to the variable `$listener`. One thing to note with the URL, a closing slash `/` is required by the HttpListener property. If an ending `/` is not used, PowerShell will throw the following error.

```powershell
MethodInvocationException: Exception calling "Add" with "1" argument(s): "Only Uri prefixes ending in '/' are allowed. (Parameter 'uriPrefix')"
```

The `$listener` object has a number of interest properties, IsListening, Prefixes, and AuthenticationSchemes.  IsListening is boolean with a value set to True. Under Prefix it will show the assigned path. Once the GetContext() is executed we can test the server by connecting to localhost address, [http://127.0.0.1:8080/test/](http://127.0.0.1:8080/test/){:target="_blank"}.

Once, the prefix is assigned, Start() is called.  As expected, this will start the port listener for the web server. Before the GetContext is executed, we can check the `$listener` object and see it's properties.

```powershell
$listener

AuthenticationSchemeSelectorDelegate :
ExtendedProtectionSelectorDelegate   :
AuthenticationSchemes                : Anonymous
ExtendedProtectionPolicy             : ProtectionScenario=TransportSelected; PolicyEnforcement=Never;
                                       CustomChannelBinding=<null>; ServiceNames=<null>
DefaultServiceNames                  : {HTTP/laptop}
Prefixes                             : {http://+:8080/test/}
Realm                                :
IsListening                          : True
IgnoreWriteExceptions                : False
UnsafeConnectionNtlmAuthentication   : False
TimeoutManager                       : System.Net.HttpListenerTimeoutManager
```

If we check the dotnet library, we can see the `$listener` object has several properties and methods associated with it. To see an explanation of how to use the methods and properties visit the dotnet library [here][httplistener]{:target="_blank"}.

```powershell
$listener | Get-Member

   TypeName: System.Net.HttpListener

Name                                 MemberType Definition
----                                 ---------- ----------
Abort                                Method     void Abort()
BeginGetContext                      Method     System.IAsyncResult BeginGetContext(System.AsyncCallback callback, Sys…
Close                                Method     void Close()
Dispose                              Method     void IDisposable.Dispose()
EndGetContext                        Method     System.Net.HttpListenerContext EndGetContext(System.IAsyncResult async…
Equals                               Method     bool Equals(System.Object obj)
GetContext                           Method     System.Net.HttpListenerContext GetContext()
GetContextAsync                      Method     System.Threading.Tasks.Task[System.Net.HttpListenerContext] GetContext…
GetHashCode                          Method     int GetHashCode()
GetType                              Method     type GetType()
Start                                Method     void Start()
Stop                                 Method     void Stop()
ToString                             Method     string ToString()
AuthenticationSchemes                Property   System.Net.AuthenticationSchemes AuthenticationSchemes {get;set;}
AuthenticationSchemeSelectorDelegate Property   System.Net.AuthenticationSchemeSelector AuthenticationSchemeSelectorDe…
DefaultServiceNames                  Property   System.Security.Authentication.ExtendedProtection.ServiceNameCollectio…
ExtendedProtectionPolicy             Property   System.Security.Authentication.ExtendedProtection.ExtendedProtectionPo…
ExtendedProtectionSelectorDelegate   Property   System.Net.HttpListener+ExtendedProtectionSelector ExtendedProtectionS…
IgnoreWriteExceptions                Property   bool IgnoreWriteExceptions {get;set;}
IsListening                          Property   bool IsListening {get;}
Prefixes                             Property   System.Net.HttpListenerPrefixCollection Prefixes {get;}
Realm                                Property   string Realm {get;set;}
TimeoutManager                       Property   System.Net.HttpListenerTimeoutManager TimeoutManager {get;}
UnsafeConnectionNtlmAuthentication   Property   bool UnsafeConnectionNtlmAuthentication {get;set;}
```

When a request is received by the server, `$context.Request` will show a several attributes from [System.Net.HttpListenerRequest class][HttpListenerRequest]{:target="_blank"}

```powershell
$context.Request

AcceptTypes            : {text/html, application/xhtml+xml, application/xml;q=0.9, image/webp…}
UserLanguages          : {en-US, en;q=0.9, no;q=0.8}
Cookies                : {}
ContentEncoding        : System.Text.UTF8Encoding+UTF8EncodingSealed
ContentType            :
IsLocal                : True
IsWebSocketRequest     : False
KeepAlive              : True
QueryString            : {}
RawUrl                 : /test/
UserAgent              : Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)
                         Chrome/110.0.0.0 Safari/537.36 Edg/110.0.1587.63
UserHostAddress        : 127.0.0.1:8080
UserHostName           : 127.0.0.1:8080
UrlReferrer            :
Url                    : http://127.0.0.1:8080/test/
ProtocolVersion        : 1.1
ClientCertificateError :
RequestTraceIdentifier : 00000000-0000-0000-fd00-0040020000f8
ContentLength64        : 0
Headers                : {sec-ch-ua, sec-ch-ua-mobile, sec-ch-ua-platform, DNT…}
HttpMethod             : GET
InputStream            : System.IO.Stream+NullStream
IsAuthenticated        : False
IsSecureConnection     : False
ServiceName            :
TransportContext       : System.Net.HttpListenerRequestContext
HasEntityBody          : False
RemoteEndPoint         : 127.0.0.1:59296
LocalEndPoint          : 127.0.0.1:8080
```

The `$context.Request` object also has a number of useful settings.  In this example RawUrl, and HttpMethod are used to identify the uri and request method that was used. From the output it shows a `GET` was sent to `/test/`.  My local port was sent from `59296` and the request went to `127.0.0.1:8080`.

```powershell
if ($context.Request.HttpMethod -eq 'GET') {
   if ($context.Request.RawUrl -match '/test$|/test/$') {
         $context.Response.ContentType = 'application/html'
         $content = [Text.Encoding]::UTF8.GetBytes((New-Guid).Guid)
   }
}
$context.Response.OutputStream.Write($content, 0, $content.Length)
$context.Response.Close()
$listener.Stop()
```

In the example, there are several if conditions that are checked. If it's a `GET` request and if it's using the `/test/` path. When the listener Prefixes is configured, the object required a closing slash `/`.  When users use a web service that ending `/` may not be included.The if condition checks for both instances. If the request is a match, the content response is performed.

To respond to the request, a contentType is set and the content is assigned as to an array of bytes.  [$context.Response class][HttpListenerResponse]{:target="_blank"} has a property called OutputStream, this will utilize a [Stream.Write Method][writestream]{:target="_blank"} write to the response object. The write stream needs the array of bytes, an offset and the length of the bytes. In this case, the offset is set to zero. After the response is set, the response is then closed and the listener can be stopped. The stop method will to shutdown the the web server service and release the port.

```powershell
Invoke-WebRequest -uri http://127.0.0.1:8080/test/ -OutVariable web

$web | fl *

Content           : {53, 101, 50, 54…}
StatusCode        : 200
StatusDescription : OK
RawContentStream  : Microsoft.PowerShell.Commands.WebResponseContentMemoryStream
RawContentLength  : 36
RawContent        : HTTP/1.1 200 OK
                    Transfer-Encoding: chunked
                    Server: Microsoft-HTTPAPI/2.0
                    Date: Wed, 08 Mar 2023 05:24:37 GMT
                    Content-Type: application/html

                    5e2640dc-e695-41ea-823b-7be01090ed10
BaseResponse      : StatusCode: 200, ReasonPhrase: 'OK', Version: 1.1, Content:
                    System.Net.Http.HttpConnectionResponseContent, Headers:
                    {
                      Transfer-Encoding: chunked
                      Server: Microsoft-HTTPAPI/2.0
                      Date: Wed, 08 Mar 2023 05:24:37 GMT
                      Content-Type: application/html
                    }
Headers           : {[Transfer-Encoding, System.String[]], [Server, System.String[]], [Date, System.String[]],
                    [Content-Type, System.String[]]}
RelationLink      : {}
```

From the client perspective, when the web server is listening for a Context request. A web request can be performed with Invoke-WebRequest or just opening a web browsers to [http://127.0.0.1:8080/test/](http://127.0.0.1:8080/test/){:target="_blank"}. The web server will recieve the request and process it. The response in this case is a GUID string. The RawContent shows the header and  of the response. The content shows the data as an array of bytes. To convert the bytes to a string, [System.Text.Encoding][encoding]{:target="_blank"} is used resulting in a GUID.

```powershell
[System.Text.Encoding]::ASCII.GetString($web.Content)
5e2640dc-e695-41ea-823b-7be01090ed10
```

I hope you found this useful.

<!--Reference links in article-->

[httplistener]: <https://learn.microsoft.com/en-us/dotnet/api/system.net.httplistener?view=net-6.0> "HttpListener Class"
[HttpListenerRequest]: <https://learn.microsoft.com/en-us/dotnet/api/system.net.httplistener?view=net-6.0> "Incoming HTTP request to an HttpListener object"
[writestream]: <https://learn.microsoft.com/en-us/dotnet/api/system.io.stream.write?view=net-6.0> "When overridden in a derived class, writes a sequence of bytes to the current stream and advances the current position within this stream by the number of bytes written"
[HttpListenerResponse]: <https://learn.microsoft.com/en-us/dotnet/api/system.net.httplistenerresponse?view=net-6.0> "Represents a response to a request being handled by an HttpListener object."
[encoding]: <https://learn.microsoft.com/en-us/dotnet/api/system.text.encoding?view=net-6.0> "Represents a character encoding"