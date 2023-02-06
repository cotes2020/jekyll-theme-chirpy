---
layout: post
title: Using PowerShell as a web server
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
summary: Using powershell as a web server
---
* Table of Contents
{:toc}

In my day job I do a lot of work with PowerShell and recently I had a need to create a web server using this language.  For this specific use case, it need to opeate as a API service for a different set of PowerShell modules.  I found the process of using PowerShell as a web server to be very simple to setup.  In order to set this up the continer I'm working with is using PowerShell 7 and is based on the dotnet6 container on a linux container environemnt.  The real key to this service is the dotnet libaray `System.Net.HttpListener`. This allows PowerShell to open the port and operate as a webserver.  Once we establish the listener we just need to add the URI paths that need to be maintained.  There is one trick when working with a container, this has to do with the IP address of the server within the continer network.  In most cases, the servers IP address will be an unknown. In most examples pepole publish on the internet, the iP is hard coded.  This is not possible in most cases with the continer.  To enable the continer to use the 0.0.0.0 address the `+` values needes to be used where an IP would go.  The `+` is a special value, it tells the dotnet library to listen on any IP.

```powershell
$UrlPrefix = '+' # the plus is special, this tells it to listen on any IP.
				         # A specific IP can be used where the + is, but in a container the + is required.
$Port = 8080     # port 80 is typically reserved, I found I had conflicts with dotnet in a container.
$Path = '/test'
$listener = New-Object System.Net.HttpListener
$listener.Prefixes.Add(("http://{0}:{1}{2}" -f $UrlPrefix, $Port, ($path + '/') ))
$listener.Start()  # starting the listener service.
```
In this example, it shows the URLPrefix being the magic `+` in the place of the IP.  Port `8080` was used instead of 80.  I intially used 80 but at some point, I encountered a port conflict.  I think this has something to do with how dotnet starts the listener.  To get around this I selected `8080` instead.  There might be a special switch that can be diable the port 80 conflict, but I didn't see it.   The path is URL path is set to`/test`, any web calls to /test will be picked up by the web server.  `New-Object` was used to call the library and assign it to the variable `$listener`.  One thing to note with the URL path, a closing slash `/` is required.  If one is not used, PowerShell will thow the following error.

```powershell
MethodInvocationException: Exception calling "Add" with "1" argument(s): "Only Uri prefixes ending in '/' are allowed. (Parameter 'uriPrefix')"
```

Once, all the URL's are added the service can be started, but calling Start().

Listener has a number of methods assigned to it.

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
