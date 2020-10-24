---
title: Proxy Pattern in .NET Core
date: 2020-06-27T18:25:18+02:00
author: Wolfgang Ofner
categories: [Design Pattern]
tags: [ASP.NET Core MVC, 'C#', GRPC, software architecture]
---
The proxy pattern belongs to the group of structural patterns and is often used for cross cutting-concerns. Proxies can be used to control access, logging, and security features. Another advantage is separation of concern (SoC) and loose coupling of your components.

The proxy often acts as a substitute of a concrete object, as shown on the following UML diagram.

<div id="attachment_2222" style="width: 710px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2020/06/Proxy-Pattern-UML.jpg"><img aria-describedby="caption-attachment-2222" loading="lazy" class="wp-image-2222" src="/assets/img/posts/2020/06/Proxy-Pattern-UML.jpg" alt="Proxy Pattern UML" width="700" height="390" /></a>
  
  <p id="caption-attachment-2222" class="wp-caption-text">
    Proxy Pattern UML (<a href="https://en.wikipedia.org/wiki/Proxy_pattern" target="_blank" rel="noopener noreferrer">Source</a>)
  </p>
</div>

## Real-World Examples

When you purchase something and pay with a check, the check is a proxy. It allows the seller to access your bank account and withdraw a certain amount. Another example is a network proxy. All requests go through this proxy and it blocks suspicious or malicious requests. Safe requests can pass through.

In software developmen,t a proxy can be used to enable endless scrolling. The proxy pattern can be used to load content in the background while the user interface is rendered.

## When to use it

Proxies help to achieve separation of concern and the single responsible principle (SRP). They are often for cross-cutting concerns like caching, logging, or access control. The details are kept inside the proxy and therefore your classes are free of these details. Another advantage of the proxy pattern is loose coupling and DRY (don&#8217;t repeat yourself).

### Related Patterns

There are four patterns similar to the proxy pattern:

  * Prototype
  * [Decorator](https://www.programmingwithwolfgang.com/decorator-pattern-in-net-core-3-1)
  * Flyweight
  * <a href="/adapter-pattern/" target="_blank" rel="noopener noreferrer">Adapter</a>

## Implementation of the Proxy Pattern

You can find the code of the demo on <a href="https://github.com/WolfgangOfner/.NetCore-ProxyPatterns" target="_blank" rel="noopener noreferrer">Github</a>.

There are four types of proxies:

  1. Remote
  2. Smart
  3. Protective
  4. Virtual

### Remote Proxy Pattern

The remote proxy pattern is the most commonly used proxy and you might have already used it without knowing. Using a remote proxy allows you to hide all the details about the network and the communication from the client. The best example of a remote proxy in .NET Core is GRPC. The GRPC client works as a proxy and makes it look like you are calling a local method. Behind the scenes, the GRPC client is calling the GRPC server over the network. Another use case is to centralize the knowledge of the network details. That&#8217;s what Proto does in GRPC or also Swagger.

### Implementation of the Remote Proxy Pattern

To demonstrate the remote proxy, I created a GRPC server using the default Visual Studio template. In my proxy class, I am calling the server with my name and get a greeting message back. The code looks like I am calling a local method because the proxy hides all the network implementation and details.

```csharp  
public class RemoteProxy : IRemoteProxy
{
    public async Task<HelloReply> GetGreetingMessage()
    {
        // todo the GreeterService must run for this test to pass
        const string name = "Wolfgang";
        using var channel = GrpcChannel.ForAddress("https://localhost:5001");
        var client = new Greeter.GreeterClient(channel);
        
        var request = new HelloRequest
        {
            Name = name
        };

        return await client.SayHelloAsync(request);
    }
}
```

Start both projects in my demo and navigate to /RemoteProxy/HelloMessage and you will see the message in your browser.

<div id="attachment_2223" style="width: 440px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2020/06/Getting-a-message-from-the-Remote-Proxy.jpg"><img aria-describedby="caption-attachment-2223" loading="lazy" class="size-full wp-image-2223" src="/assets/img/posts/2020/06/Getting-a-message-from-the-Remote-Proxy.jpg" alt="Getting a message from the Remote Proxy" width="430" height="103" /></a>
  
  <p id="caption-attachment-2223" class="wp-caption-text">
    Getting a message from the Remote Proxy
  </p>
</div>

Make sure that the server is running, otherwise, you will get an exception. I will write another post about GRPC soon and won&#8217;t go into any details about it here.

### Smart Proxy Pattern

The smart proxy pattern helps to perform additional logic around resource access like caching, or locking shared resources, for example for file access. In the following example, the smart proxy will detect that there is already a lock around a file and reuse it.

### Implementation of the Smart Proxy

In this example, I am opening two file streams to the same file. Usually, this would throw an exception because the file is already in use. In my proxy, I will check if a FileStream is already in use and if so, return the existing one.

The implementation is pretty simple:

```csharp  
public class SmartProxy : ISmartProxy
{
    public void WriteTwiceToSameFile(string outputFile, string message)
    {
        var smartProxy = new SmartProxy();

        using var file = smartProxy.OpenWrite(outputFile);
        using var file2 = smartProxy.OpenWrite(outputFile);

        file.Write(Encoding.ASCII.GetBytes(message));
        file2.Write(Encoding.ASCII.GetBytes(message));

        file.Close();
        file2.Close();
    }
}  
```

The OpenWrite method is implemented in my proxy where I check if a stream already exists. If so, I return the existing stream, if not, I return a new stream.

```csharp  
public class SmartProxy : IFile
{
    private readonly Dictionary<string, FileStream> _openStreams = new Dictionary<string, FileStream>();

    public FileStream OpenWrite(string path)
    {
        try
        {
            var stream = File.OpenWrite(path);
            _openStreams.Add(path, stream);

            return stream;
        }
        catch (IOException)
        {
            if (_openStreams.ContainsKey(path))
            {
                var stream = _openStreams[path];

                if (stream != null && stream.CanWrite)
                {
                    return stream;
                }
            }

            throw;
        }
    }
}  
```

### Protective Proxy

The protective proxy manages access to resources and acts as a gatekeeper. This version eliminates repetitive security checks from client code and helps to achieve SOC, SRP, and DRY.

### Implementation of the Protective Proxy

In this example, I want to set up some access roles for a document. Only the author of the document is allowed to update its name and only a user with the editor role is allowed to review the document. The protective proxy checks the user roles and blocks any unauthorized action. The logic of the review and update method stay unchanged and they don&#8217;t even know anything about these security checks.

```csharp  
public class ProtectiveProxy : Document
{
    public ProtectiveProxy(string name, string content) : base(name, content)
    {
    }

    public override void UpdateName(string newName, User user)
    {
        if (user.Role != Roles.Author)
        {
            throw new UnauthorizedAccessException("Cannot update name unless in Author role.");
        }

        base.UpdateName(newName, user);
    }

    public override void CompleteReview(User editor)
    {
        if (editor.Role != Roles.Editor)
        {
            throw new UnauthorizedAccessException("Cannot review documents unless you are an Editor.");
        }

        base.CompleteReview(editor);
    }
}  
```

To test the implementation, I have some test cases in the ProtectiveProxyTests class.

### Virtual Proxy Pattern

The virtual proxy pattern is often used with expensive UI operations, for example when you have infinite scrolling like on the Facebook wall. The proxy is used as a placeholder and is also responsible for getting the data. Most of the time this proxy is used for lazy loading entities.

## Conclusion

This post described the proxy pattern and how to use its different versions. Before writing this post, I haven&#8217;t heard about the proxy and I don&#8217;t think that it is used too often. But I also believe that it is good to have heard about it and so you can use it whenever you need it.The code examples where take and inspired from <a href="https://app.pluralsight.com/library/courses/c-sharp-design-patterns-proxy/table-of-contents" target="_blank" rel="noopener noreferrer">this Pluralsight course</a>.

You can find the code of the demo on <a href="https://github.com/WolfgangOfner/.NetCore-ProxyPatterns" target="_blank" rel="noopener noreferrer">Github</a>.