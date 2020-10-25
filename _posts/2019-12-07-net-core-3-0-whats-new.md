---
title: '.Net Core 3.0 - What''s new'
date: 2019-12-07T11:14:37+01:00
author: Wolfgang Ofner
categories: [Programming]
tags: [.net core, .net core 3.0, 'C#']
---
Microsoft released with .net core 3.0 the next major version of .net core which brings improvements to the deployment, .net core WPF and WinForms applications and much more. Today, I want to showcase some of these new features.

## Build and Deployment Improvements

.net core used to create a dll as an output, even for a console application. Starting with .net core 3.0, an exe file is created by default. The downside of the default behavior is that it copies all dependencies into the output folder which can be hundreds of files in a bigger application.

### Single File Executables

A pretty neat feature of .net core is that you can create a single file executable that contains all dependencies. This makes the output way clearer. To create a single file executable, you only have to add the RuntimeIdentifier and PublishSingleFile in your csproj file.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/12/Configure-your-application-to-publish-as-a-single-file.jpg"><img loading="lazy" src="/assets/img/posts/2019/12/Configure-your-application-to-publish-as-a-single-file.jpg" alt="" /></a>
  
  <p>
    Configure your application to publish as a single file
  </p>
</div>

The RuntimeIdentifier tells the compiler for what operating system it should create the executable. This could be, for example, win10-x64 or win10-x86. Creating a single file will make the executable way bigger since all the dependencies are packed into the file.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/12/Everything-got-packed-into-the-executable.jpg"><img loading="lazy" src="/assets/img/posts/2019/12/Everything-got-packed-into-the-executable.jpg" alt="Everything got packed into the executable" /></a>
  
  <p>
    Everything got packed into the executable
  </p>
</div>

### Trim the publish output

To reduce the size of your published output, you can trim it. Trimming removes not needed dependencies.

To configure ready to run, you only have to add the PublishedTrim tag to your csproj and set it to true.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/12/Remove-not-needed-dependencies-before-the-publish.jpg"><img loading="lazy" src="/assets/img/posts/2019/12/Remove-not-needed-dependencies-before-the-publish.jpg" alt="Remove not needed dependencies before the publish" /></a>
  
  <p>
    Remove not needed dependencies before the publish
  </p>
</div>

Be careful though because if you work with reflections, dependencies might get removed although you need them. You have to test your application before using it.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/12/The-executable-is-much-smaller-after-the-trim.jpg"><img loading="lazy" src="/assets/img/posts/2019/12/The-executable-is-much-smaller-after-the-trim.jpg" alt="The executable is much smaller after the trim" /></a>
  
  <p>
    The executable is much smaller after the trim
  </p>
</div>

### Ready to run publish

With .net core 3.0, you can create ready to run executables. This uses ahead of time compilation, to have better startup time because it contains the native code and .net intermediate code. The additional .net intermediate code might make the executable bigger.

To configure ready to run, you only have to add the PublishReadyToRun tag to your csproj and set it to true.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/12/Publish-ready-to-run.jpg"><img loading="lazy" size-full" src="/assets/img/posts/2019/12/Publish-ready-to-run.jpg" alt="Publish your .net core 3.0 application ready to run" /></a>
  
  <p>
    Publish ready to run
  </p>
</div>

## .Net Core 3.0 removes JSON.Net

.Net Core 3.0 introduces the new Namespace System.Text.Json. Until now JSON.Net was the standard JSON library for .net applications and also .net core depends on it. With .net core 3.0, the namespace System.Text.Json was introduced to remove this dependency. The classes in this namespace are super fast and need only a low amount of ram. The downside is that they are not feature-rich yet. This should change in future versions. You don&#8217;t have to use them but you can.

## .Net Core 3.0 goes Desktop

With .net core 3.0, you can now create WPF and WinForms applications. Currently, they are Windows only and the tooling needs a bit more work but new desktop applications can and should use .net core. Visual Studio provides a template for the .net core WPF application.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/12/Create-a-WPF-.Net-Core-Application.jpg"><img loading="lazy" src="/assets/img/posts/2019/12/Create-a-WPF-.Net-Core-Application.jpg" alt="Create a WPF .Net Core 3.0 Application" /></a>
  
  <p>
    Create a WPF .Net Core Application
  </p>
</div>

After the application is created, you can see in the csproj file that it is a WPF application and used .net core 3.0

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/12/csproj-file-of-the-.Net-Core-WPF-project.jpg"><img loading="lazy" size-full" src="/assets/img/posts/2019/12/csproj-file-of-the-.Net-Core-WPF-project.jpg" alt="csproj file of the .Net Core 3.0 WPF project" /></a>
  
  <p>
    csproj file of the .Net Core WPF project
  </p>
</div>

The tooling is not perfect yet but if you need additional UI controls you can use the Microsoft.Toolkit.Wpf.UI.Controls namespace for WinUI features. With this namespace you can, for example, draw in a canvas in your application.

The .net core application can be built as an MSIX package which will contain most of the needed references and can be distributed via the Microsoft Store. You can find the demo .net core WPF application on <a href="https://github.com/WolfgangOfner/WPFDotNetCore" target="_blank" rel="noopener noreferrer">GitHub</a>.

## Additional Features in .Net Core 3.0

There are many more new features in .net core 3.0 like:

  * Platform-dependent Intrinsics: provide access to hardware-specific instructions and can improve performance
  * Cryptography improvements: The new ciphers AES-GCM and AES CCM were added.
  * HTTP/2 support: Note HTTP/2 is not enabled by default.
  * Serial Port support for Linux
  * Improved Garbage Collector

## Conclusion

.net core is great and with the new version, it got even better. My favorite new features are the features around the improved deployment. Also, the new namespace for JSON is nice, although not production-ready yet if you have complex requirements. I am not a desktop developer but I guess that .net core 3.0 is a major game-changer for desktop developers.

You can find the code of today&#8217;s demo on <a href="https://github.com/WolfgangOfner/CSharp-8.0" target="_blank" rel="noopener noreferrer">Github.</a>