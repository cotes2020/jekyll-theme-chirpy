---
title: '.Net Standard - Getting Started'
date: 2019-02-12T09:35:00+01:00
author: Wolfgang Ofner
categories: [Programming]
tags: [.Net, .net core, .net Standard, 'C#']
---
Today, some of my colleagues had a discussion about .Net Standard. Is it a new framework, an extension to classic .Net framework, or to .Net core? Confusion was great and in today&#8217;s post, I would like to shed light on the matter.

## What is .Net Standard?

.Net Standard is a specification which defines a set of APIs which the .Net platform has to implement. It is not another .Net platform though. You can only build libraries, not executables with it. On the following screenshot, you can see that .Net Standard contains APIs from the classic .Net framework, .Net core and Xamarin.

<div id="attachment_1579" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2019/02/What-is-Net-Standard..jpg"><img aria-describedby="caption-attachment-1579" loading="lazy" class="wp-image-1579" src="/wp-content/uploads/2019/02/What-is-Net-Standard..jpg" alt="What is .Net Standard" width="700" height="305" /></a>
  
  <p id="caption-attachment-1579" class="wp-caption-text">
    What is .Net Standard (<a href="https://app.pluralsight.com/library/courses/dotnet-standard-getting-started/table-of-contents" target="_blank" rel="noopener">Source</a>)
  </p>
</div>

The following screenshot shows that it defines a set all APIs that all .Net frameworks implement.

<div id="attachment_1580" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2019/02/Implementation-of-.Net-Standard.jpg"><img aria-describedby="caption-attachment-1580" loading="lazy" class="wp-image-1580" src="/wp-content/uploads/2019/02/Implementation-of-.Net-Standard.jpg" alt="Implementation of .Net Standard" width="700" height="338" /></a>
  
  <p id="caption-attachment-1580" class="wp-caption-text">
    Implementation of .Net Standard (<a href="https://app.pluralsight.com/library/courses/dotnet-standard-getting-started/table-of-contents" target="_blank" rel="noopener">Source</a>)
  </p>
</div>

&nbsp;

### The difference to Portable Class Libraries

Some of you might remember portable class libraries, which sound like .Net Standard. Both technologies have the same idea but a portable class library needs to be recompiled every time you want to use it for a different target. .Net Standard doesn&#8217;t have to be recompiled to be used for a different target. Check out the following screenshot to compare the differences:

<div id="attachment_1578" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2019/02/Portable-Class-Library-vs-Net-Standard.jpg"><img aria-describedby="caption-attachment-1578" loading="lazy" class="wp-image-1578" src="/wp-content/uploads/2019/02/Portable-Class-Library-vs-Net-Standard.jpg" alt="Portable Class Library vs .Net Standard" width="700" height="212" /></a>
  
  <p id="caption-attachment-1578" class="wp-caption-text">
    Portable Class Library vs .Net Standard (<a href="https://app.pluralsight.com/library/courses/dotnet-standard-getting-started/table-of-contents" target="_blank" rel="noopener">Source</a>)
  </p>
</div>

Portable class libraries are deprecated because .Net Standard is better in every way and therefore shouldn&#8217;t be used anymore.

## Choosing the right Version

A new version of .Net Standard always contains all previous APIs and additional ones. The following screenshot shows how a new version is built on all previous ones:

<div id="attachment_1582" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2019/02/Net-Standard-Versions.jpg"><img aria-describedby="caption-attachment-1582" loading="lazy" class="wp-image-1582" src="/wp-content/uploads/2019/02/Net-Standard-Versions.jpg" alt=".Net Standard Versions" width="700" height="554" /></a>
  
  <p id="caption-attachment-1582" class="wp-caption-text">
    Every version is built on the previous one (<a href="https://app.pluralsight.com/library/courses/dotnet-standard-getting-started/table-of-contents" target="_blank" rel="noopener">Source</a>)
  </p>
</div>

A .Net platform implements a specific .Net Standard version, for example .net core 1.0 implements .Net Standard 1.6. The enforce this backward compatibility, every .Net Standard version is immutable.

### Which Version to choose?

The best practice is to start with a high version number and implement all your features. Then target the lowest version possible. For example, start with 2.0 and then decrease to 1.6, then 1.5  until your project doesn&#8217;t compile anymore.

### Find out which Version a .Net Platform implements

Microsoft has some great <a href="https://docs.microsoft.com/en-us/dotnet/standard/net-standard#net-implementation-support" target="_blank" rel="noopener">documentation</a> about which .Net Standard version is implemented by which .Net framework version.

<div class="table-responsive">
  <table class="table table-striped table-bordered table-hover">
    <tr>
      <td>
        .NET Standard
      </td>
      
      <td>
        1.0
      </td>
      
      <td>
        1.1
      </td>
      
      <td>
        1.2
      </td>
      
      <td>
        1.3
      </td>
      
      <td>
        1.4
      </td>
      
      <td>
        1.5
      </td>
      
      <td>
        1.6
      </td>
      
      <td>
        2.0
      </td>
    </tr>
    
    <tr>
      <td>
        .Net Core
      </td>
      
      <td>
        1.0
      </td>
      
      <td>
        1.0
      </td>
      
      <td>
        1.0
      </td>
      
      <td>
        1.0
      </td>
      
      <td>
        1.0
      </td>
      
      <td>
        1.0
      </td>
      
      <td>
        1.0
      </td>
      
      <td>
        2.0
      </td>
    </tr>
    
    <tr>
      <td>
        .NET Framework
      </td>
      
      <td>
        4.5
      </td>
      
      <td>
        4.5
      </td>
      
      <td>
        4.5.1
      </td>
      
      <td>
        4.6
      </td>
      
      <td>
        4.6.1
      </td>
      
      <td>
        4.6.1
      </td>
      
      <td>
        4.6.1
      </td>
      
      <td>
        4.6.1
      </td>
    </tr>
  </table>
</div>

If you are looking for a specific API, you can go to <a href="https://docs.microsoft.com/en-gb/dotnet/api/" target="_blank" rel="noopener">https://docs.microsoft.com/en-gb/dotnet/api/</a> and search for it.

Version 1.6 has around 13,000 APIs whereas version 2.0 has already around 32,000 APIs which includes for example Primitives, Collections, Linq or Files

<div id="attachment_1586" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2019/02/APIs-of-.Net-Standard-2.0.jpg"><img aria-describedby="caption-attachment-1586" loading="lazy" class="wp-image-1586" src="/wp-content/uploads/2019/02/APIs-of-.Net-Standard-2.0.jpg" alt="APIs of .Net Standard 2.0" width="700" height="240" /></a>
  
  <p id="caption-attachment-1586" class="wp-caption-text">
    Some APIs of 2.0 (<a href="https://app.pluralsight.com/library/courses/dotnet-standard-getting-started/table-of-contents" target="_blank" rel="noopener">Source</a>)
  </p>
</div>

&nbsp;

## Migrating an existing project

Migrating to .Net Standard just for the sake of migrating is not the best strategy. It makes sense to migrate if the heart of your library is .Net Standard compatible and if you want to use it on different .Net platform.

### How to migrate

Open the .csproj file of the classic .Net framework project you want to migrate and delete everything. Then copy a new Project tag with the target framework of netstandard in it. If you want to migrate a .Net core project, you only have to change the target framework to netstandard. For details see the following screenshot:

<div id="attachment_1589" style="width: 431px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2019/02/Migrate-to-.Net-Standard.jpg"><img aria-describedby="caption-attachment-1589" loading="lazy" class="size-full wp-image-1589" src="/wp-content/uploads/2019/02/Migrate-to-.Net-Standard.jpg" alt="Migrate to .Net Standard" width="421" height="137" /></a>
  
  <p id="caption-attachment-1589" class="wp-caption-text">
    Migrate to .Net Standard
  </p>
</div>

If you are migrating a .Net core project, you are already done. For your .Net framework project, you have to delete the AssemblyInfo.cs and the packages.config files. Then you have to reinstall your NuGet packages. The reason why you don&#8217;t have to do that for .Net core is because it uses package referencing and not the packages.config.

## Targeting multiple platforms

If you want to target multiple frameworks, for example, .Net Standard and .Net 4.6.1, you only have to change the TargetFramework tag in the .csproj file to TargetFrameworks and separate the different framework with a semicolon.

<div id="attachment_1590" style="width: 461px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2019/02/Target-multiple-platforms.jpg"><img aria-describedby="caption-attachment-1590" loading="lazy" class="size-full wp-image-1590" src="/wp-content/uploads/2019/02/Target-multiple-platforms.jpg" alt="Target multiple platforms" width="451" height="129" /></a>
  
  <p id="caption-attachment-1590" class="wp-caption-text">
    Target multiple platforms
  </p>
</div>

If you use multiple target platforms, you can use if statements to use different code, depending on your target framework:

<div id="attachment_1591" style="width: 457px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2019/02/Execute-code-depending-on-the-target-platform.jpg"><img aria-describedby="caption-attachment-1591" loading="lazy" class="size-full wp-image-1591" src="/wp-content/uploads/2019/02/Execute-code-depending-on-the-target-platform.jpg" alt="Execute code depending on the target platform" width="447" height="91" /></a>
  
  <p id="caption-attachment-1591" class="wp-caption-text">
    Execute code depending on the target platform
  </p>
</div>

## Conclusion

In this short post, I explained the basics of .Net Standard and pointed out why it is better than the deprecated portable class library. Additionally, I showed how to migrate your existing project and how to target multiple platforms. For more information, I can highly recommend the Pluralsight course &#8220;<a href="https://app.pluralsight.com/library/courses/dotnet-standard-getting-started/table-of-contents" target="_blank" rel="noopener">.Net Standard: Getting Started</a>&#8221; by Thomas Claudius Huber.