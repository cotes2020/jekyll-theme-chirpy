---
title: 'Whats new in C# 7.0'
date: 2017-10-31T16:11:26+01:00
author: Wolfgang Ofner
categories: [Programming]
tags: [.Net, 'C#']
---
Recently I made a post, about the new features in C# 6.0 and in this post I will show you whats new in C# 7.0.

## Whats new in C# 7.0

To use all features of C# 7.0 you have to use Visual Studio 2017. In theory, you can also use Visual Studio 2015 but it&#8217;s a pain to get it running. I highly recommend using Visual Studio 2017

### out Parameter

Before C# 7.0 the variable used as out parameter needed to be declared. With C# 7.0 you don&#8217;t have to do that anymore which makes the code shorter and more readable.

<div id="attachment_306" style="width: 410px" class="wp-caption aligncenter">
  <a href="http://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/out-parameter.jpg"><img aria-describedby="caption-attachment-306" loading="lazy" class="wp-image-306" src="http://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/out-parameter.jpg" alt="new in C# 7.0 the out parameter" width="400" height="247" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/out-parameter.jpg 524w, https://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/out-parameter-300x185.jpg 300w" sizes="(max-width: 400px) 100vw, 400px" /></a>
  
  <p id="caption-attachment-306" class="wp-caption-text">
    The new out parameter
  </p>
</div>

It&#8217;s not a big change but it&#8217;s a nice simplification of the code.

### Tuples

Tuples give you the possibility to return two values at a time. These two values can be different data types. On the screenshots below you can see the call of a simple method which returns two strings.

<div id="attachment_309" style="width: 832px" class="wp-caption aligncenter">
  <a href="http://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/Tuples-call.jpg"><img aria-describedby="caption-attachment-309" loading="lazy" class="wp-image-309 size-full" src="http://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/Tuples-call.jpg" alt="Tuples call" width="822" height="110" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/Tuples-call.jpg 822w, https://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/Tuples-call-300x40.jpg 300w, https://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/Tuples-call-768x103.jpg 768w" sizes="(max-width: 822px) 100vw, 822px" /></a>
  
  <p id="caption-attachment-309" class="wp-caption-text">
    Call of the method which returns a tuple value
  </p>
</div>

&nbsp;

<div id="attachment_304" style="width: 950px" class="wp-caption aligncenter">
  <a href="http://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/Tuples-return.jpg"><img aria-describedby="caption-attachment-304" loading="lazy" class="wp-image-304 size-full" src="http://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/Tuples-return.jpg" alt="Tuples return" width="940" height="145" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/Tuples-return.jpg 940w, https://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/Tuples-return-300x46.jpg 300w, https://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/Tuples-return-768x118.jpg 768w" sizes="(max-width: 940px) 100vw, 940px" /></a>
  
  <p id="caption-attachment-304" class="wp-caption-text">
    Method returns a tuple value
  </p>
</div>

To get access the returned values use variable.returnName. The returnName is the name you defined in the signature of the method.

To use Tuples you need to target the .Net Framework 4.7. If you target a lower .Net Framework, you have to install the System.ValueTuple NuGet package which can be found [here](https://www.nuget.org/packages/System.ValueTuple/).

### Pattern matching

With the new feature pattern matching it is now possible to have switch cases for data types. For example case int: do something, case string: do something else. On the screenshot below I show how to calculate the sum of all int and double values of a list of objects. If the element is a string the program writes the string to the console.

<div id="attachment_307" style="width: 529px" class="wp-caption aligncenter">
  <a href="http://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/pattern-matching-list.jpg"><img aria-describedby="caption-attachment-307" loading="lazy" class="wp-image-307 size-full" src="http://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/pattern-matching-list.jpg" alt="pattern matching list" width="519" height="267" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/pattern-matching-list.jpg 519w, https://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/pattern-matching-list-300x154.jpg 300w" sizes="(max-width: 519px) 100vw, 519px" /></a>
  
  <p id="caption-attachment-307" class="wp-caption-text">
    List of objects containing different data types
  </p>
</div>

[ ](http://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/pattern-matching-list.jpg)

<div id="attachment_308" style="width: 610px" class="wp-caption aligncenter">
  <a href="http://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/pattern-matching.jpg"><img aria-describedby="caption-attachment-308" loading="lazy" class="wp-image-308" src="http://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/pattern-matching.jpg" alt="pattern matching" width="600" height="452" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/pattern-matching.jpg 854w, https://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/pattern-matching-300x226.jpg 300w, https://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/pattern-matching-768x578.jpg 768w" sizes="(max-width: 600px) 100vw, 600px" /></a>
  
  <p id="caption-attachment-308" class="wp-caption-text">
    Switch statement for pattern matching calculating the sum of int and double values
  </p>
</div>

### Literal improvements

The last new feature I want to talk about is the literal improvement. With C# 7.0 it is now possible to separate const numbers with an underscore to improve the readability.

<div id="attachment_305" style="width: 539px" class="wp-caption aligncenter">
  <a href="http://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/Literal-improvements.jpg"><img aria-describedby="caption-attachment-305" loading="lazy" class="wp-image-305 size-full" src="http://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/Literal-improvements.jpg" alt="Literal improvements" width="529" height="195" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/Literal-improvements.jpg 529w, https://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/Literal-improvements-300x111.jpg 300w" sizes="(max-width: 529px) 100vw, 529px" /></a>
  
  <p id="caption-attachment-305" class="wp-caption-text">
    Better readability of long numbers
  </p>
</div>

These numbers can be used as normal numbers without an underscore which means when printed the underscore won&#8217;t be printed.

### Code and further reading

You can find the code examples on [GitHub](https://github.com/WolfgangOfner/CSharp-7.0). A more extensive post about whats new in C# 7.0 can be found in the <a href="https://docs.microsoft.com/en-us/dotnet/csharp/whats-new/csharp-7" target="_blank" rel="noopener noreferrer">official documentation</a>.