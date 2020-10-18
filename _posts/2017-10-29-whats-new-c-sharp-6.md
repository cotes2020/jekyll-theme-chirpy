---
title: 'Whats new in C# 6.0'
date: 2017-10-29T17:26:58+01:00
author: Wolfgang Ofner
categories: [Programming]
tags: [.Net, 'C#']
---
C# 6.0 is around for a while but in the last couple weeks, I spoke with many programmers who don&#8217;t know anything about the new features. Therefore I want to present some of the new features in this post.

To use C# 6.0 you need at least Visual Studio 2015 which has the new Roslyn compiler which is needed for C# 6.0. Theoretically, you could install the compiler for older versions but I would recommend using the new Visual Studio versions if possible.

## Whats new in C# 6.0

I will only present a few features of C# 6.0 in this post. For all new features please see the <a href="https://docs.microsoft.com/en-us/dotnet/csharp/whats-new/csharp-6" target="_blank" rel="noopener">official documentation</a>.

### Read-only properties

the new read-only properties enable real read only-behavior. To achieve this, just remove the setter as shown below.

[<img loading="lazy" class="aligncenter size-full wp-image-284" src="http://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/read-only-property.jpg" alt="read only property" width="468" height="124" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/read-only-property.jpg 468w, https://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/read-only-property-300x79.jpg 300w" sizes="(max-width: 468px) 100vw, 468px" />](http://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/read-only-property.jpg)

&nbsp;

### String interpolation {#string-interpolation}

String interpolation is my favorite new feature of C# 6.0. This new feature replaces the string.format and makes it easier and faster to combine strings and variables. To use this place a $ in front of the quotation mark of the string. Now you can write a variable directly into the string. You only have to put the variable in curly braces.

[<img loading="lazy" class="aligncenter size-full wp-image-285" src="http://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/string-interpolation.jpg" alt="string interpolation" width="699" height="91" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/string-interpolation.jpg 699w, https://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/string-interpolation-300x39.jpg 300w" sizes="(max-width: 699px) 100vw, 699px" />](http://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/string-interpolation.jpg)

&nbsp;

### Expression-bodied function

Expression-bodied functions can help to reduce unnecessary lines of code. You can use this new feature only when the method has only a single statement. Below you can see an example on how to use expression-bodied functions.

[<img loading="lazy" class="aligncenter size-full wp-image-281" src="http://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/expression-body.jpg" alt="expression body" width="848" height="113" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/expression-body.jpg 848w, https://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/expression-body-300x40.jpg 300w, https://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/expression-body-768x102.jpg 768w" sizes="(max-width: 848px) 100vw, 848px" />](http://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/expression-body.jpg)

I don&#8217;t use this function too often because I like the method block. This makes it easier to read the code for me.

&nbsp;

### Using static {#using-static}

Using static brings some syntactic sugar to C# 6.0. You can declare a namespace static as shown below.

[<img loading="lazy" class="aligncenter size-full wp-image-286" src="http://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/using-static-2.jpg" alt="using static" width="357" height="54" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/using-static-2.jpg 357w, https://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/using-static-2-300x45.jpg 300w" sizes="(max-width: 357px) 100vw, 357px" />](http://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/using-static-2.jpg)

After you did this, it&#8217;s not necessary to qualify the class when using a method. For example, it is not necessary to use Math.PI. When you only have to use PI, the code gets easier to read.

[<img loading="lazy" class="aligncenter size-full wp-image-287" src="http://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/using-static.jpg" alt="using static" width="789" height="115" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/using-static.jpg 789w, https://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/using-static-300x44.jpg 300w, https://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/using-static-768x112.jpg 768w" sizes="(max-width: 789px) 100vw, 789px" />](http://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/using-static.jpg)

&nbsp;

### Null-conditional operator {#null-conditional-operators}

Every programmer who uses objects has encountered a null reference exception. To prevent this C# 6.0 introduces the null-conditional operator. To prevent the throwing of a null reference exception, place the Elvis operator (?) after the element which might be null. The example below shows that if the person is null, &#8220;unknown&#8221; will be returned.

[<img loading="lazy" class="aligncenter size-full wp-image-283" src="http://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/null-conditional-operators.jpg" alt="null conditional operator" width="623" height="106" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/null-conditional-operators.jpg 623w, https://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/null-conditional-operators-300x51.jpg 300w" sizes="(max-width: 623px) 100vw, 623px" />](http://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/null-conditional-operators.jpg)

Without the ?? &#8220;unknown&#8221; part, null would have been returned. If you don&#8217;t return a value in case of null, you have to make sure that the left side of the = is a nullable value.

&nbsp;

### Nameof

Nameof enables the developer to get the name of the variable. I used this feature for logging. With nameof I logged the class in which something happened. In the example below, you can see how you can achieve that.

[<img loading="lazy" class="aligncenter size-full wp-image-282" src="http://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/nameof.jpg" alt="nameof" width="835" height="92" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/nameof.jpg 835w, https://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/nameof-300x33.jpg 300w, https://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/nameof-768x85.jpg 768w" sizes="(max-width: 835px) 100vw, 835px" />](http://www.programmingwithwolfgang.com/wp-content/uploads/2017/10/nameof.jpg)

&nbsp;

### Code and further reading

You can find the code examples on <a href="https://github.com/WolfgangOfner/CSharp-6.0" target="_blank" rel="noopener">GitHub</a>. A more extensive post about whats new in C# 6.0 can be found in the <a href="https://docs.microsoft.com/en-us/dotnet/csharp/whats-new/csharp-6" target="_blank" rel="noopener">official documentation</a>.

&nbsp;