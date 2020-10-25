---
title: Cross Site Scripting (XSS) in ASP .NET Core
date: 2020-06-17T09:42:15+02:00
author: Wolfgang Ofner
categories: [ASP.NET]
tags: [.net core 3.1, ASP.NET Core MVC, 'C#', Javascript, OWASP Top 10, Security]
---
Cross Site Scripting (XSS) is an attack where attackers inject code into a website which is then executed. XSS is on place seven of the <a href="https://owasp.org/www-project-top-ten/OWASP_Top_Ten_2017/" target="_blank" rel="noopener noreferrer">OWASP Top 10 list of 2017</a> but could be easily avoided. In this post, I will talk about the concepts of cross site scripting and how you can protect your application against these attacks.

## What is Cross Site Scripting

Cross site scripting is the injection of malicious code in a web application, usually, Javascript but could also be CSS or HTML. When attackers manage to inject code into your web application, this code often gets also saved in a database. This means every user could be affected by this. For example, if an attacker manages to inject Javascript into the product name on Amazon. Every user who opens the infected product would load the malicious code.

## Consequences of XSS Attacks

There are many possible consequences for your users if your website got attacked by cross site scripting:

  * Attackers could read your cookies and therefore gain access to your private accounts like social media or bank
  * Users may be redirected to malicious sites
  * Attackers could modify the layout of the website to lure users into unintentional actions
  * Users could be annoyed which will lead to damage to your reputation and probably a loss of revenue
  * Often used in combination with other attacks like [cross site request forgery (CSRF)](https://www.programmingwithwolfgang.com/cross-site-request-forgery-csrf-in-asp-net-core/)

## Best Practices against Cross Site Scripting Attacks

Preventing XSS attacks is pretty simple if you follow these best practices:

  * Validate every user input, either reject or sanitize unknown character, for example, < or > which can be used to create <script> tags
  * Test every input from an external source
  * Use HttpOnly for cookies so it is not readable by Javascript (therefore an attacker can&#8217;t use Javascript to read your cookies)
  * Use markdown instead of HTML editors

## Cross Site Scripting in ASP .NET Core

ASP .NET Core Is already pretty safe out of the box due to automatically encoding HTML, for example < gets encoded into &lt. Let&#8217;s have a look at two examples where XSS attacks can happen and how to prevent them. You can find the code for the demo on <a href="https://github.com/WolfgangOfner/MVC-XssDemo" target="_blank" rel="noopener noreferrer">Github</a>.

### ASP .NET Core 3.1 Demo

XSS can occur when you display text which a user entered. ASP .NET Core automatically encodes text when you use @Model, but displays the code as it if if you use @Html.Raw.

#### Preventing XSS Attacks in forms

The following code creates a form where the user can enter his user name. The input is displayed once in a safe way and once in an unsafe way.

```csharp  
@model Customer

<div asp-validation-summary="All"></div>

<form asp-action="Index">
    <div class="form-group">
        <label asp-for="UserName">Please enter your user name</label>
        <input type="text" class="form-control" asp-for="UserName" value="User">
    </div>
    <button type="submit" class="mt-md-1 btn btn-primary">Submit</button>
</form>

<br />

@if (!string.IsNullOrEmpty(Model.UserName))
{
    <div class="row">
        <p>Safe output: @(Model.UserName)</p>
    </div>
    <div class="row">
        <p>Unsafe output: @Html.Raw(Model.UserName)</p>
    </div>
}  
```

When a user enters his user name everything is fine. But when an attacker enters Javascript, the Javascript will be executed when the text is rendered inside the unsafe output

tag. When you enter the following code as your name:

```javascript  
<script>alert(&#8216;attacked&#8217;)</script>  
```

and click submit, an alert windows will be displayed.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/06/The-injected-code-got-executed.jpg"><img loading="lazy" src="/assets/img/posts/2020/06/The-injected-code-got-executed.jpg" alt="The injected code got executed causing Cross Site Scripting" /></a>
  
  <p>
    The injected code got executed
  </p>
</div>

When you click on OK, the text will be rendered into the safe output line and nothing will be displayed in the unsafe output line because the browser interprets the Javascript.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/06/The-Javascript-is-displayed-as-text-in-the-safe-output-line.jpg"><img aria-describedby="caption-attachment-2191" loading="lazy" class="size-full wp-image-2191" src="/assets/img/posts/2020/06/The-Javascript-is-displayed-as-text-in-the-safe-output-line.jpg" alt="The Javascript is displayed as text in the safe output line, no Cross Site Scripting possible" /></a>
  
  <p>
    The Javascript is displayed as text in the safe output line
  </p>
</div>

#### Preventing XSS Attacks in Query Parameters

Another way to inject code is through query parameters. If your application ready query parameters but doesn&#8217;t sanitize them, Javascript in it will be executed. The following code contains two forms. When you click on the button a query parameter will be read and printed to an alert box.

```csharp  
@using System.Text.Encodings.Web

@model Customer
@inject JavaScriptEncoder JavaScriptEncoder

<h2>Unsafe Javascript</h2>
<form asp-action="JavascriptAttack">
    @Html.HiddenFor(m => m.UserId)
    <div class="form-group">
        <label for="userName">Please enter your user id</label>
        <input type="text" class="form-control" id="userName" name="userName" value="User">
    </div>
    <button type="submit" class="mt-md-1 btn btn-primary" onclick="alert('Saving user name for account with id: @Context.Request.Query["userId"]');">Submit</button>
</form>
<br />
<h2>Safe Javascript</h2>
<form asp-action="JavascriptAttack">
    @Html.HiddenFor(m => m.UserId)
    <div class="form-group">
        <label for="userName2">Please enter your user id</label>
        <input type="text" class="form-control" id="userName2" name="userName2" value="User">
    </div>
    <button type="submit" class="mt-md-1 btn btn-primary" onclick="alert('Saving user name for account with id: @JavaScriptEncoder.Encode(Context.Request.Query["UserId"])');">Submit</button>
</form>
```

The first submit button will execute Javascript whereas the second one uses the JavaScriptEncode to encode the text first. To simulate an attack replace the value of the UserId with the following code and click enter:

```javascript  
%27);alert(&#8216;You got attacked&#8217;);//  
```

Click the submit button of the unsafe form and you will see two Javascript alerts. The first one saying &#8220;Saving user name for account with id: &#8221; and then a second one saying &#8220;You got attacked&#8221;.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/06/The-Javascript-got-executed.jpg"><img loading="lazy" src="/assets/img/posts/2020/06/The-Javascript-got-executed.jpg" alt="The Javascript got executed Cross Site Scripting" /></a>
  
  <p>
    The Javascript got executed
  </p>
</div>

When you click the submit button of the safe form, you will see the Javascript as text.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/06/The-Javascript-is-displayed-as-text-and-not-executed.jpg"><img loading="lazy" src="/assets/img/posts/2020/06/The-Javascript-is-displayed-as-text-and-not-executed.jpg" alt="The Javascript is displayed as text and not executed and no Cross Site Scripting" /></a>
  
  <p>
    The Javascript is displayed as text and not executed
  </p>
</div>

In reality, an attacker wouldn&#8217;t display an alert box but try to access your cookies or redirect you to a malicious website.

## Conclusion

This post showed what cross site scripting attacks are and how they can be executed. ASP .NET Core makes it very easy to prevent these attacks and to offer a great user experience to your users.

You can find the code for the demo on <a href="https://github.com/WolfgangOfner/MVC-XssDemo" target="_blank" rel="noopener noreferrer">Github</a>.