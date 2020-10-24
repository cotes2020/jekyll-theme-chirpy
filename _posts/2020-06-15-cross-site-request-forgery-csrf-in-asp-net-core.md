---
title: Cross Site Request Forgery (CSRF) in ASP .NET Core
date: 2020-06-15T08:42:05+02:00
author: Wolfgang Ofner
categories: [ASP.NET]
tags: [.net core 3.1, ASP.NET Core MVC, 'C#', CSRF, OWASP Top 10, Security]
---
Cross Site Request Forgery, also known as session riding is an exploit where attackers trick users to send requests that they don&#8217;t know about and don&#8217;t want to do. It was on the OWASP Top 10 every year, except in 2017. Although it is not on the current list, it is still important that developers take care of it and don&#8217;t leave any vulnerabilities in their application. Today I will describe what Cross Site Request Forgery is and how it can be prevented in ASP .NET Core MVC using .NET Core 3.1

## What is Cross Site Request Forgery (CSRF)

Attackers using cross site request forgery try to trick users to send malicious requests to a website that trusts the user. This is possible because websites trust the browser of a user using cookies. The goal of the attack is to use your identity cookie to execute requests. For example, you log in to your Facebook account. Facebook then sets a cookie in your browser to identify you. An attacker could now trick you to send a request to Facebook. This request could be sending a message to thousands of users promoting a malicious website.

## How can Attackers perform a Cross Site Request Forgery Attack?

The most common way to perform a cross site request forgery attack is by luring users to a malicious website using phishing emails. Attacks send millions of emails claiming that you won something and that you have to click on a link to collect your price. On the malicious site, you are asked to click on a button to accept your prize. This button click then sends the malicious request.

### Requirements for an Attack

To successfully perform a cross site request forgery attack, the following requirements have to be met:

  * The user must have visited the attacked site (Facebook in the example above)
  * The user must visit the site of the attacker (usually via phishing emails)

The attackers try to make money with these attacks, for example, promoting something or sending links to malicious websites.

## Preventing Cross Site Request Forgery Attacks

There are some simple tricks you can use to prevent CSRF attacks:

  * Use SameSite cookies, at least lax but preferably strict
  * Use antiforgery tokens
  * Always use HTTP Post with forms

HttpOnly is good practice if you don&#8217;t need to access the cookie using Javascript. Be aware that encrypting the cookie does not help against CSRF attacks.

### Using the SameSite Setting

In ASP .NET Core 2.1 and higher, you can use CookieOptions to set the SameSite attribute. Use at least lax but use strict wherever possible. To set the setting use the following code:

```csharp  
Response.Cookies.Append("UserId", "1234", new CookieOptions  
{  
    SameSite = SameSiteMode.Lax  
});  
```

### Using Antiforgery Tokens

The ASP .NET Core server uses two randomly generated antiforgery tokens. The first one is sent as a cookie and the second one is places as a hidden form field. When the form is submitted both tokens are sent back to the server. If a request does not send both tokens, the request is not accepted.

The ASP .NET Core tag helper automatically includes the antiforgery token into a form field. You can create a form for a name using this code:

```html  
<form asp-action="SaveProfile">
   <div class="form-group">  
      <label for="name">Enter your name</label>  
      <input type="text" class="form-control" asp-for="Name" id="name" />  
   </div>
   <button type="submit" class="btn btn-primary">Submit</button>  
</form>
```

When you look at the HTML code of the form, you can see the generated field for the token.

<div id="attachment_2183" style="width: 534px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2020/06/The-auto-generated-antiforgery-token-in-the-form.jpg"><img aria-describedby="caption-attachment-2183" loading="lazy" class="size-full wp-image-2183" src="/assets/img/posts/2020/06/The-auto-generated-antiforgery-token-in-the-form.jpg" alt="The auto-generated antiforgery token in the form to prevent cross site request forgery attacks" width="524" height="134" /></a>
  
  <p id="caption-attachment-2183" class="wp-caption-text">
    The auto-generated antiforgery token in the form
  </p>
</div>

You can also see two cookies in your browser. In Chrome you can see the cookies when you open the Developer Tools (F12) and then click on Application and then on Cookies.

<div id="attachment_2184" style="width: 669px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2020/06/The-User-Id-and-Anti-Forgery-Cookie.jpg"><img aria-describedby="caption-attachment-2184" loading="lazy" class="size-full wp-image-2184" src="/assets/img/posts/2020/06/The-User-Id-and-Anti-Forgery-Cookie.jpg" alt="The User Id and Anti Forgery Cookie" width="659" height="297" /></a>
  
  <p id="caption-attachment-2184" class="wp-caption-text">
    The User Id and Anti Forgery Cookie
  </p>
</div>

The last step is to tell the server to check the antiforgery token. You can do this by using the ValidateAntiForgeryToken attribute on an action.

```csharp  
[HttpPost]  
[ValidateAntiForgeryToken]  
public IActionResult SaveProfile(Customer model)  
{  
    // do something with the model
    
    return View();  
}  
```

## Conclusion

Today, I explained what cross site request forgery (CSRF) attacks are and how easy it is to protect your application against it using built-in ASP .NET Core functionality. If you want to read more about security risks, I recommend you to take a look at the <a href="https://owasp.org/www-project-top-ten/OWASP_Top_Ten_2017/" target="_blank" rel="noopener noreferrer">OWAS Top 10</a>.