---
title: 'How I learned programming - Part 4'
date: 2017-10-17T21:37:49+02:00
author: Wolfgang Ofner
categories: [Programming]
tags: [ASP.NET MVC, CSS, HTML, Javascript, learning, PHP]
---
Welcome to part 4 of how I learned programming. In my last post, I wrote about how I learned Java and C++ in Canada and how the teacher there impressed me. You can find this post <a href="/how-i-learned-programming-part-3/" target="_blank" rel="noopener">here</a>.

In this post, I will talk about how I got into web development and what problems I encountered while learning.

## HTML, CSS and Javascript&#8230; Ugh

Back in 2013 web development wasn&#8217;t fun at all. Especially when you are a beginner. HTML5 wasn&#8217;t standardized yet and there was a big problem with the browser compatibility. Especially with IE6. You could design a website, test it with every browser and it looks the same. Then you try Internet Explorer and it looked like something completely different and totally broken.

Anyways, I started learning HTML, CSS and Javascript first in theory in class and then with an assignment.

### Creating a company homepage

The first assignment was to create a homepage for a company. The following requirements needed to be implemented:

  * Use a 3 column grid layout with Bootstrap
  * Use the HTML5 semantic tags (aside, nav, header, footer,&#8230;)
  * Header: Logo + Headline with Web fonts  <span class="fontstyle0">(</span><span class="fontstyle0">https://www.google.com/fonts</span><span class="fontstyle0">)</span>
  * Nav: navigation elements with CSS transition (highlighting)
  * Aside: Welcome box
  * Section: 2x article incl. Bootstrap icons and placeholder text
  * Footer
  * Implement a register form (doesn&#8217;t have to send any data to the server)
  * Show the geolocation in the welcome box
  * Use rounded corners, shadows and other CSS attributes
  * Use Bootstrap, jQuery and jQueryUI

Doing this assignment wasn&#8217;t too hard. When learning HTML the <a href="https://www.w3schools.com/" target="_blank" rel="noopener">w3school homepage</a> is your best friend. Your first couple projects will look like websites from the 90&#8217;s. <a href="https://github.com/WolfgangOfner/Uni-CompanyHomepage" target="_blank" rel="noopener">Here</a> you can find my 90&#8217;s company website.

## Next step: PHP

Next, I learned PHP. PHP is a scripting language and contrary to Javascript, the code is executed on the server. For the assignment, I had to implement an URL shortener, like <a href="https://bitly.com/" target="_blank" rel="noopener">bitly.com</a>. The following tasks had to be implemented:

  * <span style="text-decoration: underline;">Non logged in users</span> 
      * can open the start page and can call shortened URLs
      * can log in or register
      * shortened URL is valid for 24 hours

  * <span style="text-decoration: underline;">A logged in user can</span> 
      * show / edit own URLs
      * show / edit user profile
      * logout
      * shortened URL is valid indefinitely

  * <span style="text-decoration: underline;">The starting page offers the following functions</span> 
      * Displays information about the service
      * Offers to shorten entered URLs
      * Shortened URLs consists of the server address and an 8 character long alpha numeric unique sequence

  * <span style="text-decoration: underline;">Shortening URLs</span> 
      * redirect with .htacess / web.config
      * the page contains the key for the shortened URL as a query string
      * redirect with correct HTTP response code

  * <span style="text-decoration: underline;">Signup / Login</span> 
      * user signs up with Email, username and password
      * optionally user can enter the first and last name, address and birthday
      * login works with username or email and password

  * <span style="text-decoration: underline;">Show / edit own URLS</span> 
      * List contains creation date and number of redirects
      * URLs can be activated / deactivated or deleted (with an AJAX request, which I have not implemented though)

  * <span style="text-decoration: underline;">Show / edit user profile</span> 
      * all field from the sing up are displayed and can be edited
      * for critical changes, the user has to re-enter his password (changing email, username or password)

  * <span style="text-decoration: underline;">Misc</span> 
      * user Separation of Concern
      * use useful HTML tags
      * use different architectural layers (UI, business logic, database,&#8230;)
      * implement OWASP best practices
      * use jQuery and Bootstrap

### My implementation

This assignment was fun to do but it also showed that PHP can be a pain if it&#8217;s mixed too much with HTML. Also setting up the database took my some time because we had to use PHP_pdo to connect the database. I can&#8217;t remember the details but for whatever reason, I couldn&#8217;t get it working with PHP 5.5 and so had to use 5.4. I had to use Visual Studio and installing the PHP tools didn&#8217;t work at the first time either.

For the assignment, I implemented most of the features. I didn&#8217;t implement the deletion of the URLs with Ajax and also didn&#8217;t apply all security guidelines. You can find my solution [here](https://github.com/WolfgangOfner/Uni-PHP-URL-Shortener).

## Learning ASP.NET MVC

The last part of learning web development was ASP.NET MVC. In theory, it&#8217;s pretty simple. You have Separation of Concern and the Controller takes the user input, modifies the Model and then sends the View to the user&#8217;s browser. ASP.NET is great because you have the full Visual Studio support and intelli sense and it also has already most of the security features built in. I understood all theoretical parts but I just didn&#8217;t get how it works in code. I sat in front of my Visual Studio and had no idea what was going on. Therefore I didn&#8217;t do the assignment. Around a year later I tried again to understand it. I made some progress but gave up again. A bit later I convinced myself that it can&#8217;t be that hard and tried again. This time I understood everything.

Now I can&#8217;t even explain why I didn&#8217;t understand it back then. But it shows that even if you don&#8217;t understand something the first or second time, don&#8217;t give up. After I started understanding it, I really liked it.

If you are wondering about the assignment. Here are the tasks of the assignment:

### Create a portal for blogging using ASP.NET MVC

  * <span style="text-decoration: underline;">anonymous users:</span> 
      * see an overview of the blogs
      * go through the archive (/archive/month/year)
      * read blogs
      * filter tags
      * search for tags or blogs

  * <span style="text-decoration: underline;">registered / logged in users</span> 
      * create a blog
      * edit a blog
      * display / edit user profile

  * <span style="text-decoration: underline;">Blog entry</span> 
      * title
      * friendly URL
      * content
      * author
      * tag(s)
      * timestamp

  * <span style="text-decoration: underline;">Misc</span> 
      * user Separation of Concern
      * use useful HTML tags
      * use different architectural layers (UI, business logic, database,&#8230;)
      * implement OWASP best practices
      * use <span class="fontstyle0">WYSIWYG-Editor,</span> jQuery and Bootstrap

&nbsp;

This is the story how I got into web development. In the last part of this series, I will tell you how I teach myself new stuff at home.

Next: <a href="/learned-programming-part-5/" target="_blank" rel="noopener">Part 5</a>

Previous: <a href="/how-i-learned-programming-part-3/" target="_blank" rel="noopener">Part 3</a>