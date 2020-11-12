---
title: We moved to Jekyll
date: 2020-10-26
author: Wolfgang Ofner
categories: [Miscellaneous ]
tags: [Jekyll, Azure Static Web App, Docker, Github Action, WordPress]
---
When I started my blog, I was looking for something simple so I can write my posts and don't have to manage much behind the scenes. WordPress was the most known blogging platform back then and I also got a cheap hosting package with WordPress included. Taking WordPress back then was a no-brainer but in the last three years since then, I was never really happy with it. Today, I want to talk about the reasons for the migration to Jekyll, what other options I considered, and what disadvantages the migration has.

## Why migrating away from WordPress
WordPress is the most popular blogging framework and offers many plugins which can be easily installed. Additionally, it offers a nice admin backend where you can manage everything from comments, posts, or Google Analytics. I have never used WordPress before but with the marketplace, Google search, and some try and error, I had the first version of my blog running within a couple of hours. A couple of months later, I changed the theme and made some CSS changes. This was when the "problems" started. 

WordPress is based on PHP which means that you can edit the whole code and create submodules of it so you can safely upgrade themes. I am not a PHP fan, therefore I never wanted to get into it to change the small details that I didn't like. Over time I had some layout imperfections like a horizontal scrolling bar although there was nothing to scroll to. I couldn't figure out where it came from (I guess from one of the plugins) but as I said, I didn't want to get into PHP.

The second and way bigger reason why I decided to move away from WordPress is its performance. I am really fortunate that I grew a reader base from all over the world over the last three years. This leads to performance problems for the users though. My WordPress instance is running on a server in central Europ which leads to quite some latency for Asian or American readers. Additionally, due to the load WordPress (maybe also the server) took too long to process the request. 

## Migrating away but where to?
Over the last years, I was thinking a couple of times to migrate to a different system. My first idea was to write it myself with ASP.NET Core (inspired by Scott Hanselman) but I discarded the idea quickly because it would be quite some work and I didn't have any good and affordable hosting options. Most cheap hosting providers only offer support for WordPress or other PHP frameworks but barely support for .NET Core.
The next idea was to migrate to Azure App Service to have a better insight into what's going on and the possibility to replicate it to several locations. This wasn't an option due to the pricing. Azure App Service is good when you need computing power but with a blog website, you don't have any computing except loading some text. This would be total overkill and one instance would cost around 70$ a month (most of it was the MySQL DB). 70$ a month is not an option, especially compared to the 35$ a year I am paying with my existing WordPress instance.

In May this year, Microsoft announced Azure Static Web Apps which offer free hosting for Javascript frameworks. I took this as an opportunity to look into React (React because I didn't like Angular 2) but I still don't like Javascript and writing everything myself in React was not appealing for me.
Microsoft improved Azure Static Web Apps and With all the options above ruled out, I decided to go with Jekyll and Azure Static Web App.

## Why use Jekyll and Azure Static Web App?
The biggest advantage of Jekyll is that it supports HTML, Javascript, and CSS and generates static sites. This means that it is blazing fast since there are no computing or database lookups needed anymore. The second big advantage is that it is not PHP and so I can easily change everything to my liking. The next advantage is the simple integration with Azure Static App. The deployment gets created, more or less, automatically and I have full control over the whole process.

I am fortunate that I have readers from all over the world. So far, they all had to connect to a server in Europe. With Azure Static Web App, my blog gets replicated on five servers the US, Europe, and Asia which reduces the latency for none Europeans. I only have Google Analytics data for a couple of days but so far the average page load time went down by 52%. The further the user is away, the bigger the performance increase is. For example, the page load time for US-based users decreased by 76%.

## How to migrate from WordPress to Jekyll
Migrating from WordPress to Jekyll is easy, in theory. I used the Jekyll Exporter plugin to export my posts and pictures. The problem is, that the plugin only works when there are no other plugins active. I am using around 10 plugins and deactivating them took around 30 min and gave me many 404 and 500 error pages. It was a bit annoying but also made me happy to leave WordPress behind. Once I had my posts and pictures, I was surprised how well the export was. All posts got converted to markdown and could already be used. I edited mainly the images so they don't use the WordPress CSS classes anymore and look nice on mobile and desktop and removed some boiler-plate text. It took me a couple of hours and some regex to get everything cleaned up.

### Docker is awesome
This migration reminded me of how awesome Docker is and how it can improve the efficiency of developers. The theme I am using needs besides Ruby some other Linux programs I have never heard about and also which I didn't want to install on my Windows 10 WSL. Instead of going through the kinda complicated build and run process, I use Docker to run and build the whole website. The only downside of this approach is that I have to build the website before checking in. In the near future, I want to extend the Github Action so it uses Docker to build my website before the deployment.

### New Features
I tried to change as little as possible but the theme I am using has some nice eye candy features. The biggest improvement is the built-in dark mode which enables users to switch between a light and dark theme. Another nice feature is the listing of posts inside a category including their publish dates.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/11/Display-all-Posts-of-a-Category.jpg"><img loading="lazy" src="/assets/img/posts/2020/11/Display-all-Posts-of-a-Category.jpg" alt="Display all Posts of a Category" /></a>
  
  <p>
   Display all posts of a category
  </p>
</div>

The last feature I really like is the table of content on the right side which shows you automatically where you are at the moment and also lets you create links to headlines.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/11/Table-of-contents-in-Jekyll.jpg"><img loading="lazy" src="/assets/img/posts/2020/11/Table-of-contents-in-Jekyll.jpg" alt="Table of contents in Jekyll" /></a>
  
  <p>
   Table of contents in Jekyll
  </p>
</div>

## Problems with Jekyll
To be honest, I had no major problem. The biggest problems were probably some CSS changes to make something look nice on mobile and desktop. Another inconvenience was that Azure Static Web App doesn't support root domain (programmingwithwolfgang.com) yet. The documentation suggests to use Cloudflare and configure a redirect from programmingwithwolfgang.com to wwww.programmingwithwolfgang.com. It should work but I didn't want to add Cloudflare yet. 

Fortunately, my current hoster has a setting to redirect non-www requests to www.* and since my package is running for another 11 months, I am using this for now. The Azure Static Web App team already said that the support for root domains should be implemented soon.

## Conslusion
Moving from WordPress to Jekyll was surprisingly easy and so far, I don't regret it at all. Jekyll gives me the ability to change every detail of my website without going into WordPress and PHP and also allows me to run the site locally using Docker which helps a lot with productivity when trying new features. The best part of moving away from WordPress to Jekyll on Azure Static Web App is that the website is way faster and also replicated all over the world which should increase the user experience significantly.

You can find the code of my website on <a href="https://github.com/WolfgangOfner/ProgrammingWithWolfgangWebsite" target="_blank" rel="noopener noreferrer">GitHub</a>.