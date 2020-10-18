---
title: ASP.NET Core logging to a database with NLog
date: 2019-10-01T11:18:13+02:00
author: Wolfgang Ofner
categories: [ASP.NET]
tags: [ASP.NET Core MVC, Logging, NLog, SQL]
---
ASP.NET core has seen rapid development in the last years. Additionally, there were some breaking changes since version 1, for example, the project.json file got removed. Unfortunately, the documentation is lacking behind this rapid development. I had exactly this problem when I wanted to use NLog to log to my database. There was plenty of documentation but none of it worked because it was for .net core 1.x.

Today, I want to talk about how I implemented NLog with ASP.NET core 2.2 and how I configured it to log to my database. You can find the source code for the following demo on <a href="https://github.com/WolfgangOfner/MVC-Nlog" target="_blank" rel="noopener noreferrer">GitHub</a>.

## Installing Nlog

I created a new web project with .net core and the MVC template and added the NLog.Web.AspNetCore NuGet package. Next, I create a new file NLog.config in the root folder of my solution. This file will contain all the configs for NLog. Now it is time to fill the config file with some content. You can find the source code for this demo on <a href="https://github.com/WolfgangOfner/MVC-Nlog" target="_blank" rel="noopener noreferrer">Github</a>.

### Implementing the Nlog config

To get started, create a database and then the Log table. You can find the script to create the table at the bottom of my config file. Now let&#8217;s inspect the config file:

<div id="attachment_1768" style="width: 442px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2019/10/NLog-internal-logging.jpg"><img aria-describedby="caption-attachment-1768" loading="lazy" class="size-full wp-image-1768" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2019/10/NLog-internal-logging.jpg" alt="NLog internal logging" width="432" height="120" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2019/10/NLog-internal-logging.jpg 432w, https://www.programmingwithwolfgang.com/wp-content/uploads/2019/10/NLog-internal-logging-300x83.jpg 300w" sizes="(max-width: 432px) 100vw, 432px" /></a>
  
  <p id="caption-attachment-1768" class="wp-caption-text">
    NLog internal logging
  </p>
</div>

The first section of the file is for internal logs of Nlog. These logs come in handy when you have a problem with Nlog. There you can configure what level of logging you want and where the log file should be created. You can also configure whether the file should be reloaded on save with autoReload.

<div id="attachment_1769" style="width: 522px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2019/10/Configure-the-database-connection.jpg"><img aria-describedby="caption-attachment-1769" loading="lazy" class="size-full wp-image-1769" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2019/10/Configure-the-database-connection.jpg" alt="Configure the database connection" width="512" height="132" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2019/10/Configure-the-database-connection.jpg 512w, https://www.programmingwithwolfgang.com/wp-content/uploads/2019/10/Configure-the-database-connection-300x77.jpg 300w" sizes="(max-width: 512px) 100vw, 512px" /></a>
  
  <p id="caption-attachment-1769" class="wp-caption-text">
    Configure the database connection
  </p>
</div>

The next section is for configuring the database connection. The variables are read from the appsettings.json from the NlogConnection section. You can see the appsettings.json section on the following screenshot.

<div id="attachment_1770" style="width: 304px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2019/10/Settings-for-Nlog-from-appsettings.json_.jpeg"><img aria-describedby="caption-attachment-1770" loading="lazy" class="size-full wp-image-1770" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2019/10/Settings-for-Nlog-from-appsettings.json_.jpeg" alt="Settings for Nlog from appsettings.json" width="294" height="117" /></a>
  
  <p id="caption-attachment-1770" class="wp-caption-text">
    Settings for Nlog from appsettings.json
  </p>
</div>

The commandText section defines the insert statement. This is straight forward and you don&#8217;t have to edit anything.

<div id="attachment_1772" style="width: 474px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2019/10/Setting-up-the-insert-statement-for-logging.jpg"><img aria-describedby="caption-attachment-1772" loading="lazy" class="size-full wp-image-1772" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2019/10/Setting-up-the-insert-statement-for-logging.jpg" alt="Setting up the insert statement for logging" width="464" height="261" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2019/10/Setting-up-the-insert-statement-for-logging.jpg 464w, https://www.programmingwithwolfgang.com/wp-content/uploads/2019/10/Setting-up-the-insert-statement-for-logging-300x169.jpg 300w" sizes="(max-width: 464px) 100vw, 464px" /></a>
  
  <p id="caption-attachment-1772" class="wp-caption-text">
    Setting up the insert statement for logging
  </p>
</div>

The last section lets you specify rules about your log. You can configure which logger should log where. In my example, every logger logs messages with the log level Info and higher into the database. Another example could be to log information from one logger to the database and the information from another one to a file.

<div id="attachment_1773" style="width: 417px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2019/10/Rules-for-logging.jpg"><img aria-describedby="caption-attachment-1773" loading="lazy" class="size-full wp-image-1773" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2019/10/Rules-for-logging.jpg" alt="Rules for logging" width="407" height="53" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2019/10/Rules-for-logging.jpg 407w, https://www.programmingwithwolfgang.com/wp-content/uploads/2019/10/Rules-for-logging-300x39.jpg 300w" sizes="(max-width: 407px) 100vw, 407px" /></a>
  
  <p id="caption-attachment-1773" class="wp-caption-text">
    Rules for logging
  </p>
</div>

## Using Nlog

Using Nlog in your application is really simple. First, you have to tell your WebHost to use Nlog in the CreateWebHostBuilder by simply adding .UseNlog() at the end of the statement.

<div id="attachment_1774" style="width: 477px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2019/10/Use-Nlog-in-the-WebHostBuilder.jpg"><img aria-describedby="caption-attachment-1774" loading="lazy" class="size-full wp-image-1774" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2019/10/Use-Nlog-in-the-WebHostBuilder.jpg" alt="Use Nlog in the WebHostBuilder" width="467" height="106" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2019/10/Use-Nlog-in-the-WebHostBuilder.jpg 467w, https://www.programmingwithwolfgang.com/wp-content/uploads/2019/10/Use-Nlog-in-the-WebHostBuilder-300x68.jpg 300w" sizes="(max-width: 467px) 100vw, 467px" /></a>
  
  <p id="caption-attachment-1774" class="wp-caption-text">
    Use Nlog in the WebHostBuilder
  </p>
</div>

That&#8217;s all you have to do. Now you can already use the logger in your application. To use the logger, inject the ILogger interface with the type of the class which uses it. The ILogger interface provides useful methods like LogInformation() or LogCritical(). Call one of the methods and insert your log message.

<div id="attachment_1775" style="width: 492px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2019/10/Use-ILogger-to-log-messages.jpg"><img aria-describedby="caption-attachment-1775" loading="lazy" class="size-full wp-image-1775" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2019/10/Use-ILogger-to-log-messages.jpg" alt="Use ILogger to log messages" width="482" height="407" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2019/10/Use-ILogger-to-log-messages.jpg 482w, https://www.programmingwithwolfgang.com/wp-content/uploads/2019/10/Use-ILogger-to-log-messages-300x253.jpg 300w" sizes="(max-width: 482px) 100vw, 482px" /></a>
  
  <p id="caption-attachment-1775" class="wp-caption-text">
    Use ILogger to log messages
  </p>
</div>

## Testing the implementation

To test that the logging is working, you have only to start your application and call one of the controllers which do some logging. Then you can check in your database in the Log table and you should see the log entries there.

On the following screenshot, you can see that I called the Index and the Privacy method once which create a log entry for both calls.

<div id="attachment_1776" style="width: 710px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2019/10/Log-entries-in-the-Log-table.jpg"><img aria-describedby="caption-attachment-1776" loading="lazy" class="wp-image-1776" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2019/10/Log-entries-in-the-Log-table.jpg" alt="Log entries in the Log table" width="700" height="51" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2019/10/Log-entries-in-the-Log-table.jpg 866w, https://www.programmingwithwolfgang.com/wp-content/uploads/2019/10/Log-entries-in-the-Log-table-300x22.jpg 300w, https://www.programmingwithwolfgang.com/wp-content/uploads/2019/10/Log-entries-in-the-Log-table-768x56.jpg 768w" sizes="(max-width: 700px) 100vw, 700px" /></a>
  
  <p id="caption-attachment-1776" class="wp-caption-text">
    Log entries in the Log table
  </p>
</div>

# Conclusion

This post showed how simple it is to set up and use NLog with your ASP.NET MVC Core application. All you have to do is installing the NuGet, adding the nlog.config file and use it in your application.

You can find more information about NLog on their <a href="https://nlog-project.org/" target="_blank" rel="noopener noreferrer">website</a> and you can find the source code of today&#8217;s demo on <a href="https://github.com/WolfgangOfner/MVC-Nlog" target="_blank" rel="noopener noreferrer">Github</a>.