---
title: Create a Bot Application with Azure
date: 2018-10-14T16:02:01+02:00
author: Wolfgang Ofner
categories: [Cloud]
tags: [Azure, Bot, 'C#', Microsoft Bot Framework]
---
I&#8217;ve been curious about how chat bots work and this week I finally took the time to sit down and look into it. In this and the following post, I will talk about what I learned and I how solved some problems. Before I could start, I had to decide which platform I want to use. Therefore I looked at solutions in AWS and Azure. Since I am a fan of Azure and AWS only supports English as the user language, I decided to go with Azure for my chat bot application.

## Create your first Azure Chat Bot Application

You can create a bot application with Visual Studio or directly in the Azure Portal. If you want to use Visual Studio, you need Visual Studio 2017 and the Bot Builder Template which you can download <a href="https://botbuilder.myget.org/feed/aitemplates/package/vsix/BotBuilderV4.fbe0fc50-a6f1-4500-82a2-189314b7bea2" target="_blank" rel="noopener">here</a>.  To create a bot application in the Azure Portal, follow these steps:

  1. In the Azure portal click on +Create a resource, search for Web App Bot and click Create.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/10/Create-a-Web-App-Bot-in-the-Azure-Portal.jpg"><img loading="lazy" src="/assets/img/posts/2018/10/Create-a-Web-App-Bot-in-the-Azure-Portal.jpg" alt="Create a Web App Bot in the Azure Portal" /></a>
  
  <p>
    Create a Web App Bot in the Azure Portal
  </p>
</div>

<ol start="2">
  <li>
    Enter the basic information for your bot.
  </li>
  <li>
    Select a template. For this demo, I will select the Basic Bot template because it is simple but also supports language understanding.
  </li>
  <li>
    Select the LUIS App location. It depends on this setting, which luis website you have to use later to configure luis. For example, if you select Europe the URL is <a href="https://eu.luis.ai" target="_blank" rel="noopener">https://eu.luis.ai</a>, if you select US the URL is <a href="https://luis.ai" target="_blank" rel="noopener">https://luis.ai</a>.
  </li>
  <li>
    Click Create.
  </li>
</ol>

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/10/Create-your-new-bot-application.jpg"><img loading="lazy" src="/assets/img/posts/2018/10/Create-your-new-bot-application.jpg" alt="Create your new bot application" /></a>
  
  <p>
    Create your new bot application
  </p>
</div>

## Test your bot

After the bot application is deployed, open it and select Test in Web Chat under the Bot management menu. Type in Hi and wait for a response. Sometimes you don&#8217;t get an answer on the first try. Then you have to enter a second message.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/10/Test-your-bot.jpg"><img loading="lazy" src="/assets/img/posts/2018/10/Test-your-bot.jpg" alt="Test your bot" /></a>
  
  <p>
    Test your bot
  </p>
</div>

Congratulation, you just talked to your own bot for the first time.

## Edit the bot answers

In this section, I will show you how to edit existing answers of your bot and how to deploy it to Azure.

### Edit your bot in Visual Studio

To edit the source code of your code, download it. Click on Download Bot source code on the Build blade under the Bot management menu.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/10/Download-the-source-code-of-your-bot.jpg"><img loading="lazy" src="/assets/img/posts/2018/10/Download-the-source-code-of-your-bot.jpg" alt="Download the source code of your bot" /></a>
  
  <p>
    Download the source code of your bot
  </p>
</div>

To edit an existing answer, follow these steps:

  1. Open the solution you just downloaded in Visual Studio 2017.
  2. Open GreetingDialog.cs under Dialogs/Greeting and find the line with Text = &#8220;What is your name?&#8221;. In my case, it was on line 104 but Microsoft updates the default bot quite often, so it might be somewhere else for you.
  3. Change the string to whatever you want, for example, &#8220;Howdy, tell me your name&#8221;.
  4. Save the file.

### Deploy your bot to Azure

There are different ways to deploy your bot application to Azure. The simplest is to right-click your solution and select Publish. There you can already see all settings being set and you only have to click Publish. This approach is fine when you are alone but I want to show you a more sophisticated way which includes version control and automatic deployments. I will use GitHub as my version control and every time I check code in, it will be automatically deployed. To configure Azure to do that, follow these steps:

  1. Open the App Service of your bot application and select Deployment options under the Deployment menu.
  2. Add a source for your version control, in my case GitHub, but you could also use Bitbucket, Team Services or many more.
  3. Enter your credentials, select a project and configure the branch you want to use for the deployment.
  4. Click OK.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/10/Add-GitHub-as-your-deployment-source.jpg"><img loading="lazy" src="/assets/img/posts/2018/10/Add-GitHub-as-your-deployment-source.jpg" alt="Add GitHub as your deployment source" /></a>
  
  <p>
    Add GitHub as your deployment source
  </p>
</div>

<ol start="5">
  <li>
    Next, push your bot application to your GitHub project.
  </li>
  <li>
    After you pushed your changes to GitHub, you will see the deployment under the Deployment options.
  </li>
</ol>

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/10/Sync-GitHub-with-Azure.jpg"><img loading="lazy" src="/assets/img/posts/2018/10/Sync-GitHub-with-Azure.jpg" alt="Sync GitHub with Azure" /></a>
  
  <p>
    Sync GitHub with Azure
  </p>
</div>

<ol start="7">
  <li>
    With the changes deployed, it is time to test them.
  </li>
  <li>
    Open your Web App Bot Application and select the Test in Web Chat blade under the Bot management menu.
  </li>
  <li>
    Type in Hi and you should see the text, you changed previously.
  </li>
</ol>

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/10/Test-the-changes-in-your-bot-application.jpg"><img loading="lazy" src="/assets/img/posts/2018/10/Test-the-changes-in-your-bot-application.jpg" alt="Test the changes in your bot application" /></a>
  
  <p>
    Test the changes in your bot application
  </p>
</div>

As you can see, the changes worked and the bot uses the new phrase to &#8220;ask&#8221; for the name of the user.

## Conclusion

Today, I showed you how to create a simple chat bot with Azure and how to deploy it using GitHub. In my <a href="/azure-bot-language-understanding/" target="_blank" rel="noopener">next post</a>, I will show you how to enable language understanding with luis.

You can find the code of the demo on <a href="https://github.com/WolfgangOfner/Azure-ChatBot" target="_blank" rel="noopener">GitHub</a>.

&nbsp;