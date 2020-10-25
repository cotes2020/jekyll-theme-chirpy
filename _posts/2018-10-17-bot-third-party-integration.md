---
title: Integrate your Bot in Third-Party Applications
date: 2018-10-17T12:12:52+02:00
author: Wolfgang Ofner
categories: [Cloud]
tags: [Azure, Bot, Microsoft Bot Framework, Slack]
---
It is really easy to integrate your bot, hosted on Azure to third-party tools like Slack or Skype. Today, I will show how to integrate your bot with Slack, Facebook, and your website. For this demo, I will use the code from my <a href="/extending-answers-chat-bot/" target="_blank" rel="noopener">last post</a>, in which I showed how to return different types of answers to enhance the experience of the users. You can find the code on <a href="https://github.com/WolfgangOfner/Azure-ChatBot-Return-Demo" target="_blank" rel="noopener">GitHub</a>.

## Integrate your Bot in a third-party application

You can integrate your bot with several third-party applications. To see a list, which applications are available, go to your bot and open the Channels blade under the Bot management menu.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/10/Available-channels-for-the-bot.jpg"><img loading="lazy" src="/assets/img/posts/2018/10/Available-channels-for-the-bot.jpg" alt="Available channels for the bot" /></a>
  
  <p>
    Available channels
  </p>
</div>

The available channels are:

  * Web
  * Email (Office 365)
  * Facebook Messenger
  * GroupMe
  * Kik
  * Skype for Business
  * Slack
  * Telegram
  * Twilio

### Integrate the Bot with Slack

Follow these steps to integrate your bot with Slack:

  1. Create a new Slack App and enter a name for your app.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/10/Create-a-Slack-App.jpg"><img aria-describedby="caption-attachment-1480" loading="lazy" class="size-full wp-image-1480" src="/assets/img/posts/2018/10/Create-a-Slack-App.jpg" alt="Create a Slack App" /></a>
  
  <p>
    Create a Slack App
  </p>
</div>

  1. Select OAuth & Permissions and Click Add New Redirect URL.
  2. Enter https://slack.botframework.com and click Add.
  3. Click Save URLs.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/10/Add-a-redirect-URL-to-Slack.jpg"><img loading="lazy" src="/assets/img/posts/2018/10/Add-a-redirect-URL-to-Slack.jpg" alt="Add a redirect URL to Slack" /></a>
  
  <p>
    Add a redirect URL to Slack
  </p>
</div>

  1. Select Bot Users and click Add a User.
  2. Optionally change the Display name and Default username.
  3. Switch the slider for Always Show My Bot as Online to On.
  4. Click Add Bot User.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/10/Add-a-bot-user.jpg"><img loading="lazy" src="/assets/img/posts/2018/10/Add-a-bot-user.jpg" alt="Add a bot user" /></a>
  
  <p>
    Add a bot user
  </p>
</div>

  1. Select Event Subscriptions and switch the slider to On.
  2. Enter the as Request URL https://slack.botframework.com/api/Events/{YourBotHandle}, in my case this is https://slack.botframework.com/api/Events/WolfgangBotMediaDemo.
  3. Click Add Workspace Event and add the following events: 
      * member\_join\_channel
      * member\_left\_channel
      * message.channels
      * message.groups
      * message.im
      * message.mpim
  4. Click Save Changes.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/10/Add-events.jpg"><img loading="lazy" src="/assets/img/posts/2018/10/Add-events.jpg" alt="Add events" /></a>
  
  <p>
    Add events
  </p>
</div>

#### Enter your credentials

  1. Select Basic Information. There, you can find your App ID, Client ID, Client Secret, Signing Secret, and Verification Token.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/10/The-Slack-app-credentials.jpg"><img loading="lazy" src="/assets/img/posts/2018/10/The-Slack-app-credentials.jpg" alt="The Slack app credentials" /></a>
  
  <p>
    The Slack app credentials
  </p>
</div>

  1. Click on Install your App to your workspace and then on Install App to Workspace.
  2. On the next window, click Authorize and your app gets installed.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/10/Install-your-App-to-Slack.jpg"><img aria-describedby="caption-attachment-1490" loading="lazy" class="size-full wp-image-1490" src="/assets/img/posts/2018/10/Install-your-App-to-Slack.jpg" alt="Install your App to Slack" /></a>
  
  <p>
    Install your App to Slack
  </p>
</div>

  1. Open your bot in the Azure portal and select the Channels blade under the Bot management menu.
  2. Click on Slack.
  3. Enter your Client ID, Client Secret, and Verification Token from Slack.
  4. Click Save.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/10/Enter-the-Slack-credentials-in-the-Azure-Portal.jpg"><img loading="lazy" src="/assets/img/posts/2018/10/Enter-the-Slack-credentials-in-the-Azure-Portal.jpg" alt="Enter the Slack credentials in the Azure Portal" /></a>
  
  <p>
    Enter the Slack credentials in the Azure Portal
  </p>
</div>

  1. A new window opens.
  2. Click Authorize.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/10/Authorize-the-App-in-Slack.jpg"><img aria-describedby="caption-attachment-1491" loading="lazy" class="size-full wp-image-1491" src="/assets/img/posts/2018/10/Authorize-the-App-in-Slack.jpg" alt="Authorize the App in Slack" /></a>
  
  <p>
    Authorize the App in Slack
  </p>
</div>

That&#8217;s it. Now you can chat with your chatbot in Slack. Note that some answers might be displayed differently than on the web. For example, the image carousel is displayed as several images underneath each other.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/10/Testing-the-bot-in-Slack.jpg"><img loading="lazy" src="/assets/img/posts/2018/10/Testing-the-bot-in-Slack.jpg" alt="Testing the bot in Slack" /></a>
  
  <p>
    Testing the bot in Slack
  </p>
</div>

## Integrate the Bot with Facebook Messenger

Connecting the bot to the Facebook Messenger is as simple as it was with Slack. The only downside is that Facebook has to approve your bot before you can use it. This usually only takes a couple of hours.

To integrate your bot with the Facebook Messenger, follow these steps:

  1. Create a new <a href="https://developers.facebook.com/quickstarts" target="_blank" rel="noopener">Facebook App</a>.
  2. Under Settings &#8211;> Basic, you can find your App ID.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/10/Facebook-App-ID-and-Secret.jpg"><img loading="lazy" src="/assets/img/posts/2018/10/Facebook-App-ID-and-Secret.jpg" alt="Facebook App ID and Secret" /></a>
  
  <p>
    Facebook App ID and Secret
  </p>
</div>

  1. Click on Dashboard and then on Set Up on the Messenger Tilde.
  2. In the Token Generation section, create a new page and then save the Page Access Token.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/10/Facebook-Page-Token.jpg"><img aria-describedby="caption-attachment-1494" loading="lazy" class="size-full wp-image-1494" src="/assets/img/posts/2018/10/Facebook-Page-Token.jpg" alt="Facebook Page Token" /></a>
  
  <p>
    Facebook Page Token
  </p>
</div>

  1. Click on Setup Webhooks and check the following Subscription Fields: 
      * messages
      * messaging_postback
      * messaging_optins
      * message_delieveries
  2. Before you close the window, go back to your bot in the Azure portal and open the Channels blade under the Bot management menu.
  3. Click on Facebook Messenger
  4. Enter the App ID, App Secret, Page ID, and Page Access Token. You can find the Page ID on your Page under the About section.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/10/Configure-the-Facebook-Messenger-Channel.jpg"><img loading="lazy" src="/assets/img/posts/2018/10/Configure-the-Facebook-Messenger-Channel.jpg" alt="Configure the Facebook Messenger Channel" /></a>
  
  <p>
    Configure the Facebook Messenger Channel
  </p>
</div>

  1. Copy the Callback URL and Verify token to the open window on the Facebook page.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/10/Set-up-the-Callback-URL-on-Facebook.jpg"><img loading="lazy" src="/assets/img/posts/2018/10/Set-up-the-Callback-URL-on-Facebook.jpg" alt="Set up the Callback URL on Facebook" /></a>
  
  <p>
    Set up the Callback URL on Facebook
  </p>
</div>

  1. Click on Save in the Azure Portal.
  2. Click on Verify and Save on the Facebook page.
  3. Subscribe your page to the Webhook.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/10/Subscribe-your-page-to-the-Webhook.jpg"><img loading="lazy" src="/assets/img/posts/2018/10/Subscribe-your-page-to-the-Webhook.jpg" alt="Subscribe your page to the Webhook" /></a>
  
  <p>
    Subscribe your page to the Webhook
  </p>
</div>

  1. Scroll down and send a request for approval. Once your application is approved, people can test it. As long as it&#8217;s not approved, only you can use the chat.

### Integrate your bot into your website

Integrating your bot into your website is as simple as it could be. In the Azure Portal in your bot, click on Channels under the Bot management and select Web Chat. There you have your Secret keys and the code to embed the code on your website. You only have to replace the string YOUR\_SECRET\_HERE with your actual secret and you are good to go.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/10/Integrate-your-application-in-your-website.jpg"><img loading="lazy" src="/assets/img/posts/2018/10/Integrate-your-application-in-your-website.jpg" alt="Integrate your application into your website" /></a>
  
  <p>
    Integrate your application into your website
  </p>
</div>

## Conclusion

In this post, I showed how to integrate your Azure bot with Slack, Facebook Messenger, and your own website. Besides the approval from Facebook, it only takes a couple of minutes. Keep in mind that different chats display some replies differently.

For more information on Azure bot, check out <a href="/tag/bot/" target="_blank" rel="noopener">my other posts about bots</a>.