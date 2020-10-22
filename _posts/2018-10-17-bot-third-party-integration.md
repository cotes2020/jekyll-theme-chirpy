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

<div id="attachment_1479" style="width: 710px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2018/10/Available-channels-for-the-bot.jpg"><img aria-describedby="caption-attachment-1479" loading="lazy" class="wp-image-1479" src="/assets/img/posts/2018/10/Available-channels-for-the-bot.jpg" alt="Available channels for the bot" width="700" height="389" /></a>
  
  <p id="caption-attachment-1479" class="wp-caption-text">
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

<div id="attachment_1480" style="width: 564px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2018/10/Create-a-Slack-App.jpg"><img aria-describedby="caption-attachment-1480" loading="lazy" class="size-full wp-image-1480" src="/assets/img/posts/2018/10/Create-a-Slack-App.jpg" alt="Create a Slack App" width="554" height="454" /></a>
  
  <p id="caption-attachment-1480" class="wp-caption-text">
    Create a Slack App
  </p>
</div>

  1. Select OAuth & Permissions and Click Add New Redirect URL.
  2. Enter https://slack.botframework.com and click Add.
  3. Click Save URLs.

<div id="attachment_1481" style="width: 711px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2018/10/Add-a-redirect-URL-to-Slack.jpg"><img aria-describedby="caption-attachment-1481" loading="lazy" class="wp-image-1481" src="/assets/img/posts/2018/10/Add-a-redirect-URL-to-Slack.jpg" alt="Add a redirect URL to Slack" width="701" height="324" /></a>
  
  <p id="caption-attachment-1481" class="wp-caption-text">
    Add a redirect URL to Slack
  </p>
</div>

  1. Select Bot Users and click Add a User.
  2. Optionally change the Display name and Default username.
  3. Switch the slider for Always Show My Bot as Online to On.
  4. Click Add Bot User.

<div id="attachment_1482" style="width: 710px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2018/10/Add-a-bot-user.jpg"><img aria-describedby="caption-attachment-1482" loading="lazy" class="wp-image-1482" src="/assets/img/posts/2018/10/Add-a-bot-user.jpg" alt="Add a bot user" width="700" height="481" /></a>
  
  <p id="caption-attachment-1482" class="wp-caption-text">
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

<div id="attachment_1483" style="width: 686px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2018/10/Add-events.jpg"><img aria-describedby="caption-attachment-1483" loading="lazy" class="wp-image-1483" src="/assets/img/posts/2018/10/Add-events.jpg" alt="Add events" width="676" height="700" /></a>
  
  <p id="caption-attachment-1483" class="wp-caption-text">
    Add events
  </p>
</div>

#### Enter your credentials

  1. Select Basic Information. There, you can find your App ID, Client ID, Client Secret, Signing Secret, and Verification Token.

<div id="attachment_1488" style="width: 661px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2018/10/The-Slack-app-credentials.jpg"><img aria-describedby="caption-attachment-1488" loading="lazy" class="wp-image-1488" src="/assets/img/posts/2018/10/The-Slack-app-credentials.jpg" alt="The Slack app credentials" width="651" height="700" /></a>
  
  <p id="caption-attachment-1488" class="wp-caption-text">
    The Slack app credentials
  </p>
</div>

  1. Click on Install your App to your workspace and then on Install App to Workspace.
  2. On the next window, click Authorize and your app gets installed.

<div id="attachment_1490" style="width: 457px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2018/10/Install-your-App-to-Slack.jpg"><img aria-describedby="caption-attachment-1490" loading="lazy" class="size-full wp-image-1490" src="/assets/img/posts/2018/10/Install-your-App-to-Slack.jpg" alt="Install your App to Slack" width="447" height="449" /></a>
  
  <p id="caption-attachment-1490" class="wp-caption-text">
    Install your App to Slack
  </p>
</div>

  1. Open your bot in the Azure portal and select the Channels blade under the Bot management menu.
  2. Click on Slack.
  3. Enter your Client ID, Client Secret, and Verification Token from Slack.
  4. Click Save.

<div id="attachment_1489" style="width: 710px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2018/10/Enter-the-Slack-credentials-in-the-Azure-Portal.jpg"><img aria-describedby="caption-attachment-1489" loading="lazy" class="wp-image-1489" src="/assets/img/posts/2018/10/Enter-the-Slack-credentials-in-the-Azure-Portal.jpg" alt="Enter the Slack credentials in the Azure Portal" width="700" height="348" /></a>
  
  <p id="caption-attachment-1489" class="wp-caption-text">
    Enter the Slack credentials in the Azure Portal
  </p>
</div>

  1. A new window opens.
  2. Click Authorize.

<div id="attachment_1491" style="width: 446px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2018/10/Authorize-the-App-in-Slack.jpg"><img aria-describedby="caption-attachment-1491" loading="lazy" class="size-full wp-image-1491" src="/assets/img/posts/2018/10/Authorize-the-App-in-Slack.jpg" alt="Authorize the App in Slack" width="436" height="296" /></a>
  
  <p id="caption-attachment-1491" class="wp-caption-text">
    Authorize the App in Slack
  </p>
</div>

That&#8217;s it. Now you can chat with your chatbot in Slack. Note that some answers might be displayed differently than on the web. For example, the image carousel is displayed as several images underneath each other.

<div id="attachment_1492" style="width: 503px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2018/10/Testing-the-bot-in-Slack.jpg"><img aria-describedby="caption-attachment-1492" loading="lazy" class="wp-image-1492" src="/assets/img/posts/2018/10/Testing-the-bot-in-Slack.jpg" alt="Testing the bot in Slack" width="493" height="700" /></a>
  
  <p id="caption-attachment-1492" class="wp-caption-text">
    Testing the bot in Slack
  </p>
</div>

## Integrate the Bot with Facebook Messenger

Connecting the bot to the Facebook Messenger is as simple as it was with Slack. The only downside is that Facebook has to approve your bot before you can use it. This usually only takes a couple of hours.

To integrate your bot with the Facebook Messenger, follow these steps:

  1. Create a new <a href="https://developers.facebook.com/quickstarts" target="_blank" rel="noopener">Facebook App</a>.
  2. Under Settings &#8211;> Basic, you can find your App ID.

<div id="attachment_1493" style="width: 710px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2018/10/Facebook-App-ID-and-Secret.jpg"><img aria-describedby="caption-attachment-1493" loading="lazy" class="wp-image-1493" src="/assets/img/posts/2018/10/Facebook-App-ID-and-Secret.jpg" alt="Facebook App ID and Secret" width="700" height="216" /></a>
  
  <p id="caption-attachment-1493" class="wp-caption-text">
    Facebook App ID and Secret
  </p>
</div>

  1. Click on Dashboard and then on Set Up on the Messenger Tilde.
  2. In the Token Generation section, create a new page and then save the Page Access Token.

<div id="attachment_1494" style="width: 564px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2018/10/Facebook-Page-Token.jpg"><img aria-describedby="caption-attachment-1494" loading="lazy" class="size-full wp-image-1494" src="/assets/img/posts/2018/10/Facebook-Page-Token.jpg" alt="Facebook Page Token" width="554" height="291" /></a>
  
  <p id="caption-attachment-1494" class="wp-caption-text">
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

<div id="attachment_1495" style="width: 710px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2018/10/Configure-the-Facebook-Messenger-Channel.jpg"><img aria-describedby="caption-attachment-1495" loading="lazy" class="wp-image-1495" src="/assets/img/posts/2018/10/Configure-the-Facebook-Messenger-Channel.jpg" alt="Configure the Facebook Messenger Channel" width="700" height="538" /></a>
  
  <p id="caption-attachment-1495" class="wp-caption-text">
    Configure the Facebook Messenger Channel
  </p>
</div>

  1. Copy the Callback URL and Verify token to the open window on the Facebook page.

<div id="attachment_1496" style="width: 710px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2018/10/Set-up-the-Callback-URL-on-Facebook.jpg"><img aria-describedby="caption-attachment-1496" loading="lazy" class="wp-image-1496" src="/assets/img/posts/2018/10/Set-up-the-Callback-URL-on-Facebook.jpg" alt="Set up the Callback URL on Facebook" width="700" height="431" /></a>
  
  <p id="caption-attachment-1496" class="wp-caption-text">
    Set up the Callback URL on Facebook
  </p>
</div>

  1. Click on Save in the Azure Portal.
  2. Click on Verify and Save on the Facebook page.
  3. Subscribe your page to the Webhook.

<div id="attachment_1497" style="width: 710px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2018/10/Subscribe-your-page-to-the-Webhook.jpg"><img aria-describedby="caption-attachment-1497" loading="lazy" class="wp-image-1497" src="/assets/img/posts/2018/10/Subscribe-your-page-to-the-Webhook.jpg" alt="Subscribe your page to the Webhook" width="700" height="158" /></a>
  
  <p id="caption-attachment-1497" class="wp-caption-text">
    Subscribe your page to the Webhook
  </p>
</div>

  1. Scroll down and send a request for approval. Once your application is approved, people can test it. As long as it&#8217;s not approved, only you can use the chat.

### Integrate your bot into your website

Integrating your bot into your website is as simple as it could be. In the Azure Portal in your bot, click on Channels under the Bot management and select Web Chat. There you have your Secret keys and the code to embed the code on your website. You only have to replace the string YOUR\_SECRET\_HERE with your actual secret and you are good to go.

<div id="attachment_1498" style="width: 710px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2018/10/Integrate-your-application-in-your-website.jpg"><img aria-describedby="caption-attachment-1498" loading="lazy" class="wp-image-1498" src="/assets/img/posts/2018/10/Integrate-your-application-in-your-website.jpg" alt="Integrate your application into your website" width="700" height="504" /></a>
  
  <p id="caption-attachment-1498" class="wp-caption-text">
    Integrate your application into your website
  </p>
</div>

## Conclusion

In this post, I showed how to integrate your Azure bot with Slack, Facebook Messenger, and your own website. Besides the approval from Facebook, it only takes a couple of minutes. Keep in mind that different chats display some replies differently.

For more information on Azure bot, check out <a href="/tag/bot/" target="_blank" rel="noopener">my other posts about bots</a>.