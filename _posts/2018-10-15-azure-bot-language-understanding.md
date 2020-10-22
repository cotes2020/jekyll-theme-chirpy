---
title: Azure Bot with language understanding
date: 2018-10-15T12:00:04+02:00
author: Wolfgang Ofner
categories: [Cloud]
tags: [Azure, Bot, 'C#', Luis, Microsoft Bot Framework]
---
With the Azure bot framework and luis, Microsoft offers two great features to implement language understanding in your chat bot. This is the second part of my chat bot series and in this post, I will show you how to add a language understanding to your chat bot using luis. You can find the first part <a href="/bot-application-azure/" target="_blank" rel="noopener">here</a>.

## Adding new answers to your Azure bot

If you created your Azure bot the same way as described in part 1, it will already have luis integration. To add new language features, go to the <a href="https://eu.luis.ai/home" target="_blank" rel="noopener">luis homepage</a> and login with your Azure credentials. After you logged in, you should see your chat bot. Open it and you will see 4 already existing intents.

<div id="attachment_1450" style="width: 710px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2018/10/List-of-predefined-intents.jpg"><img aria-describedby="caption-attachment-1450" loading="lazy" class="wp-image-1450" src="/assets/img/posts/2018/10/List-of-predefined-intents.jpg" alt="List of predefined intents" width="700" height="277" /></a>
  
  <p id="caption-attachment-1450" class="wp-caption-text">
    List of predefined intents
  </p>
</div>

An intent is something the user wants to do. To teach your app what the user wants, you add a new intent and then add different utterances. Utterances are different phrases the user might use. For example, if the user is asking about the weather, he might ask &#8220;what is the weather today&#8221;, &#8220;Is it hot&#8221; or &#8220;Do I need an umbrella&#8221;. The more utterances you add to an intent, the better the application understands what the user wants. A rule of thumb is that there should be at least 5 utterances to train the Azure bot.

### Create a new Intent

To create a new intent, click on + Create new intent and enter a name, for example, Weather. Now type in examples (utterances), the user might say.

<div id="attachment_1451" style="width: 619px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2018/10/Add-a-new-intent.jpg"><img aria-describedby="caption-attachment-1451" loading="lazy" class="wp-image-1451" src="/assets/img/posts/2018/10/Add-a-new-intent.jpg" alt="Add a new intent" width="609" height="700"  /></a>
  
  <p id="caption-attachment-1451" class="wp-caption-text">
    Add a new intent
  </p>
</div>

After you are done, click on Train in the right top corner and after the training on Publish.

### Handle the new intent in your Azure bot application

Now that you have created a new intend, your bot has to handle it. To do something with the new intent, you have to add it to the switch statement in the OnTurnAsync method in the BasicBot.cs file. The simplest way to add the new intent is to add a new case with &#8220;Weather&#8221; and then return a message.

<div id="attachment_1452" style="width: 603px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2018/10/Handle-the-intent-in-the-bot.jpg"><img aria-describedby="caption-attachment-1452" loading="lazy" class="size-full wp-image-1452" src="/assets/img/posts/2018/10/Handle-the-intent-in-the-bot.jpg" alt="Handle the intent in the bot" width="593" height="185" /></a>
  
  <p id="caption-attachment-1452" class="wp-caption-text">
    Handle the intent in the bot
  </p>
</div>

Additionally, I remove the welcome message by commenting the following code:

<div id="attachment_1453" style="width: 710px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2018/10/Remove-welcome-message.jpg"><img aria-describedby="caption-attachment-1453" loading="lazy" class="wp-image-1453" src="/assets/img/posts/2018/10/Remove-welcome-message.jpg" alt="Remove welcome message" width="700" height="272" /></a>
  
  <p id="caption-attachment-1453" class="wp-caption-text">
    Remove welcome message
  </p>
</div>

Check in your code and after it is deployed, you can test it.

<div id="attachment_1455" style="width: 585px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2018/10/Testing-the-weather-intent.jpg"><img aria-describedby="caption-attachment-1455" loading="lazy" class="size-full wp-image-1455" src="/assets/img/posts/2018/10/Testing-the-weather-intent.jpg" alt="Testing the weather intent" width="575" height="275" /></a>
  
  <p id="caption-attachment-1455" class="wp-caption-text">
    Testing the weather intent
  </p>
</div>

The new intent is working, but as you can see, the bot is not really smart. It would be nice if I could get different answers depending on the city.

## Making the Azure bot smarter

Currently, the bot always gives the same answer if the user asks for the weather. It would be nice if the bot would give a different answer, depending on the city, the user is asking about.

### Add an entity to the intent

The bot should react to the city, the user enters. Therefore the bot has to know that the city is a special input. In Luis this is called an entity. After defining an entity, the C# code can behave differently, depending on the value of the intent.

To add the entity, add a new utterance to the intent, for example, is it sunny in zurich? Then click on zurich and enter City to create the City entity. In the next window set the Entity type to simple and click done.

<div id="attachment_1456" style="width: 698px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2018/10/Define-an-entity-in-luis.jpg"><img aria-describedby="caption-attachment-1456" loading="lazy" class="size-full wp-image-1456" src="/assets/img/posts/2018/10/Define-an-entity-in-luis.jpg" alt="Define an entity in luis" width="688" height="567" /></a>
  
  <p id="caption-attachment-1456" class="wp-caption-text">
    Define an entity in luis
  </p>
</div>

Then I enter all previously used utterance but this time with a city. After all utterances are entered, I click on each city and select the city entity. This marks it as an entity.

<div id="attachment_1457" style="width: 710px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2018/10/Applying-the-city-entity.jpg"><img aria-describedby="caption-attachment-1457" loading="lazy" class="wp-image-1457" src="/assets/img/posts/2018/10/Applying-the-city-entity.jpg" alt="Applying the city entity" width="700" height="539" /></a>
  
  <p id="caption-attachment-1457" class="wp-caption-text">
    Applying the city entity
  </p>
</div>

Click on Train and then on Publish

### Handle the new entity in your bot code

The following code shows how to get the entity of the luisResult and then give an answer, according to the entity.

<div id="attachment_1458" style="width: 710px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2018/10/Handle-the-entity-in-the-bot.jpg"><img aria-describedby="caption-attachment-1458" loading="lazy" class="wp-image-1458" src="/assets/img/posts/2018/10/Handle-the-entity-in-the-bot.jpg" alt="Handle the entity in the Azure bot" width="700" height="250" /></a>
  
  <p id="caption-attachment-1458" class="wp-caption-text">
    Handle the entity in the Azure bot
  </p>
</div>

Obviously, this code is not really useful, it is only supposed to show what you can do with entities. A real world example could be to compare the entity with a list of cities where you have a store and if the entity is a city where you have a store, you call a weather service and show the weather for the day.

## Conclusion

In this post, I showed how to implement language understanding in your Azure Bot using luis. Additionally, I showed how to use entities and make the answers of your Azure bot smarter. You can find the code of the demo on <a href="https://github.com/WolfgangOfner/Azure-ChatBot" target="_blank" rel="noopener">GitHub</a>.

In <a href="/extending-answers-chat-bot/" target="_blank" rel="noopener">part 3</a> of this series, I will talk about different reply types like videos, images or files.