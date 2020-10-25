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

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/10/List-of-predefined-intents.jpg"><img loading="lazy" src="/assets/img/posts/2018/10/List-of-predefined-intents.jpg" alt="List of predefined intents" /></a>
  
  <p>
    List of predefined intents
  </p>
</div>

An intent is something the user wants to do. To teach your app what the user wants, you add a new intent and then add different utterances. Utterances are different phrases the user might use. For example, if the user is asking about the weather, he might ask &#8220;what is the weather today&#8221;, &#8220;Is it hot&#8221; or &#8220;Do I need an umbrella&#8221;. The more utterances you add to an intent, the better the application understands what the user wants. A rule of thumb is that there should be at least 5 utterances to train the Azure bot.

### Create a new Intent

To create a new intent, click on + Create new intent and enter a name, for example, Weather. Now type in examples (utterances), the user might say.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/10/Add-a-new-intent.jpg"><img loading="lazy" src="/assets/img/posts/2018/10/Add-a-new-intent.jpg" alt="Add a new intent" width="609" height="700"  /></a>
  
  <p>
    Add a new intent
  </p>
</div>

After you are done, click on Train in the right top corner and after the training on Publish.

### Handle the new intent in your Azure bot application

Now that you have created a new intend, your bot has to handle it. To do something with the new intent, you have to add it to the switch statement in the OnTurnAsync method in the BasicBot.cs file. The simplest way to add the new intent is to add a new case with &#8220;Weather&#8221; and then return a message.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/10/Handle-the-intent-in-the-bot.jpg"><img loading="lazy" src="/assets/img/posts/2018/10/Handle-the-intent-in-the-bot.jpg" alt="Handle the intent in the bot" /></a>
  
  <p>
    Handle the intent in the bot
  </p>
</div>

Additionally, I remove the welcome message by commenting the following code:

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/10/Remove-welcome-message.jpg"><img loading="lazy" src="/assets/img/posts/2018/10/Remove-welcome-message.jpg" alt="Remove welcome message" /></a>
  
  <p>
    Remove welcome message
  </p>
</div>

Check in your code and after it is deployed, you can test it.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/10/Testing-the-weather-intent.jpg"><img loading="lazy" src="/assets/img/posts/2018/10/Testing-the-weather-intent.jpg" alt="Testing the weather intent" /></a>
  
  <p>
    Testing the weather intent
  </p>
</div>

The new intent is working, but as you can see, the bot is not really smart. It would be nice if I could get different answers depending on the city.

## Making the Azure bot smarter

Currently, the bot always gives the same answer if the user asks for the weather. It would be nice if the bot would give a different answer, depending on the city, the user is asking about.

### Add an entity to the intent

The bot should react to the city, the user enters. Therefore the bot has to know that the city is a special input. In Luis this is called an entity. After defining an entity, the C# code can behave differently, depending on the value of the intent.

To add the entity, add a new utterance to the intent, for example, is it sunny in zurich? Then click on zurich and enter City to create the City entity. In the next window set the Entity type to simple and click done.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/10/Define-an-entity-in-luis.jpg"><img loading="lazy" src="/assets/img/posts/2018/10/Define-an-entity-in-luis.jpg" alt="Define an entity in luis" /></a>
  
  <p>
    Define an entity in luis
  </p>
</div>

Then I enter all previously used utterance but this time with a city. After all utterances are entered, I click on each city and select the city entity. This marks it as an entity.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/10/Applying-the-city-entity.jpg"><img loading="lazy" src="/assets/img/posts/2018/10/Applying-the-city-entity.jpg" alt="Applying the city entity" /></a>
  
  <p>
    Applying the city entity
  </p>
</div>

Click on Train and then on Publish

### Handle the new entity in your bot code

The following code shows how to get the entity of the luisResult and then give an answer, according to the entity.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/10/Handle-the-entity-in-the-bot.jpg"><img loading="lazy" src="/assets/img/posts/2018/10/Handle-the-entity-in-the-bot.jpg" alt="Handle the entity in the Azure bot" /></a>
  
  <p>
    Handle the entity in the Azure bot
  </p>
</div>

Obviously, this code is not really useful, it is only supposed to show what you can do with entities. A real world example could be to compare the entity with a list of cities where you have a store and if the entity is a city where you have a store, you call a weather service and show the weather for the day.

## Conclusion

In this post, I showed how to implement language understanding in your Azure Bot using luis. Additionally, I showed how to use entities and make the answers of your Azure bot smarter. You can find the code of the demo on <a href="https://github.com/WolfgangOfner/Azure-ChatBot" target="_blank" rel="noopener">GitHub</a>.

In <a href="/extending-answers-chat-bot/" target="_blank" rel="noopener">part 3</a> of this series, I will talk about different reply types like videos, images or files.