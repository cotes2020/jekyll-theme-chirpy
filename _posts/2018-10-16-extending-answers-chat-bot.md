---
title: Extending the answers of the Chat Bot
date: 2018-10-16T12:00:50+02:00
author: Wolfgang Ofner
categories: [Cloud]
tags: [Azure, Bot, 'C#', Microsoft Bot Framework]
---
The Azure chat bot supports different answer types like videos, images or links. In my last post, I showed you how to make your bot smarter and understand the user&#8217;s input. In this post, I will show you how to return different types of answers.

## Adding different reply types to the chat bot

To make your chat bot more appealing to the users, you should interact with your users with more than just text. The following demo is done with the Azure Bot Framework v3. For every return type, I create a simple intent which does nothing but return the desired type. The code can be downloaded from <a href="https://github.com/WolfgangOfner/Azure-ChatBot-Return-Demo" target="_blank" rel="noopener">GitHub</a>.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/10/Create-a-new-chat-bot.jpg"><img loading="lazy" src="/assets/img/posts/2018/10/Create-a-new-chat-bot.jpg" alt="Create a new chat bot" /></a>
  
  <p>
    Create a new chat bot
  </p>
</div>

For every return type, I will create a new dialog and call this dialog in the intent. The calling method looks like this:

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/10/Forwarding-to-a-new-dialog.jpg"><img loading="lazy" src="/assets/img/posts/2018/10/Forwarding-to-a-new-dialog.jpg" alt="Forwarding to a new dialog" /></a>
  
  <p>
    Forwarding to a new dialog
  </p>
</div>

### Reply with an internet video

Replying with a video hosted somewhere on the internet is pretty simple. You only have to create a message, add a text to this message and then a VideoCard as attachment. Before you can send the reply, you have to convert it to JSON.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/10/Return-a-video-from-the-internet.jpg"><img aria-describedby="caption-attachment-1463" loading="lazy" class="size-full wp-image-1463" src="/assets/img/posts/2018/10/Return-a-video-from-the-internet.jpg" alt="Return a video from the internet" /></a>
  
  <p>
    Return a video from the internet
  </p>
</div>

When you test your chat bot, you will see the video as the reply of the chat bot.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/10/Testing-the-internet-video-return.jpg"><img aria-describedby="caption-attachment-1464" loading="lazy" class="size-full wp-image-1464" src="/assets/img/posts/2018/10/Testing-the-internet-video-return.jpg" alt="Testing the internet video return" /></a>
  
  <p>
    Testing the internet video return
  </p>
</div>

### Reply with a Youtube video

Replying with a Youtube video is almost the same as replying with a normal video. Instead of a VideoCard, a normal Attachment is returned.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/10/Return-a-Youtube-video.jpg"><img aria-describedby="caption-attachment-1465" loading="lazy" class="size-full wp-image-1465" src="/assets/img/posts/2018/10/Return-a-Youtube-video.jpg" alt="Return a Youtube video" /></a>
  
  <p>
    Return a Youtube video
  </p>
</div>

When testing the reply, you can see the embedded Youtube video in the chat.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/10/Testing-the-Youtube-video-reply.jpg"><img aria-describedby="caption-attachment-1466" loading="lazy" class="size-full wp-image-1466" src="/assets/img/posts/2018/10/Testing-the-Youtube-video-reply.jpg" alt="Testing the Youtube video reply" /></a>
  
  <p>
    Testing the Youtube video reply
  </p>
</div>

### Reply with a file from the internet

Returning a file from the internet is basically the same as returning a Youtube video. Additionally, a name is added to the returning Attachment.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/10/Return-a-file-from-the-internet.jpg"><img loading="lazy" src="/assets/img/posts/2018/10/Return-a-file-from-the-internet.jpg" alt="Return a file from the internet" /></a>
  
  <p>
    Return a file from the internet
  </p>
</div>

Testing the file reply returns a pdf file which you can download.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/10/Testing-the-file-reply.jpg"><img aria-describedby="caption-attachment-1468" loading="lazy" class="size-full wp-image-1468" src="/assets/img/posts/2018/10/Testing-the-file-reply.jpg" alt="Testing the file reply" /></a>
  
  <p>
    Testing the file reply
  </p>
</div>

### Reply with an image from the internet

Replying with an image is the same as the pdf reply. The only difference is the ContentType which is image/png (or jpeg, gif and so on).

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/10/Return-an-image-from-the-internet.jpg"><img loading="lazy" src="/assets/img/posts/2018/10/Return-an-image-from-the-internet.jpg" alt="Return an image from the internet" /></a>
  
  <p>
    Return an image from the internet
  </p>
</div>

The test shows the image in the chat.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/10/Testing-the-image-reply.jpg"><img aria-describedby="caption-attachment-1470" loading="lazy" class="size-full wp-image-1470" src="/assets/img/posts/2018/10/Testing-the-image-reply.jpg" alt="Testing the image reply" /></a>
  
  <p>
    Testing the image reply
  </p>
</div>

### Reply an image carousel

The image carousel is a great feature to enhance the user experience in your chat bot. To reply with an image carousel add a HeroCards to the list of Attachment. To make the code easier, I create a helper method, which creates the HeroCard and returns it as Attachment.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/10/HeroCard-helper-method.jpg"><img loading="lazy" src="/assets/img/posts/2018/10/HeroCard-helper-method.jpg" alt="HeroCard helper method" /></a>
  
  <p>
    HeroCard helper method
  </p>
</div>

The HeroCard is a big image with the same text and a button. Additionally, you could create a ThumbnailCard, which has a small image, some text, and a button. Here is the helper method for the ThumbnailCard:

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/10/ThumbnailCard-helper-method.jpg"><img loading="lazy" src="/assets/img/posts/2018/10/ThumbnailCard-helper-method.jpg" alt="ThumbnailCard helper method" /></a>
  
  <p>
    ThumbnailCard helper method
  </p>
</div>

Next, I set the AttachmentLayout to AttachmentaLayoutTypes.Carousel and add a List of Attachment to the reply Attachments:

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/10/Return-an-image-carousel-of-herocards-and-thumbnailcards.jpg"><img loading="lazy" src="/assets/img/posts/2018/10/Return-an-image-carousel-of-herocards-and-thumbnailcards.jpg" alt="Return an image carousel of herocards and thumbnailcards" /></a>
  
  <p>
    Return an image carousel of HeroCards and ThumbnailCards
  </p>
</div>

I don&#8217;t like mixing a HeroCard with a ThumbnailCard but for the sake of this demo, I combined them.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/10/Testing-the-HeroCard-and-ThumbnailCard-reply.jpg"><img loading="lazy" src="/assets/img/posts/2018/10/Testing-the-HeroCard-and-ThumbnailCard-reply.jpg" alt="Testing the HeroCard and ThumbnailCard reply" /></a>
  
  <p>
    Testing the HeroCard and ThumbnailCard reply
  </p>
</div>

On the left side, you can see the HeroCard and on the right one, the ThumbnailCard.

## Conclusion

In this post, I showed how to have different types of answers to enhance the user experience with your chat bot. In the next part, I will show you how to deploy your chat bot to Slack and to Facebook.