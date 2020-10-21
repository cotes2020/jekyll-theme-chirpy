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

<div id="attachment_1461" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/10/Create-a-new-chat-bot.jpg"><img aria-describedby="caption-attachment-1461" loading="lazy" class="wp-image-1461" src="/wp-content/uploads/2018/10/Create-a-new-chat-bot.jpg" alt="Create a new chat bot" width="700" height="696" /></a>
  
  <p id="caption-attachment-1461" class="wp-caption-text">
    Create a new chat bot
  </p>
</div>

For every return type, I will create a new dialog and call this dialog in the intent. The calling method looks like this:

<div id="attachment_1462" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/10/Forwarding-to-a-new-dialog.jpg"><img aria-describedby="caption-attachment-1462" loading="lazy" class="wp-image-1462" src="/wp-content/uploads/2018/10/Forwarding-to-a-new-dialog.jpg" alt="Forwarding to a new dialog" width="700" height="99" /></a>
  
  <p id="caption-attachment-1462" class="wp-caption-text">
    Forwarding to a new dialog
  </p>
</div>

### Reply with an internet video

Replying with a video hosted somewhere on the internet is pretty simple. You only have to create a message, add a text to this message and then a VideoCard as attachment. Before you can send the reply, you have to convert it to JSON.

<div id="attachment_1463" style="width: 849px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/10/Return-a-video-from-the-internet.jpg"><img aria-describedby="caption-attachment-1463" loading="lazy" class="size-full wp-image-1463" src="/wp-content/uploads/2018/10/Return-a-video-from-the-internet.jpg" alt="Return a video from the internet" width="839" height="433" /></a>
  
  <p id="caption-attachment-1463" class="wp-caption-text">
    Return a video from the internet
  </p>
</div>

When you test your chat bot, you will see the video as the reply of the chat bot.

<div id="attachment_1464" style="width: 618px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/10/Testing-the-internet-video-return.jpg"><img aria-describedby="caption-attachment-1464" loading="lazy" class="size-full wp-image-1464" src="/wp-content/uploads/2018/10/Testing-the-internet-video-return.jpg" alt="Testing the internet video return" width="608" height="677" /></a>
  
  <p id="caption-attachment-1464" class="wp-caption-text">
    Testing the internet video return
  </p>
</div>

### Reply with a Youtube video

Replying with a Youtube video is almost the same as replying with a normal video. Instead of a VideoCard, a normal Attachment is returned.

<div id="attachment_1465" style="width: 527px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/10/Return-a-Youtube-video.jpg"><img aria-describedby="caption-attachment-1465" loading="lazy" class="size-full wp-image-1465" src="/wp-content/uploads/2018/10/Return-a-Youtube-video.jpg" alt="Return a Youtube video" width="517" height="465" /></a>
  
  <p id="caption-attachment-1465" class="wp-caption-text">
    Return a Youtube video
  </p>
</div>

When testing the reply, you can see the embedded Youtube video in the chat.

<div id="attachment_1466" style="width: 643px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/10/Testing-the-Youtube-video-reply.jpg"><img aria-describedby="caption-attachment-1466" loading="lazy" class="size-full wp-image-1466" src="/wp-content/uploads/2018/10/Testing-the-Youtube-video-reply.jpg" alt="Testing the Youtube video reply" width="633" height="320" /></a>
  
  <p id="caption-attachment-1466" class="wp-caption-text">
    Testing the Youtube video reply
  </p>
</div>

### Reply with a file from the internet

Returning a file from the internet is basically the same as returning a Youtube video. Additionally, a name is added to the returning Attachment.

<div id="attachment_1467" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/10/Return-a-file-from-the-internet.jpg"><img aria-describedby="caption-attachment-1467" loading="lazy" class="wp-image-1467" src="/wp-content/uploads/2018/10/Return-a-file-from-the-internet.jpg" alt="Return a file from the internet" width="700" height="346" /></a>
  
  <p id="caption-attachment-1467" class="wp-caption-text">
    Return a file from the internet
  </p>
</div>

Testing the file reply returns a pdf file which you can download.

<div id="attachment_1468" style="width: 642px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/10/Testing-the-file-reply.jpg"><img aria-describedby="caption-attachment-1468" loading="lazy" class="size-full wp-image-1468" src="/wp-content/uploads/2018/10/Testing-the-file-reply.jpg" alt="Testing the file reply" width="632" height="169" /></a>
  
  <p id="caption-attachment-1468" class="wp-caption-text">
    Testing the file reply
  </p>
</div>

### Reply with an image from the internet

Replying with an image is the same as the pdf reply. The only difference is the ContentType which is image/png (or jpeg, gif and so on).

<div id="attachment_1469" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/10/Return-an-image-from-the-internet.jpg"><img aria-describedby="caption-attachment-1469" loading="lazy" class="wp-image-1469" src="/wp-content/uploads/2018/10/Return-an-image-from-the-internet.jpg" alt="Return an image from the internet" width="700" height="404" /></a>
  
  <p id="caption-attachment-1469" class="wp-caption-text">
    Return an image from the internet
  </p>
</div>

The test shows the image in the chat.

<div id="attachment_1470" style="width: 648px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/10/Testing-the-image-reply.jpg"><img aria-describedby="caption-attachment-1470" loading="lazy" class="size-full wp-image-1470" src="/wp-content/uploads/2018/10/Testing-the-image-reply.jpg" alt="Testing the image reply" width="638" height="431" /></a>
  
  <p id="caption-attachment-1470" class="wp-caption-text">
    Testing the image reply
  </p>
</div>

### Reply an image carousel

The image carousel is a great feature to enhance the user experience in your chat bot. To reply with an image carousel add a HeroCards to the list of Attachment. To make the code easier, I create a helper method, which creates the HeroCard and returns it as Attachment.

<div id="attachment_1471" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/10/HeroCard-helper-method.jpg"><img aria-describedby="caption-attachment-1471" loading="lazy" class="wp-image-1471" src="/wp-content/uploads/2018/10/HeroCard-helper-method.jpg" alt="HeroCard helper method" width="700" height="177" /></a>
  
  <p id="caption-attachment-1471" class="wp-caption-text">
    HeroCard helper method
  </p>
</div>

The HeroCard is a big image with the same text and a button. Additionally, you could create a ThumbnailCard, which has a small image, some text, and a button. Here is the helper method for the ThumbnailCard:

<div id="attachment_1472" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/10/ThumbnailCard-helper-method.jpg"><img aria-describedby="caption-attachment-1472" loading="lazy" class="wp-image-1472" src="/wp-content/uploads/2018/10/ThumbnailCard-helper-method.jpg" alt="ThumbnailCard helper method" width="700" height="175" /></a>
  
  <p id="caption-attachment-1472" class="wp-caption-text">
    ThumbnailCard helper method
  </p>
</div>

Next, I set the AttachmentLayout to AttachmentaLayoutTypes.Carousel and add a List of Attachment to the reply Attachments:

<div id="attachment_1473" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/10/Return-an-image-carousel-of-herocards-and-thumbnailcards.jpg"><img aria-describedby="caption-attachment-1473" loading="lazy" class="wp-image-1473" src="/wp-content/uploads/2018/10/Return-an-image-carousel-of-herocards-and-thumbnailcards.jpg" alt="Return an image carousel of herocards and thumbnailcards" width="700" height="499" /></a>
  
  <p id="caption-attachment-1473" class="wp-caption-text">
    Return an image carousel of HeroCards and ThumbnailCards
  </p>
</div>

I don&#8217;t like mixing a HeroCard with a ThumbnailCard but for the sake of this demo, I combined them.

<div id="attachment_1474" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/10/Testing-the-HeroCard-and-ThumbnailCard-reply.jpg"><img aria-describedby="caption-attachment-1474" loading="lazy" class="wp-image-1474" src="/wp-content/uploads/2018/10/Testing-the-HeroCard-and-ThumbnailCard-reply.jpg" alt="Testing the HeroCard and ThumbnailCard reply" width="700" height="652" /></a>
  
  <p id="caption-attachment-1474" class="wp-caption-text">
    Testing the HeroCard and ThumbnailCard reply
  </p>
</div>

On the left side, you can see the HeroCard and on the right one, the ThumbnailCard.

## Conclusion

In this post, I showed how to have different types of answers to enhance the user experience with your chat bot. In the next part, I will show you how to deploy your chat bot to Slack and to Facebook.