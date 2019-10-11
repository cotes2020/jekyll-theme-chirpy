---
title: "Customize the Favicon"
date: 2019-08-11 00:34:00 +0800
categories: [Blogging, Tutorial]
tags: [favicon]
toc: false
---

The image files of [Favicons](https://www.favicon-generator.org/about/) are placed in `assets/img/favicons`. You may need to replace them with your own. So let's see how to customize these Favicons.

Whit a square image (PNG, JPG or GIF) in hand, open the site [*Favicon & App Icon Generator*](https://www.favicon-generator.org/) and upload your original image.

![upload-image]({{ site.baseurl }}/assets/img/sample/upload-image.png)

Click button <kbd>Create Favicon</kbd> and wait a moment for the website to generate the icons of various sizes automatically.

![download-icons]({{ site.baseurl }}/assets/img/sample/download-icons.png)

Download the generated package and extract, then remove the following two of them:

- browserconfig.xml
- manifest.json
 
Now, copy the rest (`.PNG` and `.ICO`) to cover the original files in folder `assets/img/favicons`.

In the end, rebuild your site so that the icon becomes your custom edition.


