---
title: Customize the Favicon
author: Cotes Chung
date: 2019-08-11 00:34:00 +0800
categories: [Blogging, Tutorial]
tags: [favicon]
toc: false
---

In [**Chirpy**](https://github.com/cotes2020/jekyll-theme-chirpy/), the image files of [Favicons](https://www.favicon-generator.org/about/) are placed in `assets/img/favicons/`. You may need to replace them with your own. So let's see how to customize these Favicons.

With a square image (PNG, JPG or GIF) in hand, open the site [*Favicon & App Icon Generator*](https://www.favicon-generator.org/) and upload your original image.

![upload-image](/assets/img/sample/upload-image.png)

Click button <kbd>Create Favicon</kbd> and wait a moment for the website to generate the icons of various sizes automatically.

![download-icons](/assets/img/sample/download-icons.png){: width="600"}

Download the generated package, unzip and delete the following two from the extracted files:

- browserconfig.xml
- manifest.json
 
Now, copy the remaining image files (`.PNG` and `.ICO`) from the extracted `.zip` file to cover the original files in the folder `assets/img/favicons/`.

The following table helps you understand the changes to the icon file:

> ✓ means keep, ✗ means delete.

| File(s)             | From Favicon & App Icon Generator | From Chirpy |
|---------------------|:---------------------------------:|:-----------:|
| `*.PNG`             | ✓                                 | ✗           |
| `*.ICO`             | ✓                                 | ✗           |
| `browserconfig.xml` | ✗                                 | ✓           |
| `manifest.json`     | ✗                                 | ✓           |


The next time you build the site, the icon will be replaced with a customized edition.
