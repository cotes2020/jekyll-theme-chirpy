---
title: Customize the Favicon
author: cotes
date: 2019-08-11 00:34:00 +0800
categories: [Blogging, Tutorial]
tags: [favicon]
---

The [favicons](https://www.favicon-generator.org/about/) of [**Chirpy**](https://github.com/cotes2020/jekyll-theme-chirpy/) are placed in the directory `assets/img/favicons/`{: .filepath}. You may want to replace them with your own. The following sections will guide you to create and replace the default favicons.

## Generate the favicon

Prepare a square image (PNG, JPG, or SVG) with a size of 512x512 or more, and then go to the online tool [**Real Favicon Generator**](https://realfavicongenerator.net/) and click the button <kbd>Pick your favicon image</kbd> to upload your image file.

In the next step, the webpage will show all usage scenarios. You can keep the default options, scroll to the bottom of the page, and click the button <kbd>Next →</kbd> to generate the favicon.

## Download & Replace

Download the generated package, unzip and delete the following file(s) from the extracted files:

- `site.webmanifest`{: .filepath}

And then copy the remaining image files (`.PNG`{: .filepath}, `.ICO`{: .filepath} and `.SVG`{: .filepath}) to cover the original files in the directory `assets/img/favicons/`{: .filepath} of your Jekyll site. If your Jekyll site doesn't have this directory yet, just create one.

The following table will help you understand the changes to the favicon files:

| File(s) | From Online Tool | From Chirpy |
| ------- | :--------------: | :---------: |
| `*.PNG` |        ✓         |      ✗      |
| `*.ICO` |        ✓         |      ✗      |
| `*.SVG` |        ✓         |      ✗      |


<!-- markdownlint-disable-next-line -->
>  ✓ means keep, ✗ means delete.
{: .prompt-info }

The next time you build the site, the favicon will be replaced with a customized edition.
