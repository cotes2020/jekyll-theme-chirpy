Hux Blog User Manual
====================

* Basics
	* [Getting Started](#getting-started)
	* [Development](#development)
	* [Config](#configs)
	* [Posts](#posts)
* Components
	* [SideBar](#sidebar)
	* [Mini About Me](#mini-about-me)
	* [Featured Tags](#featured-tags)
	* [Friends](#friends)
	* [Keynote Layout](#keynote-layout)
* Misc
	* [Comment](#comment)
	* [Analytics](#analytics)
	* [SEO Title](#seo-title)
* [FAQ](#faq)
* [Releases](#releases) 


### Getting Started

1. You will need [Ruby](https://www.ruby-lang.org/en/) and [Bundler](https://bundler.io/) to use [Jekyll](https://jekyllrb.com/). Following [Using Jekyll with Bundler](https://jekyllrb.com/tutorials/using-jekyll-with-bundler/) to fullfill the enviromental requirement.

2. Installed dependencies in the `Gemfile`:

```sh
$ bundle install 
```

3. Serve the website (`localhost:4000` by default):

```sh
$ bundle exec jekyll serve  # alternatively, npm start
```

### Development

To modify the theme, you will need [Grunt](https://gruntjs.com/). There are numbers of tasks you can find in the `Gruntfile.js`, includes minifing JavaScript, compiling `.less` to `.css`, adding banners to keep the Apache 2.0 license intact, watching for changes, etc. 

Yes, they were inherited and are extremely old-fashioned. There is no modularization and transpilation, etc.

Critical Jekyll-related code are located in `_include/` and `_layouts/`. Most of them are [Liquid](https://github.com/Shopify/liquid/wiki) templates.

This theme uses the default code syntax highlighter of jekyll, [Rouge](http://rouge.jneen.net/), which is compatible with Pygments theme so just pick any pygments theme css (e.g. from [here](http://jwarby.github.io/jekyll-pygments-themes/languages/javascript.html) and replace the content of `highlight.less`.


### Configs

You can easily customize the blog by modifying `_config.yml`:

```yml
# Site settings
title: Hux Blog             # title of your website
SEOTitle: Hux Blog          # check out docs for more detail
description: "Cool Blog"    # ...

# SNS settings      
github_username: huxpro     # modify this account to yours
weibo_username: huxpro      # the footer woule be auto-updated.

# Build settings
paginate: 10                # nums of posts in one page
```

For more options, please check out [Jekyll - Official Site](http://jekyllrb.com/). 
Most of them are very descriptive so feel brave to dive into code directly as well. 


### Posts

Posts are simply just Markdown files in the `_posts/`. 
Metadata of posts are listed in a YAML style _front-matter_.

For instance, [Hello 2015])(https://huangxuan.me/2015/01/29/hello-2015/) has the front-matter of this:

```yml
---
layout:     post
title:      "Hello 2015"
subtitle:   " \"Hello World, Hello Blog\""
date:       2015-01-29 12:00:00
author:     "Hux"
header-img: "img/post-bg-2015.jpg"
catalog: true
tags:
    - Life
    - Meta
---
```

> Note: `tags` section can also be written as `tags: [Life, Meta]`.

After [Rake](https://github.com/ruby/rake) is introduced, we can use the command below to simplify the post creation:

```
rake post title="Hello 2015" subtitle="Hello World, Hello Blog"
```

This command will automatially generate a sample post similar as above under the `_posts/` folder.

There are a bunch of _advanced_ configs:

1. a _text style_ header like [this](https://huangxuan.me/2019/09/08/spacemacs-workflow/) with

```yml
header-style: text 
```

2. Turning on Latex support:

```yml
mathjax: true
```

3. Adding a mask to the header picture:

```yml
header-mask: 0.3
```

Etc.


### SideBar

![](http://huangxuan.me/img/blog-sidebar.jpg)

**SideBar** provides possible modules to show off more personal information.

```yml
# Sidebar settings
sidebar: true   # default true
sidebar-about-description: "your description here"
sidebar-avatar: /img/avatar-hux.jpg     # use absolute URL.
```

Modules *[Featured Tags](#featured-tags)*, *[Mini About Me](#mini-about-me)* and *[Friends](#friends)* are turned on by default and you can add your own. The sidebar is naturally responsive, i.e. be pushed to bottom in a smaller screen (`<= 992px`, according to [Bootstarp Grid System](http://getbootstrap.com/css/#grid))  


### Mini About Me

**Mini-About-Me** displays your avatar, description and all SNS buttons if  `sidebar-avatar` and `sidebar-about-description` variables are set. 

It would be hidden in a smaller screen when the entire sidebar are pushed to bottom. Since there is already SNS portion there in the footer.

### Featured Tags

**Featured-Tags** is similar to any cool tag features in website like [Medium](http://medium.com).
Started from V1.4, this module can be used even when sidebar is off and displayed always in the bottom. 

```yml
# Featured Tags
featured-tags: true  
featured-condition-size: 1     # A tag will be featured if the size of it is more than this condition value
```

The only thing need to be paid attention to is `featured-condition-size`, which indicate a criteria that tags need to have to be able to "featured". Internally, a condition `{% if tag[1].size > {{site.featured-condition-size}} %}` are made.

### Friends

Friends is a common feature of any blog. It helps with SEO if you have a bi-directional hyperlinks with your friends sites.
This module can live when sidebar is off as well.

Friends information is configured as a JSON string in `_config.yml`

```yml
# Friends
friends: [
    {
        title: "Foo Blog",
        href: "http://foo.github.io/"
    },
    {
        title: "Bar Blog",
        href: "http://bar.github.io"
    }
]
```


### Keynote Layout

![](http://huangxuan.me/img/blog-keynote.jpg)

There is a increased trend to use Open Web technology for keynotes and presentations via Reveal.js, Impress.js, Slides, Prezi etc. I consider a modern blog should have first-class support to embed these HTML based presentation so **Keynote layout** are made.

To use, in the **front-matter**:

```yml
---
layout:     keynote
iframe:     "http://huangxuan.me/js-module-7day/"
---
```

The `iframe` element will be automatically resized to adapt different form factors and device orientation. 
Because most of the keynote framework prevent the browser default scroll behavior. A bottom-padding is set to help user and imply user that more content could be presented below.


### Comment

> Help Wanted: Moving to a Github-based solution.

Currently, [Disqus](http://disqus.com) <del> and [Duoshuo](http://duoshuo.com)</del> are supported as third party discussion system.

First of all, you need to sign up and get your own account. **Repeat, DO NOT use mine!** (I have set Trusted Domains) It is deathly simple to sign up and you will get the full power of management system. Please give it a try!

Second, from V1.5, you can easily complete your comment configuration by just adding your **short name** into `_config.yml`:

```yml
duoshuo_username: _your_duoshuo_short_name_
# OR
disqus_username: _your_disqus_short_name_
```

**To the old version user**, it's better that you pull the new version, otherwise you have to replace code in `post.html`, `keynote.html` and `about.html` on your own.

<del>Furthermore, Duoshuo support Sharing. if you only wanna use Duoshuo comment without sharing, you can set `duoshuo_share: false`. </del>


### Analytics

From V1.5, Google Analytics and Baidu Tongji are supported with a simple config away:

```yml
# Baidu Analytics
ba_track_id: 4cc1f2d8f3067386cc5cdb626a202900

# Google Analytics
ga_track_id: 'UA-49627206-1'            # Format: UA-xxxxxx-xx
ga_domain: huangxuan.me
```

Just checkout the code offered by Google/Baidu, and copy paste here, all the rest is already done for you.

(Google might ask for meta tag `google-site-verification`)


### SEO Title

Before V1.4, site setting `title` is not only used for displayed in Home Page and Navbar, but also used to generate the `<title>` in HTML.
It's possible that you want the two things different. For me, my site-title is **“Hux Blog”** but I want the title shows in search engine is **“黄玄的博客 | Hux Blog”** which is multi-language.

So, the SEO Title is introduced to solve this problem, you can set `SEOTitle` different from `title`, and it would be only used to generate HTML `<title>` and setting DuoShuo Sharing.


FAQ
---

#### cannot load such file -- jekyll-paginate

This blog started in Jekyll 2 time when `jekyll-paginate` is standard. With Jekyll 3, it's a plugin we included in `_config.yml`.

Make sure you installed it via plain `gem` CLI or Bundler.


Releases
--------

#### V1.8.2

- Merged #333, #336 from @JinsYin.
- Add `Gemfile` due to increasing Bundler using.
- TODO: `multilingual` could be more automative via configurations and convention.
- Drop the entire `portfolio` page until a big rewrite of a better `project` page.

#### V1.8.1

- Improve multi-lingual implementation, see `about.html` or `_posts/2017-07-12-upgrading-eleme-to-pwa.markdown` for a example of uses.

#### V1.8

- Brand new [Archive](https://huangxuan.me/archive/) page! It combines previous Archive and Tag page and it's backward-cmpatible.
Shout out to [@kitian616/jekyll-TeXt-theme](https://github.com/kitian616/jekyll-TeXt-theme) of bringing this idea. 
- Improve engineering by extracting duplicated liquid templates into reuseable includes. This was proposed in #74 by @Kaijun but postponed for entirely 2.5 years! I wasn't able to merge his PR directly because of long-time divegence but the credit is belonging to @Kaijun.
- Improved code block. Long-wanted line number are supported out of the box (thanks for @SmilingParadise's help from Sina Weibo), the default theme is updated to Atom One Dark as well (checkout FQA for how to change to your farovite theme)
- MathJax support by @Voleking in #80. I choose to use the SVG renderer though. See [Mathjax, kramdown and Octopress](https://www.lucypark.kr/blog/2013/02/25/mathjax-kramdown-and-octopress/) for writing and escaping details.
- Open Graph Protocol support by @Android-KitKat in #253
- `header-img-credit` and `header-img-credit-href`
- `nav-style: invert` and `header-style: text`

#### V1.7

- PWA / Service Worker support.

#### v1.6

- Change cdn to cdnjs for better HTTPS support

#### V1.5.2

- Feeling annoyed to delete my blog post after clone or pull? Try **Boilerplate (Beta)** to help you get started quickly and easily merge update.
- `-apple-system` is added in font rule, which display beautiful new font **San Francisco** in iOS 9 by default.
- Fixed [issue#15](https://github.com/Huxpro/huxpro.github.io/issues/15) about code wrap.

#### V1.5.1

- **[Comment](#comment)** support [**Disqus**](http://disqus.com) officially, thanks to @rpsh.

#### V1.5

- **[Comment](#comment)** and **[Analytics](#analytics)** is configurable now! We also add **Google Analytics support** and drop tencents. Both documents is updated.

#### V1.4

- **[Featured Tags](#featured-tags)** is now independent of [SideBar](#sidebar). Both documents is updated.
- New **[SEO Title](#seo-title)** for SEO usage which is differ from the site title

#### V1.3.1

- Support **PingFang (苹方)**, the new Chinese font presented by [OS X El Capitan](http://www.apple.com/cn/osx/whats-new/)


#### V1.3

- Big Improvement to the **Navigation Menu** *(especially in Android)*:  Dropping the old, stuttering, low-performance [Bootstrap collapse.js](http://getbootstrap.com/javascript/#collapse),  replaced with an own wrote, [jank free](http://jankfree.org/) navbar menu in a pretty high-performance implementation of [Google Material Design](https://www.google.com/design/spec/material-design/introduction.html).

<img src="http://huangxuan.me/img/blog-md-navbar.gif" width="320" />


#### V1.2

- Brand new **[Keynote Layout](#keynote-layout)** is provided for easily posting beautiful HTML presentations you have created with this blog


#### V1.1

- We now support a clean and gorgeous **[SideBar](#sidebar)** for displaying more info
- **[Friends](#friends)** is also added as a common feature of blog help you do SEO

#### V1.0

- Full-feature **Tag** support
- **Mobile first** user experience optimization
- **Typographic optimization** for Chinese Fonts
- **Network optimizaition** for China, dropping Google webfont, using local CDN
- Using [Github Flavored Markdown](https://help.github.com/articles/github-flavored-markdown/)
- Using Baidu, Tencent/QQ analytics
- Using [DuoShuo](http://duoshuo.com/) as the Disqus-like third party discussion system