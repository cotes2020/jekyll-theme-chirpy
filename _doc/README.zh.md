# Hux blog 模板

### [我的博客在这里 &rarr;](http://huxpro.github.io)


### 关于收到"Page Build Warning"的email

由于jekyll升级到3.0.x,对原来的pygments代码高亮不再支持，现只支持一种-rouge，所以你需要在 `_config.yml`文件中修改`highlighter: rouge`.另外还需要在`_config.yml`文件中加上`gems: [jekyll-paginate]`.

同时,你需要更新你的本地jekyll环境.

使用`jekyll server`的同学需要这样：

1. `gem update jekyll` # 更新jekyll
2. `gem update github-pages` #更新依赖的包

使用`bundle exec jekyll server`的同学在更新jekyll后，需要输入`bundle update`来更新依赖的包.

参考文档：[using jekyll with pages](https://help.github.com/articles/using-jekyll-with-pages/) & [Upgrading from 2.x to 3.x](http://jekyllrb.com/docs/upgrading/2-to-3/)


## 关于模板(beta)

我的博客仓库——`huxpro.github.io`，是经常修改的，而且还会有人乱提交代码，因此给大家做了一个稳定版的模板。大家可以直接fork模板——`huxblog-boilerplate`,要改的地方我都说明了。或者可以直接下载zip到本地自己去修改。

```
$ git clone git@github.com:Huxpro/huxblog-boilerplate.git
```

**[在这里预览模板 &rarr;](http://huangxuan.me/huxblog-boilerplate/)**

## 各版本特性

##### New Feature (V1.5.2)

* 当你fork了我的仓库之后，还要删掉里面的关于我的文档是不是感到略烦躁呢？**Boilerplate** 模板将帮助你快速开始，方便合并与更新。
* `-apple-system`被添加到了字体规则里面了，这套字体格式能将iOS9默认的新字体**San Francisco**表现的非常漂亮。
* 解决了代码过长自动换行的bug,替换为横向滚动条。详情请见[issue#15](https://github.com/Huxpro/huxpro.github.io/issues/15)

###### 其他历史版本个人觉得没有必要了解，看看英文就行了。



## 支持

* 你可以自由的fork。如果你能将主题作者和 github 的地址保留在你的页面底部，我将非常感谢你。
* 如果你喜欢我的这个博客模板，请在`huxpro.github.io`这个repository点个赞——右上角**star**一下。

## 说明文档

* 开始
	* [环境要求](#environment)
	* [开始](#get-started)
	* [写一篇博文](#write-posts)
* 组件
	* [侧边栏](#sidebar)
	* [迷你关于我](#mini-about-me)
	* [推荐标签](#featured-tags)
	* [好友链接](#friends)
	* [HTML5 演示文档布局](#keynote-layout)
* 评论与 Google/Baidu Analytics
	* [评论](#comment)
	* [网站分析](#analytics) 
* 高级部分
	* [自定义](#customization)
	* [标题底图](#header-image)
	* [搜索展示标题-头文件](#seo-title)

#### Environment

如果你安装了jekyll，那你只需要在命令行输入`jekyll serve`就能在本地浏览器预览主题。你还可以输入`jekyll serve --watch`，这样可以边修改边自动运行修改后的文件。

经 [@BrucZhaoR](https://github.com/BruceZhaoR)的测试，好像两个命令都是可以的自动运行修改后的文件的，刷新后可以实时预览。官方文件是建议安装bundler，这样你在本地的效果就跟在github上面是一样的。详情请见这里：https://help.github.com/articles/using-jekyll-with-pages/#installing-jekyll


#### Get Started

你可以通用修改 `_config.yml`文件来轻松的开始搭建自己的博客:

```
# Site settings
title: Hux Blog             # 你的博客网站标题
SEOTitle: Hux Blog			# 在后面会详细谈到
description: "Cool Blog"    # 随便说点，描述一下

# SNS settings      
github_username: huxpro     # 你的github账号
weibo_username: huxpro      # 你的微博账号，底部链接会自动更新的。

# Build settings
# paginate: 10              # 一页你准备放几篇文章
```

Jekyll官方网站还有很多的参数可以调，比如设置文章的链接形式...网址在这里：[Jekyll - Official Site](http://jekyllrb.com/) 中文版的在这里：[Jekyll中文](http://jekyllcn.com/).

#### write-posts

要发表的文章一般以markdown的格式放在这里`_posts/`，你只要看看这篇模板里的文章你就立刻明白该如何设置。

yaml 头文件长这样:

```
---
layout:     post
title:      "Hello 2015"
subtitle:   "Hello World, Hello Blog"
date:       2015-01-29 12:00:00
author:     "Hux"
header-img: "img/post-bg-2015.jpg"
tags:
    - Life
---

```

在引入[Rake](https://github.com/ruby/rake)工具之后，我们可以使用命令：

```
rake post title="Hello 2015" subtitle="Hello World, Hello Blog"
```

来自动生成上面的文章模板。

#### SideBar

看右边:
![](http://huangxuan.me/img/blog-sidebar.jpg)

设置是在 `_config.yml`文件里面的`Sidebar settings`那块。
```
# Sidebar settings
sidebar: true  #添加侧边栏
sidebar-about-description: "简单的描述一下你自己"
sidebar-avatar: /img/avatar-hux.jpg     #你的大头贴，请使用绝对地址.
```

侧边栏是响应式布局的，当屏幕尺寸小于992px的时候，侧边栏就会移动到底部。具体请见bootstrap栅格系统 <http://v3.bootcss.com/css/>


#### Mini About Me

Mini-About-Me 这个模块将在你的头像下面，展示你所有的社交账号。这个也是响应式布局，当屏幕变小时候，会将其移动到页面底部，只不过会稍微有点小变化，具体请看代码。

#### Featured Tags

看到这个网站 [Medium](http://medium.com) 的推荐标签非常的炫酷，所以我将他加了进来。
这个模块现在是独立的，可以呈现在所有页面，包括主页和发表的每一篇文章标题的头上。

```
# Featured Tags
featured-tags: true  
featured-condition-size: 1     # A tag will be featured if the size of it is more than this condition value
```

唯一需要注意的是`featured-condition-size`: 如果一个标签的 SIZE，也就是使用该标签的文章数大于上面设定的条件值，这个标签就会在首页上被推荐。
 
内部有一个条件模板 `{% if tag[1].size > {{site.featured-condition-size}} %}` 是用来做筛选过滤的.


#### Friends

好友链接部分。这会在全部页面显示。

设置是在 `_config.yml`文件里面的`Friends`那块，自己加吧。

```
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


#### Keynote Layout

HTML5幻灯片的排版：

![](http://huangxuan.me/img/blog-keynote.jpg)

这部分是用于占用html格式的幻灯片的，一般用到的是 Reveal.js, Impress.js, Slides, Prezi 等等.我认为一个现代化的博客怎么能少了放html幻灯的功能呢~

其主要原理是添加一个 `iframe`，在里面加入外部链接。你可以直接写到头文件里面去，详情请见下面的yaml头文件的写法。

```
---
layout:     keynote
iframe:     "http://huangxuan.me/js-module-7day/"
---
```

iframe在不同的设备中，将会自动的调整大小。保留内边距是为了让手机用户可以向下滑动，以及添加更多的内容。


#### Comment

博客不仅支持多说[Duoshuo](http://duoshuo.com)评论系统，也支持[Disqus](http://disqus.com)评论系统。

`Disqus`优点是：国际比较流行，界面也很大气、简介，如果有人评论，还能实时通知，直接回复通知的邮件就行了；缺点是：评论必须要去注册一个disqus账号，分享一般只有Facebook和Twitter，另外在墙内加载速度略慢了一点。想要知道长啥样，可以看以前的版本点[这里](http://brucezhaor.github.io/about.html) 最下面就可以看到。

`多说` 优点是：支持国内各主流社交软件(微博，微信，豆瓣，QQ空间 ...)一键分享按钮功能，另外登陆比较方便，管理界面也是纯中文的，相对于disqus全英文的要容易操作一些；缺点是：就是界面丑了一点。
当然你是可以自定义界面的css的，详情请看多说开发者文档 http://dev.duoshuo.com/docs/5003ecd94cab3e7250000008 。

**首先**，你需要去注册一个账号，不管是disqus还是多说的。**不要直接使用我的啊！**

**其次**，你只需要在下面的yaml头文件中设置一下就可以了。

```
duoshuo_username: _你的用户名_
# 或者
disqus_username: _你的用户名_
```

**最后**多说是支持分享的，如果你不想分享，请这样设置：`duoshuo_share: false`。你可以同时使用两个评论系统，不过个人感觉怪怪的。

#### Analytics

网站分析，现在支持百度统计和Google Analytics。需要去官方网站注册一下，然后将返回的code贴在下面：

```
# Baidu Analytics
ba_track_id: 4cc1f2d8f3067386cc5cdb626a202900

# Google Analytics
ga_track_id: 'UA-49627206-1'            # 你用Google账号去注册一个就会给你一个这样的id
ga_domain: huangxuan.me			# 默认的是 auto, 这里我是自定义了的域名，你如果没有自己的域名，需要改成auto。
```

#### Customization

如果你喜欢折腾，你可以去自定义我的这个模板的 code，[Grunt](gruntjs.com)已经为你准备好了。（感谢 Clean Blog）

JavaScript 的压缩混淆、Less 的编译、Apache 2.0 许可通告的添加与 watch 代码改动，这些任务都揽括其中。简单的在命令行中输入 `grunt` 就可以执行默认任务来帮你构建文件了。如果你想搞一搞 JavaScript 或 Less 的话，`grunt watch` 会帮助到你的。

**如果你可以理解 `_include/` 和 `_layouts/`文件夹下的代码（这里是整个界面布局的地方），你就可以使用 Jekyll 使用的模版引擎 [Liquid](https://github.com/Shopify/liquid/wiki)的语法直接修改/添加代码，来进行更有创意的自定义界面啦！**

#### Header Image

标题底图是可以自己选的，看看几篇示例post你就知道如何设置了。在
  [issue #6 ](https://github.com/Huxpro/huxpro.github.io/issues/6) 中我被问到：怎么样才能让标题底图好看呢？
  
标题底图的选取完全是看个人的审美了，我也帮不了你。每一篇文章可以有不同的底图，你想放什么就放什么，最后宽度要够，大小不要太大，否则加载慢啊。

但是需要注意的是本模板的标题是**白色**的，所以背景色要设置为**灰色**或者**黑色**，总之深色系就对了。当然你还可以自定义修改字体颜色，总之，用github pages就是可以完全的个性定制自己的博客。

#### SEO Title

我的博客标题是 **“Hux Blog”** 但是我想要在搜索的时候显示 **“黄玄的博客 | Hux Blog”** ，这个就需要SEO Title来定义了。

其实这个SEO Title就是定义了<head><title>标题</title></head>这个里面的东西和多说分享的标题，你可以自行修改的。

## 致谢

1. 这个模板是从这里[IronSummitMedia/startbootstrap-clean-blog-jekyll](https://github.com/IronSummitMedia/startbootstrap-clean-blog-jekyll)  fork 的。 感谢这个作者
2. 感谢[@BrucZhaoR](https://github.com/BruceZhaoR)的中文翻译 

3. 感谢 Jekyll、Github Pages 和 Bootstrap!



