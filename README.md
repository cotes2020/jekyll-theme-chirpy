# Sokranotes Blog

[![license](https://badgen.net/github/license/Sokranotes/Sokranotes.github.io?color=green)](https://github.com/Sokranotes/Sokranotes.github.io/blob/master/LICENSE)

Thinking	Recording	Learning

[![Sokranotes](https://sokranotes.github.io/assets/img/avatar.jpg)](https://sokranotes.github.io/)


## 博客搭建过程

### 1 Windows下安装依赖

下载[RubyInstaller](https://rubyinstaller.org/)并安装

注意版本：Jekyll 4.0.2依赖的Ruby版本**Ruby(>= 2.4, < 3.0)**

### 2 本地安装Jekyll并创建项目

```shell
// 1 安装Jekyll
gem install jekyll
// 2 在cmd对应路径下新建一个Jekyll项目myblog
jekyll new myblog
// 3 切换路径
cd myblog
// 4 解决依赖问题
bundle add webrick
// 5 启动demo
jekyll serve

//本地预览
bundle exec jekyll s
//运行
jekyll serve
```


### 3 个性化修改

在Jekyll的主题Chirpy的基础之上进行了一点小修改

1. 侧边栏颜色及文字颜色，修改配色
2. tags中展开显示，提高可读性，借鉴了 [LOFFER](https://github.com/FromEndWorld/LOFFER) 的设计
3. 侧边栏下方图标修改，去掉多余的图标，并进行了分隔
4. 文章分享修改
5. 添加MathJax以更好的支持LaTeX公式
6. 开启基于Disqus的评论（评论需要科学上网）

### 4 部署到GitHub

1. 确认本地修改完成之后，记录过的文件
2. fork [jekyll-theme-chirpy](https://github.com/cotes2020/jekyll-theme-chirpy) 并克隆到本地
3. 在项目根目录下安装依赖
    `$ bundle`
4. 文件初始化，在Windows下需要在Git command中运行
    `$ bash tools/init.sh`
5. 修改
  1. 项目名称
  2. `_config.yml`配置文件
6. 推送一个commit以触发GitHub Actions workflow，build完毕且成功远端会自动出现一个新分支`gh-pages`
7. 选择`gh-pages`分支作为Page的发布源




## 文章头部格式说明
```
---
layout: post
title: Hello World!
date: 2021-7-13 17:02:35 +0800
Author: Sokranotes
tags: [blog building, ]
comments: true
categories: technology blog_building
toc: true
typora-root-url: ..
---
```


说明：

1. title：文章标题
2. data：文章显示的日期，日期 时间 时区(默认为0时区)
3. toc：文章侧边目录是否开启
4. categories: 设置categories，子目录直接在后面添加，以空格隔开，最多支持二级目录
5. 图片位置说明：上传图片不能置于/_post路径下，需放在/assets目录下
6. typora-root-url：设置在typora中的根目录，方便typora中图片显示


## 参考

1. [Jekyll](https://jekyllrb.com/)
2. [Chirpy](https://github.com/cotes2020/jekyll-theme-chirpy)
3. [LOFFER](https://github.com/FromEndWorld/LOFFER)
4. [给 Jekyll 博客添加 Latex 公式支持](https://bryceyang.github.io/add-eqution-support-in-jekyll/)
5. [Windows下安装Jekyll](https://jekyllrb.com/docs/installation/windows/)
6. [快速指南](http://jekyllcn.com/docs/quickstart/)
7. [依赖问题](https://github.com/jekyll/jekyll/issues/8523)

## 致谢

:tada: ​[Chirpy](https://github.com/cotes2020/jekyll-theme-chirpy)

:tada: [LOFFER](https://github.com/FromEndWorld/LOFFER)

:tada: ​[Jekyll](https://jekyllrb.com/) 

:tada: ​[Bootstrap](https://getbootstrap.com/)

:tada: ​[Font Awesome](https://fontawesome.com/)

:tada: ​Other wonderful tools (their copyright information can be found in the relevant files).

:tada: [JetBrains][https://www.jetbrains.com/?from=Sokranotes.github.io] for providing the open source license.

:tada: Thanks to all the volunteers who contributed to those project

:tada: Thanks to​ those guys who submitted the issues or unmerged PR because they reported bugs, shared ideas.



## License

This work is published under [CC-BY-SA-4.0](https://github.com/Sokranotes/Sokranotes.github.io/blob/master/LICENSE) License.

