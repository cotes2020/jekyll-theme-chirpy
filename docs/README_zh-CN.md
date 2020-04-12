# Chirpy

🌏 [English](../README.md) • 简体中文

[![Build Status](https://github.com/cotes2020/jekyll-theme-chirpy/workflows/build/badge.svg?event=push)](https://github.com/cotes2020/jekyll-theme-chirpy/actions?query=event%3Apush)
[![GitHub license](https://img.shields.io/github/license/cotes2020/jekyll-theme-chirpy.svg)](https://github.com/cotes2020/jekyll-theme-chirpy/blob/master/LICENSE)
[![996.icu](https://img.shields.io/badge/link-996.icu-%23FF4D5B.svg)](https://996.icu)

一个不一样的响应式设计 Jekyll 主题（内附神秘工具），采用响应式设计，方便记录、管理、分享你的知识和经验。

[![Devices Mockup](https://raw.githubusercontent.com/cotes2020/jekyll-theme-chirpy/master/assets/img/sample/devices-mockup.png)](https://chirpy.cotes.info)

## 目录

* [功能预览](#功能预览)
* [安装](#安装)
* [运行指南](#运行指南)
* [参与贡献](#参与贡献)
* [感谢](#感谢)
* [赞助](#赞助)
* [许可证书](#许可证书)

## 功能预览

* 自动暗夜模式
* 文章最后修改日期
* 文章目录
* 自动推荐相关文章
* 语法高亮
* 二级目录
* 搜索
* Atom 订阅
*  Disqus 评论
* Google 分析
* GA 浏览报告（高级功能）
* SEO 优化
* 网站性能优化


## 安装

### 准备工作

按照 [Jekyll 官方文档](https://jekyllrb.com/docs/installation/) 完成基础环境的安装 (Ruby，RubyGem，Bundler 和 Jekyll)。

为了使用项目内免费提供的脚本工具提升你的写作体验，如果你的机器系统是 Debian 或者 macOS，则需要确保安装了 [GNU coreutils](https://www.gnu.org/software/coreutils/)。否则，通过以下方式获得：

* Debian

 ```console
 $ sudo apt-get install coreutils
 ```

* macOS

 ```console
 $ brew install coreutils
 ```

接着，[fork](https://github.com/cotes2020/jekyll-theme-chirpy/fork) 一份代码，然后克隆你 Fork 的仓库到本地机器上。

```console
$ git clone git@github.com:USER/jekyll-theme-chirpy.git -b master
```

把上述的`USER` 替换为你的 GitHub username。



### 安装 Jekyll 插件

本地首次运行或编译，请在项目根目录下运行:

```terminal
$ bundle install
```
`bundle` 命令会自动安装 `Gemfile` 内声明的依赖插件.



## 运行指南

### 文件目录

下面是主要的文件目录：

```sh
jekyll-theme-chirpy/
├── _data
├── _includes      
├── _layouts
├── _posts          # posts stay here
├── _scripts
├── .travis.yml     # remove it
├── .github         # remove this, too
├── assets      
├── tabs
│   └── about.md    # the ABOUT page
├── .gitignore
├── 404.html
├── Gemfile
├── LICENSE
├── README.md
├── _config.yml     # configuration file
├── tools           # script tools
├── docs
├── feed.xml
├── index.html
├── robots.txt
└── sitemap.xml
```


你需要将以下文件或目录删除:

- .travis.yml
- .github


### 配置文件

根据个人需要去修改 `_config.yml` 的变量，大部分都有注释介绍用法。

* 头像
    
    示例的头像文件放置在：`/assets/img/sample/avatar.jpg`. 把它换成你自己的头像，路径不限定，越小越好。(压缩图像体积可上这个网站：*<https://tinypng.com/>* ).

* 时区

    时区由 `timezone` 定义，默认为 `亚洲/上海`，如果肉身翻墙要换城市可在此列表找到： [TimezoneConverter](http://www.timezoneconverter.com/cgi-bin/findzone/findzone) 或者 [Wikipedia](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones).


###  本地运行

使用以下工具可轻松运行:

```terminal
$ bash tools/run.sh
```

访问本地服务： <http://localhost:4000>

如果你想在本地服务运行后，把修改源文件的更改实时刷新，可使用选项 `-r` (或 `--realtime`)，不过要先安装依赖 [**fswatch**](http://emcrisostomo.github.io/fswatch/) 。

###  部署到 GitHub Pages

部署开始前，把  `_config.yml` 的 `url` 改为 `https://<username>.github.io`(或者你的私有域名，如：`https://yourdomain.com`)。另外，如果你想使用 [Project 类型网站](https://help.github.com/en/github/working-with-github-pages/about-github-pages#types-of-github-pages-sites)，修改配置文件的 `baseurl` 为项目名称，以斜杠开头，如：`/project`。

#### 方法 1: 由 GitHub Pages 生成站点

依照本方法，你可以直接把源码推送到远端仓库。

> **注**: 如果你想使用任何不在这个[列表](https://pages.github.com/versions/)上的插件，越过此方法，直接看 [*方法 2: 本地构建*](#方法-2-本地构建).

**1**. 仓库改名为:

|站点类型 | 仓库名称|
|:---|:---|
|User or Organization | `<username>.github.io`|
|Project| `<username>.github.io` 以外的名字，譬如 `project`|

**2**. 提交本地更改，然后运行:

```console
$ bash tools/init.sh
```

>**注**: *最后更新* 列表根据文章的 git 修改记录生成，所以运行前先把 `_posts` 目录的修改提交。

它会自动生成文章的 *最后修改日期* 和 *分类 / 标签* 页面.

**3**. 推送到 `origin/master` 然后到 GitHub 网页为该项目开启 Pages 服务。

**4**. 网站将运行在：

|站点类型 | 网站 URL |
|:---|:---|
|User or Organization | `https://<username>.github.io/`|
|Project| `https://<username>.github.io/project/`|


#### 方法 2: 本地构建

由于安全原因，GitHub Pages 不允许第三方插件运行，如果你想突破规则，就要本地构建站点内容。

**1**. 到 GitHub 网页，创建一个新的仓库，根据以下规则命名: 

|站点类型 | 仓库名称|
|:---|:---|
|User or Organization | `<username>.github.io`|
|Project| `<username>.github.io` 以外的名字， 例如 `project`|

然后 Clone 新仓库到本地。

**2**. 构建站点:

```console
$ bash tools/build.sh -d /path/to/local/project/
```
> `project` 为新仓库名称。

生成的静态文件将会在 `/path/to/local/project`. 把新仓库的修改提交并推送到远端 `master` 分支.

**3**. 回到 GithHub 网页，为该仓库开启 Pages 服务。

**4**. 网站将运行在:

|站点类型 | 站点 URL |
|:---|:---|
|User or Organization | `https://<username>.github.io/`|
|Project| `https://<username>.github.io/project/`|


### 文档

更多细节及更佳的阅读体验，请参阅 [线上教程](https://chirpy.cotes.info/categories/tutorial/)。 与此同时，[Wiki](https://github.com/cotes2020/jekyll-theme-chirpy/wiki) 也有一份教程的拷贝。


## 参与贡献

三人行必有我师，欢迎提报告 bug, 帮助改进代码质量，或者提交新功能。具体操作规则请参考[此文](.github/CONTRIBUTING.md)，谢谢🙏。

## 感谢

这个主题的开发主要基于 [Jekyll](https://jekyllrb.com/) 生态、[Bootstrap](https://getbootstrap.com/)、[Font Awesome](https://fontawesome.com/) 和其他一些出色的工具 (相关文件中可以找到这些工具的版权信息).

:tada:感谢所有参与代码贡献的小伙伴, 他们的 GayHub ID 在这个[列表](https://github.com/cotes2020/jekyll-theme-chirpy/graphs/contributors)。 另外, 提交过 issues(或者未被合并 PR)的高富帅和白富美也不会被遗忘,他/她们帮助报告 bug、分享新点子或者启发了我写出更通俗易懂的文档。



## 赞助

如果您喜欢这个主题或者它对您有帮助，请考虑打赏作者：在 [项目主页](https://github.com/cotes2020/jekyll-theme-chirpy) 点击按钮 <kbd>:heart:Sponsor</kbd> 选择适合的链接即可完成（国内一般选第二个链接，支付宝/微信赞助），您的打赏将会极大地鼓励作者，并帮助作者更好地维护项目！


## 许可证书

本项目开源，基于 [MIT](https://github.com/cotes2020/jekyll-theme-chirpy/blob/master/LICENSE) 许可。
