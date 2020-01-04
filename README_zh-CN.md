# Jekyll Theme Chirpy

[![Build Status](https://travis-ci.com/cotes2020/jekyll-theme-chirpy.svg?branch=master)](https://travis-ci.com/cotes2020/jekyll-theme-chirpy)
[![GitHub license](https://img.shields.io/github/license/cotes2020/jekyll-theme-chirpy.svg)](https://github.com/cotes2020/jekyll-theme-chirpy/blob/master/LICENSE)
[![996.icu](https://img.shields.io/badge/link-996.icu-%23FF4D5B.svg)](https://996.icu)

[English](README.md) | 中文

![devices-mockup](https://raw.githubusercontent.com/cotes2020/jekyll-theme-chirpy/master/assets/img/sample/devices-mockup.png)


一个不一样的 Jekyll 主题，采用响应式设计，方便记录、管理、分享你的知识和经验。[**在线 Demo** »](https://chirpy.cotes.info)


## 功能预览

* 自动暗夜模式
* 文章最后修改日期
* 文章目录
* 自动推荐相关文章
* Disqus 评论
* 语法高亮
* 二级目录
* 搜索
* HTML 压缩
* Atom 订阅
* Google 分析
* 浏览数展示（高级功能）
* SEO 优化
* 网站性能优化

## 开始

### 准备工作

按照 [Jekyll 官方文档](https://jekyllrb.com/docs/installation/) 完成基础环境的安装 (Ruby, RubyGem, Bundler 和 Jekyll)。为了使用项目内免费提供的脚本工具，你还需要安装 [Python](https://www.python.org/downloads/)( >= 3.5) 和 [ruamel.yaml](https://pypi.org/project/ruamel.yaml/).

接着，[fork](https://github.com/cotes2020/jekyll-theme-chirpy/fork) 一份代码，然后克隆你的 Fork 版到本地机器上。


### 安装 Jekyll 插件

在根目录下运行:

```terminal
$ bundle install
```
`bundle` 命令会自动安装 `Gemfile` 内声明的依赖插件.


### 文件目录

下面是主要的文件目录：

```sh
jekyll-theme-chirpy/
├── _data
├── _includes      
├── _layouts
├── _posts          # posts stay here
├── _scripts
│   └── travis      # CI stuff, remove it
├── .travis.yml     # remove this, too
├── .github         # remove it
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
├── feed.xml
├── index.html
├── robots.txt
└── sitemap.xml
```


你需要将以下文件或目录删除:

- .travis.yml
- .github
- _scripts/travis


### 配置文件

根据个人需要去修改 `_config.yml` 的变量，大部分都有注释介绍用法。

* 头像
    
    示例的头像文件放置在：`/assets/img/sample/avatar.jpg`. 把它换成你自己的头像，路径不限定，越小越好。(压缩图像体积可上这个网站：*<https://tinypng.com/>* ).

* 时区

    时区由 `timezone` 定义，默认为 `亚洲/上海`，如果肉身翻墙要换城市可在此列表找到： [TimezoneConverter](http://www.timezoneconverter.com/cgi-bin/findzone/findzone) 或者 [Wikipedia](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones).

* Atom 订阅

    Atom 订阅路径是:

    ```
    <SITE_URL>/feed.xml
    ```

    `SITE_URL` 由变量 `url` 定义。


###  本地运行

使用以下工具可轻松运行:

```terminal
$ bash tools/run.sh
```

>**注**: *最后更新* 列表根据文章的 git 修改记录生成, 所以运行前先把 `_posts` 目录的修改提交.

访问本地服务： <http://localhost:4000>

如果你想在本地服务运行后，把修改源文件的更改实时刷新，可使用选项 `-r` (或 `--realtime`)，不过要先安装依赖 [**fswatch**](http://emcrisostomo.github.io/fswatch/) 。

###  部署到 GitHub Pages

部署开始前，把  `_config.yml` 的 `url` 改为 `https://<username>.github.io`(或者你的私有域名，如：`https://yourdomain.com`).

#### 方法 1: 由 GitHub Pages 生成站点

依照本方法，你可以直接把源码推送到远端仓库。

> **注**: 如果你想使用任何不在这个[列表](https://pages.github.com/versions/)上的插件，越过此方法，直接看 [*方法 2: 本地构建*](#方法-2-本地构建).

**1**. 仓库改名为:

|站点类型 | 仓库名称|
|:---|:---|
|User or Organization | `<username>.github.io`|
|Project| `<username>.github.io` 以外的名字, 譬如 `project`|

**2**. 提交本地更改, 然后运行:

```console
$ bash tools/init.sh
```

它会自动生成文章的 *最后修改日期* 和 *分类 / 标签* 页面.

**3**. 推送到 `origin/master` 然后到 GitHub 网页为该项目开启 Pages 服务。

**4**. 网站将运行在：

|站点类型 | 网站 URL |
|:---|:---|
|User or Organization | `https://<username>.github.io/`|
|Project| `https://<username>.github.io/project/`|


#### 方法 2: 本地构建

由于安全原因，GitHub Pages 不允许第三方插件运行，如果你想突破规则，就要本地构建站点内容。

**1**. 到 GitHub 网页, 创建一个新的仓库，根据以下规则命名: 

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

如果你想使用 Project 网站, 修改配置文件的 `baseurl` 为项目名称, 以斜杠开头，如：`/project`。或者，在上述命令行后面加参数`-b /project`，`project` 替换为新仓库名称。

生成的静态文件将会在 `/path/to/local/project`. 把新仓库的修改提交并推送到远端 `master` 分支.

**3**. 回到 GithHub 网页，为该仓库开启 Pages 服务。

**4**. 网站将运行在:

|站点类型 | 站点 URL |
|:---|:---|
|User or Organization | `https://<username>.github.io/`|
|Project| `https://<username>.github.io/project/`|


## 文档

更多细节及更佳的阅读体验, 请参阅 [线上教程](https://chirpy.cotes.info/categories/tutorial/)。 与此同时, [Wiki](https://github.com/cotes2020/jekyll-theme-chirpy/wiki) 也有一份教程的拷贝.


## 赞助

想要打赏作者吗？在 [项目主页](https://github.com/cotes2020/jekyll-theme-chirpy) 点击按钮 <kbd>❤️Sponsor</kbd> 选择 *支付宝、微信* 链接 <https://cotes.gitee.io/alipay-wechat-donation> 即可完成，您的打赏将会鼓励作者去更好地完成开源项目！

## License 许可

本项目开源，基于 [MIT](https://github.com/cotes2020/jekyll-theme-chirpy/blob/master/LICENSE) 许可。
