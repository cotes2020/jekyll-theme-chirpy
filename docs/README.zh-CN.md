<div align="right">

  中文(简体) /
  [EN](https://github.com/cotes2020/jekyll-theme-chirpy#readme)

</div>

<div align="center">

  # Chirpy Jekyll Theme

  [![Gem Version](https://img.shields.io/gem/v/jekyll-theme-chirpy?color=brightgreen)](https://rubygems.org/gems/jekyll-theme-chirpy)
  [![Build Status](https://github.com/cotes2020/jekyll-theme-chirpy/workflows/build/badge.svg?branch=master&event=push)](https://github.com/cotes2020/jekyll-theme-chirpy/actions?query=branch%3Amaster+event%3Apush)
  [![Codacy Badge](https://app.codacy.com/project/badge/Grade/4e556876a3c54d5e8f2d2857c4f43894)](https://www.codacy.com/gh/cotes2020/jekyll-theme-chirpy/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=cotes2020/jekyll-theme-chirpy&amp;utm_campaign=Badge_Grade)
  [![GitHub license](https://img.shields.io/github/license/cotes2020/jekyll-theme-chirpy.svg)](https://github.com/cotes2020/jekyll-theme-chirpy/blob/master/LICENSE)
  [![996.icu](https://img.shields.io/badge/link-996.icu-%23FF4D5B.svg)](https://996.icu)

  一款简约而强大、采用响应式设计的 Jekyll 主题，适合展示专业写作内容。

  [**线上体验 →**](https://chirpy.cotes.info)

  [![Devices Mockup](https://cdn.jsdelivr.net/gh/cotes2020/chirpy-images@0a003683c0c3ca549d12c309f9b3e03ea20981e5/commons/devices-mockup.png)](https://chirpy.cotes.info)

</div>

## 功能一览

- 本地化外观语言
- 可配置的主题颜色
- 文章置顶
- 文章最后修改日期
- 文章目录
- 自动推荐相关文章
- 语法高亮
- 二级目录
- 数学表达式
- Mermaid 图表
- 搜索
- Atom 订阅
- Disqus 评论
- Google 分析
- GA 浏览报告（高级功能）
- SEO 优化
- 网站性能优化

## 前提要求

参考 [Jekyll Docs](https://jekyllrb.com/docs/installation/) 安装 `Ruby`，`RubyGems`，`Jekyll` 和 `Bundler`。

## 安装

### 创建新仓库

有两种方式可以创建一个使用本主题的仓库：

- [**使用 Chirpy Starter**](#使用-chirpy-starter) - 易于版本升级，隔离无关的主题项目文件，让您的仓库舒适清爽。
- [**从 GitHub 上 Fork**](#从-github-上-fork) - 对个性化二次开发友好，但是难于升级。除非您决定魔改此款主题或者参与代码贡献，否则不推荐使用此方式。

#### 使用 Chirpy Starter

[使用 Chirpy Starter][use-starter] 来快速创建 Jekyll 站点，命名为 `<GH_USERNAME>.github.io`，其中 `GH_USERNAME` 是您的 GitHub username。

#### 从 GitHub 上 Fork

[Fork **Chirpy**](https://github.com/cotes2020/jekyll-theme-chirpy/fork) 并改名为 `<GH_USERNAME>.github.io`。友情提示：默认分支的代码处于开发状态，如果您想博客更加稳定，请切换到最新的 [Tag][latest-tag] 开始写作。

接着执行文件初始化:

```console
$ bash tools/init.sh
```

> **注**：如果您不打算部署到 GitHub Pages， 在上述命令后附加参数选项 `--no-gh`。

上述脚本将会:

1. 从您的仓库中删除了:

    - `.travis.yml`
    - `_posts` 下的文件
    - `docs` 目录

2. 如果使用了参数 `--no-gh`，则会怒删文件夹 `.github`。否则，配置 GitHub Actions：把 `.github/workflows/pages-deploy.yml.hook` 的后缀 `.hook` 去除，然后删除 `.github` 里的其他目录和文件。

3. 从 `.gitignore` 中删除 `Gemfile.lock`。

4. 自动提交一个 Commit 以保存上述文件的更改。

### 安装依赖：

首次运行本项目，需要先安装依赖：

```console
$ bundle
```

## 使用

### 配置文件

根据个人需要去修改 `_config.yml` 的变量，大部分都有注释介绍用法。典型的几个选项是：

- `url`
- `avatar`
- `timezone`
- `lang`

### 自定义样式

如果您需要自定义样式， 拷贝主题的文件 `assets/css/style.scss` 到您站点的相同路径上，然后在该文件末尾添加样式。

自 [`v4.1.0`][chirpy-4.1.0] 起，如果您想覆盖文件 `_sass/addon/variables.scss` 里定义的 SASS 变量的默认值，新建文件 `_sass/variables-hook.scss`，然后重写您需要的变量即可。

### 本地运行

发布之前，在本地预览:

```terminal
$ bundle exec jekyll s
```

或者用 Docker 运行:

```terminal
$ docker run -it --rm \
    --volume="$PWD:/srv/jekyll" \
    -p 4000:4000 jekyll/jekyll \
    jekyll serve
```

稍候片刻，即可访问本地服务：_<http://localhost:4000>_

### 部署

部署开始前，把  `_config.yml` 的 `url` 改为 `https://<username>.github.io`（或者您的私有域名，如：`https://yourdomain.com`）。另外，如果您想使用 [Project 类型网站](https://help.github.com/en/github/working-with-github-pages/about-github-pages#types-of-github-pages-sites)，修改配置文件的 `baseurl` 为项目名称，以斜杠开头，如：`/project`。

现在您可以选择下列其中一个方式去站点部署。

#### 使用 GitHub Actions 部署

由于安全原因，默认的 GitHub Pages 的构建强制加了 `safe`参数，这导致了我们不能使用插件去创建所需的附加页面。因此，我们可以使用 GitHub Actions 去构建站点，把站点文件存储在一个新分支上，再指定该分支作为 Pages 服务的源。

快速检查 GitHub Actions 构建需要的文件:

- 确保您的 Jekyll 站点存在文件 `.github/workflows/pages-deploy.yml`。没有的话，新建并填入「[示例 Workflow][workflow]」的内容，注意参数 `on.push.branches` 的值必须和您的仓库默认分支名相同。

- 检查您的 Jekyll 站点是否有文件 `tools/deploy.sh`。没有的话， 从本仓库拷贝到您的 Jekyll 项目。

- 再者，如果您已经把文件 `Gemfile.lock` 提交到了仓库里面，并且您运行本项目的操作系统不是 Linux，需要添加新的平台信息：

  ```console
  $ bundle lock --add-platform x86_64-linux
  ```

完成上述条目后，到 GitHub 把您的仓库命名为 `<GH-USERNAME>.github.io`，然后开始发布：

1. 推送任意一个 commit 到 `origin/master` 以触发 GitHub Actions workflow。一旦 build 完毕并且成功，远端将会自动出现一个新分支 `gh-pages` 用来存储构建的站点文件。

2. 回到 GitHub 上的仓库，选择标签 _Settings_ → 点击左侧导航栏的 _Pages_ → _GitHub Pages_ 选择分支 `gh-pages` 的 `/(root)` 作为「[发布源][pages-src]」:

  ![gh-pages-sources](https://cdn.jsdelivr.net/gh/cotes2020/chirpy-images@0a003683c0c3ca549d12c309f9b3e03ea20981e5/posts/20190809/gh-pages-sources.png)

3. 按照 GitHub 指示的地址去访问您的网站。

#### 手动构建部署

在 GitHub 之外的平台，就没法享受 **GitHub Actions** 的便利了。因此您需要在本地构建站点，然后推送站点文件到服务器上。

在项目根目录，运行:

```console
$ JEKYLL_ENV=production bundle exec jekyll b
```

或者通过 Docker 构建:

```terminal
$ docker run -it --rm \
    --env JEKYLL_ENV=production \
    --volume="$PWD:/srv/jekyll" \
    jekyll/jekyll \
    jekyll build
```

生成的静态文件将会在 `_site`， 把内部的文件上传到服务器即可。

### 升级

这取决于您如何使用这个 theme：

- 如果您是使用 theme gem（`Gemfile` 会有 `gem "jekyll-theme-chirpy"`），编辑 `Gemfile` 并更新 them gem 的版本号，譬如：

  ```diff
  - gem "jekyll-theme-chirpy", "~> 3.2", ">= 3.2.1"
  + gem "jekyll-theme-chirpy", "~> 3.3", ">= 3.3.0"
  ```

  接着执行以下命令：

  ```console
  $ bundle update jekyll-theme-chirpy
  ```

  随着 theme 版本的升级，运行站点的必要文件（详见 [Chirpy Starter][starter]）以及配置选项会出现变化，请参阅「[升级指南](https://github.com/cotes2020/jekyll-theme-chirpy/wiki/Upgrade-Guide)」的改动细节去保持您仓库中的相关文件同步到最新版本。

- 如果您是以 fork 的方式使用（您站点的 `Gemfile` 会有 `gemspec`），那么合并上游 [最新的 tag][latest-tag] 到您的 Repo 以完成升级。期间很有可能会产生冲突 (conflicts)，请务必耐心谨慎地解决它们。

## 文档

若想要更多细节以及更佳的阅读体验，请参阅「[线上教程](https://chirpy.cotes.info/categories/tutorial/)」。 与此同时，「[Wiki](https://github.com/cotes2020/jekyll-theme-chirpy/wiki)」也有一份教程的拷贝。请注意，Demo 网站和 Wiki 上的文档都是基于最新的发行版本，而 `master` 分支的功能往往领先于现有文档。

## 参与贡献

三人行必有我师，欢迎提报告 bug，帮助改进代码质量，或者提交新功能。具体操作规则请参考「[贡献指南](../.github/CONTRIBUTING.md)」，谢谢 🙏。

## 鸣谢

此款主题的开发主要基于 [Jekyll](https://jekyllrb.com/) 生态、[Bootstrap](https://getbootstrap.com/)、[Font Awesome](https://fontawesome.com/) 和其他一些出色的工具 (相关文件中可以找到这些工具的版权信息)。头像和图标的设计来自于 [Clipart Max](https://www.clipartmax.com/middle/m2i8b1m2K9Z5m2K9_ant-clipart-childrens-ant-cute/)。

:tada: 感谢所有参与代码贡献的小伙伴，他们的 GayHub ID 在这个「[列表](https://github.com/cotes2020/jekyll-theme-chirpy/graphs/contributors)」。 另外，提交过 issues（或者未被合并 PR）的高富帅和白富美也不会被遗忘，他/她们帮助报告 bug、分享新点子或者启发了我写出更通俗易懂的文档。

还有，感谢 [JetBrains][jb] 提供开源 License！

## 赞助

如果您喜欢此款主题或者它对您有帮助，请考虑打赏作者，您的支持将会极大地鼓励作者，并帮助作者更好地维护项目！

[![Buy Me a Coffee](https://img.shields.io/badge/-请作者喝杯咖啡-ff813f?logo=buy-me-a-coffee&logoColor=white)](https://www.buymeacoffee.com/coteschung)
[![Wechat Pay](https://img.shields.io/badge/-微信打赏作者-brightgreen?logo=wechat&logoColor=white)][cn-donation]
[![Alipay](https://img.shields.io/badge/-支付宝打赏作者-blue?logo=alipay&logoColor=white)][cn-donation]

## 许可证书

本项目开源，基于 [MIT](https://github.com/cotes2020/jekyll-theme-chirpy/blob/master/LICENSE) 许可。

[starter]: https://github.com/cotes2020/chirpy-starter
[use-starter]: https://github.com/cotes2020/chirpy-starter/generate
[workflow]: https://github.com/cotes2020/jekyll-theme-chirpy/blob/master/.github/workflows/pages-deploy.yml.hook
[chirpy-4.1.0]: https://github.com/cotes2020/jekyll-theme-chirpy/releases/tag/v4.1.0
[pages-src]: https://docs.github.com/en/github/working-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site
[latest-tag]: https://github.com/cotes2020/jekyll-theme-chirpy/tags

<!-- ReadMe links -->

[jb]: https://www.jetbrains.com/?from=jekyll-theme-chirpy
[cn-donation]: https://cotes.gitee.io/alipay-wechat-donation/
