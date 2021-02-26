# Chirpy

Language: [English](https://github.com/cotes2020/jekyll-theme-chirpy#readme) | 简体中文

[![Gem Version](https://img.shields.io/gem/v/jekyll-theme-chirpy?color=brightgreen)](https://rubygems.org/gems/jekyll-theme-chirpy)
[![Build Status](https://github.com/cotes2020/jekyll-theme-chirpy/workflows/build/badge.svg?branch=master&event=push)](https://github.com/cotes2020/jekyll-theme-chirpy/actions?query=branch%3Amaster+event%3Apush)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/8220b926db514f13afc3f02b7f884f4b)](https://app.codacy.com/manual/cotes2020/jekyll-theme-chirpy?utm_source=github.com&utm_medium=referral&utm_content=cotes2020/jekyll-theme-chirpy&utm_campaign=Badge_Grade_Dashboard)
[![GitHub license](https://img.shields.io/github/license/cotes2020/jekyll-theme-chirpy.svg)](https://github.com/cotes2020/jekyll-theme-chirpy/blob/master/LICENSE)
[![996.icu](https://img.shields.io/badge/link-996.icu-%23FF4D5B.svg)](https://996.icu)

一个不一样的 Jekyll 主题，采用响应式设计，方便记录、管理、分享您的知识和经验。[懂的进 »](https://chirpy.cotes.info)

[![Devices Mockup](https://cdn.jsdelivr.net/gh/cotes2020/chirpy-images/commons/devices-mockup.png)](https://chirpy.cotes.info)

> :bulb: 中文版文档存在更新不及时的风险（开源文档以英文为主，请见谅）。如果发现中、英文内容不匹配的情况，一切以英文版内容为准。如果您愿意的话，可提交 issue 提醒作者更新中文版 README，谢谢。

## 目录

- [功能一览](#功能一览)
- [前提要求](#前提要求)
- [安装](#安装)
- [使用](#使用)
- [文档](#文档)
- [参与贡献](#参与贡献)
- [鸣谢](#鸣谢)
- [赞助](#赞助)
- [许可证书](#许可证书)

## 功能一览

- 文章置顶
- 可配置的全局主题颜色
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

参考 [Jekyll Docs](https://jekyllrb.com/docs/installation/) 安装 `Ruby`，`RubyGems`，`Jekyll` 和 `Bundler`，Docker 粉可免。


## 安装

有二法可得此主题:

- **从 RubyGems 安装** - 易于版本升级，隔离无关的主题项目文件，让您的仓库舒适清爽。
- **从 GitHub 上 Fork** - 对个性化二次开发友好，但是难于升级，只适合专业开发人员使用。

### RubyGems 安装

在您的 Jekyll 站点的 `Gemfile` 添加:

```ruby
gem "jekyll-theme-chirpy"
```

然后，添加这行到您的 Jekyll 站点的 `_config.yml`:

```yaml
theme: jekyll-theme-chirpy
```

接着执行:

```console
$ bundle
```

最后, 拷贝额外所需主题的 gem 文件（详见 [starter 项目][starter] 的文件目录）至您的 Jekyll 站点, 然后把主题的 `_config.yml` 全部内容附加到您的 Jekyll 站点的同名文件。

> **提示**: 定位主题的 gem 文件，可以执行:
>
```console
$ bundle info --path jekyll-theme-chirpy
```

或者您可以 [使用 starter template][use-starter] 来快速创建 Jekyll 站点，以省去复制主题 gem 文件的时间。

### 在 GitHub 上 Fork

[Fork **Chirpy**](https://github.com/cotes2020/jekyll-theme-chirpy/fork) 然后克隆到本地。(友情提示：默认分支的代码处于开发状态，如果您想博客更加稳定，请切换到最新的 [Tag](https://github.com/cotes2020/jekyll-theme-chirpy/tags) 开始写作。)

安装依赖：

```console
$ bundle
```

接着执行文件初始化:

```console
$ bash tools/init.sh
```

> 如果您不打算部署到 GitHub Pages, 在上述命令后附加参数选项 `--no-gh`。

上述脚本完成了以下工作:

1. 从您的仓库中删除了:
    - `.travis.yml`
    - `_posts` 下的文件
    - `docs` 目录

2. 如果使用了参数 `--no-gh`，则会怒删 `.github`。否则，将会配置 GitHub Actions：把 `.github/workflows/pages-deploy.yml.hook` 的后缀 `.hook` 去除，然后删除 `.github` 里的其他目录和文件。

3. 自动提交一个 Commit 以保存上述文件的更改。

## 使用

### 配置文件

根据个人需要去修改 `_config.yml` 的变量，大部分都有注释介绍用法。典型的几个选项是：

- `url`
- `avatar`
- `timezone`
- `lang`

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

访问本地服务：<http://localhost:4000>

### 部署

部署开始前，把  `_config.yml` 的 `url` 改为 `https://<username>.github.io`(或者您的私有域名，如：`https://yourdomain.com`)。另外，如果您想使用 [Project 类型网站](https://help.github.com/en/github/working-with-github-pages/about-github-pages#types-of-github-pages-sites)，修改配置文件的 `baseurl` 为项目名称，以斜杠开头，如：`/project`。

现在您可以选择下列其中一个方式去站点部署。

#### 部署到 GitHub Pages

由于安全原因，GitHub Pages 的构建强制加了 `safe`参数，这导致了我们不能使用插件去创建所需的附加页面。因此，我们可以使用 GitHub Actions 去构建站点，把站点文件存储在一个新分支上，再指定该分支作为 Pages 服务的源。

快速检查 GitHub Actions 构建需要的文件:

- 确保您的 Jekyll 站点存在文件 `.github/workflows/pages-deploy.yml`。没有的话，新建并填入[示例工作流][workflow]的内容, 注意参数 `on.push.branches` 的值必须和您的仓库默认分支名相同。
- 检查您的 Jekyll 站点是否有文件 `tools/test.sh` 和 `tools/deploy.sh`. 没有的话, 从本仓库拷贝到您的 Jekyll 项目.

在 GitHub 把您的仓库命名为 `<GH-USERNAME>.github.io`，然后开始发布：

1. 推送任意一个 commit 到 `origin/master` 以触发 GitHub Actions workflow。一旦 build 完毕并且成功，远端将会自动出现一个新分支 `gh-pages` 用来存储构建的站点文件。

2. 回到 GitHub 上的仓库， 通过 _Settings_ → _Options_ → _GitHub Pages_ 选择分支 `gh-pages` 作为[_发布源_](https://docs.github.com/en/github/working-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site):

    ![gh-pages-sources](https://cdn.jsdelivr.net/gh/cotes2020/chirpy-images/posts/20190809/gh-pages-sources.png)

3. 按照 GitHub 指示的地址去访问您的网站。

#### 部署到其他 Pages 平台

在 GitHub 之外的平台，例如 GitLab，就没法享受 **GitHub Actions** 的便利了。因此我们需要在本地构建站点（或者在其他第三方 CI 平台），然后推送站点文件到服务器上。

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

## 文档

若想要更多细节以及更佳的阅读体验，请参阅 [线上教程](https://chirpy.cotes.info/categories/tutorial/)。 与此同时，[Wiki](https://github.com/cotes2020/jekyll-theme-chirpy/wiki) 也有一份教程的拷贝。

## 参与贡献

三人行必有我师，欢迎提报告 bug, 帮助改进代码质量，或者提交新功能。具体操作规则请参考 [贡献指南](../.github/CONTRIBUTING.md)，谢谢 🙏。

## 鸣谢

这个主题的开发主要基于 [Jekyll](https://jekyllrb.com/) 生态、[Bootstrap](https://getbootstrap.com/)、[Font Awesome](https://fontawesome.com/) 和其他一些出色的工具 (相关文件中可以找到这些工具的版权信息).

:tada: 感谢所有参与代码贡献的小伙伴, 他们的 GayHub ID 在这个[列表](https://github.com/cotes2020/jekyll-theme-chirpy/graphs/contributors)。 另外, 提交过 issues(或者未被合并 PR) 的高富帅和白富美也不会被遗忘,他/她们帮助报告 bug、分享新点子或者启发了我写出更通俗易懂的文档。

还有，感谢 [JetBrains][jb] 提供开源 License！

[![JB-logo](https://cdn.jsdelivr.net/gh/cotes2020/chirpy-images/commons/jetbrains.svg)][jb]

## 赞助

如果您喜欢这个主题或者它对您有帮助，请考虑打赏作者，您的支持将会极大地鼓励作者，并帮助作者更好地维护项目！

[![Ko-fi](https://img.shields.io/badge/-请作者喝杯咖啡-ff5e5b?logo=ko-fi&logoColor=white)](https://ko-fi.com/coteschung)
[![Wechat Pay](https://img.shields.io/badge/-微信打赏作者-brightgreen?logo=wechat&logoColor=white)][cn-donation]
[![Alipay](https://img.shields.io/badge/-支付宝打赏作者-blue?logo=alipay&logoColor=white)][cn-donation]

## 许可证书

本项目开源，基于 [MIT](https://github.com/cotes2020/jekyll-theme-chirpy/blob/master/LICENSE) 许可。

[starter]: https://github.com/cotes2020/chirpy-starter
[use-starter]: https://github.com/cotes2020/chirpy-starter/generate
[workflow]: https://github.com/cotes2020/jekyll-theme-chirpy/blob/master/.github/workflows/pages-deploy.yml.hook

<!-- ReadMe links -->

[jb]: https://www.jetbrains.com/?from=jekyll-theme-chirpy
[cn-donation]: https://cotes.gitee.io/alipay-wechat-donation/
