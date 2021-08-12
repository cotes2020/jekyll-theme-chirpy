<div align="right">
  <a href="https://github.com/cotes2020/jekyll-theme-chirpy#readme">EN</a> |
  <a href="https://github.com/cotes2020/jekyll-theme-chirpy/blob/master/docs/README.zh-CN.md">中文 (简体)</a>
</div>

<div align="center">
  <h1>Chirpy Jekyll Theme</h1>
  <p>
    <a href="https://rubygems.org/gems/jekyll-theme-chirpy">
    <img alt="Gem Version" src="https://img.shields.io/gem/v/jekyll-theme-chirpy?color=brightgreen"></img>
  </a>
    <a href="https://github.com/cotes2020/jekyll-theme-chirpy/actions?query=branch%3Amaster+event%3Apush">
    <img alt="Build Status" src="https://github.com/cotes2020/jekyll-theme-chirpy/workflows/build/badge.svg?branch=master&event=push"></img>
  </a>
    <a href="https://app.codacy.com/manual/cotes2020/jekyll-theme-chirpy?utm_source=github.com&utm_medium=referral&utm_content=cotes2020/jekyll-theme-chirpy&utm_campaign=Badge_Grade_Dashboard">
    <img alt="Codacy Badge" src="https://api.codacy.com/project/badge/Grade/8220b926db514f13afc3f02b7f884f4b"></img>
  </a>
    <a href="https://github.com/cotes2020/jekyll-theme-chirpy/blob/master/LICENSE">
    <img alt="GitHub license" src="https://img.shields.io/github/license/cotes2020/jekyll-theme-chirpy.svg"></img>
  </a>
    <a href="https://996.icu">
    <img alt="996.icu" src="https://img.shields.io/badge/link-996.icu-%23FF4D5B.svg"></img>
  </a>
  </p>
</div>

A minimal, sidebar, responsive web design Jekyll theme that focuses on text presentation. Designed to help you easily record and share your knowledge.

[Live Demo »](https://chirpy.cotes.info)

<p align="center">
  <a href="https://chirpy.cotes.info">
    <img alt="Devices Mockup" src="https://cdn.jsdelivr.net/gh/cotes2020/chirpy-images/commons/devices-mockup.png"></img>
  </a>
</p>

## Features

- Localized Layout
- Configurable Theme Mode
- Pinned Posts
- Hierarchical Categories
- Last Modified Date for Posts
- Table of Contents
- Automatically Recommend Related Posts
- Syntax Highlighting
- Mathematical Expressions
- Mermaid Diagram & Flowchart
- Search
- Atom Feeds
- Disqus Comments
- Google Analytics
- GA Pageviews Reporting (Advanced)
- SEO & Performance Optimization

## Prerequisites

Follow the [Jekyll Docs](https://jekyllrb.com/docs/installation/) to complete the installation of `Ruby`, `RubyGems`, `Jekyll` and `Bundler`. Please note that the version of `Ruby` must meet the requirements of the theme on [RubyGems.org](https://rubygems.org/gems/jekyll-theme-chirpy).

## Installation

There are two ways to get the theme:

- **[Install from RubyGems](#install-from-rubygems)** - Easy to update, isolate irrelevant project files so you can focus on writing.
- **[Fork on GitHub](#fork-on-github)** - Convenient for custom development, but difficult to update, only suitable for web developers.

### Install from RubyGems

Add this line to your Jekyll site's `Gemfile`:

```ruby
gem "jekyll-theme-chirpy"
```

And add this line to your Jekyll site's `_config.yml`:

```yaml
theme: jekyll-theme-chirpy
```

And then execute:

```console
$ bundle
```

Next, go to the installed local theme path:

```console
$ cd "$(bundle info --path jekyll-theme-chirpy)"
```

And then copy the critical files (for details, see [starter project][starter]) from the theme's gem to your Jekyll site.

> ⚠️ **Watch out for duplicate files!**
>
> If your Jekyll site is created by the `jekyll new` command, there will be `index.markdown` and `about.markdown` in the root directory of your site. Please be sure to remove them, otherwise they will overwrite the `index.html` and `_tabs/about.html` from this project, resulting in blank or messy pages.

As an alternative, which we recommend, you can create a Jekyll site [**using the starter template**][use-starter] to save time copying files from the theme's gem. We've prepared everything you need there!

### Fork on GitHub

[Fork **Chirpy**](https://github.com/cotes2020/jekyll-theme-chirpy/fork) on GitHub and then clone your fork to local. (Please note that the default branch code is in development.  If you want the blog to be stable, please switch to the [latest tag](https://github.com/cotes2020/jekyll-theme-chirpy/tags) and start writing.)

Install gem dependencies by:

```console
$ bundle
```

And then execute:

```console
$ bash tools/init.sh
```

> **Note**: If you don't plan to deploy your site on GitHub Pages, append parameter option `--no-gh` at the end of the above command.

What it does is:

1. Remove some files or directories from your repository:
    - `.travis.yml`
    - files under `_posts`
    - folder `docs`

2. If you use the `--no-gh` option, the directory `.github` will be deleted. Otherwise, setup the GitHub Action workflow by removing the extension `.hook` of `.github/workflows/pages-deploy.yml.hook`, and then remove the other files and directories in the folder `.github`.

3. Automatically create a commit to save the changes.

## Usage

### Configuration

Update the variables of `_config.yml` as needed. Some of them are typical options:

- `url`
- `avatar`
- `timezone`
- `lang`

### Customing Stylesheet

If you need to customize stylesheet, copy the theme's `assets/css/style.scss` to the same path on your Jekyll site, and then add the custom style at the end of the style file.

Starting from `v4.1.0`, if you want to overwrite the SASS variables defined in `_sass/addon/variables.scss`, create a new file `_sass/variables-hook.scss` and assign new values to the target variable in it.

### Running Local Server

You may want to preview the site contents before publishing, so just run it by:

```console
$ bundle exec jekyll s
```

Or run the site on Docker with the following command:

```terminal
$ docker run -it --rm \
    --volume="$PWD:/srv/jekyll" \
    -p 4000:4000 jekyll/jekyll \
    jekyll serve
```

Open a browser and visit to _<http://localhost:4000>_.

### Deployment

Before the deployment begins, checkout the file `_config.yml` and make sure the `url` is configured correctly. Furthermore, if you prefer the [**project site**](https://help.github.com/en/github/working-with-github-pages/about-github-pages#types-of-github-pages-sites) and don't use a custom domain, or you want to visit your website with a base URL on a web server other than **GitHub Pages**, remember to change the `baseurl` to your project name that starting with a slash, e.g, `/project-name`.

Now you can choose ONE of the following methods to deploy your Jekyll site.

#### Deploy on GitHub Pages

For security reasons, GitHub Pages build runs on `safe` mode, which restricts us from using plugins to generate additional page files. Therefore, we can use **GitHub Actions** to build the site, store the built site files on a new branch, and use that branch as the source of the GH Pages service.

Quickly check the files needed for GitHub Actions build:

- Ensure your Jekyll site has the file `.github/workflows/pages-deploy.yml`. Otherwise, create a new one and fill in the contents of the [workflow file][workflow], and the value of the `on.push.branches` should be the same as your repo's default branch name.
- Ensure your Jekyll site has file `tools/test.sh` and `tools/deploy.sh`. Otherwise, copy them from this repo to your Jekyll site.

And then rename your repository to `<GH-USERNAME>.github.io` on GitHub.

Now publish your Jekyll site by:

1. Push any commit to remote to trigger the GitHub Actions workflow. Once the build is complete and successful, a new remote branch named `gh-pages` will appear to store the built site files.

2. Browse to your repo's landing page on GitHub and select the branch `gh-pages` as the [publishing source](https://docs.github.com/en/github/working-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site) through _Settings_ → _Options_ → _GitHub Pages_:

    ![gh-pages-sources](https://cdn.jsdelivr.net/gh/cotes2020/chirpy-images/posts/20190809/gh-pages-sources.png)

3. Visit your website at the address indicated by GitHub.

#### Deploy on Other Platforms

On platforms other than GitHub, we cannot enjoy the convenience of **GitHub Actions**. Therefore, we should build the site locally (or on some other 3rd-party CI platform) and then put the site files on the server.

Go to the root of the source project, build your site by:

```console
$ JEKYLL_ENV=production bundle exec jekyll b
```

Or build the site with Docker by:

```terminal
$ docker run -it --rm \
    --env JEKYLL_ENV=production \
    --volume="$PWD:/srv/jekyll" \
    jekyll/jekyll \
    jekyll build
```

Unless you specified the output path, the generated site files will be placed in folder `_site` of the project's root directory. Now you should upload those files to your web server.

## Documentation

For more details and a better reading experience, please check out the [tutorials on the demo site](https://chirpy.cotes.info/categories/tutorial/). In the meanwhile, a copy of the tutorial is also available on the [Wiki](https://github.com/cotes2020/jekyll-theme-chirpy/wiki). Please note that the tutorials on the demo website or Wiki are based on the latest release, and the features of `master` branch usually ahead of the documentation.

## Contributing

The old saying, "Two heads are better than one." Consequently, welcome to report bugs, improve code quality or submit a new feature. For more information, see [contributing guidelines](.github/CONTRIBUTING.md).

## Credits

This theme is mainly built with [Jekyll](https://jekyllrb.com/) ecosystem, [Bootstrap](https://getbootstrap.com/), [Font Awesome](https://fontawesome.com/) and some other wonderful tools (their copyright information can be found in the relevant files). The avatar and favicon design comes from [Clipart Max](https://www.clipartmax.com/middle/m2i8b1m2K9Z5m2K9_ant-clipart-childrens-ant-cute/).

:tada: Thanks to all the volunteers who contributed to this project, their GitHub IDs are on [this list](https://github.com/cotes2020/jekyll-theme-chirpy/graphs/contributors). Also, I won't forget those guys who submitted the issues or unmerged PR because they reported bugs, shared ideas, or inspired me to write more readable documentation.

Last but not least, thank [JetBrains][jb] for providing the open source license.

## Sponsoring

If you like this theme or find it helpful, please consider sponsoring me, because it will encourage and help me better maintain the project, I will be very grateful!

[![Buy Me a Coffee](https://img.shields.io/badge/-Buy%20Me%20a%20Coffee-ff813f?logo=buy-me-a-coffee&logoColor=white)](https://www.buymeacoffee.com/coteschung)
[![Wechat Pay](https://img.shields.io/badge/-Tip%20Me%20on%20WeChat-brightgreen?logo=wechat&logoColor=white)][cn-donation]
[![Alipay](https://img.shields.io/badge/-Tip%20Me%20on%20Alipay-blue?logo=alipay&logoColor=white)][cn-donation]

## License

This work is published under [MIT](https://github.com/cotes2020/jekyll-theme-chirpy/blob/master/LICENSE) License.

[starter]: https://github.com/cotes2020/chirpy-starter
[use-starter]: https://github.com/cotes2020/chirpy-starter/generate
[workflow]: https://github.com/cotes2020/jekyll-theme-chirpy/blob/master/.github/workflows/pages-deploy.yml.hook

<!-- ReadMe links -->

[jb]: https://www.jetbrains.com/?from=jekyll-theme-chirpy
[cn-donation]: https://cotes.gitee.io/alipay-wechat-donation/
