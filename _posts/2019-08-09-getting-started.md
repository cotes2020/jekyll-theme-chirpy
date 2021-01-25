---
title: Getting Started
author: Cotes Chung
date: 2019-08-09 20:55:00 +0800
categories: [Blogging, Tutorial]
tags: [getting started]
pin: true
---

## Prerequisites

Follow the [Jekyll Docs](https://jekyllrb.com/docs/installation/) to complete the installation of `Ruby`, `RubyGems`, `Jekyll` and `Bundler`.

## Installation

There are two ways to get the theme:

  - Install from [RubyGems](https://rubygems.org/gems/jekyll-theme-chirpy)
  - Fork from GitHub

### Install From Rubygems

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

Finally, copy the missing files (refer to the [starter project][starter] for the detailed file directory structure) from the theme's gem to your Jekyll site, and append all the variables of the theme's `_config.yml` to your Jekyll site.

> **Hint**: To locate the theme’s gem, execute:
>
```console
$ bundle info --path jekyll-theme-chirpy
```

Or you can [use the starter template][use-starter] to create a Jekyll site to save time copying contents from theme's gem.

[starter]: https://github.com/cotes2020/chirpy-starter
[use-starter]: https://github.com/cotes2020/chirpy-starter/generate

### Fork From GitHub

[Fork **Chirpy**](https://github.com/cotes2020/jekyll-theme-chirpy/fork) from GitHub and clone your fork to local.

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

  2. If you use the `--no-gh` option, the directory `.github` will be deleted. Otherwise, setup the GitHub Action workflow by removing extension `.hook` of `.github/workflows/pages-deploy.yml.hook`, and then remove the other files and directories in folder `.github`.

  3. Automatically create a commit to save the changes.


## Usage

### Configuration

Update the variables of `_config.yml` as needed. Some of them are typical options:

  - `url`
  - `avatar`
  - `timezone`
  - `theme_mode`

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

Before the deployment begins, checkout the file `_config.yml` and make sure the `url` is configured correctly. Furthermore, if you prefer the [**project site**](https://help.github.com/en/github/working-with-github-pages/about-github-pages#types-of-github-pages-sites) and don't use a custom domain, or you want to visit your website with a base url on a web server other than **GitHub Pages**, remember to change the `baseurl` to your project name that starting with a slash, e.g, `/project-name`.

Now you can now choose ONE of the following methods to deploy your website.

#### Deploy on GitHub Pages

For security reasons, GitHub Pages build runs on `safe` mode, which restricts us from using plugins to generate additional page files. Therefore, we can use **GitHub Actions** to build the site, store the built site files on a new branch, and use that branch as the source of the GH Pages service.

Ensure your Jekyll site has the file `/.github/workflows/pages-deploy.yml`.
Otherwise, create a new one and fill in the contents of the [workflow file][workflow], and the value of the `on.push.branches` should be the same as your repo's default branch name.

[workflow]:https://github.com/cotes2020/jekyll-theme-chirpy/blob/master/.github/workflows/pages-deploy.yml.hook

Rename your repoistory to `<GH-USERNAME>.github.io` on GitHub.

And then publish your site by:

  1. Push any commit to remote to trigger the GitHub Actions workflow. Once the build is complete and successful, a new remote branch named `gh-pages` will appear to store the built site files.

  2. Browse to your repo's landing page on GitHub and select the branch `gh-pages` as the [publishing source](https://docs.github.com/en/github/working-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site) throught _Settings_ → _Options_ → _GitHub Pages_:

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
