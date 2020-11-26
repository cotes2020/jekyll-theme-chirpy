---
title: Getting Started
author: Cotes Chung
date: 2019-08-09 20:55:00 +0800
categories: [Blogging, Tutorial]
tags: [getting started]
pin: true
---


## Installation

[Fork **Chirpy**](https://github.com/cotes2020/jekyll-theme-chirpy/fork) on GitHub, rename the repository to `USERNAME.github.io` (where `USERNAME` is your GitHub username), and then open terminal and clone the fork to local by:

```terminal
$ git clone https://github.com/USERNAME/USERNAME.github.io.git -b master --single-branch
```

### Setting up the local envrionment

If you would like to run or build the project on your local machine, please follow the [Jekyll Docs](https://jekyllrb.com/docs/installation/) to complete the installation of `Ruby`, `RubyGems`, `Jekyll` and `Bundler`. 

Before running or building for the first time, please complete the installation of the Jekyll plugins. Go to the root directory of project and run:

```terminal
$ bundle install
```

`bundle` will automatically install all the dependencies specified by `Gemfile`.

### Setting up Docker environment (optional)

If you're a loyal fan of [**Docker**](https://www.docker.com/) or just too lazy to install the packages mentioned in [_Setting up the local envrionment_](#setting-up-the-local-envrionment), please make sure you have **Docker Engine** installed and running, and then get Docker image `jekyll/jekyll` from Docker Hub by the following command:

```console
$ docker pull jekyll/jekyll
```

## Usage

### Initialization

Go to the root directory of the project and start initialization:

```console
$ bash tools/init.sh
```

> **Note**: If you not intend to deploy it on GitHub Pages, append parameter option `--no-gh` at the end of the above command.

What it does is:

1. Remove some files or directories from your repository:

    - `.travis.yml`
    - files under `_posts`
    - folder `docs`

2. If you use the `--no-gh` option, the directory `.github` will be deleted. Otherwise, setup the GitHub Action workflow by removing extension `.hook` of `.github/workflows/pages-deploy.yml.hook`, and then remove the other files and directories in folder `.github`. 

3. Automatically create a commit to save the changes.

### Configuration

Generally, go to `_config.yml` and configure the variables as needed. Some of them are typical options:

- `url`
- `avatar`
- `timezone`
- `theme_mode`

### Run Locally

You may want to preview the site contents before publishing, so just run it by:

```terminal
$ bundle exec jekyll s
```

Then open a browser and visit to <http://localhost:4000>.

### Run on Docker

Run the site on Docker with the following command:

```terminal
$ docker run --rm -it \
    --volume="$PWD:/srv/jekyll" \
    -p 4000:4000 jekyll/jekyll \
    jekyll serve
```


### Deployment

Before the deployment begins, checkout the file `_config.yml` and make sure the `url` is configured correctly. Furthermore, if you prefer the [_project site_](https://help.github.com/en/github/working-with-github-pages/about-github-pages#types-of-github-pages-sites) and don't use a custom domain, or you want to visit your website with a base url on a web server other than **GitHub Pages**, remember to change the `baseurl` to your project name that starting with a slash. For example, `/project`.

Assuming you have already gone through the [initialization](#initialization), you can now choose ONE of the following methods to deploy your website.

#### Deploy on GitHub Pages

For security reasons, GitHub Pages build runs on `safe` mode, which restricts us from using plugins to generate additional page files. Therefore, we can use **GitHub Actions** to build the site, store the built site files on a new branch, and use that branch as the source of the Pages service.

1. Push any commit to `origin/master` to trigger the GitHub Actions workflow. Once the build is complete and successful, a new remote branch named `gh-pages` will appear to store the built site files.

2. Browse to your repo's landing page on GitHub and select the branch `gh-pages` as the [publishing source](https://docs.github.com/en/github/working-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site) throught _Settings_ → _Options_ → _GitHub Pages_:
    ![gh-pages-sources](/assets/img/sample/gh-pages-sources.png){: width="650" class="normal"}

3. Visit your website at the address indicated by GitHub.

#### Deploy on Other Platforms

On platforms other than GitHub, we cannot enjoy the convenience of **GitHub Actions**. Therefore, we should build the site locally (or on some other 3rd-party CI platform) and then put the site files on the server.

Go to the root of the source project, build your site by:

```console
$ JEKYLL_ENV=production bundle exec jekyll b
```

Or, build the site with Docker by:

```terminal
$ docker run -it --rm \
    --env JEKYLL_ENV=production \
    --volume="$PWD:/srv/jekyll" \
    jekyll/jekyll \
    jekyll build
```

Unless you specified the output path, the generated site files will be placed in folder `_site` of the project's root directory. Now you should upload those files to your web server.
