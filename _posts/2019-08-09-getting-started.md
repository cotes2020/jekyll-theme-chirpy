---
title: Getting Started
author: Cotes Chung
date: 2019-08-09 20:55:00 +0800
categories: [Blogging, Tutorial]
tags: [getting started]
---

## Prerequisites

Follow the [Jekyll Docs](https://jekyllrb.com/docs/installation/) to complete the installtion of basic environment (`Ruby `, `RubyGems` and `Bundler`). 

To improve the writing experience, we need to use some script tools. If your machine is running Debian or macOS, make sure that [GNU coreutils](https://www.gnu.org/software/coreutils/) is installed. Otherwise, install by:

* Debian

```console
$ sudo apt-get install coreutils
```

* macOS

```console
$ brew install coreutils
```


## Jekyll Plugins

[Fork **Chirpy** from GitHub](https://github.com/cotes2020/jekyll-theme-chirpy/fork), then clone your forked repo to local:

```console
$ git clone git@github.com:USER/jekyll-theme-chirpy.git -b master
```

and replace the `USER` above to your GitHub username.

The first time you run or build the project on local machine, perform the installation of Jekyll plugins. Go to the root of repo and run:

```terminal
$ bundle install
```

`bundle` will automatically install all the dependent Jekyll Plugins that listed in the `Gemfile`.


## Directory Structure

The main files and related brief introductions are listed below.

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
├── feed.xml
├── index.html
├── robots.txt
└── sitemap.xml
```


As mentioned above, some files or directories should be removed from your repo:

- .travis.yml
- .github


## Configuration

Generally, go to `_config.yml` and configure the variables as needed. Some of them are typical options:

* `url`
    
    Set to your website url and there should be no slash symbol at the tail. Format: `<protocol>://<domain>`.


* `avatar`
    
    It defines the image file location of avatar. The sample image is `/assets/img/sample/avatar.jpg`, and should be replaced by your own one(a square image). Notice that a huge image file will increase the load time of your site, so keep your avatar image size as samll as possible(may be *<https://tinypng.com/>* will help).

* `timezone`

    To ensure that the posts' release date matches the city you live in, please modify the field `timezone` correctly. A list of all available values can be found on [TimezoneConverter](http://www.timezoneconverter.com/cgi-bin/findzone/findzone) or [Wikipedia](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones).

* `theme_mode`
  
    There are three options for the theme color scheme:
    
    - **dual**  - The default color scheme will follow the system settings, but if the system does not support dark mode, or the browser does not support `Media Queries Level 5`, the theme will be displayed as `light` mode by default. Anyway, the bottom left corner of the Sidebar will provide a button for users to switch color schemes.

    - **dark**  - Always show dark mode.
    - **light** - Always show light mode.


##  Run Locally

You may want to preview the site content before publishing, so just run the script tool:

```terminal
$ bash tools/run.sh
```

Open a brower and visit <http://localhost:4000>.

Few days later, you may find that the file changes does not refresh in real time by using `run.sh`. Don't worry, the advanced option `-r` (or `--realtime`) will solve this problem, but it requires [**fswatch**](http://emcrisostomo.github.io/fswatch/) to be installed on your machine.

##  Deploying to GitHub Pages

Before the deployment begins, checkout the file `_config.yml` and make sure that the `url` has been configured. What's more, if you prefer the [Project site on GitHub](https://help.github.com/en/github/working-with-github-pages/about-github-pages#types-of-github-pages-sites) and also use the default domain `<username>.github.io`, remember to change the `baseurl` to your project name that starting with a slash. For example, `/project`.


### Option 1: Built by GitHub Pages

By deploying the site in this way, you're allowed to push the source code directly to the remote.

> **Note**: If you want to use any third-party Jekyll plugins that not in [this list](https://pages.github.com/versions/), stop reading the current approach and go to [*Option 2: Build locally*](#option-2-build-locally).

**1**. Rename the repository to:

|Site Type | Repo's Name|
|:---|:---|
|User or Organization | `<username>.github.io`|
|Project| Any one except `<username>.github.io`, let's say `project`|

**2**. Commit the changes of the repo first, then run the initialization script:

```console
$ bash tools/init.sh
```

> Please note that the *Recent Update* list requires the latest git-log date of posts, thus make sure the changes in `_posts` have been committed before running this command.

it will automatically generates the *Latest Modified Date* and *Categories / Tags* page for the posts and submit a commit. Its output is similar to the following log:

```terminal
[INFO] Success to update lastmod for 4 post(s).
[INFO] Succeed! 3 category-pages created.
[INFO] Succeed! 4 tag-pages created.
[Automation] Updated the Categories, Tags, Lastmod for post(s).
 11 files changed, 46 insertions(+), 3 deletions(-)
 ...
Updated the Categories, Tags, Lastmod for post(s).
```


**3**. Push the changes to `origin/master` then go to GitHub website and enable GitHub Pages service for the repo.

**4**. Check it out:

|Site Type | Site URL |
|:---|:---|
|User or Organization | `https://<username>.github.io/`|
|Project| `https://<username>.github.io/project/`|


### Option 2: Build Locally

For security reasons, GitHub Pages runs on `safe` mode, which means the third-party Jekyll plugins or custom scripts won't work. If you want to use any another plugins that not in the [whitelist](https://pages.github.com/versions/), **you have to generate the site locally rather than on GitHub Pages**.

**1**. Browse to GitHub website, create a brand new repo named: 

|Site Type | Repo's Name|
|:---|:---|
|User or Organization | `<username>.github.io`|
|Project| Any one except `<username>.github.io`, let's say `project`|

and clone it.

**2**. In the root of the source project, build your site by:

```console
$ bash tools/build.sh -d /path/to/local/project/
```

The generated static files will be placed in the root of `/path/to/local/project`. Commit and push the changes to the `master` branch on GitHub.

**3**. Go to GitHub website and enable Pages service for the new repository.

**4**. Visit at:

|Site Type | Site URL |
|:---|:---|
|User or Organization | `https://<username>.github.io/`|
|Project| `https://<username>.github.io/project/`|

### Finishing work

No matter which way you choose to deploy the website on GitHub, please enforce the `HTTPS` for it. See official docs: [Configuring a publishing source for your GitHub Pages site](https://help.github.com/en/github/working-with-github-pages/securing-your-github-pages-site-with-https).
