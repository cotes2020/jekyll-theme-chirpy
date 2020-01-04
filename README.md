# Jekyll Theme Chirpy

[![Build Status](https://travis-ci.com/cotes2020/jekyll-theme-chirpy.svg?branch=master)](https://travis-ci.com/cotes2020/jekyll-theme-chirpy)
[![GitHub license](https://img.shields.io/github/license/cotes2020/jekyll-theme-chirpy.svg)](https://github.com/cotes2020/jekyll-theme-chirpy/blob/master/LICENSE)
[![996.icu](https://img.shields.io/badge/link-996.icu-%23FF4D5B.svg)](https://996.icu)

English | [中文](README_zh-CN.md)

![devices-mockup](https://raw.githubusercontent.com/cotes2020/jekyll-theme-chirpy/master/assets/img/sample/devices-mockup.png)

A minimal, portfolio, sidebar, bootstrap Jekyll theme with responsive web design and focuses on text exhibition. It will help you easily record, manage and share your knowledge and experience. Hope you like it! [**Live Demo** »](https://chirpy.cotes.info)


## Features

* Auto Dark Mode
* Posts' Last Modified Date
* Table of Contents
* Recommand Related Post Automatically
* Disqus Comments
* Syntax highlighting
* Two Level Categories
* Search
* HTML Compress
* Atom Feeds
* Google Analytics
* GA Pageviews (Advanced)
* SEO Tag
* Performance Optimization

## Quick Start

### Preparation

Follow the [Jekyll Docs](https://jekyllrb.com/docs/installation/) to complete the installtion of basic environment (Ruby, RubyGem, Bundler and Jekyll). In addition, to use the funny script tools, we also need to install [Python](https://www.python.org/downloads/)(version 3.5 or abover) and [ruamel.yaml](https://pypi.org/project/ruamel.yaml/).

Next, [fork](https://github.com/cotes2020/jekyll-theme-chirpy/fork) **Chirpy** and then clone your forked repo locally.


### Install Jekyll Plugins

Go to the root of repo and run:

```terminal
$ bundle install
```

`bundle` will install all the dependent Jekyll Plugins listed in file `Gemfile` automatically.


### File Structure

The main files and related brief introductions are listed below.

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


As mentioned above, some files or directories should be removed from your repo:

- .travis.yml
- .github
- _scripts/travis


### Configuration

Customize the variables in file `_config.yml` as needed.

* Avatar
    
    The sample avatar is `/assets/img/sample/avatar.jpg`. It should be replaced by your own one. Notice that a huge image file will increase the load time of your site, so keep your avatar size as samll as possible(may be *<https://tinypng.com/>* will help).

* TimeZone

    To ensure that the posts' release date matches the city you live in, please modify the field `timezone` correctly. A list of all available values can be found on [TimezoneConverter](http://www.timezoneconverter.com/cgi-bin/findzone/findzone) or [Wikipedia](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones).

* Atom Feed

    The Atom feed url of your site will be:

    ```
    <SITE_URL>/feed.xml
    ```

    The `SITE_URL` was defined by variable `url` of `_config.yml`.


###  Run Locally

You may want to preview the site before publishing, so just run the script tool:

```terminal
$ bash tools/run.sh
```

>**Note**: The *Recent Update* list requires the latest git-log date of posts, thus make sure the changes in `_posts` have been committed before running this command.

Open a brower and visit <http://localhost:4000>.

Few days later, you may find that the file changes does not refresh in real time by using `run.sh`. Don't worry, the advanced option `-r` (or `--realtime`) will solve this problem, but it requires [**fswatch**](http://emcrisostomo.github.io/fswatch/) to be installed on your machine.

###  Deploying to GitHub Pages

Before the deployment begins, ensure the `url` in file `_config.yml` has been set to `https://<username>.github.io`(or the custom domain, if you have. e.g. `https://yourdomain.com`).

#### Option 1: Built by GitHub Pages

By deploying the site in this way, you're allowed to push the source code directly to the remote.

> **Note**: If you want to use any third-party Jekyll plugins that not in [this list](https://pages.github.com/versions/), stop reading the current approach and go to [*Option 2: Build locally*](#option-2-build-locally).

**1**. Rename the repository to:

|Site Type | Repo's Name|
|:---|:---|
|User or Organization | `<username>.github.io`|
|Project| any one except `<username>.github.io`, let's say `project`|

**2**. Commit the changes of the repo first, then run the initialization script:

```console
$ bash tools/init.sh
```

it will automatically generates the *Latest Modified Date* and *Categories / Tags* page for the posts.

**3**. Push the changes to `origin/master` then go to GitHub website and enable GitHub Pages service for the repo.

**4**. Check it out:

|Site Type | Site URL |
|:---|:---|
|User or Organization | `https://<username>.github.io/`|
|Project| `https://<username>.github.io/project/`|


#### Option 2: Build Locally

For security reasons, GitHub Pages runs on `safe` mode, which means the third-party Jekyll plugins or custom scripts won't work. If you want to use any another plugins that not in the [whitelist](https://pages.github.com/versions/), **you have to generate the site locally rather than on GitHub Pages**.

**1**. Browse to GitHub website, create a brand new repo named: 

|Site Type | Repo's Name|
|:---|:---|
|User or Organization | `<username>.github.io`|
|Project| any one except `<username>.github.io`, let's say `project`|

and clone it.

**2**. In the root of the source project, build your site by:

```console
$ bash tools/build.sh -d /path/to/local/project/
```

If you prefer to the Project site, change `baseurl` of file `_config.yml` to your project name, starting with a slash. e.g. `/project`. Or, simply add argument `-b /project` behide the command above.

The generated static files will be placed in the root of `/path/to/local/project`. Commit and push the changes to the `master` branch on GitHub.

**3**. Go to GitHub website and enable Pages service for the new repository.

**4**. Visit at:

|Site Type | Site URL |
|:---|:---|
|User or Organization | `https://<username>.github.io/`|
|Project| `https://<username>.github.io/project/`|

and enjoy!


## Documentation

For more information, please check out the [tutorial](https://chirpy.cotes.info/categories/tutorial/). In the meanwhile, a copy of the tutorial is also available on the [Wiki](https://github.com/cotes2020/jekyll-theme-chirpy/wiki).

## Sponsor

Want to buy me a coffee? Click the button <kbd>❤️Sponsor</kbd> at the top of the [Home Page](https://github.com/cotes2020/jekyll-theme-chirpy) and choose a link that suits you to donate. I'd really appreciate it and take it as encouragement to work on better projects.



## License

This work is published under [MIT](https://github.com/cotes2020/jekyll-theme-chirpy/blob/master/LICENSE) License.
