---
title: Getting Started
date: 2019-08-09 20:55:00 +0800
categories: [Blogging, Tutorial]
tags: [getting started]
---

## Preparation

Follow the [Jekyll Docs](https://jekyllrb.com/docs/installation/) to complete the installtion of basic environment (Ruby, RubyGem, Bundler and Jekyll). In order to use the script tools to save time, we also need to install [Python](https://www.python.org/downloads/)(version 3.5 or abover) and [ruamel.yaml](https://pypi.org/project/ruamel.yaml/).

In addition, if your machine is running Debian or macOS, make sure you have the [GNU coreutils](https://www.gnu.org/software/coreutils/) installed. Otherwise, get it by:

* Debian

```console
$ sudo apt-get install coreutils
```

* macOS

```console
$ brew install coreutils
```


## Install Jekyll Plugins

Go to the root of repo and run:

```terminal
$ bundle install
```

`bundle` will install all the dependent Jekyll Plugins listed in file `Gemfile` automatically.


## File Structure

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


## Configuration

Customize the variables in file `_config.yml` as needed.

* Avatar
    
    The sample avatar is `/assets/img/sample/avatar.jpg`. It should be replaced by your own one. Notice that a huge image file will increase the load time of your site, so keep your avatar size as samll as possible(may be *<https://tinypng.com/>* will help).

* TimeZone

    To ensure that the posts' release date matches the city you live in, please modify the field `timezone` correctly. A list of all available values can be found on [TimezoneConverter](http://www.timezoneconverter.com/cgi-bin/findzone/findzone) or [Wikipedia](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones).


##  Run Locally

You may want to preview the site before publishing, so just run the script tool:

```terminal
$ bash tools/run.sh
```

Open a brower and visit <http://localhost:4000>.

Few days later, you may find that the file changes does not refresh in real time by using `run.sh`. Don't worry, the advanced option `-r` (or `--realtime`) will solve this problem, but it requires [**fswatch**](http://emcrisostomo.github.io/fswatch/) to be installed on your machine.

##  Deploying to GitHub Pages

Before the deployment begins, ensure the `url` in file `_config.yml` has been set to `https://<username>.github.io`(or the custom domain, if you have. e.g. `https://yourdomain.com`).


### Option 1: Built by GitHub Pages

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

> Please note that the *Recent Update* list requires the latest git-log date of posts, thus make sure the changes in `_posts` have been committed before running this command.

it will automatically generates the *Latest Modified Date* and *Categories / Tags* page for the posts.

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
|Project| any one except `<username>.github.io`, let's say `project`|

and clone it.

**2**. In the root of the source project, build your site by:

```console
$ bash tools/build.sh -d /path/to/local/project/
```

> If you prefer to the Project site, change `baseurl` of file `_config.yml` to your project name, starting with a slash. e.g. `/project`. Or, simply add argument `-b /project` behide the command above.

The generated static files will be placed in the root of `/path/to/local/project`. Commit and push the changes to the `master` branch on GitHub.

**3**. Go to GitHub website and enable Pages service for the new repository.

**4**. Visit at:

|Site Type | Site URL |
|:---|:---|
|User or Organization | `https://<username>.github.io/`|
|Project| `https://<username>.github.io/project/`|

and enjoy!
