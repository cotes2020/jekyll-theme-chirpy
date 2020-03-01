# Jekyll Theme Chirpy

[![Build Status](https://github.com/cotes2020/jekyll-theme-chirpy/workflows/build/badge.svg?event=push)](https://github.com/cotes2020/jekyll-theme-chirpy/actions?query=event%3Apush)
[![GitHub license](https://img.shields.io/github/license/cotes2020/jekyll-theme-chirpy.svg)](https://github.com/cotes2020/jekyll-theme-chirpy/blob/master/LICENSE)
[![996.icu](https://img.shields.io/badge/link-996.icu-%23FF4D5B.svg)](https://996.icu)

Language: English | [简体中文](docs/README_zh-CN.md)

A minimal, portfolio, sidebar, bootstrap Jekyll theme with responsive web design and focuses on text exhibition. It will help you easily record, manage and share your knowledge and experience. 

You will get the following features:

* Auto Dark Mode
* Posts' Last Modified Date
* Table of Contents
* Automatically Recommend Related Posts
* Syntax highlighting
* Two Level Categories
* Search
* Atom Feeds
* Disqus Comments
* Google Analytics
* GA Pageviews (Advanced)
* SEO Tag
* Performance Optimization

[**Live Demo** »](https://chirpy.cotes.info)

![devices-mockup](https://raw.githubusercontent.com/cotes2020/jekyll-theme-chirpy/master/assets/img/sample/devices-mockup.png)

## Table of Contents

* [Installing](#installing)
* [Usage](#usage)
* [Contributing](#contributing)
* [Credits](#credits)
* [Support](#support)
* [License](#license)


## Installing

### Prerequisites

Follow the [Jekyll Docs](https://jekyllrb.com/docs/installation/) to complete the installtion of basic environment (`Ruby `, `RubyGems` and `Bundler`). In order to use the script tools to save time, we also need to install [Python](https://www.python.org/downloads/)(version 3.5 or abover) and [ruamel.yaml](https://pypi.org/project/ruamel.yaml/).

In addition, if your machine is running Debian or macOS, make sure the [GNU coreutils](https://www.gnu.org/software/coreutils/) is installed. Otherwise, get it by:

* Debian

```console
$ sudo apt-get install coreutils
```

* macOS

```console
$ brew install coreutils
```


### Jekyll Plugins

[Fork **Chirpy** from GitHub](https://github.com/cotes2020/jekyll-theme-chirpy/fork), then clone your forked repo to local:

```console
$ git clone git@github.com:USER/jekyll-theme-chirpy.git
```

replace the `USER` above to your GitHub username.

The first time you run or build the project on your machine, perform the installation of Jekyll plugins. Go to the root of repo and run:

```terminal
$ bundle install
```

`bundle` will automatically install all the dependent Jekyll Plugins that listed in the `Gemfile`.


## Usage


### Directory Structure

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
├── docs
├── feed.xml
├── index.html
├── robots.txt
└── sitemap.xml
```


As mentioned above, some files or directories should be removed from your repo:

- .travis.yml
- .github


### Customization

Basically, go to `_config.yml` and customize the variables as needed, some of them are typical options:

* `url`
	
	Set to your website domain and there should be no slash symbol at the tail.


* `avatar`
    
    It defines the image file location of avatar. The sample image is `/assets/img/sample/avatar.jpg`, and should be replaced by your own one(a square image). Notice that a huge image file will increase the load time of your site, so keep your avatar image size as samll as possible(may be *<https://tinypng.com/>* will help).

* `timezone`

    To ensure that the posts' release date matches the city you live in, please modify the field `timezone` correctly. A list of all available values can be found on [TimezoneConverter](http://www.timezoneconverter.com/cgi-bin/findzone/findzone) or [Wikipedia](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones).

* `theme_mode`
  
	There are three options for the theme color scheme:
	
	- **dual** 	- The default color scheme will follow the system settings, but if the system does not support dark mode, or the browser does not support `Media Queries Level 5`, the theme will be displayed as `light` mode by default. Anyway, the bottom left corner of the Sidebar will provide a button for users to switch color schemes.

	- **dark**	- Always show dark mode.
	- **light**	- Always show light mode.


###  Run Locally

You may want to preview the site before publishing, so just run the script tool:

```terminal
$ bash tools/run.sh
```

Open a modern brower and visit at <http://localhost:4000>.

Few days later, you may find that the file modification(e.g. edits to a post) does not refresh in real time by using `run.sh`. Don't worry, the advanced option `-r` (or `--realtime`) will solve this problem, but it requires [**fswatch**](http://emcrisostomo.github.io/fswatch/) to be installed on your machine. Type `-h` for more information.

###  Deploying to GitHub Pages

Before the deployment begins, ensure the `url` in file `_config.yml` has been set to `https://<username>.github.io`(or the custom domain, if you have. e.g. `https://yourdomain.com`). What's more, if you prefer to the [Project site](https://help.github.com/en/github/working-with-github-pages/about-github-pages#types-of-github-pages-sites), change `baseurl` of file `_config.yml` to your project name, starting with a slash. e.g. `/project`.

#### Option 1: Built by GitHub Pages

By deploying the site in this way, you're allowed to push the source code directly to the remote.

> **Note**: If you want to use any third-party Jekyll plugins that not in [this list](https://pages.github.com/versions/), stop reading the current approach and go to [*Option 2: Build locally*](#option-2-build-locally).

**1**. Rename the repository to:

|Site Type | Repo's Name|
|:---|:---|
|User or Organization | `<username>.github.io`|
|Project| any one except `<username>.github.io`, let's say `project`|

**2**. Commit the changes of the repo first, then run the initialization script:

```terminal
$ bash tools/init.sh
```

>**Note**: The *Recent Update* requires the posts' latest git-log date, so make sure the changes in `_posts` have been committed before running this command.

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

The generated static files will be placed in the root of `/path/to/local/project`. Commit and push the changes to the `master` branch on GitHub.

**3**. Go to GitHub website and enable Pages service for the new repository.

**4**. Visit at:

|Site Type | Site URL |
|:---|:---|
|User or Organization | `https://<username>.github.io/`|
|Project| `https://<username>.github.io/project/`|

and enjoy!

### Documentation

For more details and the better reading experience, please check out the [tutorial in demo site](https://chirpy.cotes.info/categories/tutorial/). In the meanwhile, a copy of the tutorial is also available on the [Wiki](https://github.com/cotes2020/jekyll-theme-chirpy/wiki).

## Contributing

The old saying: "Tow heads are better than one. Five heads are better than two." So, welcome to report bugs, improve code quality or submit a new feature. For more information, see [contributing guidelines](.github/CONTRIBUTING.md).


## Credits

This theme is mainly built with [Jekyll](https://jekyllrb.com/) ecosystem, [Bootstrap](https://getbootstrap.com/), [Font Awesome](https://fontawesome.com/) and some other wonderful tools(their copyright information can be found in the relevant files).

:tada:Thanks to all the volunteers who contributed to this project, their GitHub IDs are on [this list](https://github.com/cotes2020/jekyll-theme-chirpy/graphs/contributors). Also, I won't forget those guys who submitted the issues or unmerged PR because they reported bugs, shared ideas or inspired me to write more readable documentation.


## Support

If you enjoy this theme or find it helpful, please consider becoming my sponsor, I'd really appreciate it! Click the button <kbd>:heart:Sponsor</kbd> at the top of the [Home Page](https://github.com/cotes2020/jekyll-theme-chirpy) and choose a link that suits you to donate. This will encourage me and help me maintain this project.


## License

This work is published under [MIT](https://github.com/cotes2020/jekyll-theme-chirpy/blob/master/LICENSE) License.