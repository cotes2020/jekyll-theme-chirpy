---
layout: post
title: "Jekyll a simple static site generator"
date: 2023-07-14 14:26:00
categories: static site
tags: Jekyll
image:
  path: /assets/img/headers/jekyll.png
---
Jekyll is a static site generator. It takes text written in your favorite markup language and uses layouts to create a static website. You can tweak the site’s look and feel, URLs, the data displayed on the page, and more.....!

### Prerequisites:
- Ruby version 2.5.0 or higher
- RubyGems
- GCC and Make
- Node.js & Git

### Install dependencies:

##### Install Ruby and other prerequisites:

```sh
sudo apt-get install ruby-full build-essential zlib1g-dev
```

Add environment variables to your `~/.bashrc (or) ~/.zshrc` file to configure the gem installation path:
```sh
echo '# Install Ruby Gems to ~/gems' >> ~/.zshrc 
echo 'export GEM_HOME="$HOME/gems"' >> ~/.zshrc
echo 'export PATH="$HOME/gems/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

Install Jekyll and Bundler:
```sh
gem install jekyll bundler
```

Install Node.js
```sh
curl -fsSL https://deb.nodesource.com/setup_20.x | bash - &&\
sudo apt-get install -y nodejs
```

### Create a new site using chirpy Starter:

Sign in to GitHub to fork [Chirpy](https://github.com/cotes2020/jekyll-theme-chirpy/fork), and then rename it to `USERNAME.github.io`, where `USERNAME` represents your GitHub username.

Clone your site to local machine. In order to build JavaScript files later, we need to install Node.js, and then run the tool:

```sh
git clone https://github.com/MBN02/mbn02.github.io.git

cd repo-name

bash tools/init
```
The above command will:

1. Check out the code to the latest tag (to ensure the stability of your site: as the code for the default branch is under development).
2. Remove non-essential sample files and take care of GitHub-related files.
3. Build JavaScript files and export to assets/js/dist/, then make them tracked by Git.
4. Automatically create a new commit to save the changes above.

Before running local server for the first time, go to the root directory of your site and run:
```sh 
cd repo-name

bundle
```

### Configuration:
Update the variables of _config.yml as needed.
```sh
timezone: Asia/Kolkata
title: # the main title
tagline: # it will display as the sub-title
description: >- # used by seo meta and the atom feed
url: "https://username.github.io"
github:
  username: # change to your github username
twitter:
  username: # change to your twitter username
social:
avatar: # the avatar on sidebar, support local or CORS resources
```

You may want to preview the site contents before publishing, so just run it by:
```sh
bundle exec jekyll s
```
You can access your local server using: `http://127.0.0.1:4000/`

### Writing a New Post:
Create a new file named `YYYY-MM-DD-TITLE.EXTENSION` and put it in the `_posts` of the root directory. Please note that the EXTENSION must be one of md and markdown.

```sh
2023-07-15-cert-manager-installation.md
2023-07-15-Jekyll-static-site.md
```

Fill the Front Matter as below at the top of the post:
```sh
---
title: "Jekyll a simple static site generator"
date: 2023-07-14 14:26:00
categories: static site
tags: Jekyll
image:
  path: /assets/img/headers/jekyll.png
---
```
Once youre happy with the changes,you can commit them to gitlab and this will trigger a github action.
```sh
git add .
git commit -m "Initial commit"
git push
```

### Configuring custom domain for Github pages

In your domain register create A records, point your apex domain to the IP addresses for GitHub Pages.

```sh
185.199.108.153
185.199.109.153
185.199.110.153
185.199.111.153

Example:
Type   Name         Data         TTL
 A	     @	  185.199.108.153	600 seconds
```

Create a CNAME record pointing to USERNAME.github.io
```
Type     Name        Data             TTL
CNAME	 www	   mbn02.github.io.	   1 Hour
```

*In github --> Project --> Settings --> Pages* 

Add your domain name under `Custom Domain`
```sh
www.mkbn.in
```


### Reference Links:
- [Jekyll docs](https://jekyllrb.com/)

- [chirpy-starter](https://github.com/cotes2020/chirpy-starter)

- [Customize the Favicon](https://chirpy.cotes.page/posts/customize-the-favicon/)