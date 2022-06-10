# Tempus Spatium

[![Automatic build](https://github.com/Booodaness/tempus-spatium/actions/workflows/pages-deploy.yml/badge.svg)](https://github.com/Booodaness/tempus-spatium/actions/workflows/pages-deploy.yml)
[![Website](https://img.shields.io/website?down_color=critical&down_message=down&up_color=success&up_message=up&url=https%3A%2F%2Fbooodaness.github.io%2Ftempus-spatium%2F)](https://booodaness.github.io/tempus-spatium/)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/m/Booodaness/tempus-spatium)](https://github.com/Booodaness/tempus-spatium/commits/main)
[![License](https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-important)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

![Logo](assets/img/site/avatar.png)

## About

A blog for physics, mathematics and philosophy, maintained by Siddhartha Bhattacharjee. Detailed description in the blog's [about](https://booodaness.github.io/tempus-spatium/about/) page.

## Viewing the site

### On the internet

This website is hosted at https://booodaness.github.io/tempus-spatium/.

### Locally

To run this site as a local server, follow the steps below:

1. Make sure you have [Git](https://git-scm.com/) installed. Open Git bash and clone this repository into your system:

```
cd <parent_directory>

git clone https://github.com/Booodaness/tempus-spatium
```

2. Download the latest stable release of [Ruby](https://www.ruby-lang.org/en/downloads/).

3. Install [Bundler](https://bundler.io/).

4. Navigate to the clone of this repository made in step 1:

```
cd <parent_directory>/tempus-spatium
```

5. Open Git bash and run `bundle install`, followed by:

```
$ bundle exec jekyll serve
```

You will get an output resembling:

```
Configuration file: <parent_directory>/tempus-spatium/_config.yml
 Theme Config file: <parent_directory>/tempus-spatium/_config.yml
            Source: <parent_directory>/tempus-spatium
       Destination: <parent_directory>/tempus-spatium/_site
 Incremental build: disabled. Enable with --incremental
      Generating...
                    done in <seconds> seconds.
 Auto-regeneration: enabled for '<parent_directory>/tempus-spatium'
    Server address: http://127.0.0.1:4000/tempus-spatium/
  Server running... press ctrl-c to stop.
```

6. Open your browser and go to `http://127.0.0.1:4000/tempus-spatium/`.

## Development

### Facilities used

This project is powered by various facilities. Given below are links to their documentation:

1. [Jekyll](https://jekyllrb.com/docs/)

2. [jekyll-theme-chirpy](https://github.com/cotes2020/jekyll-theme-chirpy)

3. [GitHub Pages](https://docs.github.com/en/pages)

### Contributing

#### Posts

1. Visit the concerned post on the [blog](https://booodaness.github.io/tempus-spatium/).

2. Click the 'Suggest edits' link in the post's metadata.

3. Follow the steps below.

#### General

1. [Fork this repository](https://github.com/Booodaness/tempus-spatium/fork).

2. Make changes and push them to the fork's remote.

3. Start a pull request.

4. Optionally, document the changes in the request's review page.

## Using the work

### Adaptation

Except where otherwise noted, the blog posts on this site are [licensed](LICENCE.md) under [Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).

Hence, any individual or group of individuals can reuse and modify the material as long as:

1. The purpose is non-commercial.

2. Credit is given to the author by mentioning their name (Siddhartha Bhattacharjee) and linking to the original content.

3. The new material must be distributed under the same license.

### Citation

#### Posts

Blog posts can be cited as BibTeX,

```
@article{bhattacharjee:2022,
  title={<post_title>},
  author={Siddhartha Bhattacharjee},
  journal={Tempus Spatium},
  url={https://booodaness.github.io/tempus-spatium/<post_name>},
  year={2022},
  publisher={GitHub Pages}}
```

Where, `post_title` is the title of the post and `post_name` is the part of its URL succeeding `https://booodaness.github.io`.

#### Repository

To cite this repository as a dataset, use the BibTeX,

```
@misc{Bhattacharjee_Tempus_Spatium,
author = {Bhattacharjee, Siddhartha},
title = {{Tempus Spatium}},
url = {https://booodaness.github.io/tempus-spatium/}
}
```

See [CITATION.cff](CITATION.cff) for more details.
