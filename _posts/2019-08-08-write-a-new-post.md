---
title: "Writing a New Post"
date: 2019-08-08 14:10:00 +0800
categories: [Blogging, Tutorial]
tags: [writing]
---

## Naming and Path

Create a new file named with the format `YYYY-MM-DD-title.md` then put it into `_post` of the root directory.

## Front Matter

Basically, you need to fill the [Front Matter](https://jekyllrb.com/docs/front-matter/) as below at the top of the post:

```yaml
---
title: TITLE
date: YYYY-MM-DD HH:MM:SS +/-TTTT
categories: [TOP_CATEGORIE, SUB_CATEGORIE]
tags: [TAG]     # TAG names should always be lowercase
---
```

> **Note**: The posts' ***layout*** has been set to `post` by default, so there is no need to add the variable ***layout*** in Front Matter block.

- **Timezone of date**

    In order to accurately record the release date of a post, you should not only setup the `timezone` of `_config.yml` but also provide the the post's timezone in field `date` of its Front Matter block. Format: `+/-TTTT`, e.g. `+0800`.

- **Categories and Tags**

    The `categories` of each post is designed to contain up to two elements, and the number of elements in `tags` can be zero to infinity.

    The list of posts belonging to the same category/tag is recorded on a separate page. The number of such *category*/*tag* type pages is equal to the number of `categories`/`tags` for all posts, they must match perfectly. 

    let's say there is a post with front matter:
```yaml
categories: [Animal, Insect]
tags: bee
```

    then we should have two *category* type pages placed in folder `categories` of root and one *tag* type page placed in folder `tags`  of root:
```terminal
jekyll-theme-chirpy
├── categories
│   ├── animal.html
│   └── tutorial.html
├── tags
│   └── bee.html
```
    
    and the content of a *category* type page is
```yaml
---
layout: category
title: CATEGORY_NAME        # e.g. Insect
category: CATEGORY_NAME     # e.g. Insect
---
```

    the content of a *tag* type page is
```yaml
---
layout: tag
title: TAG_NAME             # e.g. bee
category: TAG_NAME          # e.g. bee
---
```

    With the increasing number of posts, the number of categories and tags will increase several times!  If we still manually create these *category*/*tag* type files, it will obviously be a super time-consuming job, and it is very likely to miss some of them(i.e. when you click on the missing `category` or `tag` link from a post or somewhere, it will complain to you '404'). The good news is that we got a lovely script tool to finish the pages creation stuff: `tools/init.sh`. See its usage [here]({{ "/posts/getting-started/#option-1-built-by-github-pages" | relative_url }}).

## Table of Contents

By default, the **T**able **o**f **C**ontents (TOC) is displayed on the right panel of the post. If you want to turn it off globally, go to `_config.yml` and set the variable `toc` to `false`. If you want to turn off TOC for specific post, add the following to post's [Front Matter](https://jekyllrb.com/docs/front-matter/):

```yaml
---
toc: false
---
```


## Comments

Similar to TOC, the [Disqus](https://disqus.com/) comments is loaded by default in each post, and the global switch is defined by variable `comments` in file `_config.yml` . If you want to close the comment for specific post, add the following to the **Front Matter** of the post:

```yaml
---
comments: false
---
```


## Code Block

Markdown symbols <code class="highlighter-rouge">```</code> can easily create a code block as following examples.

```
This is a common code snippet, without syntax highlight and line number.
```

## Specific Language

Using <code class="highlighter-rouge">```Language</code> you will get code snippets with line Numbers and syntax highlight.

> **Note**: The Jekyll style `{% raw %}{%{% endraw %} highlight LANGUAGE {% raw %}%}{% endraw %}` or `{% raw %}{%{% endraw %} highlight LANGUAGE linenos {% raw %}%}{% endraw %}` are not allowed to be used in this theme !

```yaml
# Yaml code snippet
items:
    - part_no:   A4786
      descrip:   Water Bucket (Filled)
      price:     1.47
      quantity:  4
```

#### Liquid codes

If you want to display the **Liquid** snippet, surround the liquid code with `{% raw %}{%{% endraw %} raw {%raw%}%}{%endraw%}` and `{% raw %}{%{% endraw %} endraw {%raw%}%}{%endraw%}` .

{% raw %}
```liquid
{% if product.title contains 'Pack' %}
  This product's title contains the word Pack.
{% endif %}
```
{% endraw %}

## Learn More
For more knowledge about Jekyll posts, visit the [Jekyll Docs: Posts](https://jekyllrb.com/docs/posts/).

