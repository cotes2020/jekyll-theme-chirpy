---
title: Add flow chart and mathematical expression on Jekyll tech blog (Feat. Mermaid, MathJax)
author: Beanie
date: 2022-05-05 09:13:00 +0800
categories: [etc, 기술블로그]
tags: []
layout: post
current: post
class: post-template
subclass: 'post'
navigation: True
cover:  assets/img/post_images/jekyll_cover.jpeg
---

## Add flow chart using Mermaid
&nbsp;
### Mermaid?

Mermaid is a JavaScript library which draws flowchart diagrams from script.

```
graph LR
  A(landing page)-->B[check auto login]
  B-->C(login page)
  B-->D(main page)
```

In this way, it intuitively converts a written script into a diagram.

<div class="mermaid">
  graph LR;
  A(landing page)-->B[check auto login];
  B-->C(login page);
  B-->D(main page);
</div>

&nbsp;

### Rendering Mermaid in jekyll

This tech blog was made using the Jekyll Chipy theme.
According to guide document for the Jekyll Chirpy theme, when writing a post, we can use mermaid by adding `'''mermaid` if the following is inserted on posting head.
```yaml
---
mermaid: true
---
```

However, for some reason, it didn't work. So I looked for another way.

Searching for methods to render Mermaid on Jekyll gives the following two results.
* [jekyll-mermaid](https://github.com/jasonbellamy/jekyll-mermaid)
* [jekyll-spaceship](https://github.com/jeffreytse/jekyll-spaceship)

But, both of these methods didn't work well either.

So, I decided to just embed Mermaid directly in the html file.

&nbsp;

**Embedding MermaidPermalink**

Checking the Mermaid-js file, I could check the CDN of the corresponding js file.
Mermaid was successfuly rendered after entering this CDN in each html document. I inserted it in _includes/head.html for this blogpost, but it should be sufficient to just add it on common html elements.

```html
<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
<script>mermaid.initialize({startOnLoad:true});</script>
```

And then, mermaid can be called within .md files like the following.

```markdown
<div class="mermaid">
  graph LR;
  A(landing page)-->B[Check auto login];
  B-->C(login page);
  B-->D(main);
</div>
```

&nbsp;

## Adding MathJax mathematical expression
&nbsp;

### MathJax?
MathJax is a cross-browser JavaScript library that uses MathML, LaTeX, and ASCIIMathML markup to display mathematical notation in a web browser. MathJax is provided as open source software under the Apache License.
### Rendering MathJax in jekyll
&nbsp;
Similar to flow chart, it can be used by adding the following code to _includes/head.html.

```html
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
      inlineMath: [['$','$']]
    }
  });
</script>
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
```

Then, formulas written in latex grammar are rendered well!
