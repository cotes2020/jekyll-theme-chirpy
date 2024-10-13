---
layout: category
title: Data
---

{% assign posts = site.categories['Data'] %}
{% for post in posts %}
  <h2><a href="{{ post.url }}">{{ post.title }}</a></h2>
{% endfor %}
