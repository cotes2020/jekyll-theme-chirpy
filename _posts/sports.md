---
layout: category
title: Sports
permalink: /categories/sports/
---

{% assign posts = site.categories['Sports'] %}
{% for post in posts %}
  <h2><a href="{{ post.url }}">{{ post.title }}</a></h2>
{% endfor %}
