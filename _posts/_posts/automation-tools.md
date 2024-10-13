---
layout: category
title: Automation Tools
---

{% assign posts = site.categories['Automation Tools'] %}
{% for post in posts %}
  <h2><a href="{{ post.url }}">{{ post.title }}</a></h2>
{% endfor %}
