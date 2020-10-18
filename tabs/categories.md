---
title: Categories
type: categories
# All the Categories of posts
# v2.0
# https://github.com/cotes2020/jekyll-theme-chirpy
# © 2017-2019 Cotes Chung
# MIT License
---

{% assign HEAD_PREFIX = "h_" %}
{% assign LIST_PREFIX = "l_" %}

{% assign group_index = 0 %}

{% assign sort_categories = site.categories | sort %}

{% for category in sort_categories %}
  {% assign category_name = category | first %}
  {% assign posts_of_category = category | last %}
  {% assign first_post = posts_of_category | first %}

  <div class="card categories">
    <div class="card-header d-flex justify-content-between hide-border-bottom"
        id="{{ HEAD_PREFIX }}{{ group_index }}">
      <span>
        <a href="{{ site.baseurl }}/categories/{{ category_name | replace: ' ', '-' | downcase | url_encode }}/"
          class="ml-1 mr-2">
          {{ category_name }}
        </a>
        <!-- content count -->
        {% assign top_posts_size = site.categories[category_name] | size %}
        <span class="text-muted small font-weight-light">
            {{ top_posts_size }}
            post{% if top_posts_size > 1 %}s{% endif %}
        </span> 
      </span>
    </div>
  </div>
    {% assign group_index = group_index | plus: 1 %}

{% endfor %}
