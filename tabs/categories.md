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

  {% if category_name == first_post.categories[0] %}
    {% assign sub_categories = "" | split: "" %}

    {% for post in posts_of_category %}
      {% assign second_category = post.categories[1] %}
      {% if second_category %}
        {% unless sub_categories contains second_category %}
          {% assign sub_categories = sub_categories | push: second_category %}
        {% endunless %}
      {% endif %}
    {% endfor %}

    {% assign sub_categories = sub_categories | sort %}
    {% assign sub_categories_size = sub_categories | size %}

  <div class="card categories">
    <!-- top-category -->
    <div class="card-header d-flex justify-content-between hide-border-bottom"
        id="{{ HEAD_PREFIX }}{{ group_index }}">
      <span>
      {% if sub_categories_size > 0 %}
        <i class="far fa-folder-open fa-fw"></i>
      {% else %}
        <i class="far fa-folder fa-fw"></i>
      {% endif %}
        <a href="{{ site.baseurl }}/categories/{{ category_name | replace: ' ', '-' | downcase | url_encode }}/"
          class="ml-1 mr-2">
          {{ category_name }}
        </a>

        <!-- content count -->
        {% assign top_posts_size = site.categories[category_name] | size %}
        <span class="text-muted small font-weight-light">
          {% if sub_categories_size > 0 %}
            {{ sub_categories_size }}
            {% if sub_categories_size > 1 %}categories{% else %}category{% endif %},
          {% endif %}
            {{ top_posts_size }}
            post{% if top_posts_size > 1 %}s{% endif %}
        </span>
      </span>

      <!-- arrow -->
      {% if sub_categories_size > 0%}
      <a href="#{{ LIST_PREFIX }}{{ group_index }}" data-toggle="collapse" aria-expanded="true"
          class="category-trigger hide-border-bottom">
        <i class="fas fa-fw fa-angle-down"></i>
      </a>
      {% else %}
      <span data-toggle="collapse" class="category-trigger hide-border-bottom disabled">
        <i class="fas fa-fw fa-angle-right"></i>
      </span>
      {% endif %}

    </div> <!-- .card-header -->

    <!-- Sub-categories -->
    {% if sub_categories_size > 0 %}
    <div id="{{ LIST_PREFIX }}{{ group_index }}" class="collapse show" aria-expanded="true">
      <ul class="list-group">
        {% for sub_category in sub_categories %}
        <li class="list-group-item">
          <i class="far fa-folder fa-fw"></i>
          <a href="{{ site.baseurl }}/categories/{{ sub_category | replace: ' ', '-' | downcase | url_encode }}/"
            class="ml-1 mr-2">{{ sub_category }}</a>
          {% assign posts_size = site.categories[sub_category] | size %}
          <span class="text-muted small font-weight-light">{{ posts_size }}
            post{% if posts_size > 1 %}s{% endif %}
          </span>
        </li>
        {% endfor %}
      </ul>
    </div>
    {% endif %}

  </div> <!-- .card -->

    {% assign group_index = group_index | plus: 1 %}

  {% endif %}
{% endfor %}
