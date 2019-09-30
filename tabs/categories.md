---
title: Categories

# All the Categories of posts
# © 2017-2019 Cotes Chung
# MIT License
---

{% assign sort_categories = site.categories | sort %}

{% for category in sort_categories %}
  {% assign category_name = category | first %}
  {% assign posts_of_category = category | last %}
  {% assign first_post = posts_of_category[0] %}

  {% if category_name == first_post.categories[0] %}
    {% assign sub_categories = "" %}
    {% for post in posts_of_category %}
      {% if post.categories.size > 1 %}
        {% assign sub_categories = sub_categories | append: post.categories[1] | append: "|" %}
      {% endif %}
    {% endfor %}

    {% assign sub_categories = sub_categories | split: "|" | uniq | sort %}
    {% assign sub_categories_size = sub_categories | size %}

  <div class="card categories">
    <!-- top-category -->
    <div class="card-header d-flex justify-content-between hide-border-bottom" id="h_{{ category_name }}">
      <span>
      {% if sub_categories_size > 0 %}
        <i class="far fa-folder-open fa-fw"></i>
      {% else %}
        <i class="far fa-folder fa-fw"></i>
      {% endif %}
        <a href="{{ site.baseurl }}/categories/{{ category_name | replace: ' ', '-' | downcase }}/">{{ category_name }}</a>
        <!-- content count -->
        {% assign top_posts_size = site.categories[category_name] | size %}
        <span class="text-muted small font-weight-light pl-2">
        {% if sub_categories_size > 0 %}
          {{ sub_categories_size }}
          {% if sub_categories_size > 1 %}categories{% else %}category{% endif %},
        {% endif %}
          {{ top_posts_size }}
          post{% if top_posts_size > 1 %}s{% endif %}
        </span>
      </span>

      <!-- arrow -->
      <a href="#l_{{ category_name }}" data-toggle="collapse" aria-expanded="true" class="category-trigger hide-border-bottom">
        {% if sub_categories_size > 0%}
        <i class="fas fa-angle-up"></i>
        {% else %}
        <i class="fas fa-angle-down disabled"></i>
        {% endif %}
      </a>

    </div> <!-- .card-header -->

    <!-- Sub-categories -->
    {% if sub_categories_size > 0 %}
    <div id="l_{{ category_name }}" class="collapse show" aria-expanded="true">
      <ul class="list-group">
        {% for sub_category in sub_categories %}
        <li class="list-group-item">
          <i class="far fa-folder fa-fw"></i>&nbsp;<a href="{{ site.baseurl }}/categories/{{ sub_category | replace: ' ', '-' | downcase }}/">{{ sub_category }}</a>
          {% assign posts_size = site.categories[sub_category] | size %}
          <span class="text-muted small font-weight-light pl-2">{{ posts_size }}
            post{% if posts_size > 1 %}s{% endif %}
          </span>
        </li>
        {% endfor %}
      </ul>
    </div>
    {% endif %}

  </div> <!-- .card -->

  {% endif %}
{% endfor %}

<script src="{{ site.baseurl }}/assets/js/dist/category-collapse.min.js" async></script>
