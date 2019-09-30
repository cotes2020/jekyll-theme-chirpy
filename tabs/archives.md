---
title: Archives

# The Archives of posts.
# © 2017-2019 Cotes Chung
# MIT License
---

<div id="archives" class="pl-xl-2">
{% for post in site.posts %}
  {% capture this_year %}{{ post.date | date: "%Y" }}{% endcapture %}
  {% capture pre_year %}{{ post.previous.date | date: "%Y" }}{% endcapture %}
  {% if forloop.first %}
    {% assign last_day = "" %}
    {% assign last_month = "" %}
  <span class="lead">{{this_year}}</span>
  <ul class="list-unstyled">
  {% endif %}
    <li>
      <div>
        {% capture this_day %}{{ post.date | date: "%d" }}{% endcapture %}
        {% capture this_month %}{{ post.date | date: "%b" }}{% endcapture %}

        {% comment %}
        {% if last_day and last_day == this_day %}
          {% assign same_day = true %}
        {% else %}
          {% assign last_day = this_day %}
          {% assign same_day = false %}
        {% endif %}

        {% if last_month and last_month == this_month %}
          {% assign same_month = true %}
        {% else %}
          {% assign last_month = this_month %}
          {% assign same_month = false %}
        {% endif %}

        <span class="date day {% if same_day and same_month %}text-white{% endif %}">{{ this_day }}</span>
        <span class="date month small {% if same_day and same_month %}text-white{% else %}text-muted{% endif %}">
          {{ this_month }}
        </span>
        {% endcomment %}

        <span class="date day">{{ this_day }}</span>
        <span class="date month small text-muted">{{ this_month }}</span>
        <a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a>
      </div>
    </li>
  {% if forloop.last %}
  </ul>
  {% elsif this_year != pre_year %}
  </ul>
  <span class="lead">{{pre_year}}</span>
  <ul class="list-unstyled">
    {% assign last_day = "" %}
    {% assign last_month = "" %}
  {% endif %}
{% endfor %}
</div>