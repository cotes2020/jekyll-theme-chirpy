---
layout: page
# The Archives of posts.
---

{% include lang.html %}

{% assign df_strftime_m = site.data.locales[lang].df.archives.strftime | default: '/ %m' %}
{% assign df_dayjs_m = site.data.locales[lang].df.archives.dayjs | default: '/ MM' %}

<div id="archives" class="pl-xl-3">

{% for post in site.posts %}
  {% capture cur_year %}{{ post.date | date: "%Y" }}{% endcapture %}

  {% if cur_year != last_year %}
    {% unless forloop.first %}</ul>{% endunless %}
    <div class="year lead">{{ cur_year }}</div>
    <ul class="list-unstyled">
    {% assign last_year = cur_year %}
  {% endif %}

  <li>
  {% assign ts = post.date | date: '%s' %}
    <span class="date day" data-ts="{{ ts }}" data-df="DD">{{ post.date | date: "%d" }}</span>
    <span class="date month small text-muted ms-1" data-ts="{{ ts }}" data-df="{{ df_dayjs_m }}">
      {{ post.date | date: df_strftime_m }}
    </span>
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
  </li>

  {% if forloop.last %}</ul>{% endif %}

{% endfor %}

</div>
