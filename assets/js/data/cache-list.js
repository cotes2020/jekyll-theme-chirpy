---
layout: compress

# The list to be cached by PWA
# Chirpy v2.2
# https://github.com/cotes2020/jekyll-theme-chirpy
# © 2020 Cotes Chung
# MIT Licensed
---

const include = [

  /*--- CSS ---*/

  '{{ "/assets/css/home.css" | relative_url }}',
  '{{ "/assets/css/categories.css" | relative_url }}',
  '{{ "/assets/css/tags.css" | relative_url }}',
  '{{ "/assets/css/archives.css" | relative_url }}',
  '{{ "/assets/css/page.css" | relative_url }}',
  '{{ "/assets/css/post.css" | relative_url }}',
  '{{ "/assets/css/category-tag.css" | relative_url }}',
  '{{ "/assets/css/lib/bootstrap-toc.min.css" | relative_url }}',

  /*--- Javascripts ---*/

  '{{ "/assets/js/home.min.js" | relative_url }}',
  '{{ "/assets/js/page.min.js" | relative_url }}',
  '{{ "/assets/js/post.min.js" | relative_url }}',

  /*--- HTML ---*/

  /* Tabs */
  {% for tab in site.data.tabs %}
    {% capture item %}
      {%- unless tab.name == 'Home' -%}
        /tabs/{{ tab.name | downcase }}
      {%- endunless -%}
      {{- "/" -}}
    {% endcapture %}
    '{{ item | relative_url }}',
  {% endfor %}

  /*--- Icons ---*/

  {%- capture icon_url -%}
    {{ "/assets/img/favicons" | relative_url }}
  {%- endcapture -%}
  '{{ icon_url }}/favicon.ico',
  '{{ icon_url }}/apple-icon.png',
  '{{ icon_url }}/apple-icon-precomposed.png',
  '{{ icon_url }}/apple-icon-57x57.png',
  '{{ icon_url }}/apple-icon-60x60.png',
  '{{ icon_url }}/apple-icon-72x72.png',
  '{{ icon_url }}/apple-icon-76x76.png',
  '{{ icon_url }}/apple-icon-114x114.png',
  '{{ icon_url }}/apple-icon-120x120.png',
  '{{ icon_url }}/apple-icon-144x144.png',
  '{{ icon_url }}/apple-icon-152x152.png',
  '{{ icon_url }}/apple-icon-180x180.png',
  '{{ icon_url }}/android-icon-192x192.png',
  '{{ icon_url }}/favicon-32x32.png',
  '{{ icon_url }}/favicon-96x96.png',
  '{{ icon_url }}/favicon-16x16.png',
  '{{ icon_url }}/ms-icon-144x144.png',
  '{{ icon_url }}/manifest.json',
  '{{ icon_url }}/browserconfig.xml',

  /*--- Others ---*/

  '{{ "/assets/js/data/search.json" | relative_url }}',
  '{{ "/404.html" | relative_url }}',

  '{{ "/app.js" | relative_url }}',
  '{{ "/sw.js" | relative_url }}'
];

const exclude = [
  {%- if site.google_analytics.pv.proxy_url and site.google_analytics.pv.enabled -%}
    '{{ site.google_analytics.pv.proxy_url }}',
  {%- endif -%}
  '/assets/js/data/pageviews.json',
  '/img.shields.io/'
];
