---
layout: compress
# The list to be cached by PWA
---

const resource = [
  /* --- CSS --- */
  '{{ "/assets/css/:THEME.css" | replace: ':THEME', site.theme | relative_url }}',

  /* --- PWA --- */
  '{{ "/app.js" | relative_url }}',
  '{{ "/sw.js" | relative_url }}',

  /* --- HTML --- */
  '{{ "/index.html" | relative_url }}',
  '{{ "/404.html" | relative_url }}',

  {% for tab in site.tabs %}
    '{{ tab.url | relative_url }}',
  {% endfor %}

  /* --- Favicons & compressed JS --- */
  {% assign cache_list = site.static_files | where: 'swcache', true  %}
  {% for file in cache_list %}
    '{{ file.path | relative_url }}'{%- unless forloop.last -%},{%- endunless -%}
  {% endfor %}
];

/* The request url with below domain will be cached */
const allowedDomains = [
  {% if site.google_analytics.id != empty and site.google_analytics.id %}
    'www.googletagmanager.com',
    'www.google-analytics.com',
  {% endif %}

  '{{ site.url | split: "//" | last }}',

  {% if site.img_cdn contains '//' and site.img_cdn %}
    '{{ site.img_cdn | split: '//' | last | split: '/' | first }}',
  {% endif %}

  'fonts.gstatic.com',
  'fonts.googleapis.com',
  'cdn.jsdelivr.net',
  'polyfill.io'
];

/* Requests that include the following path will be banned */
const denyUrls = [];
