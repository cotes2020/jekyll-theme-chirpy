---
layout: compress

# The list to be cached by PWA
---

const resource = [

  /* --- CSS --- */
  '{{ "/assets/css/style.css" | relative_url }}',

  /* --- JavaScripts --- */
  {% assign js_path = "/assets/js" | relative_url %}
  '{{ js_path }}/dist/home.min.js',
  '{{ js_path }}/dist/page.min.js',
  '{{ js_path }}/dist/post.min.js',
  '{{ js_path }}/dist/categories.min.js',
  '{{ js_path }}/data/search.json',
  '{{ "/app.js" | relative_url }}',
  '{{ "/sw.js" | relative_url }}',

  /* --- HTML --- */
  '{{ "/index.html" | relative_url }}',
  '{{ "/404.html" | relative_url }}',
  {% for tab in site.tabs %}
    '{{ tab.url }}',
  {% endfor %}

  /* --- Icons --- */
  {% assign icon_url = "/assets/img/favicons" | relative_url %}
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
  '{{ icon_url }}/browserconfig.xml'
];

/* The request url with below domain will be cached */
const allowedDomains = [
  {% if site.google_analytics.id != '' %}
    'www.googletagmanager.com',
    'www.google-analytics.com',
  {% endif %}

  '{{ site.url | split: "//" | last }}',

  'fonts.gstatic.com',
  'fonts.googleapis.com',
  'cdn.jsdelivr.net',
  'polyfill.io'
];

/* Requests that include the following path will be banned */
const denyUrls = [
  {% if site.google_analytics.pv.cache_path %}
    '{{ site.google_analytics.pv.cache_path | absolute_url }}'
  {% endif %}
];
