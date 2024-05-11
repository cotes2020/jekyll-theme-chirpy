---
layout: compress
permalink: '/:path/swconf.js'
# Note that this file will be fetched by the ServiceWorker, so it will not be cached.
---

const swconf = {
  {% if site.pwa.cache.enabled %}
    cacheName: 'chirpy-{{ "now" | date: "%s" }}',

    {%- comment -%} Resources added to the cache during PWA installation. {%- endcomment -%}
    resources: [
      '{{ "/assets/css/:THEME.css" | replace: ':THEME', site.theme | relative_url }}',
      '{{ "/" | relative_url }}',
      {% for tab in site.tabs %}
        '{{- tab.url | relative_url -}}',
      {% endfor %}

      {% assign cache_list = site.static_files | where: 'swcache', true %}
      {% for file in cache_list %}
        '{{ file.path | relative_url }}'{%- unless forloop.last -%},{%- endunless -%}
      {% endfor %}
    ],

    {%- comment -%} The request url with below path will not be cached. {%- endcomment -%}
    denyPaths: [
      {% for path in site.pwa.cache.deny_paths %}
        {% unless path == empty %}
          '{{ path | relative_url }}'{%- unless forloop.last -%},{%- endunless -%}
        {% endunless  %}
      {% endfor %}
    ],
    purge: false
  {% else %}
    purge: true
  {% endif %}
};
