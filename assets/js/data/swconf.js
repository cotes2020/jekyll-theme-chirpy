---
layout: compress
permalink: '/:path/swconf.js'
# Note that this file will be fetched by the ServiceWorker, so it will not be cached.
---

const swconf = {
  {%- if site.pwa.cache.enabled %}
    cacheName: 'chirpy-{{ "now" | date: "%s" }}',

    {%- comment -%} The specific list of URLs automatically cached during PWA installation. {% endcomment %}
    precacheUrls: [
      '{{ "/assets/css/:THEME.css" | replace: ':THEME', site.theme | relative_url }}',
      '{{ "/" | relative_url }}',
      {% for tab in site.tabs -%}
        '{{ tab.url | relative_url }}',
      {% endfor %}

      {%- assign cache_list = site.static_files | where: 'precache', true -%}
      {% for file in cache_list -%}
        '{{- file.path | relative_url -}}'{%- unless forloop.last -%},{% endunless %}
      {% endfor -%}
    ],

    {%- comment -%} Define URLs that should be excluded from the cache. {%- endcomment -%}

    {%- assign deny_urls = '' -%}

    {%- if site.comments.provider -%}
      {%- case site.comments.provider -%}
        {%- when 'disqus' -%}
          {%- assign deny_urls = deny_urls | append: 'disqus\.com::' -%}
        {%- when 'utterances' -%}
          {%- assign deny_urls = deny_urls | append: 'utteranc\.es::' -%}
        {%- when 'giscus' -%}
          {%- assign deny_urls = deny_urls | append: 'giscus\.app::' -%}
      {%- endcase -%}
    {%- endif -%}

    {%- comment -%} Analytics & Pageviews providers {% endcomment -%}

    {%- if site.analytics.google.id -%}
      {%- assign deny_urls = deny_urls | append: 'googletagmanager\.com::' -%}
    {%- endif %}

    {%- if site.analytics.goatcounter.id -%}
      {%- assign deny_urls = deny_urls | append: '\.zgo\.at::' -%}
    {%- endif %}

    {%- if site.analytics.cloudflare.id -%}
      {%- assign deny_urls = deny_urls | append: '\.cloudflareinsights\.com::' -%}
    {%- endif %}

    {%- if site.analytics.fathom.id -%}
      {%- assign deny_urls = deny_urls | append: '\.usefathom\.com::' -%}
    {%- endif -%}

    {%- if site.analytics.umami.id and site.analytics.umami.domain -%}
      {%- assign umami_domain = site.analytics.umami.domain | replace: '.', '\.' -%}
      {%- assign deny_urls = deny_urls | append: umami_domain | append: '::' -%}
    {%- endif -%}

    {%- if site.analytics.matomo.id and site.analytics.matomo.domain -%}
      {%- assign matomo_domain = site.analytics.matomo.domain | replace: '.', '\.' -%}
      {%- assign deny_urls = deny_urls | append: matomo_domain | append: '::' -%}
    {%- endif -%}

    {%- if site.pageviews.provider -%}
      {%- case site.pageviews.provider -%}
        {%- when 'goatcounter' -%}
          {%- assign deny_urls = deny_urls | append: '\.goatcounter\.com::' -%}
      {%- endcase -%}
    {%- endif -%}

    {%- for path in site.pwa.cache.deny_paths -%}
      {%- assign url = path | absolute_url | replace: '/', '\/' -%}
      {%- assign deny_urls = deny_urls | append: url | append: '::' -%}
    {%- endfor %}

    denyUrls: {{ deny_urls | split: '::' | jsonify }},
    purge: false
  {%- else %}
    purge: true
  {%- endif %}
};
